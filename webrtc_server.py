#!/usr/bin/env python3
import os
import cv2
import numpy as np
import asyncio
import threading
import json

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

# ─── GLOBALS & SETUP ───────────────────────────────────────────────────────────

ROOT = os.path.dirname(__file__)
relay = MediaRelay()  # to share one camera track across multiple peers

# ─── CAMERA TRACK (with buffer‐flushing thread) ─────────────────────────────────

class CameraTrack(VideoStreamTrack):
    """
    VideoStreamTrack that continuously reads frames from the webcam in a background thread,
    keeping only the most recent frame.  recv() immediately wraps that latest frame
    into a VideoFrame for WebRTC.
    """

    def __init__(self):
        super().__init__()

        # Open default camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

        self.frame = None
        self.lock = threading.Lock()

        # Start reader thread
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def _reader_thread(self):
        """
        Continuously grab frames from OpenCV, store the latest in self.frame.
        """
        while True:
            ret, img = self.cap.read()
            if not ret:
                continue
            # Store a copy under lock
            with self.lock:
                self.frame = img.copy()

    async def recv(self):
        """
        Called by aiortc when it needs the next frame. We take the very latest frame
        from self.frame, convert to rgb, wrap in VideoFrame, timestamp it, and return it.
        """
        pts, time_base = await self.next_timestamp()

        with self.lock:
            img = self.frame.copy() if self.frame is not None else None

        if img is None:
            # Send a black frame if camera hasn't produced one yet
            black = np.zeros((480, 640, 3), dtype="uint8")
            rgb = cv2.cvtColor(black, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        new_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame

# ─── HTTP ROUTES ───────────────────────────────────────────────────────────────

pcs = set()  # keep track of active PeerConnections

async def index(request):
    """
    Serve webrtc.html, the page containing the JS to negotiate WebRTC.
    """
    path = os.path.join(ROOT, "webrtc.html")
    content = open(path, "r").read()
    return web.Response(content_type="text/html", text=content)

async def offer(request):
    """
    Receive SDP offer from browser, create a low‐latency H.264 answer, and return it.
    """
    params = await request.json()
    offer_sdp = params["sdp"]
    offer_type = params["type"]

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        print("Connection state is", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    # Attach our CameraTrack
    camera = CameraTrack()
    pc.addTrack(relay.subscribe(camera))

    # Apply the incoming offer from the browser
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))

    # Create an SDP answer
    answer = await pc.createAnswer()

    # --- SDP Munging for low‐latency H.264 ---
    sdp_text = answer.sdp

    # 1) Force preference of H264.  Browsers often put VP8 first in m=video; we reorder:
    #    Find the “m=video” line, then ensure “H264/90000” appears before “VP8/90000”.
    sdp_lines = sdp_text.splitlines()
    new_sdp_lines = []
    for line in sdp_lines:
        if line.startswith("m=video"):
            # If this line mentions VP8 first, swap H264 in front
            # Example original: "m=video 9 UDP/TLS/RTP/SAVPF 96 98 100" (96=VP8, 98=VP9,100=H264)
            # We want “100 96 98” so H264 (pt=100) is first
            parts = line.split(" ")
            # Identify payload types (e.g. “96”, “98”, “100”)
            pts = parts[3:]
            # Known mapping: “VP8”=pt 96, “H264” often=pt 100
            if "96" in pts and "100" in pts:
                # Move “100” (H264) in front of “96”:
                pts.insert(0, pts.pop(pts.index("100")))
            new_sdp_lines.append(" ".join(parts[:3] + pts))
        else:
            new_sdp_lines.append(line)
    sdp_text = "\r\n".join(new_sdp_lines) + "\r\n"

    # 2) Insert a low‐bitrate limit to reduce buffering (e.g. 200 kbps):
    sdp_lines = sdp_text.splitlines()
    final_sdp = []
    for line in sdp_lines:
        final_sdp.append(line)
        if line.startswith("a=mid:video"):
            # Right after the video mid, specify:
            final_sdp.append("b=AS:200")  # ~200 kbps max for video
    sdp_text = "\r\n".join(final_sdp) + "\r\n"

    # Apply the modified SDP as our local description
    await pc.setLocalDescription(RTCSessionDescription(sdp=sdp_text, type=answer.type))

    payload = {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }
    return web.Response(content_type="application/json", text=json.dumps(payload))

# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    host = "0.0.0.0"
    port = 8000
    print(f"Starting WebRTC server at http://{host}:{port}")
    web.run_app(app, host=host, port=port)

if __name__ == "__main__":
    main()
