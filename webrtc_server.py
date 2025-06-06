# webrtc_server.py

import os
import cv2
import numpy as np
import json
import threading
import asyncio

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

import detection  # we explicitly start its thread below

ROOT = os.path.dirname(__file__)
relay = MediaRelay()


class CameraTrack(VideoStreamTrack):
    """
    Pulls 640×480 BGR frames from detection.LatestFrame.buffer (no resizing needed).
    Throttles to 15 FPS to keep encoding smooth.
    """
    def __init__(self):
        super().__init__()
        self._last_ts = None

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        now = asyncio.get_event_loop().time()

        # Throttle to ~15 FPS
        if self._last_ts is not None:
            elapsed = now - self._last_ts
            if elapsed < (1/15):
                await asyncio.sleep((1/15) - elapsed)

        frame = detection.LatestFrame.get()
        if frame is None:
            black = np.zeros((480, 640, 3), dtype="uint8")
            rgb = cv2.cvtColor(black, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        new_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
        new_frame.pts = pts
        new_frame.time_base = time_base

        self._last_ts = asyncio.get_event_loop().time()
        return new_frame


pcs = set()


async def index(request):
    path = os.path.join(ROOT, "webrtc.html")
    content = open(path, "r").read()
    return web.Response(content_type="text/html", text=content)


async def offer(request):
    params = await request.json()
    offer_sdp  = params["sdp"]
    offer_type = params["type"]

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state_change():
        print("Connection state →", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    # Add our CameraTrack (reads 640×480 annotated frames directly)
    camera = CameraTrack()
    pc.addTrack(relay.subscribe(camera))

    # 1) Apply incoming SDP offer
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    )

    # 2) Create & send back SDP answer (let VP8/VP9 be negotiated)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


def main():
    # ── STEP A: Start detection loop in the same process ─────────────────────────
    threading.Thread(target=detection.detection_loop, daemon=True).start()
    print("⟳ Detection thread started. LatestFrame.buffer → 640×480 frames.")

    # ── STEP B: Launch WebRTC server ─────────────────────────────────────────────
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)

    host = "0.0.0.0"
    port = 8000
    print(f"WebRTC server listening on http://{host}:{port}")
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
