# detection.py

import os
import cv2
import csv
import yaml
import torch
import numpy as np
from datetime import datetime
from collections import deque, defaultdict
from scipy.spatial import distance
import threading

# ─── HYPERPARAMETERS ───────────────────────────────────────────────────────────
distance_threshold = 100   # px for re-identifying the same “Person”
infer_every       = 2     # run YOLO once every 2 frames → ~15 FPS inference

# ─── SHARED 640×480 FRAME BUFFER FOR WEBRTC ─────────────────────────────────────
class LatestFrame:
    """
    Holds exactly one (640×480) BGR frame at any time.
    detection_loop() writes into .buffer; WebRTC’s CameraTrack reads from it.
    """
    buffer = None
    lock   = threading.Lock()

    @classmethod
    def set(cls, frame_640x480):
        with cls.lock:
            cls.buffer = frame_640x480.copy()

    @classmethod
    def get(cls):
        with cls.lock:
            return None if cls.buffer is None else cls.buffer.copy()


# ─── CONFIG & MODEL LOADING ─────────────────────────────────────────────────────
THIS_DIR       = os.path.dirname(__file__)
CONFIG_PATH    = os.path.join(THIS_DIR, "config.json")
ICONS_DIR      = os.path.join(THIS_DIR, "icons")
DETECTIONS_DIR = os.path.join(THIS_DIR, "detections")

# Point to your custom PPE checkpoint, not generic COCO
MODEL_PATH     = os.path.join(THIS_DIR, "model.pt")
assert os.path.exists(MODEL_PATH), "model.pt missing! put your PPE model here."

assert os.path.exists(CONFIG_PATH), "config.json missing!"
os.makedirs(DETECTIONS_DIR, exist_ok=True)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

CONF_THRESHOLD    = config["detection_logic"].get("min_confidence", 0.5)
VIOLATION_CLASSES = [v.lower() for v in config.get("violation_classes", [])]

CSV_LOG    = os.path.join(THIS_DIR, config["log_behavior"]["csv_path"])
LOG_FIELDS = config["log_behavior"]["fields"]
os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)
if not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

print(f"Loaded config: {config.get('name', '<unnamed>')}")


# ─── LOAD YOLOv8n (PPE) ON GPU (FP16 if available) ───────────────────────────────
from ultralytics import YOLO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = YOLO(MODEL_PATH)
model.to(device)
if device.type.startswith("cuda"):
    model.model.half()

print("Using device:", device)
print("Model classes:", model.names)


# ─── PPE CLASS INDICES (match your PPE .pt) ──────────────────────────────────────
PPE_CLASSES = {
    "helmet_OK":  0, "helmet_NOT": 2,
    "mask_OK":    1, "mask_NOT":  3,
    "vest_OK":    7, "vest_NOT":  4,
}

# ─── ICON LOADING ────────────────────────────────────────────────────────────────
icon_size = (60, 60)

def load_icon(fname):
    path = os.path.join(ICONS_DIR, fname)
    if not os.path.exists(path):
        print("Warning: missing icon:", path)
        return None
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # preserve alpha channel
    if icon is None:
        print("Warning: failed to load:", path)
        return None
    return cv2.resize(icon, icon_size)


icons = {
    0: load_icon("helmet_green.png"),
    2: load_icon("helmet_red.png"),
    1: load_icon("mask_green.png"),
    3: load_icon("mask_red.png"),
    7: load_icon("safetyvest_green.png"),
    4: load_icon("safetyvest_red.png"),
}


def overlay_icon(frame, icon, x, y):
    """
    Overlay a 60×60 RGBA icon onto the BGR frame at (x,y). Clips if off-screen.
    """
    if icon is None:
        return
    h, w = icon.shape[:2]
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + w, y1 + h
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return

    roi = frame[y1:y2, x1:x2]
    if icon.shape[2] == 4:
        alpha = icon[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (alpha * icon[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)
    else:
        roi[:, :, :] = icon[:, :, :3]


def overlaps(boxA, boxB, threshold=0.1):
    """
    Return True if boxA overlaps boxB by > threshold fraction of boxA’s area.
    box: (x1,y1,x2,y2)
    """
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = max(1, (ax2 - ax1) * (ay2 - ay1))
    return (inter_area / areaA) > threshold


def draw_corner_lines(frame, box, color, L=20, t=3):
    """
    Draw “corner” lines for a bounding box (x1,y1,x2,y2).
    """
    x1, y1, x2, y2 = box
    # Top-left
    cv2.line(frame, (x1, y1),       (x1 + L, y1), color, t)
    cv2.line(frame, (x1, y1),       (x1, y1 + L), color, t)
    # Top-right
    cv2.line(frame, (x2, y1),       (x2 - L, y1), color, t)
    cv2.line(frame, (x2, y1),       (x2, y1 + L), color, t)
    # Bottom-left
    cv2.line(frame, (x1, y2),       (x1 + L, y2), color, t)
    cv2.line(frame, (x1, y2),       (x1, y2 - L), color, t)
    # Bottom-right
    cv2.line(frame, (x2, y2),       (x2 - L, y2), color, t)
    cv2.line(frame, (x2, y2),       (x2, y2 - L), color, t)


def assign_person_ids(current_boxes, person_tracks, next_id):
    """
    Assign a stable ID to each “Person” box by centroid distance.
    Returns (assigned_map, updated_tracks, updated_next_id).
    """
    assigned = {}
    for pid, track in person_tracks.items():
        if not track:
            continue
        last_box = track[-1]
        lx = (last_box[0] + last_box[2]) / 2
        ly = (last_box[1] + last_box[3]) / 2
        best_idx, best_dist = -1, float("inf")
        for i, box in enumerate(current_boxes):
            if i in assigned:
                continue
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            d = distance.euclidean((lx, ly), (cx, cy))
            if d < best_dist and d < distance_threshold:
                best_dist = d
                best_idx = i
        if best_idx != -1:
            assigned[best_idx] = pid
            person_tracks[pid].append(current_boxes[best_idx])

    # Any boxes not assigned → new IDs
    for i, box in enumerate(current_boxes):
        if i not in assigned:
            person_tracks[next_id] = deque([box], maxlen=5)
            assigned[i] = next_id
            next_id += 1

    # Clean up empty tracks
    to_del = [pid for pid, track in person_tracks.items() if not track]
    for pid in to_del:
        del person_tracks[pid]

    return assigned, person_tracks, next_id


# ─── SHARED STATE (tracking, history) ────────────────────────────────────────────
person_tracks  = {}                             # { person_id: deque([last_boxes]) }
next_person_id = 0                              # next available person ID
status_history = defaultdict(lambda: deque(maxlen=5))


# ─── DETECTION LOOP (runs in its own thread) ─────────────────────────────────────
def detection_loop():
    global person_tracks, next_person_id, status_history

    # Open 640×480 camera @ 30 FPS. We'll downsize to 640×360 for YOLO.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,         30)
    assert cap.isOpened(), "Failed to open camera."

    print("⟳ Detection loop started…")
    frame_count     = 0
    last_annotated  = None  # will hold the last 640×360 annotated frame

    while True:
        ret, full_frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # 1) Downscale → 640×360 for YOLO inference
        small = cv2.resize(full_frame, (640, 360))

        if frame_count % infer_every == 0:
            # 2) Run YOLO on 640×360
            results = model.predict(source=small, conf=CONF_THRESHOLD, verbose=False)

            people = []
            items  = []
            detections_for_logging = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    conf_v = float(box.conf.item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    detections_for_logging.append((cls_id, conf_v, x1, y1, x2, y2))

                    if cls_id == 5:        # “Person” class
                        people.append((x1, y1, x2, y2))
                    elif cls_id in {0,1,2,3,4,7}:  # PPE classes
                        items.append((x1, y1, x2, y2, cls_id))

            # 3) Assign stable IDs to persons
            assigned, person_tracks, next_person_id = assign_person_ids(
                people, person_tracks, next_person_id
            )

            violation_found = False
            violation_label = None
            violation_conf  = 0.0
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 4) Start with a fresh copy of small for annotation
            annotated = small.copy()

            # 5) Draw each person’s bounding box, ID, and stack icons inside
            for idx, (x1_p, y1_p, x2_p, y2_p) in enumerate(people):
                pid = assigned[idx]

                # 5a) Define head/torso regions for PPE overlap test
                head  = (x1_p, y1_p, x2_p, y1_p + (y2_p - y1_p)//3)
                torso = (x1_p, y1_p + (y2_p - y1_p)//3, x2_p, y1_p + 2*(y2_p - y1_p)//3)
                statuses = {"helmet": False, "mask": False, "vest": False}

                # 5b) Check each PPE “OK” box for overlap
                for (ix1, iy1, ix2, iy2, cls_j) in items:
                    if cls_j == PPE_CLASSES["helmet_OK"] and overlaps((ix1,iy1,ix2,iy2), head):
                        statuses["helmet"] = True
                    elif cls_j == PPE_CLASSES["mask_OK"] and overlaps((ix1,iy1,ix2,iy2), head):
                        statuses["mask"] = True
                    elif cls_j == PPE_CLASSES["vest_OK"] and overlaps((ix1,iy1,ix2,iy2), torso):
                        statuses["vest"] = True

                # 5c) Smooth over last 5 frames
                status_history[pid].append(statuses)
                smoothed = {k: any(hist[k] for hist in status_history[pid]) for k in statuses}

                # 5d) Draw corner‐style box around this person
                draw_corner_lines(annotated, (x1_p, y1_p, x2_p, y2_p), (0,255,255))

                # 5e) Put “ID {pid}” just above top-left
                cv2.putText(
                    annotated,
                    f"ID {pid}",
                    (x1_p, max(20, y1_p - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,255),
                    2
                )

                # 5f) Check if any PPE missing → violation
                if not all(smoothed.values()):
                    violation_found = True
                    missing = [k for k,v in smoothed.items() if not v]
                    violation_label = "_".join(missing)
                    for (c_k, conf_k, *_) in detections_for_logging:
                        nm = model.names[c_k].lower()
                        if nm in missing and conf_k > violation_conf:
                            violation_conf = conf_k

                # 5g) Stack three icons **inside** the person’s box
                #     – 5 px of horizontal padding from the right side of the box:
                x_icon = x2_p - icon_size[0] - 5
                y_icon = y1_p + 5

                #    Order: helmet, mask, vest
                icon_order = []
                #   helmet OK → 0, helmet NOT → 2
                icon_order.append(icons[PPE_CLASSES["helmet_OK"]] if smoothed["helmet"] else icons[PPE_CLASSES["helmet_NOT"]])
                #   mask   OK → 1, mask   NOT → 3
                icon_order.append(icons[PPE_CLASSES["mask_OK"]]   if smoothed["mask"]   else icons[PPE_CLASSES["mask_NOT"]])
                #   vest   OK → 7, vest   NOT → 4
                icon_order.append(icons[PPE_CLASSES["vest_OK"]]   if smoothed["vest"]   else icons[PPE_CLASSES["vest_NOT"]])

                for ic in icon_order:
                    if ic is not None:
                        overlay_icon(annotated, ic, x_icon, y_icon)
                    y_icon += icon_size[1] + 5  # stack downward

            # 6) If still no violation, do the “legacy” check
            if not violation_found:
                for (c_k, conf_k, _, _, _, _) in detections_for_logging:
                    nm = model.names[c_k].lower()
                    if nm in VIOLATION_CLASSES and conf_k > violation_conf:
                        violation_found = True
                        violation_label = nm
                        violation_conf  = conf_k

            # 7) Save + log violation if found
            if violation_found:
                fname = f"{violation_label}_{ts}.jpg"
                outp  = os.path.join(DETECTIONS_DIR, fname)
                cv2.imwrite(outp, annotated)
                entry = {
                    "timestamp":  ts,
                    "label":      violation_label,
                    "confidence": f"{violation_conf:.2f}",
                    "status":     "violation"
                }
                with open(CSV_LOG, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(entry)
                print(f"[{frame_count}] Violation saved: {fname} (conf={violation_conf:.2f})")

            # 8) Store this annotated 640×360 for reuse on non-inference frames
            last_annotated = annotated.copy()

        else:
            # Not an inference frame. If we have a last_annotated, keep reusing it:
            if last_annotated is not None:
                annotated = last_annotated.copy()
            else:
                # before the first inference, just show the raw small
                annotated = small.copy()

        # ── 9) Now “pad” annotated(640×360) → 640×480 (top+bottom black bars)
        top_bot_pad = (480 - 360) // 2  # = 60 pixels each
        annotated_padded = cv2.copyMakeBorder(
            annotated,
            top_bot_pad, top_bot_pad,  # top, bottom
            0, 0,                      # left, right
            borderType=cv2.BORDER_CONSTANT,
            value=(0,0,0)
        )  # → result is exactly 640×480

        # 10) Publish into the shared buffer
        LatestFrame.set(annotated_padded)

        # (Loop repeats immediately; CameraTrack in WebRTC will pull as often as it can.)


if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    print("⟳ Detection thread running. LatestFrame.buffer → 640×480 frames soon.")
    while True:
        cv2.waitKey(1000)
