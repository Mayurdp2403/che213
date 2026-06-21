import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
import csv
import time

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION — only edit this section
# ══════════════════════════════════════════════════════════════════
CONFIG = {
    # Put your video file name here to test it, or use 0/1 for a live webcam
    "video_source"    : r"C:\Users\Ravi Kasotiya\Videos\WhatsApp Video 2026-04-04 at 4.17.53 PM.mp4", 
    "frame_width"     : 640,
    "frame_height"    : 480,

    # --- Tripwire ---
    "tripwire_y"      : 184,             # horizontal counting line (click to reposition)

# --- Region of Interest (ROI) ---
# --- Region of Interest (ROI) ---
    # Assuming the code squashes this vertical video into a 640-pixel wide window:
    # The fluid column takes up about the middle 60% of the frame.
    "roi_left"        : 140,             # Crops out the left black framing
    "roi_right"       : 500,             # Crops out the right black framing

    # --- Drop/Bubble detection ---
    "min_area"        : 15,              # Set low because the bubbles are quite small
    "max_area"        : 1000,            # Prevents giant flashes of light from counting
    "min_aspect"      : 60,              # x100 (0.6 ratio). Filters out the tall, thin vertical glare stripe
    "max_aspect"      : 150,             # x100 (1.5 ratio). Bubbles are relatively round

    # --- Image Processing & Morphology ---
    "use_clahe"       : 1,               # Keep ON. The fluid is bright; this will make the dark bubble edges highly visible.
    "blur_size"       : 7,               # Very light blur. 21 (your old setting) would completely erase these tiny bubbles.
    "dilate_iter"     : 2,               # Light dilation just to make the tracking points solid.
    "erode_iter"      : 0,               # Keep at 0 to protect the small objects.

    # --- Tracker ---
    "max_disappeared" : 20,              # frames a drop can vanish before being dropped
    "max_distance"    : 150,             # Max pixels a drop can move in 1 frame

    # --- Output ---
    "log_file"        : "bubble_log.csv",
}

# ══════════════════════════════════════════════════════════════════
#  CENTROID TRACKER
# ══════════════════════════════════════════════════════════════════
class CentroidTracker:
    def __init__(self, max_disappeared=20, max_distance=150):
        self.next_id         = 0
        self.objects         = OrderedDict()   # id → (cx, cy)
        self.disappeared     = OrderedDict()   # id → frames missing
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance    

    def register(self, centroid):
        self.objects[self.next_id]     = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, input_rects):
        if len(input_rects) == 0:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        new_centroids = np.array(
            [(x + w // 2, y + h // 2) for (x, y, w, h) in input_rects],
            dtype="float"
        )

        if len(self.objects) == 0:
            for c in new_centroids:
                self.register(tuple(c))
            return self.objects

        obj_ids   = list(self.objects.keys())
        obj_cents = list(self.objects.values())
        D         = dist.cdist(np.array(obj_cents), new_centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            
            if D[r, c] > self.max_distance:
                continue

            obj_id = obj_ids[r]
            self.objects[obj_id]     = tuple(new_centroids[c])
            self.disappeared[obj_id] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(D.shape[0])) - used_rows:
            obj_id = obj_ids[r]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        for c in set(range(D.shape[1])) - used_cols:
            self.register(tuple(new_centroids[c]))

        return self.objects


# ══════════════════════════════════════════════════════════════════
#  MOG2 BACKGROUND SUBTRACTION (UPDATED FOR BUBBLES)
# ══════════════════════════════════════════════════════════════════
def detect_moving_blobs(frame, bg_subtractor, params):
    # 1. Apply Region of Interest (ROI) Crop
    mask_roi = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.rectangle(mask_roi, (params["roi_left"], 0), 
                            (params["roi_right"], frame.shape[0]), 
                            255, -1)
    
    # 2. Convert to grayscale & isolate ROI
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.bitwise_and(gray, gray, mask=mask_roi)

    # 3. Contrast Enhancement (CLAHE) - Makes faint bubble edges much darker/visible
    if params["use_clahe"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_roi = clahe.apply(gray_roi)

    # 4. Heavy Blur - Merges the top/bottom edges of the hollow bubble
    b_size = params["blur_size"]
    b_size = b_size if b_size % 2 != 0 else b_size + 1 # Ensure odd number
    blurred = cv2.GaussianBlur(gray_roi, (b_size, b_size), 0)

    # 5. Apply MOG2
    motion_mask = bg_subtractor.apply(blurred)
    _, motion_mask = cv2.threshold(motion_mask, 200, 255, cv2.THRESH_BINARY)

    # 6. Morphology (Cleanup)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) 
    
    # Close gaps inside the hollow bubble before eroding
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel_large) 
    motion_mask = cv2.erode(motion_mask,  kernel_small, iterations=params["erode_iter"])
    motion_mask = cv2.dilate(motion_mask, kernel_large, iterations=params["dilate_iter"])

    # 7. Find Outlines
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 8. Filter by Area AND Aspect Ratio
    valid_rects = []
    for c in contours:
        area = cv2.contourArea(c)
        if params["min_area"] <= area <= params["max_area"]:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / float(max(1, h)) # prevent div by zero
            
            # Check if the shape is relatively square/round, not a long sliver of glare
            if params["min_aspect"] <= aspect_ratio <= params["max_aspect"]:
                valid_rects.append((x, y, w, h))

    return valid_rects, motion_mask


# ══════════════════════════════════════════════════════════════════
#  TRACKBAR HELPERS
# ══════════════════════════════════════════════════════════════════
def nothing(_): pass

def create_trackbars(win):
    cv2.createTrackbar("Tripwire Y",  win, CONFIG["tripwire_y"],  CONFIG["frame_height"], nothing)
    cv2.createTrackbar("ROI Left",    win, CONFIG["roi_left"],    CONFIG["frame_width"], nothing)
    cv2.createTrackbar("ROI Right",   win, CONFIG["roi_right"],   CONFIG["frame_width"], nothing)
    cv2.createTrackbar("Min Area",    win, CONFIG["min_area"],    1000,  nothing) 
    cv2.createTrackbar("Max Area",    win, CONFIG["max_area"],    50000, nothing)
    cv2.createTrackbar("Min Aspect",  win, CONFIG["min_aspect"],  300, nothing) # x100
    cv2.createTrackbar("Max Aspect",  win, CONFIG["max_aspect"],  300, nothing) # x100
    cv2.createTrackbar("CLAHE On/Off",win, CONFIG["use_clahe"],   1, nothing)
    cv2.createTrackbar("Blur Size",   win, CONFIG["blur_size"],   21, nothing)
    cv2.createTrackbar("Dilate Iter", win, CONFIG["dilate_iter"], 10,    nothing)
    cv2.createTrackbar("Erode Iter",  win, CONFIG["erode_iter"],  5,     nothing)

def read_trackbars(win):
    return {
        "tripwire_y"  : cv2.getTrackbarPos("Tripwire Y",  win),
        "roi_left"    : cv2.getTrackbarPos("ROI Left",    win),
        "roi_right"   : max(cv2.getTrackbarPos("ROI Left", win) + 10, cv2.getTrackbarPos("ROI Right", win)), # Ensure right > left
        "min_area"    : max(1, cv2.getTrackbarPos("Min Area",  win)),
        "max_area"    : cv2.getTrackbarPos("Max Area",    win),
        "min_aspect"  : cv2.getTrackbarPos("Min Aspect",  win) / 100.0,
        "max_aspect"  : cv2.getTrackbarPos("Max Aspect",  win) / 100.0,
        "use_clahe"   : bool(cv2.getTrackbarPos("CLAHE On/Off", win)),
        "blur_size"   : max(1, cv2.getTrackbarPos("Blur Size", win)),
        "dilate_iter" : max(0, cv2.getTrackbarPos("Dilate Iter", win)), 
        "erode_iter"  : max(0, cv2.getTrackbarPos("Erode Iter",  win)), 
    }


# ══════════════════════════════════════════════════════════════════
#  MOUSE CALLBACK
# ══════════════════════════════════════════════════════════════════
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        CONFIG["tripwire_y"] = y
        print(f"  [TRIPWIRE] Moved to Y = {y}")


# ══════════════════════════════════════════════════════════════════
#  HUD OVERLAY (UPDATED WITH ROI LINES)
# ══════════════════════════════════════════════════════════════════
def draw_hud(frame, drop_count, fps, tw, tracked, frame_no, n_blobs, params):
    h, w = frame.shape[:2]

    # Tripwire line (orange)
    cv2.line(frame, (0, tw), (w, tw), (0, 165, 255), 2)
    cv2.putText(frame, f"TRIPWIRE Y={tw}", (w - 185, tw - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # ROI Lines (Blue)
    rl = params["roi_left"]
    rr = params["roi_right"]
    cv2.line(frame, (rl, 0), (rl, h), (255, 0, 0), 2)
    cv2.line(frame, (rr, 0), (rr, h), (255, 0, 0), 2)
    cv2.putText(frame, "ROI Left", (rl + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(frame, "ROI Right", (rr + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Centroid dot + ID for each tracked drop
    for obj_id, (cx, cy) in tracked.items():
        cx, cy = int(cx), int(cy)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"#{obj_id}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # Semi-transparent info panel
    panel = frame.copy()
    cv2.rectangle(panel, (8, 8), (355, 115), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"Bubble Count : {drop_count}", (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"Blobs       : {n_blobs}",    (18, 67),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)
    cv2.putText(frame, f"Frame       : {frame_no}",   (18, 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS         : {fps:.1f}",    (18, 107),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print(f"  Connecting to source: {CONFIG['video_source']}...")
    
    cap = cv2.VideoCapture(CONFIG["video_source"])
    
    # If using a webcam, set properties. (Ignored by video files)
    if isinstance(CONFIG["video_source"], int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"  [ERROR] Cannot open source {CONFIG['video_source']}")
        return

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"  Source connected ✓")
    print(f"  FPS        : {vid_fps:.1f}")

    # ── Initialize MOG2 Background Subtractor ─────────────────────
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=10, detectShadows=False)

    # ── Tracker & counting state ──────────────────────────────────
    tracker        = CentroidTracker(CONFIG["max_disappeared"], CONFIG["max_distance"])
    prev_positions = {}    
    crossed_ids    = set() 
    drop_count     = 0
    frame_no       = 0

    # ── CSV log ───────────────────────────────────────────────────
    log_f   = open(CONFIG["log_file"], "w", newline="")
    log_csv = csv.writer(log_f)
    log_csv.writerow(["bubble_no", "timestamp", "frame", "cx", "cy"])

    # ── Windows & controls ────────────────────────────────────────
    WIN_MAIN = "Bubble Counter  [Q=quit  R=reset  P=pause  click=tripwire]"
    WIN_MASK = "Motion Mask  (white = moving bubble)"
    WIN_CTRL = "Controls"
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 480, 450)
    create_trackbars(WIN_CTRL)
    cv2.setMouseCallback(WIN_MAIN, on_mouse)

    delay  = max(1, int(1000 / vid_fps))
    t_prev = time.time()
    fps    = 0.0

    print("\n  Controls:")
    print("  Q          → quit")
    print("  R          → reset counter")
    print("  P          → pause / resume")
    print("  [ / ]      → slow / fast playback")
    print("  Left-click → move tripwire to clicked row")
    print("\n  Watch the MOTION MASK window and adjust Aspect Ratios to filter glare.")

    # ── Main loop ─────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()

        # If video ends, loop it for testing purposes
        if not ret:
            print("  [INFO] Video ended. Looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (CONFIG["frame_width"], CONFIG["frame_height"]))
        frame_no += 1

        # Read live parameter adjustments from trackbars
        params = read_trackbars(WIN_CTRL)
        
        # Sync trackbar tripwire with mouse clicks
        if params["tripwire_y"] != CONFIG["tripwire_y"]:
            cv2.setTrackbarPos("Tripwire Y", WIN_CTRL, CONFIG["tripwire_y"])
            params["tripwire_y"] = CONFIG["tripwire_y"]

        # ── DETECT moving blobs via MOG2 ──────────────────────────
        valid_rects, motion_mask = detect_moving_blobs(frame, bg_subtractor, params)

        # Draw green bounding boxes around detected blobs
        for (x, y, w, h) in valid_rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ── UPDATE tracker ────────────────────────────────────────
        tracked = tracker.update(valid_rects)

        # ── TRIPWIRE COUNTING ─────────────────────────────────────
        tw = params["tripwire_y"]

        for obj_id, (cx, cy) in tracked.items():
            cy     = int(cy)
            cx     = int(cx)
            prev_y = prev_positions.get(obj_id, cy)

            # DOWNWARD crossing
            if prev_y < tw <= cy and obj_id not in crossed_ids:
                drop_count += 1
                crossed_ids.add(obj_id)
                ts = time.strftime("%H:%M:%S")
                print(f"  ↓ BUBBLE #{drop_count:03d}  id={obj_id}  "
                      f"frame={frame_no}  pos=({cx},{cy})  [{ts}]")
                log_csv.writerow([drop_count, ts, frame_no, cx, cy])
                log_f.flush()

            prev_positions[obj_id] = cy

        # Clean up state for disappeared drops
        for dead_id in list(prev_positions.keys()):
            if dead_id not in tracked:
                del prev_positions[dead_id]

        # Remove counted IDs that no longer exist (prevents memory leak)
        crossed_ids &= set(tracked.keys())

        # FPS calculation
        t_now  = time.time()
        fps    = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-9)
        t_prev = t_now

        # Draw HUD & show windows
        draw_hud(frame, drop_count, fps, tw, tracked, frame_no, len(valid_rects), params)
        cv2.imshow(WIN_MAIN, frame)
        cv2.imshow(WIN_MASK, motion_mask)

        # Key controls
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            drop_count     = 0
            crossed_ids    = set()
            prev_positions = {}
            tracker        = CentroidTracker(CONFIG["max_disappeared"], CONFIG["max_distance"])
            print("  [RESET] Counter cleared.")
        elif key == ord('p'):
            print("  [PAUSED]  Press P to resume...")
            while True:
                k = cv2.waitKey(50) & 0xFF
                if k == ord('p'):
                    print("  [RESUMED]")
                    break
                if k == ord('q'):
                    cap.release()
                    log_f.close()
                    cv2.destroyAllWindows()
                    return
        elif key == ord(']'):
            delay = max(1, delay - 10)
        elif key == ord('['):
            delay = min(300, delay + 10)

    # ── Cleanup ───────────────────────────────────────────────────
    cap.release()
    log_f.close()
    cv2.destroyAllWindows()
    print(f"\n  ══ Final bubble count : {drop_count} ══")

if __name__ == "__main__":
    main()