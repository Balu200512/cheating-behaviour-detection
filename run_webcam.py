from ultralytics import YOLO
import cv2
import time
import threading

# ---------------- CONFIG ----------------
WEIGHT_PATH = r"c:\Users\mudug\Desktop\project\runs\detect\train3\weights\best.pt"
CONF_THRESH = 0.2
IMG_SIZE = 640
ALERT_THRESHOLD = 5.0   # seconds

# Only LEFT & RIGHT are cheating
SUSPICIOUS_SET = {"left", "right"}
NORMAL_SET = {"normal", "forward"}

# ---------------- SOUND (WINDOWS) ----------------
try:
    import winsound
    SOUND_AVAILABLE = True
except:
    SOUND_AVAILABLE = False

# ---------------- LOAD MODEL ----------------
print("Loading model:", WEIGHT_PATH)
model = YOLO(WEIGHT_PATH)

# Read class names safely
try:
    names_map = model.names if hasattr(model, "names") else model.model.names
    if isinstance(names_map, dict):
        class_names = [names_map[i] for i in range(len(names_map))]
    else:
        class_names = list(names_map)
except:
    class_names = ["up", "down", "left", "right", "normal"]

print("Classes:", class_names)

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found")

# ---------------- BEEP THREAD ----------------
beep_thread = None
beep_stop = None

def beep_loop(stop_event):
    while not stop_event.is_set():
        winsound.Beep(1000, 300)
        time.sleep(0.2)

def start_beep():
    global beep_thread, beep_stop
    if not SOUND_AVAILABLE:
        return
    if beep_thread is None or not beep_thread.is_alive():
        beep_stop = threading.Event()
        beep_thread = threading.Thread(target=beep_loop, args=(beep_stop,), daemon=True)
        beep_thread.start()

def stop_beep():
    global beep_thread, beep_stop
    if beep_stop:
        beep_stop.set()
    beep_thread = None
    beep_stop = None

# ---------------- STATE VARIABLES ----------------
suspicious_start = None
alerting = False

print("Detection started. Press 'q' to quit.")

# ---------------- MAIN LOOP ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)

        detected_label = None
        detected_conf = 0.0

        # Pick highest confidence detection
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = class_names[cls_id]

                if conf > detected_conf:
                    detected_conf = conf
                    detected_label = label

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        current_time = time.time()

        # ---------------- LOGIC ----------------
        if detected_label in SUSPICIOUS_SET:
            if suspicious_start is None:
                suspicious_start = current_time
                print(f"[DEBUG] Looking {detected_label}, timer started")

            elapsed = current_time - suspicious_start

        else:
            # NORMAL or no detection
            if suspicious_start is not None:
                print("[DEBUG] Returned to normal, timer reset")

            suspicious_start = None
            elapsed = 0.0

            if alerting:
                print("[ALERT] Stopped")
                stop_beep()
                alerting = False

        # ---------------- ALERT ----------------
        if elapsed >= ALERT_THRESHOLD:
            cv2.putText(frame, "CHEATING ALERT!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            if not alerting:
                print("[ALERT] Triggered")
                start_beep()
                alerting = True

        # ---------------- DISPLAY ----------------
        cv2.putText(frame, f"Status: {detected_label}",
                    (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, f"Elapsed: {elapsed:.1f}s",
                    (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Cheating Behaviour Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    stop_beep()
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")
