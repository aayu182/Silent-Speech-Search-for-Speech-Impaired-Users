import cv2
import os
import time
import urllib.request
import numpy as np

# ====== Settings ======
WORDS = ["weather", "news", "music", "maps", "hello"]
RECORD_SECONDS = 2
COUNTDOWN_SECONDS = 2
DATA_DIR = "data"
FPS = 30
CROP_SIZE = 96        # saved video size (pixels)
VIEW_SIZE = 288       # window size shown on screen
MODEL_PATH = "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
# ======================

# --- Face / mouth detection: MediaPipe (new Tasks API, old API, or OpenCV fallback) ---
detector_name = "OpenCV (fallback)"
legacy_mesh = None
landmarker = None
mp = None

try:
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        # Old MediaPipe (<= 0.10.x)
        legacy_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        detector_name = "MediaPipe (solutions)"
    else:
        # New MediaPipe: Tasks API needs a model file
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(MODEL_PATH):
            print("Face model download ho raha hai (ek baar, ~4 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1)
        landmarker = vision.FaceLandmarker.create_from_options(options)
        detector_name = "MediaPipe (Tasks)"
except Exception as e:
    print("MediaPipe load nahi hua:", e)
    legacy_mesh = None
    landmarker = None

haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

smooth = {"cx": None, "cy": None, "side": None}


def mouth_from_landmarks(lm, w, h):
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in (61, 291, 0, 17)])
    cx, cy = pts.mean(axis=0)
    mouth_w = np.linalg.norm(pts[0] - pts[1])
    return cx, cy, mouth_w * 2.0


def find_mouth(frame):
    """Return (cx, cy, side) of mouth region or None."""
    h, w = frame.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if legacy_mesh is not None:
        res = legacy_mesh.process(rgb)
        if res.multi_face_landmarks:
            return mouth_from_landmarks(res.multi_face_landmarks[0].landmark, w, h)

    elif landmarker is not None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_image)
        if res.face_landmarks:
            return mouth_from_landmarks(res.face_landmarks[0], w, h)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        return x + fw / 2, y + fh * 0.78, fw * 0.55
    return None


def get_mouth_crop(frame):
    """Return CROP_SIZE x CROP_SIZE mouth image, or None if no face."""
    found = find_mouth(frame)
    if found is None:
        return None
    cx, cy, side = found
    a = 0.6  # smoothing (higher = more stable, slower to follow)
    if smooth["cx"] is None:
        smooth["cx"], smooth["cy"], smooth["side"] = cx, cy, side
    else:
        smooth["cx"] = a * smooth["cx"] + (1 - a) * cx
        smooth["cy"] = a * smooth["cy"] + (1 - a) * cy
        smooth["side"] = a * smooth["side"] + (1 - a) * side

    h, w = frame.shape[:2]
    half = smooth["side"] / 2
    x1, x2 = int(max(0, smooth["cx"] - half)), int(min(w, smooth["cx"] + half))
    y1, y2 = int(max(0, smooth["cy"] - half)), int(min(h, smooth["cy"] + half))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE))


def count_videos(word):
    return len([f for f in os.listdir(os.path.join(DATA_DIR, word)) if f.endswith(".mp4")])


def show_mouth(crop, text_top, text_bottom, color=(0, 255, 0)):
    """Show ONLY the mouth crop (never the full face)."""
    if crop is None:
        view = np.zeros((VIEW_SIZE, VIEW_SIZE, 3), dtype=np.uint8)
        cv2.putText(view, "Mouth nahi dikha", (30, VIEW_SIZE // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        view = cv2.resize(crop, (VIEW_SIZE, VIEW_SIZE))
    cv2.putText(view, text_top, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(view, text_bottom, (8, VIEW_SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    cv2.imshow("Mouth Recorder", view)


for w_ in WORDS:
    os.makedirs(os.path.join(DATA_DIR, w_), exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam nahi khul raha. Check karo ki koi aur app use to nahi kar rahi.")

print("Detector:", detector_name)
print("Controls: 1-5 = word chuno | SPACE = record | q = quit")

current = 0
status = "1-5: word chuno | SPACE: record | q: quit"

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    word = WORDS[current]
    crop = get_mouth_crop(frame)
    show_mouth(crop, f"{word} (saved: {count_videos(word)})", status)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if ord("1") <= key <= ord(str(len(WORDS))):
        current = key - ord("1")
        status = f"'{WORDS[current]}' selected. SPACE dabao"

    if key == 32:  # SPACE
        start = time.time()
        while time.time() - start < COUNTDOWN_SECONDS:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            left = COUNTDOWN_SECONDS - int(time.time() - start)
            show_mouth(get_mouth_crop(frame), f"Ready... {left}", word, (0, 255, 255))
            cv2.waitKey(1)

        n = count_videos(word) + 1
        path = os.path.join(DATA_DIR, word, f"{word}_{n:03d}.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                 (CROP_SIZE, CROP_SIZE))
        last = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        lost = 0
        total = 0
        start = time.time()
        while time.time() - start < RECORD_SECONDS:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            crop = get_mouth_crop(frame)
            total += 1
            if crop is None:
                lost += 1
                crop_to_write = last
            else:
                crop_to_write = crop
                last = crop
            writer.write(crop_to_write)
            show_mouth(crop, f"REC: {word} (hoth se bolo)", "bina awaaz ke", (0, 0, 255))
            cv2.waitKey(1)
        writer.release()

        if total > 0 and lost / total > 0.3:
            os.remove(path)
            status = "Mouth kai baar nahi dikha, video delete. Dobara try karo"
        else:
            status = f"Saved: {os.path.basename(path)}"
        print(status)

cap.release()
cv2.destroyAllWindows()