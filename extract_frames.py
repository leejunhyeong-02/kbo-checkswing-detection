import cv2
import os
from pathlib import Path

VIDEO_DIR = Path("data/videos")  # 입력 영상 폴더
OUTPUT_DIR = Path("data/bat_frames")  # 입력 영상 폴더
FRAME_INTERVAL = 5  # 5프레임마다 1장 추출

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_files = sorted([f for f in VIDEO_DIR.iterdir() if f.suffix.lower() == ".mp4"])
total_images = 0

for v_idx, vf in enumerate(video_files):
    cap = cv2.VideoCapture(str(vf))
    if not cap.isOpened():
        print(f"[SKIP] {vf.name}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_idx = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx % FRAME_INTERVAL == 0:
            filename = f"{vf.stem}_f{f_idx:04d}.jpg"
            _, img_encoded = cv2.imencode('.jpg', frame)
            with open(str(OUTPUT_DIR / filename), 'wb') as f:
                f.write(img_encoded)
            saved += 1
        f_idx += 1

    cap.release()
    total_images += saved
    print(f"[{v_idx+1}/{len(video_files)}] {vf.name} | {total_frames}frames -> {saved}images")

print(f"\nDone! Total: {total_images} images saved to {OUTPUT_DIR}")
