import cv2
import os
import time
from pathlib import Path
from ultralytics import YOLO

VIDEO_DIR = Path("data/videos")  # 평가할 KBO 체크스윙 영상 폴더
MODEL_DIR = Path("models")  # homeplate.pt, best.pt 위치
HOMEPLATE_MODEL_PATH = str(MODEL_DIR / "homeplate.pt")

def main():
    print("Loading Home Plate Model...")
    hp_model = YOLO(HOMEPLATE_MODEL_PATH)
    
    video_files = list(VIDEO_DIR.glob("*.mp4"))
    print(f"Found {len(video_files)} videos. Starting test...")
    
    total_frames = 0
    total_hp_detected = 0
    
    start_time = time.time()
    for idx, vf in enumerate(video_files):
        cap = cv2.VideoCapture(str(vf))
        vf_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if vf_frames == 0:
            continue
            
        hp_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            hp_results = hp_model(frame, conf=0.3, verbose=False)
            if hp_results[0].keypoints is not None and len(hp_results[0].keypoints) > 0:
                kpts = hp_results[0].keypoints[0].xy.cpu().numpy()[0]
                if len(kpts) >= 4 and kpts[0][0] > 1 and kpts[1][0] > 1 and kpts[3][0] > 1:
                    hp_frames += 1
                    
        cap.release()
        total_frames += vf_frames
        total_hp_detected += hp_frames
        print(f"[{idx+1}/{len(video_files)}] {vf.name}: {hp_frames}/{vf_frames} -> {hp_frames/vf_frames*100:.1f}%")

    elapsed = time.time() - start_time
    rate = (total_hp_detected / total_frames * 100) if total_frames > 0 else 0
    print("\n" + "="*50)
    print(f"Total Video Files: {len(video_files)}")
    print(f"Total Frames Analyzed: {total_frames}")
    print(f"Home Plate Detected Frames: {total_hp_detected}")
    print(f"Overall Home Plate Detection Rate: {rate:.2f}%")
    print(f"Time Taken: {elapsed:.1f}s")
    print("="*50)

if __name__ == "__main__":
    main()
