# ==========================================
# 홈플레이트 프레임 단위 탐지율 측정 스크립트
# Google Colab 환경용
# ==========================================

# 1. 패키지 설치
# !pip install ultralytics

# 2. 구글 드라이브 마운트
from google.colab import drive
drive.mount('/content/drive')

import cv2
import os
import csv
import time
import unicodedata
from pathlib import Path
from ultralytics import YOLO

# ==========================================
# [경로 설정] - 기존 평가 스크립트와 동일 구조
# ==========================================
DRIVE_ROOT = Path("/content/drive/MyDrive")

# 홈플레이트 모델
HOMEPLATE_MODEL_PATH = str(DRIVE_ROOT / "homeplate.pt")

# 영상 폴더 (편집된 체크스윙 영상들)
VIDEO_DIR = DRIVE_ROOT / "KBO_edit"

# 결과 CSV
OUTPUT_CSV = DRIVE_ROOT / "homeplate_detection_rate.csv"

# ==========================================
# 홈플레이트 탐지율 측정
# ==========================================
def measure_homeplate_detection(video_path, model):
    """영상의 전체 프레임에서 홈플레이트 탐지율을 계산"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0

    total_frames = 0
    detected_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # 홈플레이트 모델 추론 (conf=0.3, 기존 평가와 동일)
        results = model(frame, conf=0.3, verbose=False)

        # 키포인트가 유효하게 감지되었는지 확인
        if (results[0].keypoints is not None
            and len(results[0].keypoints) > 0):
            kpts = results[0].keypoints[0].xy.cpu().numpy()[0]
            # 최소 4개 키포인트가 유효한 좌표(>1)를 가져야 탐지 성공
            if (len(kpts) >= 4
                and kpts[0][0] > 1
                and kpts[1][0] > 1
                and kpts[3][0] > 1):
                detected_frames += 1

    cap.release()
    return detected_frames, total_frames


def main():
    print("홈플레이트 모델 로딩 중...")
    model = YOLO(HOMEPLATE_MODEL_PATH)

    video_files = sorted([
        f for f in VIDEO_DIR.iterdir()
        if f.suffix.lower() == ".mp4"
    ])
    total_videos = len(video_files)
    print(f"총 {total_videos}개 영상에서 홈플레이트 탐지율 측정 시작...\n")

    # CSV 헤더
    fieldnames = ['filename', 'detected_frames', 'total_frames', 'detection_rate']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    all_detected = 0
    all_total = 0

    for i, video_file in enumerate(video_files):
        start_t = time.time()
        detected, total = measure_homeplate_detection(video_file, model)
        rate = (detected / total * 100) if total > 0 else 0

        all_detected += detected
        all_total += total

        elapsed = time.time() - start_t
        print(f"[{i+1}/{total_videos}] {video_file.name} | "
              f"{detected}/{total} frames ({rate:.1f}%) | {elapsed:.1f}s")

        # CSV에 영상별 결과 추가
        row = {
            'filename': video_file.name,
            'detected_frames': detected,
            'total_frames': total,
            'detection_rate': f"{rate:.2f}%"
        }
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

    # 최종 결과 출력
    overall_rate = (all_detected / all_total * 100) if all_total > 0 else 0
    print(f"\n{'='*50}")
    print(f"[최종 결과] 홈플레이트 프레임 단위 탐지율")
    print(f"{'='*50}")
    print(f"전체 프레임: {all_total}")
    print(f"탐지 프레임: {all_detected}")
    print(f"평균 탐지율: {overall_rate:.2f}%")
    print(f"{'='*50}")
    print(f"\n결과 CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
