# ==========================================
# 0. 필수 설정 (Colab 시작 시 최초 1회 실행)
# ==========================================
# !pip install ultralytics scipy

# 구글 드라이브 마운트 (창이 뜨면 승인해주세요)
from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
import os
import csv
import time
import unicodedata
from pathlib import Path
from ultralytics import YOLO
from scipy.signal import medfilt

# ==========================================
# 1. 경로 설정 (사용자님의 드라이브 구조에 맞춤)
# ==========================================
DRIVE_ROOT = Path("/content/drive/MyDrive")

# 모델 파일 (드라이브 루트에 바로 있음)
HOMEPLATE_MODEL_PATH = str(DRIVE_ROOT / "homeplate.pt")
BAT_MODEL_PATH = str(DRIVE_ROOT / "best.pt")

# 비디오 폴더 (KBO_edit)
VIDEO_DIR = DRIVE_ROOT / "KBO_edit"

# 결과 저장 경로
OUTPUT_CSV = DRIVE_ROOT / "evaluation_results_final_v5_5.csv"
RESULT_VIDEOS_DIR = DRIVE_ROOT / "KBO_result_videos"

os.makedirs(RESULT_VIDEOS_DIR, exist_ok=True)

# ==========================================
# Utility 함수들 (NFC 정규화 추가)
# ==========================================
def extract_ground_truth(filename):
    # 한글 자소 분리(NFD) 문제를 해결하기 위해 NFC로 정규화
    normalized_name = unicodedata.normalize('NFC', Path(filename).stem)
    if normalized_name.endswith("노스윙"): return "NO SWING"
    elif normalized_name.endswith("스윙"): return "SWING"
    return "UNKNOWN"

def detect_handedness(hp_keypoints):
    if len(hp_keypoints) < 4: return None
    pt0, pt1, back_point = hp_keypoints[0], hp_keypoints[1], hp_keypoints[3]
    if pt0[0] <= 1 or pt1[0] <= 1 or back_point[0] <= 1: return None
    front_avg_x = (pt0[0] + pt1[0]) / 2
    return back_point[0] > front_avg_x

# ==========================================
# 체크스윙 판정 함수 (V5.5 - Physical Filters + 2rd frame)
# ==========================================
def analyze_checkswing(video_path, homeplate_model, bat_model):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return "ERROR", {"error": "영상 열기 실패"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_data = {}
    handedness_votes = []
    hp_pts0, hp_pts1, hp_pts3 = [], [], []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        hp_results = homeplate_model(frame, conf=0.3, verbose=False)
        if hp_results[0].keypoints is not None and len(hp_results[0].keypoints) > 0:
            kpts = hp_results[0].keypoints[0].xy.cpu().numpy()[0]
            if len(kpts) >= 4 and kpts[0][0] > 1 and kpts[1][0] > 1 and kpts[3][0] > 1:
                hp_pts0.append(kpts[0])
                hp_pts1.append(kpts[1])
                hp_pts3.append(kpts[3])
                h = detect_handedness(kpts)
                if h is not None: handedness_votes.append(h)

        bat_results = bat_model(frame, conf=0.1, verbose=False)
        if bat_results[0].keypoints is not None and len(bat_results[0].keypoints) > 0:
            if bat_results[0].boxes and len(bat_results[0].boxes) > 0:
                idx = bat_results[0].boxes.conf.argmax().item()
                bat_kpts = bat_results[0].keypoints[idx].xy.cpu().numpy()[0]
                if len(bat_kpts) > 1 and bat_kpts[0][0] > 1 and bat_kpts[1][0] > 1:
                    frame_data[frame_idx] = {
                        'p0_x': bat_kpts[0][0], 'p0_y': bat_kpts[0][1],
                        'p1_x': bat_kpts[1][0], 'p1_y': bat_kpts[1][1]
                    }
        frame_idx += 1
    cap.release()

    if not frame_data or len(hp_pts3) < 3:
        return "UNDETERMINED", {"reason": "배트/플레이트 감지 부족"}

    is_left_handed = sum(handedness_votes) > len(handedness_votes) / 2 if handedness_votes else True

    # [V5.5] 다이나믹 헤드 매핑
    for f, d in frame_data.items():
        p0x, p0y, p1x, p1y = d['p0_x'], d['p0_y'], d['p1_x'], d['p1_y']
        if is_left_handed:
            if p0x < p1x: d.update({'top_x': p0x, 'top_y': p0y, 'bottom_x': p1x, 'bottom_y': p1y})
            else: d.update({'top_x': p1x, 'top_y': p1y, 'bottom_x': p0x, 'bottom_y': p0y})
        else:
            if p0x > p1x: d.update({'top_x': p0x, 'top_y': p0y, 'bottom_x': p1x, 'bottom_y': p1y})
            else: d.update({'top_x': p1x, 'top_y': p1y, 'bottom_x': p0x, 'bottom_y': p0y})

    f_indices = sorted(frame_data.keys())
    if len(f_indices) > 5:
        txs = medfilt([frame_data[f]['top_x'] for f in f_indices], 5)
        tys = medfilt([frame_data[f]['top_y'] for f in f_indices], 5)
        bxs = medfilt([frame_data[f]['bottom_x'] for f in f_indices], 5)
        bys = medfilt([frame_data[f]['bottom_y'] for f in f_indices], 5)
        for i, f in enumerate(f_indices):
            frame_data[f]['top_x'], frame_data[f]['top_y'] = txs[i], tys[i]
            frame_data[f]['bottom_x'], frame_data[f]['bottom_y'] = bxs[i], bys[i]
            
        for f in range(f_indices[0], f_indices[-1]):
            if f not in frame_data:
                prev_f = max([af for af in f_indices if af < f])
                next_f = min([af for af in f_indices if af > f])
                if next_f - prev_f <= 11:
                    r = (f - prev_f) / (next_f - prev_f)
                    frame_data[f] = {
                        'top_x': frame_data[prev_f]['top_x'] + (frame_data[next_f]['top_x'] - frame_data[prev_f]['top_x']) * r,
                        'top_y': frame_data[prev_f]['top_y'] + (frame_data[next_f]['top_y'] - frame_data[prev_f]['top_y']) * r,
                        'bottom_x': frame_data[prev_f]['bottom_x'] + (frame_data[next_f]['bottom_x'] - frame_data[prev_f]['bottom_x']) * r,
                        'bottom_y': frame_data[prev_f]['bottom_y'] + (frame_data[next_f]['bottom_y'] - frame_data[prev_f]['bottom_y']) * r,
                    }

    m0, m1 = np.median(hp_pts0, axis=0), np.median(hp_pts1, axis=0)
    mid_front = (m0 + m1) / 2
    
    frames = sorted(frame_data.keys())
    velocities = {frames[i]: (frame_data[frames[i]]['top_x'] - frame_data[frames[i-1]]['top_x']) / (frames[i]-frames[i-1]) 
                  for i in range(1, len(frames)) if frames[i]-frames[i-1] <= 3}
    swing_start = -1
    for f, v in velocities.items():
        if (is_left_handed and v < -3.5) or (not is_left_handed and v > 3.5):
            if all((is_left_handed and velocities.get(f+j, 0) < -2) or (not is_left_handed and velocities.get(f+j, 0) > 2) for j in range(3)):
                swing_start = f
                break
    if swing_start == -1: swing_start = frames[0]
    search_range = [f for f in frames if swing_start <= f <= frames[-1]]
    max_frame_idx = min(search_range, key=lambda f: frame_data[f]['top_x']) if is_left_handed else max(search_range, key=lambda f: frame_data[f]['top_x'])

    # 최종 판정 로직 (V5.5 Polish)
    max_consecutive = 0
    current_consecutive = 0
    MIN_BAT_LEN, MAX_BAT_LEN = 60, 150
    
    for f in search_range:
        d = frame_data[f]
        tx, ty, bx, by = d['top_x'], d['top_y'], d['bottom_x'], d['bottom_y']
        bat_len = ((tx - bx)**2 + (ty - by)**2)**0.5
        if not (MIN_BAT_LEN <= bat_len <= MAX_BAT_LEN):
            current_consecutive = 0
            continue
            
        is_swing_f = (tx < bx and ty > by) if is_left_handed else (tx > bx and ty > by)
        if is_swing_f:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
            
    swing_detected = max_consecutive >= 2

    # 시각화 생략 (Colab 속도 향상을 위해 필요한 경우만 켜세요)
    # ... (생략) ...

    return ("SWING" if swing_detected else "NO SWING"), {
        "max_reach_frame": max_frame_idx,
        "handedness": "좌타자" if is_left_handed else "우타자",
        "bat_frames": len(frame_data),
        "total_frames": total_frames
    }

def main():
    print(f"LOADING MODELS (GPU mode enabled if available)...")
    homeplate_model = YOLO(HOMEPLATE_MODEL_PATH)
    bat_model = YOLO(BAT_MODEL_PATH)
    
    video_files = sorted([f for f in VIDEO_DIR.iterdir() if f.suffix.lower() == ".mp4"])
    total = len(video_files)
    print(f"Found {total} mp4 files. Starting evaluation...")

    fieldnames = ['filename', 'ground_truth', 'prediction', 'match', 'handedness', 
                  'max_reach_frame', 'bat_frames', 'total_frames']
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for i, video_file in enumerate(video_files):
        gt = extract_ground_truth(video_file.name)
        start_t = time.time()
        try:
            pred, details = analyze_checkswing(video_file, homeplate_model, bat_model)
            match = "✅" if pred == gt else "❌"
            print(f"[{i+1}/{total}] {video_file.name} | GT: {gt} | Pred: {pred} | {match} ({time.time()-start_t:.1f}s)")
            row = {**{"filename": video_file.name, "ground_truth": gt, "prediction": pred, "match": (pred==gt)}, **details}
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        except Exception as e:
            print(f"Error {video_file.name}: {e}")

if __name__ == "__main__":
    main()
