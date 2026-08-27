# ==========================================
# [V5.6 Hybrid] KBO 체크스윙 통합 평가 스크립트
# 정확도 80%+ 목표 & 시각화(영상 저장) 포함 버전
# ==========================================

# 1. 패키지 설치
# !pip install ultralytics scipy

# 2. 구글 드라이브 마운트
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
# [경로 설정] 사용자 드라이브 구조
# ==========================================
DRIVE_ROOT = Path("/content/drive/MyDrive")

# 모델 (루트)
HOMEPLATE_MODEL_PATH = str(DRIVE_ROOT / "homeplate.pt")
BAT_MODEL_PATH = str(DRIVE_ROOT / "best.pt")

# 영상 폴더 (KBO_edit)
VIDEO_DIR = DRIVE_ROOT / "KBO_edit"

# 결과 저장
OUTPUT_CSV = DRIVE_ROOT / "evaluation_results_v5_6_hybrid.csv"
RESULT_VIDEOS_DIR = DRIVE_ROOT / "KBO_result_videos_v5_6"
os.makedirs(RESULT_VIDEOS_DIR, exist_ok=True)

# ==========================================
# 판정 로직 (V5.6 Hybrid: V5.4 민감도 + V5.5 매핑)
# ==========================================
def extract_ground_truth(filename):
    normalized_name = unicodedata.normalize('NFC', Path(filename).stem)
    if "노스윙" in normalized_name: return "NO SWING"
    elif "스윙" in normalized_name: return "SWING"
    return "UNKNOWN"

def detect_handedness(hp_keypoints):
    if len(hp_keypoints) < 4: return None
    pt0, pt1, back_point = hp_keypoints[0], hp_keypoints[1], hp_keypoints[3]
    if pt0[0] <= 1 or pt1[0] <= 1 or back_point[0] <= 1: return None
    front_avg_x = (pt0[0] + pt1[0]) / 2
    return back_point[0] > front_avg_x

def analyze_checkswing(video_path, homeplate_model, bat_model):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return "ERROR", {"reason": "영상 열기 실패"}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 영상 저장 설정
    output_path = RESULT_VIDEOS_DIR / f"{video_path.stem}_result.mp4"
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_data = {}
    handedness_votes = []
    hp_pts0, hp_pts1, hp_pts3 = [], [], []
    frame_idx = 0

    # 1단계: 추론 및 데이터 수집
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        hp_results = homeplate_model(frame, conf=0.3, verbose=False)
        if hp_results[0].keypoints is not None and len(hp_results[0].keypoints) > 0:
            kpts = hp_results[0].keypoints[0].xy.cpu().numpy()[0]
            if len(kpts) >= 4 and kpts[0][0] > 1 and kpts[1][0] > 1 and kpts[3][0] > 1:
                hp_pts0.append(kpts[0]); hp_pts1.append(kpts[1]); hp_pts3.append(kpts[3])
                h = detect_handedness(kpts)
                if h is not None: handedness_votes.append(h)

        bat_results = bat_model(frame, conf=0.1, verbose=False)
        if bat_results[0].keypoints is not None and len(bat_results[0].keypoints) > 0:
            if bat_results[0].boxes and len(bat_results[0].boxes) > 0:
                idx = bat_results[0].boxes.conf.argmax().item()
                bat_kpts = bat_results[0].keypoints[idx].xy.cpu().numpy()[0]
                if len(bat_kpts) > 1 and bat_kpts[0][0] > 1 and bat_kpts[1][0] > 1:
                    frame_data[frame_idx] = {'p0_x': bat_kpts[0][0], 'p0_y': bat_kpts[0][1], 'p1_x': bat_kpts[1][0], 'p1_y': bat_kpts[1][1]}
        frame_idx += 1

    if not frame_data or len(hp_pts3) < 3:
        out.release(); cap.release()
        return "UNDETERMINED", {"reason": "감지 부족"}
        
    is_left_handed = sum(handedness_votes) > len(handedness_votes) / 2 if handedness_votes else True

    # 2단계: 다이나믹 매핑 (V5.5 핵심 로직)
    for f, d in frame_data.items():
        p0x, p0y, p1x, p1y = d['p0_x'], d['p0_y'], d['p1_x'], d['p1_y']
        if is_left_handed:
            if p0x < p1x: d.update({'top_x': p0x, 'top_y': p0y, 'bottom_x': p1x, 'bottom_y': p1y})
            else: d.update({'top_x': p1x, 'top_y': p1y, 'bottom_x': p0x, 'bottom_y': p0y})
        else:
            if p0x > p1x: d.update({'top_x': p0x, 'top_y': p0y, 'bottom_x': p1x, 'bottom_y': p1y})
            else: d.update({'top_x': p1x, 'top_y': p1y, 'bottom_x': p0x, 'bottom_y': p0y})

    # 3단계: 보간 및 필터링
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

    # 4단계: 판정 (V5.6 Hybrid: 길이 필터 완화 + 2프레임 기준)
    m0, m1 = np.median(hp_pts0, axis=0), np.median(hp_pts1, axis=0)
    mid_front = (m0 + m1) / 2
    
    frames = sorted(frame_data.keys())
    velocities = {frames[i]: (frame_data[frames[i]]['top_x'] - frame_data[frames[i-1]]['top_x']) / (frames[i]-frames[i-1]) for i in range(1, len(frames)) if frames[i]-frames[i-1] <= 3}
    swing_start = frames[0]
    for f, v in velocities.items():
        if (is_left_handed and v < -3.5) or (not is_left_handed and v > 3.5):
            if all((is_left_handed and velocities.get(f+j, 0) < -2) or (not is_left_handed and velocities.get(f+j, 0) > 2) for j in range(3)):
                swing_start = f; break
    
    search_range = [f for f in frames if swing_start <= f <= frames[-1]]
    max_frame_idx = min(search_range, key=lambda f: frame_data[f]['top_x']) if is_left_handed else max(search_range, key=lambda f: frame_data[f]['top_x'])

    max_consecutive = 0
    current_consecutive = 0
    # [V5.6] 길이 필터 완화 (그루기 시각화나 블러링 대응)
    MIN_BAT_LEN, MAX_BAT_LEN = 40, 300 
    
    for f in search_range:
        d = frame_data[f]
        tx, ty, bx, by = d['top_x'], d['top_y'], d['bottom_x'], d['bottom_y']
        bat_len = ((tx - bx)**2 + (ty - by)**2)**0.5
        if not (MIN_BAT_LEN <= bat_len <= MAX_BAT_LEN):
            current_consecutive = 0; continue
            
        is_swing_f = (tx < bx and ty > by) if is_left_handed else (tx > bx and ty > by)
        if is_swing_f:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else: current_consecutive = 0
            
    swing_detected = max_consecutive >= 2 # V5.4의 고정확도 민감도 복구

    # 5단계: 시각화 및 영상 저장
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        # 홈플레이트 가이드라인
        cv2.line(frame, (int(mid_front[0]), 0), (int(mid_front[0]), height), (0, 255, 255), 3)
        
        # 배트 표시
        if f_idx in f_indices: # 보간된 프레임 포함
            d = frame_data[f_idx]
            tx, ty, bx, by = d['top_x'], d['top_y'], d['bottom_x'], d['bottom_y']
            cv2.line(frame, (int(tx), 0), (int(tx), height), (255, 0, 0), 2) # Top 가이드
            cv2.line(frame, (int(bx), 0), (int(bx), height), (0, 255, 0), 2) # Bottom 가이드
            cv2.line(frame, (int(tx), int(ty)), (int(bx), int(by)), (0, 0, 255), 4) # 배트 몸체
            cv2.circle(frame, (int(tx), int(ty)), 12, (255, 0, 0), -1) # 배트 끝 (Blue)
            cv2.circle(frame, (int(bx), int(by)), 15, (0, 255, 0), -1) # 배트 손잡이 (Green)

        # HUD 표시
        color = (0, 0, 255) if swing_detected else (0, 255, 0)
        status_text = "SWING!" if swing_detected else "NO SWING"
        cv2.putText(frame, status_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 6)
        if f_idx >= max_frame_idx and abs(f_idx - max_frame_idx) < 10:
            cv2.putText(frame, "MAX REACH!", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5)
            
        out.write(frame)

    cap.release()
    out.release()
    return ("SWING" if swing_detected else "NO SWING"), {"max_reach_frame": max_frame_idx, "handedness": "좌타자" if is_left_handed else "우타자", "total_frames": total_frames}

def main():
    print(f"모델 로딩 중...")
    homeplate_model = YOLO(HOMEPLATE_MODEL_PATH)
    bat_model = YOLO(BAT_MODEL_PATH)
    
    video_files = sorted([f for f in VIDEO_DIR.iterdir() if f.suffix.lower() == ".mp4"])
    total = len(video_files)
    print(f"총 {total}개 영상 분석 및 시각화 저장 시작...")

    fieldnames = ['filename', 'ground_truth', 'prediction', 'match', 'handedness', 'max_reach_frame', 'total_frames']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    tp, tn, fp, fn = 0, 0, 0, 0
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
            if pred == gt:
                if gt == "SWING": tp += 1
                else: tn += 1
            else:
                if gt == "SWING": fn += 1
                else: fp += 1
        except Exception as e: print(f"오류 {video_file.name}: {e}")

    acc = (tp + tn) / total * 100 if total > 0 else 0
    print(f"\n========================================\n[V5.6 하이브리드] 최종 성적\n========================================\n정확도: {acc:.2f}%\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n========================================")

if __name__ == "__main__":
    main()
