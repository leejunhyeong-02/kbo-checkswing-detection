"""
KBO 체크스윙 데이터셋 모델 평가 스크립트 (V4.9)
=========================================
V4.9: 사용자 선호 시각화 스타일(3월 16일 버전) 완전 복구
- 3중 수직 가이드라인 (Yellow: Plate, Green: Bot, Blue: Top)
- 하단 포인트 보라색 강조 및 전용 범례 박스
- V4.7+ 기하학적 판정 로직 유지 (정확도 + 비주얼 최적화)

[사용법]
1. 아래 '경로 설정' 섹션에서 본인 환경에 맞게 경로를 수정하세요.
2. MODEL_DIR 안에 homeplate.pt, best.pt 모델 파일을 위치시키세요.
3. VIDEO_DIR 안에 평가할 체크스윙 영상(.mp4)을 위치시키세요.
   - 파일명 규칙: *_스윙.mp4 (Swing) / *_노스윙.mp4 (No Swing)
"""

import cv2
import numpy as np
import os
import csv
import time
from pathlib import Path
from ultralytics import YOLO
from scipy.signal import medfilt

# ==========================================
# ★ 경로 설정 (실행 전 본인 환경에 맞게 수정)
# ==========================================

# 모델 파일 경로 (homeplate.pt, best.pt 위치)
MODEL_DIR = Path("models")

# 평가할 체크스윙 영상 폴더 경로
VIDEO_DIR = Path("data/videos")

# 결과 저장 경로 (자동 생성됨)
OUTPUT_CSV = Path("evaluation_results.csv")
RESULT_VIDEOS_DIR = Path("result_videos")

# 모델 파일 경로
HOMEPLATE_MODEL_PATH = str(MODEL_DIR / "homeplate.pt")
BAT_MODEL_PATH = str(MODEL_DIR / "best.pt")

os.makedirs(RESULT_VIDEOS_DIR, exist_ok=True)

BAT_BOTTOM_IDX = 0  # 손목
BAT_TOP_IDX = 1     # 배트 끝
MAX_REACH_DISPLAY_FRAMES = 30
MAX_REACH_TEXT_FRAMES = 10

# ==========================================
# Utility 함수들
# ==========================================
def extract_ground_truth(filename):
    name = Path(filename).stem
    if name.endswith("_노스윙"): return "NO SWING"
    elif name.endswith("_스윙"): return "SWING"
    return "UNKNOWN"

def detect_handedness(hp_keypoints):
    if len(hp_keypoints) < 4: return None
    pt0, pt1, back_point = hp_keypoints[0], hp_keypoints[1], hp_keypoints[3]
    if pt0[0] <= 1 or pt1[0] <= 1 or back_point[0] <= 1: return None
    front_avg_x = (pt0[0] + pt1[0]) / 2
    return back_point[0] > front_avg_x

# ==========================================
# 체크스윙 판정 함수 (V4.9)
# ==========================================
def analyze_checkswing(video_path, homeplate_model, bat_model):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return "ERROR", {"error": "영상 열기 실패"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1단계: 정보 수집
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
                        'p1_x': bat_kpts[1][0], 'p1_y': bat_kpts[1][1],
                        'conf': bat_results[0].boxes.conf[idx].item()
                    }
        frame_idx += 1
    cap.release()

    if not frame_data or len(hp_pts3) < 3:
        return "UNDETERMINED", {"reason": "배트 또는 플레이트 감지 부족"}

    # 타석 판정 (조기 판정하여 다이나믹 매핑에 사용)
    is_left_handed = sum(handedness_votes) > len(handedness_votes) / 2 if handedness_votes else True

    # [V5.5] 다이나믹 헤드 매핑 적용 (데이터 수집 직후 수행)
    for f, d in frame_data.items():
        p0x, p0y, p1x, p1y = d['p0_x'], d['p0_y'], d['p1_x'], d['p1_y']
        if is_left_handed:
            # 좌타자: X가 더 작은 점이 무조건 배트 끝(Top)
            if p0x < p1x:
                d.update({'top_x': p0x, 'top_y': p0y, 'bottom_x': p1x, 'bottom_y': p1y})
            else:
                d.update({'top_x': p1x, 'top_y': p1y, 'bottom_x': p0x, 'bottom_y': p0y})
        else:
            # 우타자: X가 더 큰 점이 무조건 배트 끝(Top)
            if p0x > p1x:
                d.update({'top_x': p0x, 'top_y': p0y, 'bottom_x': p1x, 'bottom_y': p1y})
            else:
                d.update({'top_x': p1x, 'top_y': p1y, 'bottom_x': p0x, 'bottom_y': p0y})

    # 배트 필터링 및 보간 (이제 top_x... 키가 모두 복구됨)
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
                        'interpolated': True
                    }

    # 홈플레이트 기하 구조 (V4.7/V4.8.1 계승)
    m0, m1, m3 = np.median(hp_pts0, axis=0), np.median(hp_pts1, axis=0), np.median(hp_pts3, axis=0)
    mid_front = (m0 + m1) / 2
    v_swing = m1 - m0
    if v_swing[0] < 0: v_swing = -v_swing

    # MAX REACH
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
    
    # [V5.5] 이미 매핑되었으므로 top_x를 바로 사용 가능
    max_frame_idx = min(search_range, key=lambda f: frame_data[f]['top_x']) if is_left_handed else max(search_range, key=lambda f: frame_data[f]['top_x'])

    # 3단계: 최종 판정 (V5.5 - Layered Physical Filters)
    max_consecutive = 0
    current_consecutive = 0
    MIN_BAT_LEN, MAX_BAT_LEN = 60, 150
    
    for f in search_range:
        d = frame_data[f]
        tx, ty, bx, by = d['top_x'], d['top_y'], d['bottom_x'], d['bottom_y']
        
        # [V5.5 필터 1] 배트 길이 검증
        bat_len = ((tx - bx)**2 + (ty - by)**2)**0.5
        if not (MIN_BAT_LEN <= bat_len <= MAX_BAT_LEN):
            current_consecutive = 0
            continue
            
        # [V5.5 필터 2] X-Y 통합 로직 (매핑된 좌표 사용)
        if is_left_handed:
            is_swing_f = (tx < bx) and (ty > by)
        else:
            is_swing_f = (tx > bx) and (ty > by)
            
        if is_swing_f:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
            
    # 최종 판정: 물리적 조건이 연속으로 2프레임 이상 만족될 때 스윙 인정 (Sensitivity 최적화)
    swing_detected = max_consecutive >= 2
# 4단계: 시각화 (V4.9: 3월 16일 버전 완전 복구)
    output_path = RESULT_VIDEOS_DIR / f"{video_path.stem}_result.mp4"
    cap = cv2.VideoCapture(str(video_path))
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    legend_x = width - 350
    legend_y_start = 30
    legend_line_height = 40

    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        # [V4.9] 수직 판정선 (사용자 선호 스타일)
        cv2.line(frame, (int(mid_front[0]), 0), (int(mid_front[0]), height), (0, 255, 255), 3) # Yellow: Plate

        # 배트 표시 (V5.5 다이나믹 매핑된 좌표 시각화)
        if f_idx in frame_data:
            d = frame_data[f_idx]
            tx, ty, bx, by = d['top_x'], d['top_y'], d['bottom_x'], d['bottom_y']
            
            # 수직 가이드라인
            cv2.line(frame, (int(tx), 0), (int(tx), height), (255, 0, 0), 2) # Blue: Top
            cv2.line(frame, (int(bx), 0), (int(bx), height), (0, 255, 0), 2) # Green: Bottom
            
            # 배트 구조
            cv2.line(frame, (int(tx), int(ty)), (int(bx), int(by)), (0, 0, 255), 4)
            cv2.circle(frame, (int(tx), int(ty)), 12, (255, 0, 0), -1)
            cv2.circle(frame, (int(bx), int(by)), 15, (0, 255, 0), -1)
            cv2.circle(frame, (int(bx), int(by)), 18, (255, 0, 255), 3) # Magenta Circle

        # HUD 텍스트
        if f_idx >= max_frame_idx:
            color = (0, 0, 255) if swing_detected else (0, 255, 0)
            status_text = "SWING!" if swing_detected else "NO SWING"
            cv2.putText(frame, status_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 6)
            if abs(f_idx - max_frame_idx) <= MAX_REACH_TEXT_FRAMES:
                cv2.putText(frame, ">>> MAX REACH! <<<", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 5)
        else:
            cv2.putText(frame, "ANALYZING...", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)

        # [V4.9] 범례 (Legend) 박스 복구
        cv2.rectangle(frame, (legend_x - 20, legend_y_start - 30), (width - 20, legend_y_start + 250), (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x - 20, legend_y_start - 30), (width - 20, legend_y_start + 250), (255, 255, 255), 2)
        
        cv2.circle(frame, (legend_x, legend_y_start), 10, (255, 0, 0), -1)
        cv2.putText(frame, ": Bat Top", (legend_x + 30, legend_y_start + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.circle(frame, (legend_x, legend_y_start + legend_line_height), 10, (0, 255, 0), -1)
        cv2.putText(frame, ": Bat Bottom", (legend_x + 30, legend_y_start + legend_line_height + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.line(frame, (legend_x - 15, legend_y_start + legend_line_height*2), (legend_x + 15, legend_y_start + legend_line_height*2), (0, 255, 255), 4)
        cv2.putText(frame, ": Homeplate Line", (legend_x + 30, legend_y_start + legend_line_height*2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        handedness_text = ("Left" if is_left_handed else "Right") + "-handed"
        cv2.putText(frame, handedness_text, (legend_x, legend_y_start + legend_line_height*4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 255), 2)

        out.write(frame)
    cap.release()
    out.release()

    # v5.3.1 fix: best_candidate 대신 search_range의 마지막(최대 도달) 프레임 정보 사용
    final_judge_candidate = frame_data[max_frame_idx]
    return ("SWING" if swing_detected else "NO SWING"), {
        "max_reach_frame": max_frame_idx,
        "judge_frame": max_frame_idx,
        "bat_top_x": f"{final_judge_candidate['top_x']:.1f}",
        "bat_bottom_x": f"{final_judge_candidate['bottom_x']:.1f}",
        "handedness": "좌타자" if is_left_handed else "우타자",
        "bat_frames": len(frame_data),
        "total_frames": total_frames
    }

def main():
    print(f"LOADING MODELS from {HOMEPLATE_MODEL_PATH} and {BAT_MODEL_PATH}...")
    homeplate_model = YOLO(HOMEPLATE_MODEL_PATH)
    bat_model = YOLO(BAT_MODEL_PATH)
    print("MODELS LOADED SUCCESSFULLY.")
    
    print(f"VIDEO_DIR: {VIDEO_DIR}")
    if not VIDEO_DIR.exists():
        print(f"ERROR: VIDEO_DIR does not exist: {VIDEO_DIR}")
        return
    
    video_files = sorted([f for f in VIDEO_DIR.iterdir() if f.suffix.lower() == ".mp4"])
    total = len(video_files)
    print(f"Found {total} mp4 files in directory.")
    
    tp, fp, tn, fn = 0, 0, 0, 0
    
    if total == 0:
        print("ERROR: No video files found. Exiting.")
        return
    
    print(f"\n🚀 V5.3 통합 평가 시작 (총 {total}개 영상)")
    print("="*60)

    # 헤더 작성 (최초 1회)
    fieldnames = ['filename', 'ground_truth', 'prediction', 'match', 'handedness', 
                  'max_reach_frame', 'judge_frame', 'bat_top_x', 'bat_bottom_x', 
                  'bat_frames', 'total_frames']
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for i, video_file in enumerate(video_files):
        gt = extract_ground_truth(video_file.name)
        start_time = time.time()
        
        try:
            pred, details = analyze_checkswing(video_file, homeplate_model, bat_model)
        except Exception as e:
            print(f"ERROR processing {video_file.name}: {e}")
            continue
        
        elapsed = time.time() - start_time
        match = "✅" if pred == gt else "❌"
        
        print(f"[{i+1}/{total}] {video_file.name:<40} | GT: {gt:<10} | Pred: {pred:<10} | {match} ({elapsed:.1f}s)")
        
        if pred == gt:
            if gt == "SWING": tp += 1
            else: tn += 1
        else:
            if gt == "SWING": fn += 1
            else: fp += 1
            
        # 결과 즉시 저장
        res = {**{"filename": video_file.name, "ground_truth": gt, "prediction": pred, "match": (pred==gt)}, **details}
        # details 에 가끔 dict 가 섞여올 수 있으므로 fieldnames 에 있는 것만 필터링
        row = {k: res.get(k, '') for k in fieldnames}
        
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    acc = (tp + tn) / total * 100 if total > 0 else 0
    print("\n" + "="*60)
    print(f"최종 정확도: {acc:.2f}% ({tp+tn}/{total})")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*60)

if __name__ == "__main__":
    main()
