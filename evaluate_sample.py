"""
KBO 체크스윙 데이터셋 모델 평가 스크립트 (V4.7)
=========================================
KBO 비디오셋 편집본 폴더의 영상들을 최종 모델로 판정하고,
파일 이름에 포함된 Ground Truth와 비교하여 성능을 평가한다.
V4.7: 플레이트 고유 좌표계(Geometric Invariant) 기반 도트곱 판정 적용.
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
# 경로 설정
# ==========================================
BASE_DIR = Path(".")  # 프로젝트 루트 디렉토리
MODEL_DIR = BASE_DIR / "model(pt)"
VIDEO_DIR = Path("data/videos")  # 평가할 KBO 체크스윙 영상 폴더
OUTPUT_CSV = BASE_DIR / "code" / "evaluation_results.csv"
RESULT_VIDEOS_DIR = BASE_DIR / "code" / "result_videos"

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
    """파일명에서 Ground Truth 라벨 추출 (확장자 제거 후 체크)"""
    name = Path(filename).stem
    if name.endswith("_노스윙"):
        return "NO SWING"
    elif name.endswith("_스윙"):
        return "SWING"
    else:
        return "UNKNOWN"

def detect_handedness(hp_keypoints):
    """홈플레이트 키포인트를 이용해 타자의 좌/우를 판별합니다."""
    if len(hp_keypoints) < 4: return None
    pt0, pt1, back_point = hp_keypoints[0], hp_keypoints[1], hp_keypoints[3]
    if pt0[0] <= 1 or pt1[0] <= 1 or back_point[0] <= 1: return None
    front_avg_x = (pt0[0] + pt1[0]) / 2
    return back_point[0] > front_avg_x

# ==========================================
# 체크스윙 판정 함수 (V4.7)
# ==========================================
def analyze_checkswing(video_path, homeplate_model, bat_model):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return "ERROR", {"error": "영상 열기 실패"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1단계: 정보 수집 (배트, 홈플레이트)
    frame_data = {}
    handedness_votes = []
    hp_pts0, hp_pts1, hp_pts3 = [], [], []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 홈플레이트 탐지 (0,1: 앞선, 3: 뒤꼭짓점)
        hp_results = homeplate_model(frame, conf=0.3, verbose=False)
        if hp_results[0].keypoints is not None and len(hp_results[0].keypoints) > 0:
            kpts = hp_results[0].keypoints[0].xy.cpu().numpy()[0]
            if len(kpts) >= 4 and kpts[0][0] > 1 and kpts[1][0] > 1 and kpts[3][0] > 1:
                hp_pts0.append(kpts[0])
                hp_pts1.append(kpts[1])
                hp_pts3.append(kpts[3])
                h = detect_handedness(kpts)
                if h is not None: handedness_votes.append(h)

        # 배트 탐지 (민감도 극대화)
        bat_results = bat_model(frame, conf=0.1, verbose=False)
        if bat_results[0].keypoints is not None and len(bat_results[0].keypoints) > 0:
            if bat_results[0].boxes and len(bat_results[0].boxes) > 0:
                idx = bat_results[0].boxes.conf.argmax().item()
                bat_kpts = bat_results[0].keypoints[idx].xy.cpu().numpy()[0]
                if len(bat_kpts) > 1 and bat_kpts[0][0] > 1 and bat_kpts[1][0] > 1:
                    frame_data[frame_idx] = {
                        'top_x': bat_kpts[BAT_TOP_IDX][0], 'top_y': bat_kpts[BAT_TOP_IDX][1],
                        'bottom_x': bat_kpts[BAT_BOTTOM_IDX][0], 'bottom_y': bat_kpts[BAT_BOTTOM_IDX][1],
                        'conf': bat_results[0].boxes.conf[idx].item()
                    }
        frame_idx += 1
    cap.release()

    if not frame_data or len(hp_pts3) < 3:
        return "UNDETERMINED", {"reason": "배트 또는 플레이트 감지 부족"}

    # 배트 필터링 및 보간
    f_indices = sorted(frame_data.keys())
    if len(f_indices) > 5:
        txs = medfilt([frame_data[f]['top_x'] for f in f_indices], 5)
        tys = medfilt([frame_data[f]['top_y'] for f in f_indices], 5)
        for i, f in enumerate(f_indices):
            frame_data[f]['top_x'], frame_data[f]['top_y'] = txs[i], tys[i]
        
        # 보간 (최대 10프레임)
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

    # 홈플레이트 기하 구조 확정
    m0, m1, m3 = np.median(hp_pts0, axis=0), np.median(hp_pts1, axis=0), np.median(hp_pts3, axis=0)
    mid_front = (m0 + m1) / 2
    v_depth = m3 - mid_front # 정면(앞)에서 뒤꼭짓점으로 향하는 벡터
    v_swing = np.array([-v_depth[1], v_depth[0]]) # 수직인 스윙축 벡터
    if v_swing[0] < 0: v_swing = -v_swing # 항상 오른쪽을 향하도록 정규화
    
    is_left_handed = sum(handedness_votes) > len(handedness_votes) / 2 if handedness_votes else True

    # 2단계: MAX REACH (속도 기반 트리거 적용)
    frames = sorted(frame_data.keys())
    velocities = {frames[i]: (frame_data[frames[i]]['top_x'] - frame_data[frames[i-1]]['top_x']) / (frames[i]-frames[i-1]) 
                  for i in range(1, len(frames)) if frames[i]-frames[i-1] <= 3}
    
    swing_start = -1
    for f, v in velocities.items():
        if (is_left_handed and v < -3.5) or (not is_left_handed and v > 3.5):
            # 연속 3프레임 이동 시 스윙 시작으로 간주
            if all((is_left_handed and velocities.get(f+j, 0) < -2) or (not is_left_handed and velocities.get(f+j, 0) > 2) for j in range(3)):
                swing_start = f
                break
    
    if swing_start == -1: swing_start = frames[0]
    search_range = [f for f in frames if swing_start <= f <= frames[-1]]
    if not search_range: return "UNDETERMINED", {"reason": "스윙 시퀀스 탐색 실패"}
    
    if is_left_handed:
        max_frame_idx = min(search_range, key=lambda f: frame_data[f]['top_x'])
    else:
        max_frame_idx = max(search_range, key=lambda f: frame_data[f]['top_x'])

    # 3단계: 최종 판정 (V4.7 기하 정밀 판정)
    best_candidate = frame_data[max_frame_idx]
    rel_vec = np.array([best_candidate['top_x'] - mid_front[0], best_candidate['top_y'] - mid_front[1]])
    dot_val = np.dot(rel_vec, v_swing)
    mag = np.linalg.norm(v_swing)
    
    # 시작점 대비 이동 거리 계산
    start_vec = np.array([frame_data[frames[0]]['top_x'] - mid_front[0], frame_data[frames[0]]['top_y'] - mid_front[1]])
    start_dot = np.dot(start_vec, v_swing)
    travel_dist = abs(dot_val - start_dot) / mag

    # 판정 기준 (여유분 50px 적용)
    if is_left_handed:
        swing_detected = dot_val < 50
    else:
        swing_detected = dot_val > -50
        
    if travel_dist < 130:
        swing_detected = False # 최소 이동 거리 미달

    # 4단계: 시각화 및 저장
    output_path = RESULT_VIDEOS_DIR / f"{video_path.stem}_result.mp4"
    cap = cv2.VideoCapture(str(video_path))
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        # 판정선 (노란색)
        m_perp = v_swing[1] / v_swing[0] if abs(v_swing[0]) > 0.1 else 1000
        x_t, x_b = int(mid_front[0] + (0-mid_front[1])*m_perp), int(mid_front[0] + (height-mid_front[1])*m_perp)
        cv2.line(frame, (x_t, 0), (x_b, height), (0, 255, 255), 2)
        
        # 배트 및 판정 표시
        if f_idx in frame_data:
            d = frame_data[f_idx]
            cv2.circle(frame, (int(d['top_x']), int(d['top_y'])), 10, (255, 0, 0), -1)
            
        if f_idx >= max_frame_idx:
            text = "SWING!" if swing_detected else "NO SWING"
            color = (0, 0, 255) if swing_detected else (0, 255, 0)
            cv2.putText(frame, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            if f_idx < max_frame_idx + 15:
                cv2.putText(frame, "MAX REACHED", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        out.write(frame)
    cap.release()
    out.release()

    return ("SWING" if swing_detected else "NO SWING"), {
        "max_reach_frame": max_frame_idx,
        "judge_frame": max_frame_idx,
        "bat_top_x": f"{best_candidate['top_x']:.1f}",
        "bat_bottom_x": f"{best_candidate['bottom_x']:.1f}",
        "handedness": "좌타자" if is_left_handed else "우타자",
        "bat_frames": len(frame_data),
        "total_frames": total_frames
    }

# ==========================================
# 메인 루프
# ==========================================
def main():
    homeplate_model = YOLO(HOMEPLATE_MODEL_PATH)
    bat_model = YOLO(BAT_MODEL_PATH)
    video_files = sorted([f for f in VIDEO_DIR.iterdir() if f.suffix.lower() == ".mp4"])
    
    results = []
    tp, fp, tn, fn = 0, 0, 0, 0
    print(f"\n🚀 판정 시작 (총 {len(video_files)}개 영상)\n" + "="*50)

    for i, video_file in enumerate(video_files[:5]):
        gt = extract_ground_truth(video_file.name)
        print(f"[{i+1}/{len(video_files)}] {video_file.name} (GT: {gt})")
        
        pred, details = analyze_checkswing(video_file, homeplate_model, bat_model)
        match = "✅" if pred == gt else "❌"
        print(f"   결과: {pred} {match}")
        
        if pred == gt:
            if gt == "SWING": tp += 1
            else: tn += 1
        else:
            if gt == "SWING": fn += 1
            else: fp += 1
            
        results.append({**{"filename": video_file.name, "ground_truth": gt, "prediction": pred, "match": (pred==gt)}, **details})

    # 통계 출력 및 CSV 저장
    acc = (tp + tn) / len(video_files) * 100 if video_files else 0
    print(f"\n" + "="*50 + f"\nAccuracy: {acc:.2f}% ({tp+tn}/{len(video_files)})\n" + "="*50)
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    main()
