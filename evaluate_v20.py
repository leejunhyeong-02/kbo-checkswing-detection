# [V20.0] New 2-Point Fixed Order Model (p0=knob, p1=tip)

import cv2
import numpy as np
import os, time, unicodedata
from pathlib import Path
from ultralytics import YOLO

# -----------------
# 파일 및 폴더 설정
# -----------------
HOMEPLATE_MODEL_PATH = "models/homeplate.pt"
BAT_MODEL_PATH = "models/best_2pt.pt"  # 새로운 2점 모델 이름
VIDEO_DIR = Path("data/videos")  # 평가할 KBO 체크스윙 영상 폴더
RESULT_VIDEOS_DIR = Path("result_videos")  # 결과 영상 저장 폴더 (자동 생성)
os.makedirs(RESULT_VIDEOS_DIR, exist_ok=True)

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

def analyze_checkswing(video_path, hp_model, bat_model):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return "ERROR", {}
    
    vid_w, vid_h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    out = cv2.VideoWriter(str(RESULT_VIDEOS_DIR / f"{video_path.stem}_v20.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (vid_w, vid_h))

    # 타석 모델로 좌타/우타 판별
    handedness_votes, hp_pts0, hp_pts1 = [], [], []
    temp_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        temp_frames.append(frame)
        
        hp_res = hp_model(frame, conf=0.3, verbose=False)
        if hp_res[0].keypoints is not None and len(hp_res[0].keypoints) > 0:
            k = hp_res[0].keypoints[0].xy.cpu().numpy()[0]
            if len(k) >= 4 and k[0][0] > 1:
                hp_pts0.append(k[0])
                hp_pts1.append(k[1])
                hv = detect_handedness(k)
                if hv is not None: handedness_votes.append(hv)

    cap.release()
    if not hp_pts0:
        out.release(); return "UNDETERMINED", {}

    mid_plate_x = (np.median(hp_pts0, axis=0)[0] + np.median(hp_pts1, axis=0)[0]) / 2
    is_left = sum(handedness_votes) > len(handedness_votes) / 2 if handedness_votes else True

    # 배트 판정 (V20: 2점 고정 모델. p0 = knob, p1 = tip)
    SPATIAL_BUFFER = 200
    bat_trajectory = [] # [f_idx, p0, p1]
    
    for f_idx, frame in enumerate(temp_frames):
        bat_res = bat_model(frame, conf=0.1, verbose=False)
        if bat_res[0].keypoints is not None and len(bat_res[0].keypoints) > 0:
            if bat_res[0].boxes and len(bat_res[0].boxes) > 0:
                idx = bat_res[0].boxes.conf.argmax().item()
                bk = bat_res[0].keypoints[idx].xy.cpu().numpy()[0]
                bbox = bat_res[0].boxes.xyxy[idx].cpu().numpy()
                
                # 라벨링 순서: 0번은 Bottom(손잡이), 1번은 Top(배트끝)
                if len(bk) >= 2 and bk[0][0] > 1 and bk[1][0] > 1:
                    bbox_center_x = (bbox[0] + bbox[2]) / 2
                    if abs(bbox_center_x - mid_plate_x) <= SPATIAL_BUFFER:
                        bat_length = ((bk[0][0] - bk[1][0])**2 + (bk[0][1] - bk[1][1])**2)**0.5
                        if 100 <= bat_length <= 250: # 배트 길이 필터
                            bat_trajectory.append({'f_idx': f_idx, 'p0': bk[0], 'p1': bk[1], 'len': bat_length})
                            
    if not bat_trajectory:
        out.release(); return "UNDETERMINED", {}

    # 스윙 체크 (2프레임 연속 p1이 p0를 교차)
    current_consecutive, max_consecutive = 0, 0
    judge_frame, judge_bat_len = -1, 0
    f_indices = [b['f_idx'] for b in bat_trajectory]
    
    for b in bat_trajectory:
        crossed = False
        p0x, p1x = b['p0'][0], b['p1'][0]
        
        if is_left:
            crossed = (p1x < p0x - 20) # 하방 마진 적용
        else:
            crossed = (p1x > p0x + 20)
            
        if crossed:
            current_consecutive += 1
            if current_consecutive >= 2 and judge_frame == -1:
                judge_frame = b['f_idx']
                judge_bat_len = b['len']
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
        else:
            current_consecutive = 0
            
    swing_detected = max_consecutive >= 2
    if judge_frame == -1:
        judge_frame = f_indices[len(f_indices)//2] if f_indices else 0

    # 시각화
    traj_dict = {b['f_idx']: b for b in bat_trajectory}
    legend_x, legend_y_start, lh = vid_w - 300, 30, 40
    
    for f_idx, frame in enumerate(temp_frames):
        cv2.line(frame, (int(mid_plate_x), 0), (int(mid_plate_x), vid_h), (0, 255, 255), 3)
        if f_idx in traj_dict:
            b = traj_dict[f_idx]
            p0, p1 = b['p0'], b['p1']
            
            # 빨강: p0 (손잡이), 녹색: p1 (배트끝)
            cv2.circle(frame, (int(p0[0]), int(p0[1])), 10, (0, 0, 255), -1)
            cv2.circle(frame, (int(p1[0]), int(p1[1])), 10, (0, 255, 0), -1)
            cv2.line(frame, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 255), 3)
            
            # 번호 표시
            cv2.putText(frame, 'p0(Knob)', (int(p0[0])+15, int(p0[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.putText(frame, 'p1(Tip)', (int(p1[0])+15, int(p1[1])+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if f_idx >= judge_frame:
            color = (0, 0, 255) if swing_detected else (0, 255, 0)
            cv2.putText(frame, "SWING!" if swing_detected else "NO SWING", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 6)
            if abs(f_idx - judge_frame) <= 5:
                cv2.putText(frame, f">>> JUDGE POINT <<<", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

        cv2.rectangle(frame, (legend_x-20, legend_y_start-30), (vid_w-20, legend_y_start+200), (0,0,0), -1)
        cv2.putText(frame, ("Left" if is_left else "Right")+"-handed", (legend_x, legend_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,255), 2)
        cv2.circle(frame, (legend_x, legend_y_start+lh), 8, (0,0,255), -1)
        cv2.putText(frame, ": p0 (Knob/Bottom)", (legend_x+20, legend_y_start+lh+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.circle(frame, (legend_x, legend_y_start+lh*2), 8, (0,255,0), -1)
        cv2.putText(frame, ": p1 (Tip/Top)", (legend_x+20, legend_y_start+lh*2+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        out.write(frame)
    out.release()
    return ("SWING" if swing_detected else "NO SWING"), {"judge_frame": judge_frame, "judge_bat_len": round(judge_bat_len, 1)}

def main():
    if not os.path.exists(BAT_MODEL_PATH):
        print(f"ERROR: 2점짜리 새 모델({BAT_MODEL_PATH})을 해당 폴더에 다운받지 않았습니다!")
        print("구글 코랩에서 'runs/pose/train/weights/best.pt' 파일을 다운로드 하세요.")
        return

    h_model = YOLO(HOMEPLATE_MODEL_PATH)
    b_model = YOLO(BAT_MODEL_PATH)
    
    video_files = sorted([f for f in VIDEO_DIR.iterdir() if f.suffix.lower() == ".mp4"])[:10]
    
    print("====================================")
    print("🔥 V20.0 테스트 시작! (새로운 2점 고정 모델) 🔥")
    print("====================================")
    
    tp, tn, fp, fn = 0, 0, 0, 0
    t0 = time.time()
    
    for i, vf in enumerate(video_files):
        gt = extract_ground_truth(vf.name)
        pred, details = analyze_checkswing(vf, h_model, b_model)
        match = (pred == gt)
        j_frame = details.get('judge_frame', '?')
        b_len = details.get('judge_bat_len', '?')
        
        print(f"[{i+1}/{len(video_files)}] {vf.name}")
        print(f" -> GT: {gt} | Pred: {pred} | Judge_F: {j_frame} | Bat_Len: {b_len}px | Result: {'✅' if match else '❌'}")
        
        if pred == "UNDETERMINED":
            fn += 1 # 스윙 못찾음
        elif match:
            if gt == "SWING": tp += 1
            else: tn += 1
        else:
            if gt == "SWING": fn += 1
            else: fp += 1
            
    print("\n--- Summary ---")
    score = (tp+tn)/len(video_files) * 100
    print(f"Accuracy: {score:.2f}% (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
    print(f"Time Taken: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
