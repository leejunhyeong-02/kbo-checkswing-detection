"""
Real-Time KBO Check Swing Automated Adjudication System
Inference Module (inference.py)

Paper: KBO 체크스윙 실시간 자동 판정 시스템: YOLO11 키포인트 탐지 및 임계선 보정 접근법
Journal: Journal of Korea Multimedia Society, Vol. 29, No. 7, 2026
Authors: Jun Hyeong Lee, Han Saem Park (Sangmyung University)
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


class CheckSwingDetector:
    def __init__(
        self,
        homeplate_model_path="models/homeplate.pt",
        bat_model_path="models/best.pt",
        conf_threshold=0.3,
        margin_px=20,
        min_consecutive_frames=2,
    ):
        """
        Initialize the check swing detector with two YOLO11n-pose models.
        """
        print(f"[Init] Loading Home Plate Model: {homeplate_model_path}")
        self.homeplate_model = YOLO(homeplate_model_path)

        print(f"[Init] Loading Bat Model: {bat_model_path}")
        self.bat_model = YOLO(bat_model_path)

        self.conf_threshold = conf_threshold
        self.margin_px = margin_px
        self.min_consecutive_frames = min_consecutive_frames

    def detect_handedness(self, hp_keypoints):
        """
        Determine batter's handedness using home plate orientation (majority voting across frames).
        """
        if len(hp_keypoints) < 5:
            return None
        pt0 = hp_keypoints[0]  # Front left corner
        pt1 = hp_keypoints[1]  # Front right corner
        pt3 = hp_keypoints[3]  # Rear peak point (catcher direction)

        if pt0[0] <= 1 or pt1[0] <= 1 or pt3[0] <= 1:
            return None

        mid_front_x = (pt0[0] + pt1[0]) / 2.0
        # If rear point is to the left of front center -> Left-handed, else Right-handed
        if pt3[0] < mid_front_x:
            return "LEFT"
        else:
            return "RIGHT"

    def run(self, video_path, output_path=None, display=False):
        """
        Run 4-step automated check swing adjudication on an input video clip.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            out_dir = Path(output_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"\n[Processing] Video: {video_path.name} ({width}x{height}, {fps:.1f} FPS, {total_frames} frames)")

        handedness_votes = []
        consecutive_cross_count = 0
        max_consecutive_cross = 0
        final_decision = "NO SWING"
        judge_frame_idx = -1

        # Frame loop
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Run dual model inference
            hp_results = self.homeplate_model.predict(frame, conf=self.conf_threshold, verbose=False)
            bat_results = self.bat_model.predict(frame, conf=self.conf_threshold, verbose=False)

            # Extract Home Plate Keypoints (5 pts)
            hp_kpts = None
            if hp_results and len(hp_results[0].keypoints.xy) > 0:
                hp_kpts = hp_results[0].keypoints.xy[0].cpu().numpy()
                h_vote = self.detect_handedness(hp_kpts)
                if h_vote:
                    handedness_votes.append(h_vote)

            # Extract Bat Keypoints (0: Knob, 1: Tip)
            knob_pt = None
            tip_pt = None
            if bat_results and len(bat_results[0].keypoints.xy) > 0:
                bat_kpts = bat_results[0].keypoints.xy[0].cpu().numpy()
                if len(bat_kpts) >= 2:
                    knob_pt = bat_kpts[0]  # (x, y)
                    tip_pt = bat_kpts[1]   # (x, y)

            # Determine dominant handedness
            current_handedness = "RIGHT"
            if handedness_votes:
                current_handedness = "LEFT" if handedness_votes.count("LEFT") > handedness_votes.count("RIGHT") else "RIGHT"

            # 4-Step Adjudication Logic
            is_crossing = False
            threshold_x = None

            if knob_pt is not None and tip_pt is not None and knob_pt[0] > 0 and tip_pt[0] > 0:
                knob_x, knob_y = knob_pt[0], knob_pt[1]
                tip_x, tip_y = tip_pt[0], tip_pt[1]

                # Step 1: Valid swing state (Tip below Knob in image coordinates)
                if tip_y > knob_y:
                    # Step 2: Perspective-corrected threshold line
                    slope_S = 0.0
                    if hp_kpts is not None and len(hp_kpts) >= 2:
                        hp0, hp1 = hp_kpts[0], hp_kpts[1]
                        dy = hp1[1] - hp0[1]
                        dx = hp1[0] - hp0[0]
                        if abs(dy) > 1e-4:
                            slope_S = dx / dy

                    threshold_x = knob_x + (tip_y - knob_y) * slope_S

                    # Step 3: Crossing check with margin
                    if current_handedness == "RIGHT":
                        if tip_x > (threshold_x + self.margin_px):
                            is_crossing = True
                    else:  # LEFT
                        if tip_x < (threshold_x - self.margin_px):
                            is_crossing = True

            # Step 4: Frame-continuity verification
            if is_crossing:
                consecutive_cross_count += 1
                if consecutive_cross_count > max_consecutive_cross:
                    max_consecutive_cross = consecutive_cross_count
                if consecutive_cross_count >= self.min_consecutive_frames and final_decision == "NO SWING":
                    final_decision = "SWING"
                    judge_frame_idx = frame_idx
            else:
                consecutive_cross_count = 0

            # -------------------------------------------------------------
            # Visual Overlay
            # -------------------------------------------------------------
            overlay = frame.copy()

            # Draw Home Plate points & front edge
            if hp_kpts is not None and len(hp_kpts) >= 2:
                p0 = tuple(map(int, hp_kpts[0]))
                p1 = tuple(map(int, hp_kpts[1]))
                cv2.line(overlay, p0, p1, (0, 255, 255), 2)  # Yellow front edge
                for pt in hp_kpts:
                    if pt[0] > 0:
                        cv2.circle(overlay, tuple(map(int, pt)), 4, (0, 255, 255), -1)

            # Draw Bat points & line
            if knob_pt is not None and tip_pt is not None and knob_pt[0] > 0 and tip_pt[0] > 0:
                p_knob = tuple(map(int, knob_pt))
                p_tip = tuple(map(int, tip_pt))
                cv2.line(overlay, p_knob, p_tip, (255, 0, 255), 3)  # Magenta bat
                cv2.circle(overlay, p_knob, 7, (0, 0, 255), -1)      # Red Knob
                cv2.circle(overlay, p_tip, 7, (0, 255, 0), -1)       # Green Tip

            # Draw Threshold Guide Line
            if threshold_x is not None and knob_pt is not None:
                cv2.line(
                    overlay,
                    (int(threshold_x), int(knob_pt[1] - 50)),
                    (int(threshold_x), int(knob_pt[1] + 150)),
                    (0, 165, 255),  # Orange threshold
                    2,
                )

            # Status Box Header
            cv2.rectangle(overlay, (20, 20), (380, 110), (0, 0, 0), -1)
            cv2.rectangle(overlay, (20, 20), (380, 110), (255, 255, 255), 1)

            status_color = (0, 0, 255) if final_decision == "SWING" else (0, 255, 0)
            cv2.putText(
                overlay,
                f"DECISION: {final_decision}",
                (35, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                3,
            )
            cv2.putText(
                overlay,
                f"Batter: {current_handedness} | Consec: {consecutive_cross_count} / {self.min_consecutive_frames}",
                (35, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

            # JUDGE POINT Alert
            if final_decision == "SWING" and (frame_idx - judge_frame_idx) <= 15:
                cv2.putText(
                    overlay,
                    ">>> JUDGE POINT <<<",
                    (width // 2 - 180, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 0, 255),
                    3,
                )

            if writer:
                writer.write(overlay)

            if display:
                cv2.imshow("KBO Check Swing Adjudication", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        print(f"[Done] Result: {final_decision} (Max Consecutive Frames: {max_consecutive_cross})")
        if output_path:
            print(f"[Saved] Output Video: {output_path}")

        return {
            "decision": final_decision,
            "handedness": current_handedness,
            "max_consecutive_frames": max_consecutive_cross,
            "judge_frame": judge_frame_idx,
            "total_frames": frame_idx,
        }


def main():
    parser = argparse.ArgumentParser(description="KBO Check Swing Automated Adjudication Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input check swing video (.mp4)")
    parser.add_argument("--homeplate_model", type=str, default="models/homeplate.pt", help="Path to home plate model (.pt)")
    parser.add_argument("--bat_model", type=str, default="models/best.pt", help="Path to bat model (.pt)")
    parser.add_argument("--output", type=str, default="output/result.mp4", help="Path to save annotated result video")
    parser.add_argument("--conf", type=float, default=0.3, help="Keypoint confidence threshold")
    parser.add_argument("--margin", type=int, default=20, help="Crossing margin in pixels")
    parser.add_argument("--min_consecutive", type=int, default=2, help="Minimum consecutive crossing frames")
    parser.add_argument("--show", action="store_true", help="Display live visualization window")

    args = parser.parse_args()

    detector = CheckSwingDetector(
        homeplate_model_path=args.homeplate_model,
        bat_model_path=args.bat_model,
        conf_threshold=args.conf,
        margin_px=args.margin,
        min_consecutive_frames=args.min_consecutive,
    )

    result = detector.run(video_path=args.video, output_path=args.output, display=args.show)
    print("\n" + "=" * 40)
    print(f" FINAL ADJUDICATION: {result['decision']}")
    print(f" Batter Type:        {result['handedness']}-handed")
    print(f" Max Consecutive:    {result['max_consecutive_frames']} frames")
    print("=" * 40)


if __name__ == "__main__":
    main()
