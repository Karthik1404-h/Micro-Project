"""
validate_pipeline.py — Visual proof that every stage works
==========================================================
Processes the first ~2 minutes of the input video and produces:

    validation_report/
        stage1_motion_mask_XXXX.jpg     — raw MOG2 output showing what Stage 1 sees
        stage2_yolo_detections_XXXX.jpg — YOLO bounding boxes on real frames
        stage3_decisions.txt            — per-frame trigger log (heuristic / CNN+LSTM)
        stage4_sample_clip.mp4          — a short H.265 encoded clip
        summary.txt                     — full statistics for all 4 stages

Run:  python validate_pipeline.py
"""

import collections
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

import config
from activity_model import ActivityEngine

# ── Settings ──────────────────────────────────────────────────────────
TEST_SECONDS   = 120            # process first 2 minutes
REPORT_DIR     = "logs/validation_report"
SNAPSHOT_EVERY = 200            # save a diagnostic image every N frames
MAX_SNAPSHOTS  = 15             # cap total saved images


def fmt(s: float) -> str:
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"


def main():
    report = Path(REPORT_DIR)
    report.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(config.INPUT_VIDEO)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: {config.INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    test_frames  = int(TEST_SECONDS * video_fps)

    print(f"\n{'═'*64}")
    print(f"  PIPELINE VALIDATION — {config.INPUT_VIDEO}")
    print(f"  Resolution : {width}×{height}  @ {video_fps:.1f} fps")
    print(f"  Full video : {total_frames:,} frames ({fmt(total_frames/video_fps)})")
    print(f"  Testing    : first {test_frames:,} frames ({TEST_SECONDS}s)")
    print(f"  Report dir : {report}/")
    print(f"{'═'*64}\n")

    # ── Stage 1 setup ─────────────────────────────────────────────────
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=config.MOG2_HISTORY,
        varThreshold=config.MOG2_VAR_THRESHOLD,
        detectShadows=config.MOG2_DETECT_SHADOWS,
    )

    # ── Stage 2 setup ─────────────────────────────────────────────────
    print("[Stage-2] Loading YOLOv8 model …")
    model = YOLO(config.YOLO_MODEL)
    model.to(config.YOLO_DEVICE)
    print(f"[Stage-2] Ready on {config.YOLO_DEVICE}")

    # ── Stage 3 setup ─────────────────────────────────────────────────
    activity_engine = ActivityEngine(
        weights_path=config.ACTIVITY_MODEL_WEIGHTS,
        device=config.YOLO_DEVICE,
        seq_length=config.ACTIVITY_SEQ_LENGTH,
        threshold=config.ACTIVITY_THRESHOLD,
    )
    stage3_mode = "CNN+LSTM" if activity_engine.model_ready else "heuristic (new-track)"

    # ── Counters ──────────────────────────────────────────────────────
    n_read = 0
    n_motion = 0                 # passed Stage 1
    n_yolo = 0                   # ran through Stage 2
    n_subjects = 0               # frames with at least 1 relevant subject
    n_triggers = 0               # Stage 3 trigger fired
    n_snapshots = 0

    motion_run = 0
    seen_track_ids = set()
    class_counter = collections.Counter()     # how many times each class seen
    decision_log = []                          # Stage 3 per-trigger log

    CLASS_NAMES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck", 14: "bird", 15: "cat", 16: "dog",
        17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
        21: "bear", 22: "zebra", 23: "giraffe",
        24: "backpack", 26: "handbag", 28: "suitcase",
    }

    COLORS = {
        0: (0,255,120), 1: (0,220,80), 2: (0,180,255), 3: (255,100,0),
        5: (180,0,255), 7: (0,80,255), 14: (0,255,255), 15: (0,200,255),
        16: (30,180,255), 17: (40,60,200), 18: (60,180,100), 19: (40,140,60),
        20: (20,60,200), 21: (0,0,220), 22: (200,200,0), 23: (160,255,0),
        24: (0,255,200), 26: (20,240,180), 28: (60,220,160),
    }

    # For Stage 4 test clip
    ffmpeg_bin   = imageio_ffmpeg.get_ffmpeg_exe()
    clip_writer  = None
    clip_frames  = 0
    clip_raw     = str(report / "stage4_sample_clip.raw.mp4")
    clip_final   = str(report / "stage4_sample_clip.mp4")
    clip_started = False

    t_start = time.time()
    print(f"\n[RUN] Processing {test_frames:,} frames …\n")

    # ── Frame Loop ────────────────────────────────────────────────────
    while n_read < test_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n_read += 1
        video_time = n_read / video_fps

        # Progress
        if n_read % 500 == 0:
            elapsed = time.time() - t_start
            fps_p = n_read / max(elapsed, 1e-6)
            print(f"  [{100*n_read/test_frames:5.1f}%]  {n_read:,}/{test_frames:,}  "
                  f"{fps_p:.0f} fps  motion={n_motion}  yolo={n_yolo}  "
                  f"triggers={n_triggers}")

        # ── STAGE 1: Motion gate ──────────────────────────────────────
        fgmask = subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        motion = any(cv2.contourArea(c) >= config.MOTION_AREA_THRESHOLD
                     for c in contours)

        if motion:
            motion_run += 1
        else:
            motion_run = 0

        passed_s1 = motion_run >= config.MIN_MOTION_FRAMES

        # Save Stage 1 diagnostic snapshot
        if n_read % SNAPSHOT_EVERY == 0 and n_snapshots < MAX_SNAPSHOTS:
            n_snapshots += 1
            # Side-by-side: original | motion mask
            mask_color = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            # Draw contours on mask
            cv2.drawContours(mask_color, contours, -1, (0, 255, 0), 1)
            status = "MOTION" if passed_s1 else "STATIC"
            cv2.putText(mask_color, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if passed_s1 else (0, 0, 255), 2)
            combo = np.hstack([frame, mask_color])
            path = report / f"stage1_frame{n_read:05d}_{status.lower()}.jpg"
            cv2.imwrite(str(path), combo, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if not passed_s1:
            continue

        n_motion += 1

        # ── STAGE 2: YOLO detection ──────────────────────────────────
        results = model.track(
            source=frame,
            conf=config.YOLO_CONFIDENCE,
            device=config.YOLO_DEVICE,
            classes=list(config.RELEVANT_CLASS_IDS),
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
        n_yolo += 1

        active_tracks = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is not None:
                    active_tracks.append(box)

        if active_tracks:
            n_subjects += 1

        # Count classes
        for t in active_tracks:
            cls_id = int(t.cls[0])
            class_counter[cls_id] += 1

        current_ids = {int(t.id[0]) for t in active_tracks}
        new_ids = current_ids - seen_track_ids
        seen_track_ids.update(current_ids)

        # Save Stage 2 diagnostic snapshot (with detections)
        if active_tracks and n_read % SNAPSHOT_EVERY < 50 and n_snapshots < MAX_SNAPSHOTS:
            n_snapshots += 1
            vis = frame.copy()
            for t in active_tracks:
                tid = int(t.id[0])
                cls_id = int(t.cls[0])
                conf = float(t.conf[0])
                x1, y1, x2, y2 = map(int, t.xyxy[0])
                color = COLORS.get(cls_id, (200, 200, 200))
                lbl = f"{CLASS_NAMES.get(cls_id, '?')} #{tid} {conf:.0%}"
                thick = 3 if tid in new_ids else 1
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
                cv2.putText(vis, lbl, (x1, max(y1-6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            tag = "NEW" if new_ids else "tracked"
            cv2.putText(vis, f"Stage-2 @ {fmt(video_time)}  [{tag}]",
                        (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            path = report / f"stage2_yolo_frame{n_read:05d}.jpg"
            cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # ── STAGE 3: Activity analysis ────────────────────────────────
        if active_tracks:
            activity_engine.push_frame(frame)

        verdict = activity_engine.is_anomalous()
        if verdict is not None:
            trigger = verdict
        else:
            trigger = bool(new_ids)

        if trigger:
            n_triggers += 1
            labels = [CLASS_NAMES.get(int(t.cls[0]), "obj")
                      for t in active_tracks if int(t.id[0]) in new_ids]
            desc = ", ".join(labels) if labels else "activity"
            decision_log.append(
                f"TRIGGER #{n_triggers:03d}  {fmt(video_time)}  "
                f"mode={stage3_mode}  objects={desc}  "
                f"new_ids={sorted(new_ids)}"
            )

            # ── STAGE 4: Write a sample clip (first event only) ──────
            if not clip_started:
                clip_started = True
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                clip_writer = cv2.VideoWriter(
                    clip_raw, fourcc, config.OUTPUT_FPS, (width, height))
                print(f"\n  [Stage-4] Recording sample clip at {fmt(video_time)}")

        # Write to sample clip if active
        if clip_writer is not None:
            clip_writer.write(frame)
            clip_frames += 1
            if clip_frames >= int(10 * video_fps):   # ~10 seconds
                clip_writer.release()
                clip_writer = None
                # Encode H.265
                try:
                    subprocess.run([
                        ffmpeg_bin, "-y", "-i", clip_raw,
                        "-c:v", "libx265", "-preset", "fast", "-crf", "26",
                        "-movflags", "+faststart", "-tag:v", "hvc1",
                        clip_final
                    ], check=True, capture_output=True)
                    Path(clip_raw).unlink(missing_ok=True)
                    sz = Path(clip_final).stat().st_size / 1024
                    print(f"  [Stage-4] Sample clip saved: {clip_final}  "
                          f"({sz:.0f} KB, H.265)")
                except subprocess.CalledProcessError:
                    # Fallback H.264
                    subprocess.run([
                        ffmpeg_bin, "-y", "-i", clip_raw,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                        "-movflags", "+faststart", clip_final
                    ], check=True, capture_output=True)
                    Path(clip_raw).unlink(missing_ok=True)
                    sz = Path(clip_final).stat().st_size / 1024
                    print(f"  [Stage-4] Sample clip saved: {clip_final}  "
                          f"({sz:.0f} KB, H.264 fallback)")

    # ── Cleanup ───────────────────────────────────────────────────────
    if clip_writer is not None:
        clip_writer.release()
        try:
            subprocess.run([
                ffmpeg_bin, "-y", "-i", clip_raw,
                "-c:v", "libx265", "-preset", "fast", "-crf", "26",
                "-movflags", "+faststart", "-tag:v", "hvc1", clip_final
            ], check=True, capture_output=True)
            Path(clip_raw).unlink(missing_ok=True)
        except Exception:
            pass

    cap.release()
    elapsed = time.time() - t_start

    # ── Write decision log ────────────────────────────────────────────
    dec_path = report / "stage3_decisions.txt"
    with open(dec_path, "w") as f:
        f.write(f"Stage-3 Activity Decisions — mode: {stage3_mode}\n")
        f.write(f"Processed: {n_read:,} frames ({fmt(n_read/video_fps)})\n\n")
        for line in decision_log:
            f.write(line + "\n")

    # ── Compute percentages ───────────────────────────────────────────
    pct_motion  = 100 * n_motion / max(n_read, 1)
    pct_subject = 100 * n_subjects / max(n_motion, 1)
    pct_trigger = 100 * n_triggers / max(n_subjects, 1)
    discard_s1  = n_read - n_motion
    discard_s2  = n_motion - n_subjects

    # ── Summary ───────────────────────────────────────────────────────
    summary_lines = [
        f"{'═'*64}",
        f"  VALIDATION REPORT — {config.INPUT_VIDEO}",
        f"  Processed  : {n_read:,} frames  ({fmt(n_read/video_fps)})  in {elapsed:.1f}s",
        f"{'═'*64}",
        f"",
        f"  ┌─ Stage 1: Pre-Filtering (Motion Gatekeeper) ──────────────┐",
        f"  │  Frames in       : {n_read:,}",
        f"  │  Motion detected : {n_motion:,}  ({pct_motion:.1f}%)",
        f"  │  Discarded       : {discard_s1:,}  ({100-pct_motion:.1f}% — no motion)",
        f"  │  Tool: MOG2 (history={config.MOG2_HISTORY}, "
        f"varThreshold={config.MOG2_VAR_THRESHOLD})",
        f"  │  VERDICT: {'✓ WORKING' if discard_s1 > 0 else '✗ NOT FILTERING'}",
        f"  └────────────────────────────────────────────────────────────┘",
        f"",
        f"  ┌─ Stage 2: Subject Verification (Object Filter) ───────────┐",
        f"  │  YOLO calls      : {n_yolo:,}",
        f"  │  Subjects found  : {n_subjects:,}  ({pct_subject:.1f}% of motion frames)",
        f"  │  Discarded       : {discard_s2:,}  (motion but no relevant subject)",
        f"  │  Unique tracks   : {len(seen_track_ids)}",
        f"  │  Tool: YOLOv8-Nano + ByteTrack on {config.YOLO_DEVICE}",
    ]
    # Class breakdown
    if class_counter:
        summary_lines.append(f"  │  ── Class breakdown (detection counts) ──")
        for cls_id, cnt in class_counter.most_common():
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            summary_lines.append(f"  │    {name:15s} : {cnt:,}")
    summary_lines += [
        f"  │  VERDICT: {'✓ WORKING' if n_subjects > 0 else '✗ NO DETECTIONS'}",
        f"  └────────────────────────────────────────────────────────────┘",
        f"",
        f"  ┌─ Stage 3: Activity Analysis (Context Engine) ─────────────┐",
        f"  │  Mode            : {stage3_mode}",
        f"  │  Event triggers  : {n_triggers}",
        f"  │  Trigger rate    : {pct_trigger:.1f}% of subject frames",
        f"  │  CNN+LSTM weights: {'LOADED' if activity_engine.model_ready else 'NOT FOUND → heuristic fallback'}",
        f"  │  Note: Train on UCF-Crime dataset to activate CNN+LSTM",
        f"  │  VERDICT: {'✓ WORKING' if n_triggers > 0 else '✗ NO TRIGGERS'}",
        f"  └────────────────────────────────────────────────────────────┘",
        f"",
        f"  ┌─ Stage 4: Keyframe Extraction & Summarisation ────────────┐",
        f"  │  Sample clip     : {clip_final if Path(clip_final).exists() else 'NOT CREATED'}",
    ]
    if Path(clip_final).exists():
        sz = Path(clip_final).stat().st_size / 1024
        summary_lines.append(f"  │  Clip size       : {sz:.0f} KB")
        summary_lines.append(f"  │  Codec           : H.265 (libx265)")
    summary_lines += [
        f"  │  Tool: OpenCV VideoWriter → FFmpeg re-encode",
        f"  │  VERDICT: {'✓ WORKING' if Path(clip_final).exists() else '✗ NO CLIP'}",
        f"  └────────────────────────────────────────────────────────────┘",
        f"",
        f"  ┌─ Funnel Summary ──────────────────────────────────────────┐",
        f"  │  {n_read:>7,} raw frames",
        f"  │     ↓  Stage 1 discards {discard_s1:,} ({100-pct_motion:.0f}%)",
        f"  │  {n_motion:>7,} motion frames",
        f"  │     ↓  Stage 2 discards {discard_s2:,}",
        f"  │  {n_subjects:>7,} subject frames",
        f"  │     ↓  Stage 3 triggers {n_triggers}",
        f"  │  {n_triggers:>7,} events → clips",
        f"  └────────────────────────────────────────────────────────────┘",
        f"",
        f"  Report files in: {report}/",
    ]

    # Write and print
    sum_path = report / "summary.txt"
    with open(sum_path, "w", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(line + "\n")

    print("\n")
    for line in summary_lines:
        print(line)
    print()

    # List generated files
    print(f"  ── Generated files ──")
    for p in sorted(report.iterdir()):
        sz = p.stat().st_size
        if sz > 1024*1024:
            print(f"    {p.name:50s}  {sz/1024/1024:.1f} MB")
        elif sz > 1024:
            print(f"    {p.name:50s}  {sz/1024:.0f} KB")
        else:
            print(f"    {p.name:50s}  {sz} B")
    print()


if __name__ == "__main__":
    main()
