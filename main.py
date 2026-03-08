"""
Activity-Aware Frame Extraction for CCTV Video Optimization
============================================================
Four-stage pipeline (funnel architecture):

  Every raw frame
      │
      ▼  Stage 1 — Pre-Filtering (Motion Gatekeeper)
      │  MOG2 background subtraction.
      │  No significant pixel change → discard immediately.
      ▼
      Stage 2 — Subject Verification (Object Filter)
      │  YOLOv8-Nano + ByteTrack object tracking.
      │  No relevant subject (person / vehicle / animal) → discard.
      ▼
      Stage 3 — Activity Analysis (Context Engine)
      │  CNN (ResNet-18) + LSTM spatial-temporal deep learning.
      │  Determines whether the subject's behaviour is anomalous.
      │  FALLBACK: if no trained weights → heuristic new-track trigger.
      ▼
      Stage 4 — Keyframe Extraction & Summarisation (Output Builder)
      │  OpenCV VideoWriter → individual .mp4 clips per event.
      │  FFmpeg re-encode to H.265 (or H.264 fallback).
      │  event_log.txt with timestamps for every event.

Output folder structure:
    event_clips/
        event_001__00h01m26s__bus_truck.mp4
        event_002__00h04m42s__car.mp4
        event_003__00h08m42s__person.mp4
        ...
        event_log.txt   ← timestamp + description for every event

Usage
-----
    python main.py
    python main.py --input cam.mp4 --clips-folder my_clips --preview
"""

import argparse
import collections
import re
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

# ──────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    # People & micro-mobility
    0:  "person",     1:  "bicycle",
    # Vehicles
    2:  "car",        3:  "motorcycle", 5:  "bus",       7:  "truck",
    # Animals  (any animal on road/property = notable security event)
    14: "bird",      15: "cat",        16: "dog",       17: "horse",
    18: "sheep",     19: "cow",        20: "elephant",  21: "bear",
    22: "zebra",     23: "giraffe",
    # Suspicious / notable objects
    24: "backpack",  26: "handbag",    28: "suitcase",
}

LABEL_COLORS = {
    0:  (0,   255, 120),  # person      – green
    1:  (0,   220,  80),  # bicycle     – lime
    2:  (0,   180, 255),  # car         – sky blue
    3:  (255, 100,   0),  # motorcycle  – orange
    5:  (180,   0, 255),  # bus         – purple
    7:  (0,    80, 255),  # truck       – dark blue
    14: (0,   255, 255),  # bird        – cyan
    15: (0,   200, 255),  # cat         – light blue
    16: (30,  180, 255),  # dog         – azure
    17: (40,   60, 200),  # horse       – navy
    18: (60,  180, 100),  # sheep       – teal
    19: (40,  140,  60),  # cow         – dark green
    20: (20,   60, 200),  # elephant    – indigo
    21: (0,     0, 220),  # bear        – red
    22: (200, 200,   0),  # zebra       – yellow
    23: (160, 255,   0),  # giraffe     – yellow-green
    24: (0,   255, 200),  # backpack    – spring green
    26: (20,  240, 180),  # handbag
    28: (60,  220, 160),  # suitcase
}


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CCTV Per-Clip Event Summariser")
    p.add_argument("--input",        default=config.INPUT_VIDEO)
    p.add_argument("--clips-folder", default=config.CLIPS_FOLDER)
    p.add_argument("--preview",      action="store_true")
    return p.parse_args()


def fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def clip_name(event_num: int, video_time: float, labels: list[str]) -> str:
    """Build a readable filename from event number, timestamp and labels."""
    h  = int(video_time // 3600)
    m  = int((video_time % 3600) // 60)
    s  = int(video_time % 60)
    ts = f"{h:02d}h{m:02d}m{s:02d}s"
    # Unique labels only, joined by underscore, filesystem-safe
    unique = list(dict.fromkeys(labels))          # preserve order, deduplicate
    label_str = "_".join(re.sub(r"[^a-z0-9]", "", l) for l in unique)[:40]
    return f"event_{event_num:03d}__{ts}__{label_str}.mp4"


def open_writer(path: str, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(path, fourcc, config.OUTPUT_FPS, (width, height))
    return w


def encode_clip(raw_path: str, out_path: str, ffmpeg_bin: str,
                codec: str = "h265") -> None:
    """Re-encode a raw mp4v file to H.265 (or H.264 fallback)."""
    if codec == "h265":
        try:
            subprocess.run([
                ffmpeg_bin, "-y", "-i", raw_path,
                "-c:v", "libx265", "-preset", "fast", "-crf", "26",
                "-movflags", "+faststart", "-tag:v", "hvc1",
                out_path
            ], check=True, capture_output=True)
            Path(raw_path).unlink(missing_ok=True)
            return
        except subprocess.CalledProcessError:
            pass   # H.265 unavailable — fall through to H.264

    subprocess.run([
        ffmpeg_bin, "-y", "-i", raw_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-movflags", "+faststart",
        out_path
    ], check=True, capture_output=True)
    Path(raw_path).unlink(missing_ok=True)


def annotate(frame: np.ndarray, tracks, video_time: float,
             new_ids: set) -> np.ndarray:
    vis = frame.copy()
    w = vis.shape[1]
    for t in tracks:
        tid    = int(t.id[0])
        cls_id = int(t.cls[0])
        conf   = float(t.conf[0])
        x1, y1, x2, y2 = map(int, t.xyxy[0])
        color  = LABEL_COLORS.get(cls_id, (200, 200, 200))
        thick  = 3 if tid in new_ids else 1
        label  = f"{CLASS_NAMES.get(cls_id,'obj')} #{tid}  {conf:.0%}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
        cv2.putText(vis, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    bar = (0, 0, 160) if new_ids else (40, 40, 40)
    cv2.rectangle(vis, (0, 0), (w, 28), bar, -1)
    cv2.putText(vis, f"◉ EVENT  {fmt(video_time)}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return vis


# ──────────────────────────────────────────────────────────────────────
#  Stage 1 : Pre-Filtering — Background subtraction gate
# ──────────────────────────────────────────────────────────────────────

def has_motion(fgmask: np.ndarray, threshold: int | None = None) -> bool:
    if threshold is None:
        threshold = config.MOTION_AREA_THRESHOLD
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) >= threshold for c in contours)


# ──────────────────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────────────────

def run(input_path: str, clips_folder: str, show_preview: bool) -> None:

    # ── Open source video ─────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pre_roll_frames  = int(config.PRE_ROLL_SECONDS  * video_fps)
    post_roll_frames = int(config.POST_ROLL_SECONDS * video_fps)
    max_event_frames = int(config.MAX_EVENT_SECONDS * video_fps)
    min_idle_frames  = int(config.MIN_IDLE_SECONDS  * video_fps)

    # Scale motion threshold to actual resolution (calibrated for 640×360)
    ref_area = 640 * 360
    actual_area = width * height
    motion_threshold = max(500, int(config.MOTION_AREA_THRESHOLD
                                    * actual_area / ref_area))

    # ── Prepare output folder ─────────────────────────────────────────
    clips_dir = Path(clips_folder)
    clips_dir.mkdir(parents=True, exist_ok=True)
    log_path  = Path("logs") / f"{Path(input_path).stem}_event_log.txt"

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()

    print(f"\n{'─'*62}")
    print(f"  Input       : {input_path}")
    print(f"  Clips saved : {clips_dir}/")
    print(f"  Video       : {width}×{height}  {video_fps:.1f}fps  "
          f"{fmt(total_frames/video_fps)}")
    print(f"  Pre/Post    : {config.PRE_ROLL_SECONDS}s / "
          f"{config.POST_ROLL_SECONDS}s    "
          f"Max clip: {config.MAX_EVENT_SECONDS}s    "
          f"Gap: {config.MIN_IDLE_SECONDS}s")
    print(f"  Motion thr  : {config.MOTION_AREA_THRESHOLD}"
          f" → scaled to {motion_threshold} for {width}×{height}")
    print(f"{'─'*62}\n")

    # ── Background subtractor ──────────────────────────────────────────
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=config.MOG2_HISTORY,
        varThreshold=config.MOG2_VAR_THRESHOLD,
        detectShadows=config.MOG2_DETECT_SHADOWS,
    )

    # ── Stage 2: YOLO + ByteTrack ──────────────────────────────────────
    print("[Stage-2] Loading YOLOv8 model …")
    model = YOLO(config.YOLO_MODEL)
    model.to(config.YOLO_DEVICE)
    print(f"[Stage-2] Model ready on {config.YOLO_DEVICE}\n")

    # ── Stage 3: Activity Analysis (CNN + LSTM) ────────────────────────
    activity_engine = ActivityEngine(
        weights_path=config.ACTIVITY_MODEL_WEIGHTS,
        device=config.YOLO_DEVICE,
        seq_length=config.ACTIVITY_SEQ_LENGTH,
        threshold=config.ACTIVITY_THRESHOLD,
    )
    stage3_mode = "CNN+LSTM" if activity_engine.model_ready else "heuristic"
    print()

    # ── State ──────────────────────────────────────────────────────────
    pre_roll_buf: collections.deque = collections.deque(maxlen=pre_roll_frames)

    seen_track_ids: set = set()
    event_active        = False
    event_num           = 0
    event_frame_count   = 0
    post_roll_remaining = 0
    idle_frames_since   = min_idle_frames   # allow events immediately at start
    motion_run          = 0

    current_writer      = None   # VideoWriter for the active event
    current_raw_path    = ""
    current_final_path  = ""
    current_labels: list[str] = []

    n_read = n_motion = n_yolo = n_activity = n_written = 0
    event_log_lines: list[str] = []

    t_start = time.time()

    def close_current_event(video_time: float) -> None:
        """Flush the writer, re-encode to H.264, log."""
        nonlocal current_writer, current_raw_path, current_final_path
        if current_writer is None:
            return
        current_writer.release()
        current_writer = None
        try:
            codec = getattr(config, "OUTPUT_CODEC", "h265")
            encode_clip(current_raw_path, current_final_path,
                        ffmpeg_bin, codec=codec)
            size_mb = Path(current_final_path).stat().st_size / 1024**2
            print(f"    → clip saved:  {Path(current_final_path).name}"
                  f"  ({size_mb:.1f} MB)")
        except subprocess.CalledProcessError as e:
            print(f"    [WARN] ffmpeg encode failed: {e}")

    # ── Frame loop ────────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        n_read    += 1
        video_time = n_read / video_fps

        if n_read % 500 == 0:
            elapsed = time.time() - t_start
            fps_p   = n_read / max(elapsed, 1e-6)
            eta     = (total_frames - n_read) / max(fps_p, 1e-6)
            pct     = 100 * n_read / max(total_frames, 1)
            print(f"  [{pct:5.1f}%]  {n_read:,}/{total_frames:,}  "
                  f"{fps_p:.0f}fps  ETA {fmt(eta)}  "
                  f"events={event_num}  clips_written={event_num}")

        # ── Stage 1: Motion gate ──────────────────────────────────────
        fgmask = subtractor.apply(frame)
        if has_motion(fgmask, motion_threshold):
            motion_run += 1
        else:
            motion_run  = 0

        tier1   = motion_run >= config.MIN_MOTION_FRAMES
        warmup  = n_read <= config.MOG2_HISTORY   # background model still learning

        pre_roll_buf.append(frame.copy())

        # Run YOLO when:
        #   • motion detected (normal path)
        #   • event in progress (need to see when subject leaves)
        #   • MOG2 warmup period (background model not stable yet)
        run_yolo = tier1 or event_active or warmup

        if not run_yolo:
            idle_frames_since += 1
            continue

        n_motion += 1
        in_idle_gap = (not event_active) and (idle_frames_since < min_idle_frames)

        # ── Stage 2: YOLO + ByteTrack ─────────────────────────────────
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

        current_ids = {int(t.id[0]) for t in active_tracks}
        new_ids     = current_ids - seen_track_ids
        seen_track_ids.update(current_ids)

        # ── Stage 3: Activity Analysis ────────────────────────────────
        # Push frame to CNN+LSTM activity engine when subjects present
        if active_tracks:
            activity_engine.push_frame(frame)
            n_activity += 1

        verdict = activity_engine.is_anomalous()
        if verdict is not None:
            # CNN+LSTM model available → use its judgement
            trigger = verdict and not in_idle_gap
        else:
            # No trained model → heuristic: new track IDs = event
            trigger = bool(new_ids) and not in_idle_gap

        if trigger and not event_active:
            event_num        += 1
            event_active      = True
            event_frame_count = 0
            post_roll_remaining = post_roll_frames

            # Label list for filename (e.g. ["person","car"])
            current_labels = [
                CLASS_NAMES.get(int(t.cls[0]), "obj")
                for t in active_tracks if int(t.id[0]) in new_ids
            ]
            desc = ", ".join(
                f"{CLASS_NAMES.get(int(t.cls[0]),'obj')} #{int(t.id[0])}"
                for t in active_tracks if int(t.id[0]) in new_ids
            )
            print(f"\n  ★ EVENT #{event_num} at {fmt(video_time)}  ←  {desc}")
            event_log_lines.append(f"EVENT #{event_num:03d}  {fmt(video_time)}  {desc}")

            # Build file paths
            fname           = clip_name(event_num, video_time, current_labels)
            current_final_path = str(clips_dir / fname)
            current_raw_path   = current_final_path + ".raw.mp4"
            current_writer  = open_writer(current_raw_path, width, height)

            # Write pre-roll
            for buf_frame in pre_roll_buf:
                current_writer.write(buf_frame)
                n_written += 1
            pre_roll_buf.clear()

        if event_active:
            # Draw annotations if preview enabled
            display_frame = (annotate(frame, active_tracks, video_time, new_ids)
                             if show_preview else frame)
            current_writer.write(frame)      # always write clean frame
            n_written += 1
            event_frame_count += 1

            # Activity-based duration: event stays alive while YOLO
            # still sees any relevant subject in the frame.
            # Post-roll countdown only begins when subjects leave.
            if active_tracks:
                post_roll_remaining = post_roll_frames
            else:
                post_roll_remaining -= 1

            if post_roll_remaining <= 0:
                print(f"    → event ended at {fmt(video_time)}")
                close_current_event(video_time)
                event_active      = False
                event_frame_count = 0
                idle_frames_since = 0
            elif event_frame_count >= max_event_frames:
                print(f"    → event capped at {fmt(video_time)}")
                close_current_event(video_time)
                event_active      = False
                event_frame_count = 0
                idle_frames_since = 0
                post_roll_remaining = 0

            if show_preview and event_active:
                cv2.imshow("CCTV Summariser  [Q = quit]", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            idle_frames_since += 1

    # ── Final cleanup ─────────────────────────────────────────────────
    if event_active and current_writer is not None:
        close_current_event(video_time)

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start

    # Write event log file
    with open(log_path, "w") as f:
        f.write(f"CCTV Event Log — {input_path}\n")
        f.write(f"Processed  : {fmt(total_frames/video_fps)} of footage\n")
        f.write(f"Total events: {event_num}\n\n")
        for line in event_log_lines:
            f.write(line + "\n")

    # Print summary
    print(f"\n{'═'*62}")
    print(f"  Completed in       : {fmt(elapsed)}")
    print(f"  Frames processed   : {n_read:,}")
    print(f"  YOLO calls (S2)    : {n_yolo:,}  "
          f"({100*n_yolo/max(n_read,1):.1f}% of frames)")
    print(f"  Activity calls (S3): {n_activity:,}")
    print(f"  Stage-3 mode       : {stage3_mode}")
    print(f"  Key events found   : {event_num}")
    print(f"  Total frames saved : {n_written:,}")
    print(f"  Output folder      : {clips_dir}/")
    print(f"  Event log          : {log_path}")
    print(f"{'═'*62}")
    print(f"\n  ── Event Timeline ──")
    for line in event_log_lines:
        print(f"    {line}")
    print()


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(
        input_path   = args.input,
        clips_folder = args.clips_folder,
        show_preview = args.preview or config.SHOW_PREVIEW,
    )
