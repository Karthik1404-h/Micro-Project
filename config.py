# ─────────────────────────────────────────────────────────────
#  Configuration – edit these values before running the script
# ─────────────────────────────────────────────────────────────

INPUT_VIDEO = "data/input1.mp4"

# Folder where individual event clips will be saved.
# Each event gets its own file: event_001_00h01m26s_bus_car.mp4
CLIPS_FOLDER = "output"

# ── Tier-1  Background Subtraction ───────────────────────────
# Minimum contour area (px²) to be considered real motion.
MOTION_AREA_THRESHOLD = 5000
MIN_MOTION_FRAMES     = 2
MOG2_HISTORY          = 500
MOG2_VAR_THRESHOLD    = 50
MOG2_DETECT_SHADOWS   = True

# ── Tier-2  YOLOv8 + ByteTrack Object Tracker ────────────────
YOLO_MODEL      = "yolov8n.pt"
YOLO_DEVICE     = "cuda"         # 'cpu' if no GPU
YOLO_CONFIDENCE = 0.45

# COCO class IDs to detect and trigger events for.
# Vehicles
VEHICLE_IDS  = {2, 3, 5, 7}            # car, motorcycle, bus, truck
# People & micro-mobility
PEOPLE_IDS   = {0, 1}                  # person, bicycle
# Animals  (any animal on a road/property is a notable event)
ANIMAL_IDS   = {14, 15, 16, 17, 18,   # bird, cat, dog, horse, sheep
                19, 20, 21, 22, 23}   # cow, elephant, bear, zebra, giraffe
# Suspicious / notable objects
OBJECT_IDS   = {24, 26, 28}           # backpack, handbag, suitcase

RELEVANT_CLASS_IDS = VEHICLE_IDS | PEOPLE_IDS | ANIMAL_IDS | OBJECT_IDS

# ── Tier-3  Activity Analysis (CNN + LSTM) ─────────────────────
# Spatial-temporal deep learning for anomaly / significant-activity
# detection.  A ResNet-18 CNN extracts per-frame spatial features;
# an LSTM analyses the temporal sequence to produce an anomaly score.
#
# When the weights file does NOT exist the pipeline automatically
# falls back to the heuristic trigger (new-track detection) so the
# system keeps working before training is done.
ACTIVITY_MODEL_WEIGHTS = "activity_model_weights.pth"
ACTIVITY_SEQ_LENGTH    = 16      # number of frames the LSTM looks at
ACTIVITY_THRESHOLD     = 0.5     # anomaly score above this = event

# ── Event Detection ───────────────────────────────────────────
# When Stage-3 falls back to heuristic mode, a "key event" fires
# when a BRAND-NEW track ID (object) appears that has never been
# seen before in the entire video.
#
# PRE_ROLL_SECONDS  : seconds of footage BEFORE the trigger (context)
# POST_ROLL_SECONDS : seconds of footage AFTER the last new detection
# MAX_EVENT_SECONDS : hard cap – one event clip cannot exceed this
# MIN_IDLE_SECONDS  : minimum quiet gap required after an event ends
#                     before the next event can start

PRE_ROLL_SECONDS  = 4
POST_ROLL_SECONDS = 5
MAX_EVENT_SECONDS = 40
MIN_IDLE_SECONDS  = 20

# ── Output Clips ────────────────────────────────────────────
OUTPUT_FPS   = 20       # FPS of every saved clip
OUTPUT_CODEC = "h265"   # 'h265' (recommended) or 'h264'; auto-fallback

# ── Display ──────────────────────────────────────────────────
SHOW_PREVIEW = False
