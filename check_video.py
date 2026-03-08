import cv2

video_path = "input_video.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video. Check the file name and path.")
else:
    fps      = cap.get(cv2.CAP_PROP_FPS)
    frames   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps / 60

    print(f"Video is VALID")
    print(f"Resolution  : {width} x {height}")
    print(f"FPS         : {fps}")
    print(f"Duration    : {duration:.1f} minutes")
    print(f"Total frames: {int(frames)}")

cap.release()
