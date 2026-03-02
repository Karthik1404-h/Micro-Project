"""Download all CCTV videos from HuggingFace dataset."""
import os, sys, time, urllib.request, shutil

BASE = "https://huggingface.co/datasets/PixelML/epstein-files-cctv-video-memory/resolve/main/videos"
DEST = "cctv_videos"

VIDEOS = [
    "EFTA00028842.mp4",   #  ~0.8 MB
    "EFTA00029996.mp4",   #  ~2.3 MB
    "EFTA00029997.mp4",   #  ~4.7 MB
    "EFTA00033226.mp4",   # ~23.8 MB
    "EFTA00033244.mp4",   # ~26.5 MB
    "EFTA00033246.mp4",   # ~23.8 MB
    "EFTA00033262.mp4",   # ~32.2 MB  (already have as cctv_prison.mp4)
    "EFTA00033280.mp4",   # ~23.8 MB
    "EFTA00033368.mp4",   # ~23.8 MB
    "EFTA00033396.mp4",   # ~26.8 MB
]

os.makedirs(DEST, exist_ok=True)

for i, name in enumerate(VIDEOS, 1):
    dest_path = os.path.join(DEST, name)

    # Skip if already downloaded
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"  [{i:2d}/{len(VIDEOS)}] SKIP  {name} (already exists)")
        continue

    url = f"{BASE}/{name}?download=true"
    print(f"  [{i:2d}/{len(VIDEOS)}] Downloading {name} ...", end="", flush=True)
    t0 = time.time()

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(dest_path, "wb") as f:
                shutil.copyfileobj(resp, f)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        elapsed = time.time() - t0
        print(f"  {size_mb:.1f} MB in {elapsed:.1f}s")
    except Exception as e:
        print(f"  FAILED: {e}")

# Also copy existing prison video if not already in folder
prison_src = "cctv_prison.mp4"
prison_dst = os.path.join(DEST, "EFTA00033262.mp4")
if os.path.exists(prison_src) and not os.path.exists(prison_dst):
    shutil.copy2(prison_src, prison_dst)
    print(f"\n  Copied {prison_src} → {prison_dst}")

# Summary
print("\n── Download Summary ──")
total = 0
for name in VIDEOS:
    p = os.path.join(DEST, name)
    if os.path.exists(p):
        sz = os.path.getsize(p) / (1024 * 1024)
        total += sz
        print(f"  ✓ {name:24s}  {sz:6.1f} MB")
    else:
        print(f"  ✗ {name:24s}  MISSING")
print(f"\n  Total: {total:.1f} MB  ({len([v for v in VIDEOS if os.path.exists(os.path.join(DEST, v))])}/{len(VIDEOS)} videos)")
