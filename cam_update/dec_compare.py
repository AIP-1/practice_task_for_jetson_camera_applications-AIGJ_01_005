"""과제 1 심화: 1080p H.264 입력에서 하드웨어 디코딩(NVDEC) vs CPU 디코딩 비교
- 시험 영상: 1920x1080 30fps H.264 30초(900프레임)를 만든다(imageio-ffmpeg의 ffmpeg, libx264).
  --src 로 직접 찍은 1080p 영상이나 RTSP 주소를 줘도 된다(RTSP는 --secs 동안만 디코딩).
- 비교하는 방식 (설치돼 있는 것만 돈다)
  gst-nvdec  : GStreamer nvv4l2decoder(NVDEC) → nvvidconv → BGRx  (하드웨어 디코딩)
  gst-cpu    : GStreamer avdec_h264(libav, CPU) → videoconvert → BGRx
  opencv-cpu : OpenCV VideoCapture(FFmpeg, CPU) → BGR (지금 cam1.py·서비스가 쓰는 방식)
- 각 방식: 처리 FPS(동기화 끔, 최대 속도), 그 프로세스의 CPU 사용률(코어 1개 = 100%), 걸린 시간
  추가로 실시간 30fps로 흘렸을 때(sync=true)의 CPU 사용률도 잰다 → 실제 카메라 1대분 부하
사용법: python -u dec_compare.py            (결과 runs/dec_compare.md)
        python -u dec_compare.py --src my1080p.mp4
"""
import argparse, os, re, resource, shutil, subprocess, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="runs/test1080p.mp4")
ap.add_argument("--secs", type=float, default=30)
a = ap.parse_args()
Path("runs").mkdir(exist_ok=True)

def ffmpeg_exe():
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")

if a.src == "runs/test1080p.mp4" and not Path(a.src).exists():
    ff = ffmpeg_exe(); print(f"[시험 영상 만들기] {ff}", flush=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=30",
                    "-f", "lavfi", "-i", "nullsrc=size=1920x1080:rate=30",
                    "-filter_complex", "[0:v]noise=alls=12:allf=t+u[v]", "-map", "[v]",
                    "-t", str(int(a.secs)), "-c:v", "libx264", "-preset", "veryfast", "-g", "30", "-bf", "0",
                    "-pix_fmt", "yuv420p", "-profile:v", "high", a.src], check=True)

def has(el):
    return subprocess.run(["gst-inspect-1.0", el], capture_output=True).returncode == 0 if shutil.which("gst-inspect-1.0") else False

def run_child(cmd):
    """자식 프로세스를 돌리고 (걸린 시간, CPU 초)를 돌려준다"""
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    return wall, cpu, p

is_rtsp = a.src.startswith("rtsp://")
if is_rtsp:
    srcpart = f"rtspsrc location={a.src} protocols=tcp latency=0 ! rtph264depay ! h264parse"
else:
    srcpart = f"filesrc location={a.src} ! qtdemux ! h264parse"

def count_frames():
    cap_cmd = [ffmpeg_exe(), "-v", "error", "-i", a.src, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"]
    p = subprocess.run(cap_cmd + ["-stats"], capture_output=True, text=True)
    m = re.findall(r"frame=\s*(\d+)", p.stderr)
    return int(m[-1]) if m else None

N = None if is_rtsp else count_frames()
if not is_rtsp and not N:                       # ffmpeg로 못 세면 OpenCV 헤더 값으로
    import cv2 as _cv; _c = _cv.VideoCapture(a.src); N = int(_c.get(_cv.CAP_PROP_FRAME_COUNT)) or None; _c.release()
print(f"입력 {a.src}, 프레임 {N}", flush=True)

pipes = {}
if has("nvv4l2decoder"):
    pipes["gst-nvdec (하드웨어)"] = f"{srcpart} ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx"
if has("avdec_h264"):
    pipes["gst-cpu (avdec_h264)"] = f"{srcpart} ! avdec_h264 ! videoconvert ! video/x-raw,format=BGRx"

rows = []
for name, p in pipes.items():
    for sync in ("false", "true"):
        tail = f" ! fakesink sync={sync}"
        cmd = ["gst-launch-1.0", "-q"] + p.split() + tail.split()
        if is_rtsp: cmd = ["timeout", "-s", "INT", str(a.secs)] + cmd
        wall, cpu, pr = run_child(cmd)
        ok = pr.returncode in (0, 124, 130) if is_rtsp else pr.returncode == 0
        fps = (N / wall) if (N and ok) else None
        rows.append((name, "최대 속도" if sync == "false" else "실시간 30fps", fps, cpu / wall * 100, wall, ok, pr.stderr.strip()[-200:]))
        print(rows[-1][:6], flush=True)

# OpenCV CPU (같은 프로세스 안에서 잼)
import cv2
for mode in ("최대 속도", "실시간 30fps"):
    cap = cv2.VideoCapture(a.src); n = 0
    r0 = resource.getrusage(resource.RUSAGE_SELF); t0 = time.time()
    while True:
        if is_rtsp and time.time() - t0 > a.secs: break
        ok, f = cap.read()
        if not ok: break
        n += 1
        if mode == "실시간 30fps":
            wait = t0 + n / 30 - time.time()
            if wait > 0: time.sleep(wait)
    wall = time.time() - t0; r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    rows.append(("opencv-cpu (FFmpeg)", mode, n / wall if wall else None, cpu / wall * 100, wall, n > 0, ""))
    print(rows[-1][:6], flush=True)

md = ["### 과제 1 심화 — 1080p 디코딩: 하드웨어(NVDEC) vs CPU\n",
      "| 방식 | 흘리는 속도 | 처리 FPS | 프로세스 CPU (코어 1개=100%) | 걸린 시간 (s) | 성공 |", "|---|---|---|---|---|---|"]
for name, mode, fps, cpu, wall, ok, err in rows:
    md.append(f"| {name} | {mode} | {fps:.1f} | {cpu:.0f}% | {wall:.1f} | {'✅' if ok else '❌ ' + err} |" if fps
              else f"| {name} | {mode} | — | {cpu:.0f}% | {wall:.1f} | {'✅' if ok else '❌ ' + err} |")
missing = [e for e in ("nvv4l2decoder", "avdec_h264") if e not in " ".join(pipes.values())]
if missing: md.append(f"\n설치돼 있지 않아 건너뜀: {', '.join(missing)}")
md.append(f"\n입력: {a.src} ({N}프레임, 1920x1080 H.264 High, GOP 30, B-프레임 없음). "
          "GStreamer는 출력까지 BGRx로 변환(OpenCV에 넘길 수 있는 형태)한 뒤 버림. "
          "CPU 사용률 = 그 프로세스의 (user+sys) CPU 시간 ÷ 걸린 시간. 보드 전력 모드 MAXN_SUPER.")
open("runs/dec_compare.md", "w").write("\n".join(md)); print("\n".join(md))
