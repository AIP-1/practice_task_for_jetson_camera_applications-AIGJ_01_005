"""과제 2-1 벤치마크: PyTorch / TensorRT FP16 / INT8 / 큰 모델(yolo11s) FP16
각 모델: 평균·p95 추론 ms, FPS, mAP50-95(coco128), 전력(VDD_IN), 메모리
사용법: nohup python -u bench2.py > runs/bench2.log 2>&1 &   (끝나면 runs/bench2.md 표)
"""
import os, re, json, time, shutil, subprocess, threading, resource
import numpy as np, cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS

os.makedirs("runs", exist_ok=True); os.makedirs("engines", exist_ok=True)
DATA, IMGSZ, N = "coco128.yaml", 640, 300
img = cv2.imread(str(ASSETS / "bus.jpg"))

def export(pt, out, **kw):
    if os.path.exists(out):
        print(f"[있음] {out}"); return out
    t = time.time(); print(f"[변환] {pt} → {out} {kw} (수 분 걸림)", flush=True)
    p = YOLO(pt).export(format="engine", imgsz=IMGSZ, device=0, workspace=2, **kw)
    shutil.move(p, out); print(f"[완료] {out}  {time.time()-t:.0f}초", flush=True)
    return out

class Tegra(threading.Thread):
    """tegrastats 로 전력(VDD_IN mW)과 RAM(MB) 기록"""
    def __init__(s):
        super().__init__(daemon=True); s.pw, s.ram, s.on = [], [], True
        s.p = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE, text=True)
    def run(s):
        for line in s.p.stdout:
            if not s.on: continue
            m = re.search(r"VDD_IN (\d+)mW", line); r = re.search(r"RAM (\d+)/", line)
            if m: s.pw.append(int(m.group(1)))
            if r: s.ram.append(int(r.group(1)))
    def stop(s): s.p.terminate()

def bench(name, path, half):
    print(f"\n=== {name} ({path}) ===", flush=True)
    m = YOLO(path, task="detect")
    for _ in range(30): m.predict(img, imgsz=IMGSZ, device=0, half=half, verbose=False)
    tg = Tegra(); tg.start(); time.sleep(0.5); tg.pw.clear(); tg.ram.clear()
    wall, inf = [], []
    for _ in range(N):
        t = time.perf_counter(); r = m.predict(img, imgsz=IMGSZ, device=0, half=half, verbose=False)
        wall.append((time.perf_counter() - t) * 1000); inf.append(r[0].speed["inference"])
    tg.on = False; tg.stop()
    v = m.val(data=DATA, imgsz=IMGSZ, batch=1, device=0, half=half, verbose=False, plots=False)
    row = {"모델": name, "추론_평균_ms": round(float(np.mean(inf)), 2), "추론_p95_ms": round(float(np.percentile(inf, 95)), 2),
           "전체_평균_ms": round(float(np.mean(wall)), 2), "FPS": round(1000 / float(np.mean(wall)), 1),
           "mAP50-95": round(float(v.box.map), 4), "mAP50": round(float(v.box.map50), 4),
           "전력_W": round(np.mean(tg.pw) / 1000, 2) if tg.pw else None,
           "보드RAM_MB": int(np.mean(tg.ram)) if tg.ram else None,
           "프로세스_최대RSS_MB": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
           "파일_MB": round(os.path.getsize(path) / 1e6, 1)}
    print("결과:", json.dumps(row, ensure_ascii=False), flush=True)
    return row

if __name__ == "__main__":
    t0 = time.time()
    fp16 = export("yolo11n.pt", "engines/yolo11n_fp16.engine", half=True)
    int8 = export("yolo11n.pt", "engines/yolo11n_int8.engine", int8=True, data=DATA)
    big = export("yolo11s.pt", "engines/yolo11s_fp16.engine", half=True)
    rows = [bench("PyTorch yolo11n (FP32)", "yolo11n.pt", False),
            bench("TensorRT yolo11n FP16", fp16, True),
            bench("TensorRT yolo11n INT8", int8, False),
            bench("TensorRT yolo11s FP16 (큰 모델)", big, True)]
    json.dump(rows, open("runs/bench2.json", "w"), ensure_ascii=False, indent=1)
    keys = list(rows[0])
    md = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n" + "\n".join("| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows)
    md += f"\n\n조건: 입력 640, batch 1, bus.jpg {N}회 반복, mAP는 coco128(128장), 과제6 서비스 동시 가동 중. 총 {time.time()-t0:.0f}초\n"
    open("runs/bench2.md", "w").write(md); print("\n" + md)
