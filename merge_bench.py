"""과제 6 심화: 세 프로그램을 하나로 합쳐 엔진(모델)을 공유하면 메모리·GPU가 얼마나 줄어드나 (실측)

작업 세 개 (과제 안내문 확인 문제 6-1과 같은 부하)
  occ  : 재실 인원 — yolo11n 입력 960, 1초에 1번
  line : 통행(선 통과) — yolo11n 입력 640, 1초에 15번
  dist : 거리 — yolo11n 입력 640, 1초에 15번 + SGBM 시차 계산(CPU)
방식
  A 따로 : 프로세스 3개가 각자 모델을 올림 (지금처럼 프로그램을 따로 띄운 경우)
  B 합침 : 프로세스 1개, 스레드 3개. line·dist는 640 모델 하나를 같이 씀(잠금), occ는 960 모델
측정 (방식마다 워밍업 20초 뒤 60초)
  보드 RAM 증가량(tegrastats, 시작 전 대비), 프로세스 RSS 합, GPU 사용률 평균, 작업별 실제 처리율·지연(p50/p95)
입력은 카메라 대신 bus.jpg(고정 이미지) — 자원 비교가 목적이라 정확도는 보지 않는다.
사용법: python -u merge_bench.py            (결과 runs/merge_bench.md)
        python -u merge_bench.py --secs 30   (짧게)
"""
import argparse, json, os, re, statistics, subprocess, sys, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--secs", type=float, default=60)
ap.add_argument("--warm", type=float, default=20)
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--worker", default=None, help="내부용: occ|line|dist|all")
ap.add_argument("--out", default=None)
a = ap.parse_args()
TASKS = {"occ": (960, 1.0, False), "line": (640, 15.0, False), "dist": (640, 15.0, True)}

def run_tasks(names, out):
    import numpy as np, cv2
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    img = cv2.imread(str(ASSETS / "bus.jpg")); img = cv2.resize(img, (1280, 720))
    gl = cv2.cvtColor(cv2.resize(img, (640, 360)), cv2.COLOR_BGR2GRAY); gr = np.roll(gl, -12, axis=1)
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7)
    models, locks = {}, {}
    for n in names:                                    # 같은 입력 크기면 모델 하나를 같이 씀
        sz = TASKS[n][0]
        if sz not in models:
            models[sz] = YOLO(a.model); locks[sz] = threading.Lock()
            models[sz].predict(img, imgsz=sz, device=0, verbose=False)
    stats = {n: [] for n in names}
    t_start = time.time() + a.warm; t_end = t_start + a.secs
    def loop(n):
        sz, hz, stereo = TASKS[n]; nxt = time.time()
        while time.time() < t_end:
            t0 = time.time()
            with locks[sz]:
                models[sz].predict(img, imgsz=sz, classes=[0], device=0, verbose=False)
            if stereo: sgbm.compute(gl, gr)
            if t0 >= t_start: stats[n].append((t0, (time.time() - t0) * 1000))
            nxt += 1 / hz
            time.sleep(max(0, nxt - time.time()))
            if time.time() - nxt > 1: nxt = time.time()     # 너무 밀리면 따라잡기 포기
    th = [threading.Thread(target=loop, args=(n,)) for n in names]
    [t.start() for t in th]; [t.join() for t in th]
    res = {}
    for n, v in stats.items():
        ms = sorted(x for _, x in v)
        res[n] = {"rate": len(v) / a.secs, "p50": ms[len(ms) // 2] if ms else None,
                  "p95": ms[int(len(ms) * 0.95)] if ms else None, "target": TASKS[n][1]}
    json.dump(res, open(out, "w"))

if a.worker:
    run_tasks(list(TASKS) if a.worker == "all" else [a.worker], a.out); sys.exit()

class Tegra(threading.Thread):
    def __init__(s):
        super().__init__(daemon=True); s.ram, s.gpu, s.on, s.rec = [], [], True, False
    def run(s):
        p = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            if not s.on: break
            m = re.search(r"RAM (\d+)/", line); g = re.search(r"GR3D_FREQ (\d+)%", line)
            if m: s.ram.append((time.time(), int(m.group(1))))
            if g: s.gpu.append((time.time(), int(g.group(1))))
        p.terminate()
    def between(s, arr, t0, t1):
        v = [x for t, x in arr if t0 <= t <= t1]; return v

def mem_mb(pids):
    """(RSS 합, PSS 합) MB. RSS는 공유 라이브러리를 프로세스마다 다시 세고, PSS는 나눠서 센다"""
    rss = pss = 0
    for pid in pids:
        try:
            for line in open(f"/proc/{pid}/smaps_rollup"):
                if line.startswith("Rss:"): rss += int(line.split()[1]) / 1024
                elif line.startswith("Pss:"): pss += int(line.split()[1]) / 1024
        except (FileNotFoundError, PermissionError): pass
    return rss, pss

tg = Tegra(); tg.start(); time.sleep(5)
Path("runs").mkdir(exist_ok=True)
rows = {}
for mode, groups in (("A 따로 (프로세스 3개)", ["occ", "line", "dist"]), ("B 합침 (프로세스 1개)", ["all"])):
    base_ram = statistics.median(x for _, x in tg.ram[-8:])      # 방식마다 직전 값을 기준으로
    print(f"{mode} 시작 전 보드 RAM {base_ram}MB", flush=True)
    procs, outs = [], []
    for g in groups:
        out = f"runs/_mb_{g}.json"; outs.append(out)
        procs.append(subprocess.Popen([sys.executable, "-u", __file__, "--worker", g, "--out", out,
                                       "--secs", str(a.secs), "--warm", str(a.warm), "--model", a.model]))
    t0 = time.time(); peak_rss = peak_pss = 0
    while any(p.poll() is None for p in procs):
        if time.time() - t0 > a.warm:
            r_, p_ = mem_mb([p.pid for p in procs]); peak_rss = max(peak_rss, r_); peak_pss = max(peak_pss, p_)
        time.sleep(1)
    t1 = time.time()
    ram = tg.between(tg.ram, t0 + a.warm, t1 - 2); gpu = tg.between(tg.gpu, t0 + a.warm, t1 - 2)
    res = {}
    for o in outs: res.update(json.load(open(o)))
    rows[mode] = {"ram_inc": statistics.median(ram) - base_ram if ram else None, "rss": peak_rss, "pss": peak_pss, "base": base_ram,
                  "gpu": statistics.mean(gpu) if gpu else None, "tasks": res}
    print(mode, json.dumps(rows[mode], ensure_ascii=False), flush=True)
    time.sleep(8)
tg.on = False

md = ["### 과제 6 심화 — 세 프로그램 합치기: 모델 공유 전후 자원 (실측)\n",
      "| 방식 | 보드 RAM 증가 (MB) | 프로세스 PSS 합 (MB) | 프로세스 RSS 합 (MB) | GPU 사용률 평균 | 재실 (1회/초) | 통행 (15회/초) | 거리 (15회/초) |",
      "|---|---|---|---|---|---|---|---|"]
for mode, r in rows.items():
    cells = []
    for n in ("occ", "line", "dist"):
        t = r["tasks"].get(n, {})
        cells.append(f"{t.get('rate', 0):.1f}회/초 · p50 {t.get('p50') or 0:.0f}ms · p95 {t.get('p95') or 0:.0f}ms")
    md.append(f"| {mode} | {r['ram_inc']:.0f} | {r['pss']:.0f} | {r['rss']:.0f} | {r['gpu']:.0f}% | " + " | ".join(cells) + " |")
A, B = list(rows.values())
md.append(f"\n합치니 PSS 합 {A['pss']:.0f} → {B['pss']:.0f}MB ({(1 - B['pss'] / A['pss']) * 100:.0f}% 감소), "
          f"RSS 합 {A['rss']:.0f} → {B['rss']:.0f}MB, 보드 RAM 증가 {A['ram_inc']:.0f} → {B['ram_inc']:.0f}MB.")
md.append(f"\n측정 조건 — 젯슨 MAXN_SUPER, {a.model} PyTorch(CUDA), 입력 bus.jpg 1280x720 고정, 거리 작업은 640x360 SGBM(64 시차) 추가, "
          f"방식마다 워밍업 {a.warm:.0f}초 뒤 {a.secs:.0f}초 측정. 보드 RAM 증가 = 측정 중 tegrastats RAM 중앙값 − 시작 전 값(다른 프로그램 포함 보드 전체). "
          "합침 방식은 통행·거리가 640 모델 하나를 잠금으로 번갈아 씀.")
open("runs/merge_bench.md", "w").write("\n".join(md)); print("\n".join(md))
