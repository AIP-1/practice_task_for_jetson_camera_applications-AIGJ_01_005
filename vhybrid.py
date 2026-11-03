"""가상 시험(합성 영상): 과제 3 심화 — 출입 계수를 직접 계수로 주기 보정하는 혼합 방식
- vcount.py와 같은 카메라·문·방 설정. 하루 운영을 '압축한 하루'로 흉내 낸다:
  사람 N명이 무작위 시각에 들어와 무작위 자리에 머물다 나간다(둘이 붙어 들어오기, 앞사람에 가려지기 포함).
- 세 방식을 같은 영상에서 동시에 돌린다.
  ① 직접 계수: 방 안 영역에 발이 있는 사람 수, 최근 5프레임 중앙값
  ② 출입 계수: ByteTrack ID가 문 선(y = +2.2, 여유 폭 0.25m)을 넘을 때 IN/OUT 누적
  ③ 혼합: ②를 기본으로 쓰되, PERIOD초마다 '직접 계수가 STABLE초 동안 같은 값이고 문 근처에 아무도 없을 때'
         누적 값을 직접 계수로 바로잡는다(보정 오프셋). 바로잡은 시각과 크기를 기록한다.
사용법: python -u vhybrid.py [--visitors 30] [--minutes 10] [--period 30] [--seed 1]
결과: runs/vhybrid.md, runs/vhybrid.mp4, runs/vhybrid.csv
"""
import argparse, csv
import numpy as np, cv2
from ultralytics import YOLO
from vscene import make_cam, cut_people, render, ray, W, H

ap = argparse.ArgumentParser()
ap.add_argument("--visitors", type=int, default=30)
ap.add_argument("--minutes", type=float, default=10)
ap.add_argument("--period", type=float, default=30, help="보정 시도 간격(초)")
ap.add_argument("--stable", type=float, default=3, help="직접 계수가 이 시간 동안 같아야 보정(초)")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--dt", type=float, default=0.2)
a = ap.parse_args()

CAM = make_cam(0, 0, 2.5, tilt_deg=25)
PEOPLE = cut_people(6)
DT, ROOM_Y, DOOR_Y, MARGIN, SPEED = a.dt, 1.8, 2.2, 0.25, 1.2
T_END = a.minutes * 60
rng = np.random.default_rng(a.seed)

# 방문 계획: (들어오는 시각, 자리 (x, y), 나가는 시각 또는 None)
PLAN = []
t_in = np.sort(rng.uniform(2, T_END * 0.8, a.visitors))
for i, t0 in enumerate(t_in):
    if i > 0 and rng.random() < 0.2:                    # 20%는 앞사람과 붙어서 들어옴
        t0 = PLAN[-1][0] + 0.3
    spot = (rng.uniform(4.0, 8.0), rng.uniform(-2.5, 1.3))
    stay = rng.uniform(40, 300)
    t_out = t0 + stay if t0 + stay < T_END - 10 else None
    PLAN.append((float(t0), spot, t_out))

def pos(k, t):
    t0, (sx, sy), t_out = PLAN[k]; door = np.array([sx, 3.4]); seat = np.array([sx, sy])
    if t < t0: return None
    d_in = np.linalg.norm(seat - door) / SPEED
    if t_out is None or t < t_out:
        return door + min(1, (t - t0) / d_in) * (seat - door)
    s = (t - t_out) / d_in
    return None if s > 1 else seat + s * (door - seat)

def foot_world(uv):
    o, d = ray(CAM, uv); s = -o[2] / d[2]; return (o + s * d)[:2]

det = YOLO("yolo11n.pt")
side, n_in, n_out, hist, rows, fixes = {}, 0, 0, [], [], []
offset, next_fix = 0, a.period
vw = cv2.VideoWriter("runs/vhybrid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 1 / DT * 2, (W // 2, H // 2))
n_stable = int(round(a.stable / DT))
for t in np.arange(0, T_END, DT):
    actors = [(p[0], p[1], k % len(PEOPLE)) for k in range(len(PLAN)) if (p := pos(k, t)) is not None]
    gt = sum(1 for (x, y, k) in actors if y < ROOM_Y)
    img = render(CAM, actors, PEOPLE)
    r = det.track(img, persist=True, classes=[0], conf=0.3, tracker="bytetrack.yaml", verbose=False)[0]
    direct, near_door = 0, False
    if r.boxes.id is not None:
        for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
            fx, fy = foot_world(((x1 + x2) / 2, y2))
            if fy < ROOM_Y: direct += 1
            if fy > ROOM_Y - 0.3: near_door = True          # 문 쪽 띠에 누가 있으면 보정 보류
            s = 1 if fy > DOOR_Y + MARGIN else (-1 if fy < DOOR_Y - MARGIN else 0)
            if s and tid in side and s != side[tid]:
                if s == -1: n_in += 1
                else: n_out += 1
            if s: side[tid] = s
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    hist.append(direct); d_s = int(np.median(hist[-5:]))
    entry = max(0, n_in - n_out)                            # 음수는 0으로 자름(기본 대책)
    hybrid = max(0, n_in - n_out + offset)
    if t >= next_fix:                                       # 보정 시도
        recent = hist[-n_stable:]
        if len(recent) == n_stable and len(set(recent)) == 1 and not near_door:
            new_off = recent[0] - (n_in - n_out)
            if new_off != offset:
                fixes.append((t, hybrid, recent[0]))
            offset = new_off; hybrid = max(0, n_in - n_out + offset)
            next_fix = t + a.period
        # 조건이 안 맞으면 다음 프레임에 다시 시도
    rows.append((t, gt, d_s, entry, hybrid))
    cv2.putText(img, f"t={t:5.1f}s GT {gt} direct {d_s} entry {entry} hybrid {hybrid}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    if int(round(t / DT)) % 2 == 0: vw.write(cv2.resize(img, (W // 2, H // 2)))
vw.release()

with open("runs/vhybrid.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t", "gt", "direct", "entry", "hybrid"]); w.writerows(rows)
R = np.array([r[1:] for r in rows], float)
def stat(c):
    e = R[:, c] - R[:, 0]
    return np.abs(e).mean(), e.mean(), (np.abs(e) <= 1).mean() * 100, np.abs(e).max(), e[-1]
md = [f"### 과제 3 심화 — 혼합 보정 (가상 시험, 압축한 하루 {a.minutes:.0f}분 · 방문 {a.visitors}명)\n",
      "| 방식 | MAE (명) | 편향 (명) | ±1명 이내 | 최대 오차 | 끝났을 때 오차 (드리프트) |", "|---|---|---|---|---|---|"]
for name, c in (("① 직접 계수", 1), ("② 출입 계수", 2), (f"③ 혼합 ({a.period:.0f}초마다 보정)", 3)):
    m, b, p, mx, last = stat(c)
    md.append(f"| {name} | {m:.2f} | {b:+.2f} | {p:.0f}% | {mx:.0f} | {last:+.0f} |")
gt_in = sum(1 for p in PLAN); gt_out = sum(1 for p in PLAN if p[2] is not None)
md.append(f"\n출입 계수 IN {n_in} / OUT {n_out} (정답 IN {gt_in} / OUT {gt_out}). 끝날 때 정답 {int(R[-1,0])}명.")
md.append(f"보정이 실제로 값을 바꾼 횟수 {len(fixes)}회: " + ", ".join(f"{t:.0f}s {h}→{d}" for t, h, d in fixes[:15]))
md.append("\n| 시각 (s) | 정답 | 직접 | 출입 | 혼합 |\n|---|---|---|---|---|")
for row in rows[:: int(60 / DT)]: md.append(f"| {row[0]:.0f} | " + " | ".join(str(int(v)) for v in row[1:]) + " |")
md.append(f"\n가상 시험(합성 영상): 카메라 1대 1280x720 높이 2.5m 25° 아래, bus.jpg에서 오린 사람(키 1.7m 세운 판), "
          f"yolo11n + ByteTrack conf 0.3, 문 선 여유 폭 {MARGIN}m, 프레임 간격 {DT}초, 보정 조건 = 직접 계수 {a.stable:.0f}초 동안 같음 + 문 근처 비어 있음, seed {a.seed}.")
open("runs/vhybrid.md", "w").write("\n".join(md)); print("\n".join(md))
