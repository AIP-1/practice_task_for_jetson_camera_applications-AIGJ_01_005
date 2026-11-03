"""가상 시험(합성 영상): 과제 3 출입 계수 vs 직접 계수 동시 운영
- 카메라 1대(높이 2.5m, 25° 아래)가 방을 봄. 문은 왼쪽 화면 밖(y = +3.4), 방 안 = y < +1.8
- 사람 6명이 차례로 들어와 자리에 서고(일부는 둘이 붙어서 들어옴), 일부는 나감. 정답 인원을 앎
- 직접 계수: 방 안 영역에 발이 있는 사람 수 → 최근 5프레임 중앙값
- 출입 계수: ByteTrack ID의 발이 문 선(y = +2.2)을 넘을 때 IN/OUT, 여유 폭 0.25m
사용법: python -u vcount.py   (결과 runs/vcount.md, 영상 runs/vcount.mp4)
"""
import numpy as np, cv2
from ultralytics import YOLO
from vscene import make_cam, cut_people, render, ray, W, H

CAM = make_cam(0, 0, 2.5, tilt_deg=25)
PEOPLE = cut_people(6)
DT, ROOM_Y, DOOR_Y, MARGIN = 0.2, 1.8, 2.2, 0.25
# 각 사람: (들어오는 시각, 서 있을 자리 (x, y), 나가는 시각 또는 None)
PLAN = [(1, (4.5, 1.0), None), (6, (6.0, -1.0), 40), (11, (5.0, -2.0), None),
        (16, (7.0, 0.5), None), (16.3, (7.0, 1.3), 46),          # 둘이 붙어서 들어옴
        (24, (5.5, 0.2), None)]                                    # 앞사람(4.5, 1.0) 뒤에 서서 가려짐
T_END, SPEED = 56, 1.2

def pos(k, t):
    t_in, (sx, sy), t_out = PLAN[k]; door = np.array([sx, 3.4])
    if t < t_in: return None
    d_in = np.linalg.norm(np.array([sx, sy]) - door) / SPEED
    if t_out is None or t < t_out:
        a = min(1, (t - t_in) / d_in); return door + a * (np.array([sx, sy]) - door)
    a = (t - t_out) / d_in
    return None if a > 1 else np.array([sx, sy]) + a * (door - np.array([sx, sy]))

def foot_world(uv):
    o, d = ray(CAM, uv); s = -o[2] / d[2]; return (o + s * d)[:2]

det = YOLO("yolo11n.pt"); side, n_in, n_out = {}, 0, 0; hist = []; rows = []
vw = cv2.VideoWriter("runs/vcount.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 1 / DT, (W // 2, H // 2))
for i, t in enumerate(np.arange(0, T_END, DT)):
    actors = [(p[0], p[1], k) for k in range(len(PLAN)) if (p := pos(k, t)) is not None]
    gt = sum(1 for (x, y, k) in actors if y < ROOM_Y)
    img = render(CAM, actors, PEOPLE)
    r = det.track(img, persist=True, classes=[0], conf=0.3, tracker="bytetrack.yaml", verbose=False)[0]
    direct = 0
    if r.boxes.id is not None:
        for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
            fx, fy = foot_world(((x1 + x2) / 2, y2))
            if fy < ROOM_Y: direct += 1
            s = 1 if fy > DOOR_Y + MARGIN else (-1 if fy < DOOR_Y - MARGIN else 0)   # 문 밖 = +1, 방 쪽 = -1
            if s and tid in side and s != side[tid]:
                if s == -1: n_in += 1
                else: n_out += 1
            if s: side[tid] = s
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"#{tid}", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    hist.append(direct); d_s = int(np.median(hist[-5:])); entry = n_in - n_out
    rows.append((t, gt, d_s, entry))
    cv2.putText(img, f"t={t:4.1f}s  GT {gt}  direct {d_s}  entry {entry} (IN {n_in}/OUT {n_out})", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    vw.write(cv2.resize(img, (W // 2, H // 2)))
vw.release()
R = np.array([(g, d, e) for _, g, d, e in rows])
def stat(c):
    e = R[:, c] - R[:, 0]; return np.abs(e).mean(), e.mean(), (np.abs(e) <= 1).mean() * 100
md = ["### 과제 3 — 출입 계수 vs 직접 계수 (가상 시험, 동시 운영)\n", "| 방식 | MAE (명) | 편향 (명) | ±1명 이내 | 끝났을 때 값 (정답 4) |", "|---|---|---|---|---|"]
for name, c in (("직접 계수 (중앙값 5프레임)", 1), ("출입 계수 (IN−OUT 누적)", 2)):
    a, b, p = stat(c); md.append(f"| {name} | {a:.2f} | {b:+.2f} | {p:.0f}% | {int(R[-1, c])} |")
md.append(f"\n출입 계수 최종 IN {n_in} / OUT {n_out} (정답 IN 6 / OUT 2).")
md.append("\n| 시각 (s) | 정답 | 직접 | 출입 |\n|---|---|---|---|")
for t, g, d, e in rows[:: int(4 / DT)]: md.append(f"| {t:.0f} | {g} | {d} | {e} |")
md.append(f"\n가상 시험(합성 영상): 카메라 1대 1280x720 높이 2.5m 25° 아래, 사람 6명(bus.jpg에서 오린 키 1.7m 세운 판), yolo11n + ByteTrack conf 0.3, 프레임 간격 {DT}초, {T_END}초.")
open("runs/vcount.md", "w").write("\n".join(md)); print("\n".join(md))
