"""가상 시험(합성 영상): 과제 5 세 지점 오차 + 10-6 접근 알람 + 10-8 이동 타겟 TTC
카메라 2대: 높이 1.2m, 간격 1.0m(y = ±0.5), 10° 아래. 사람 위치는 두 카메라 어깨 삼각측량.
사용법: python -u vprox.py   (결과 runs/vprox.md)
"""
import numpy as np, json
from vscene import make_cam, cut_people, render, pose_all, stereo_people

C0, C1 = make_cam(0, 0.5, 1.2), make_cam(0, -0.5, 1.2)
PEOPLE = cut_people(2)
DT = 0.1                                                    # 가상 프레임 간격 0.1초 (10fps)
md = []

def locate(actors):
    P0, P1 = pose_all(render(C0, actors, PEOPLE)), pose_all(render(C1, actors, PEOPLE))
    return [X[:2] for X in stereo_people(C0, C1, P0, P1)]

# ---------- A. 세 지점(1.5 / 2.5 / 4.0m) 거리 오차 ----------
md.append("### 과제 5 — 가상 영상 세 지점 거리 오차 (기준 ≤ 5%)\n\n| 실제 거리 (m) | 측정 평균 (m) | 오차 (%) | 판정 |\n|---|---|---|---|")
for Z in (1.5, 2.5, 4.0):
    est = []
    for dy in np.linspace(-0.1, 0.1, 5):                    # 좌우로 조금씩 옮겨 5번
        L = locate([(Z, dy, 0)])
        if L: est.append(L[0][0])
    m = float(np.mean(est)) if est else float("nan"); e = abs(m - Z) / Z * 100
    md.append(f"| {Z} | {m:.3f} | {e:.1f} | {'✅' if e <= 5 else '❌'} |"); print("A", Z, m, e, flush=True)

# ---------- B. 10-6 고정 타겟 접근·이탈 알람 ----------
TGT, NEAR, MARGIN = np.array([3.0, 0.0]), 1.5, 0.2
path = [(x, 0.4) for x in np.arange(7.0, 1.9, -0.1)] + [(x, 0.4) for x in np.arange(2.0, 7.1, 0.1)]
alarm, ev = False, []
gt_in = next(i for i, p in enumerate(path) if np.linalg.norm(np.array(p) - TGT) < NEAR)
gt_out = next(i for i, p in enumerate(path) if i > gt_in and np.linalg.norm(np.array(p) - TGT) > NEAR)
for i, p in enumerate(path):
    L = locate([(p[0], p[1], 0)])
    if not L: continue
    d = float(np.linalg.norm(L[0] - TGT))
    if not alarm and d < NEAR: alarm = True; ev.append(("접근", i, d))
    elif alarm and d > NEAR + MARGIN: alarm = False; ev.append(("이탈", i, d))
md.append(f"\n### 10-6 — 고정 타겟 접근 알람 (근접 {NEAR}m, 해제 {NEAR + MARGIN}m)\n\n| 알람 | 측정 시각 (s) | 정답 시각 (s) | 알람 때 측정 거리 (m) |\n|---|---|---|---|")
for kind, i, d in ev:
    g = gt_in if kind == "접근" else gt_out
    md.append(f"| {kind} | {i * DT:.1f} | {g * DT:.1f} | {d:.2f} |")
md.append(f"\n알람 {len(ev)}회(정답 2회: 접근 1 · 이탈 1). 사람이 0.4m 옆을 지나 타겟(3.0, 0.0)에 가장 가까울 때 0.4m.")
print("B", ev, flush=True)

# ---------- C. 10-8 이동 타겟 TTC ----------
md.append("\n### 10-8 — 이동 타겟(다른 사람)이 서 있는 사람에게 다가올 때\n\n| 실제 속도 (m/s) | 추정 속도 (m/s) | 충돌 위험 알람 때 거리 (m) | 근접 알람 때 거리 (m) | 충돌 위험이 먼저 울린 시간 (s) |\n|---|---|---|---|---|")
STAND = np.array([2.6, 0.9]); START = np.array([7.0, -1.0])
for v in (0.5, 1.0, 1.5):
    u = (STAND - START) / np.linalg.norm(STAND - START); tracks = {"h": [], "t": []}
    risk_t = near_t = risk_d = near_d = None; vs = []; t = 0.0; pos = START.copy()
    while np.linalg.norm(pos - STAND) > 0.7:
        L = locate([(STAND[0], STAND[1], 0), (pos[0], pos[1], 1)])
        if len(L) == 2:
            # 서 있는 사람 = 정답 위치에 가까운 쪽 (서 있는 사람은 움직이지 않음)
            h, g = sorted(L, key=lambda q: np.linalg.norm(q - STAND))
            tracks["h"].append((t, h)); tracks["t"].append((t, g))
            rec = [q for q in tracks["t"] if q[0] >= t - 1.0]                   # 최근 1초로 속도 추정
            if len(rec) >= 4:
                T = np.array([q[0] for q in rec]); Pt = np.array([q[1] for q in rec])
                vt = np.array([np.polyfit(T, Pt[:, k], 1)[0] for k in (0, 1)])
                d = float(np.linalg.norm(g - h)); va = -float(np.dot(g - h, vt)) / d
                vs.append(np.linalg.norm(vt))
                if risk_t is None and va > 0.5 and d / va < 2.0: risk_t, risk_d = t, d
                if near_t is None and d < 1.5: near_t, near_d = t, d
        pos = pos + u * v * DT; t += DT
    lead = (near_t - risk_t) if (risk_t is not None and near_t is not None) else None
    md.append(f"| {v} | {np.median(vs) if vs else float('nan'):.2f} | {'-' if risk_d is None else f'{risk_d:.2f}'} | {'-' if near_d is None else f'{near_d:.2f}'} | {'-' if lead is None else f'{lead:.1f}'} |")
    print("C", v, risk_d, near_d, lead, flush=True)
md.append("\n충돌 위험 = 접근 속도 > 0.5m/s 이고 TTC(거리 ÷ 접근 속도) < 2초. 근접 = 거리 < 1.5m. 속도는 최근 1초 위치의 직선 기울기.")
md.append("\n가상 시험(합성 영상): bus.jpg에서 오린 사람 2명(키 1.7m 세운 판), yolo11n-pose, 어깨 키포인트 신뢰도 0.5, 재투영 오차 15px 이내로 두 카메라 짝 맞춤, 프레임 간격 0.1초.")
open("runs/vprox.md", "w").write("\n".join(md)); print("\n" + "\n".join(md))
