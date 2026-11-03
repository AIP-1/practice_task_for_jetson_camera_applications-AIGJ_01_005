"""가상 시험(합성 영상): 과제 6 심화 — 거리 측정의 자기 감시 ('보정 필요' 알림)
- 정렬형 카메라 2대(기준선 0.12m, 높이 1.2m)가 정면 벽(무늬 있는 판, 3.0m)을 본다.
- 매 프레임 SGBM 시차 지도의 벽 영역 중앙값으로 벽 거리를 잰다. 처음 30프레임 평균이 기준값.
- 시험 중간에 카메라 한 대가 살짝 틀어진다(누가 건드림): 오른쪽 카메라 좌우 회전 YAW도.
  → 정렬이 깨져 시차가 밀리고 벽 거리가 틀어진다.
- 감시 규칙: |측정 − 기준| / 기준 > TH 가 N프레임 연속이면 '보정 필요'.
- 결과: 틀어지기 전 헛알람 수, 틀어진 뒤 알람까지 걸린 프레임, 회전 크기별 거리 변화.
사용법: python -u vwall.py   (결과 runs/vwall.md)
"""
import numpy as np, cv2
from vscene import make_cam, put_plane, W, H, F

B, Z_WALL, TH, N = 0.12, 3.0, 0.03, 5
rng = np.random.default_rng(0)
tex = (rng.random((300, 400)) * 255).astype(np.uint8)
tex = cv2.cvtColor(cv2.GaussianBlur(cv2.resize(tex, (800, 600), interpolation=cv2.INTER_NEAREST), (3, 3), 0), cv2.COLOR_GRAY2BGR)

def yaw(c, deg):
    """카메라를 제자리에서 좌우로 deg 만큼 돌림"""
    t_ = np.deg2rad(deg); Ry = np.array([[np.cos(t_), 0, np.sin(t_)], [0, 1, 0], [-np.sin(t_), 0, np.cos(t_)]])
    R = Ry @ c["R"]; t = -R @ c["C"]
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    return {**c, "R": R, "t": t, "P": K @ np.hstack([R, t[:, None]])}

def shot(c):
    img = np.full((H, W, 3), 128, np.uint8)
    put_plane(img, c, [[Z_WALL, 2.0, 2.4], [Z_WALL, -2.0, 2.4], [Z_WALL, -2.0, 0.0], [Z_WALL, 2.0, 0.0]], tex)
    noise = rng.normal(0, 4, img.shape)
    return cv2.cvtColor(np.clip(img + noise, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7, P1=8 * 49, P2=32 * 49,
                             uniquenessRatio=10, speckleWindowSize=50, speckleRange=2)
def wall_dist(cl, cr):
    d = sgbm.compute(shot(cl), shot(cr)).astype(np.float32) / 16
    roi = d[H // 2 - 80:H // 2 + 80, W // 2 - 120:W // 2 + 120]
    v = roi[roi > 0.5]
    return F * B / np.median(v) if len(v) > 100 else np.nan

cl0 = make_cam(0, +B / 2, 1.2, tilt_deg=0); cr0 = make_cam(0, -B / 2, 1.2, tilt_deg=0)

# ① 회전 크기별 벽 거리 변화
rows = []
for deg in (0, 0.05, 0.1, 0.2, 0.3, 0.5):
    z = np.nanmedian([wall_dist(cl0, yaw(cr0, deg)) for _ in range(3)])
    rows.append((deg, z, (z - Z_WALL) / Z_WALL * 100))
    print(f"회전 {deg}° → 벽 {z:.3f}m ({rows[-1][2]:+.1f}%)", flush=True)

# ② 시간 흐름 시험: 120프레임, 60프레임째 오른쪽 카메라가 0.2° 틀어짐
YAW_BUMP, T_BUMP, T_END = 0.2, 60, 120
zs, alarm_at, false_alarms, streak = [], None, 0, 0
for k in range(T_END):
    z = wall_dist(cl0, yaw(cr0, YAW_BUMP if k >= T_BUMP else 0))
    zs.append(z)
    if k < 30: continue
    base = np.nanmean(zs[:30])
    off = abs(z - base) / base > TH
    streak = streak + 1 if off else 0
    if streak >= N:
        if k < T_BUMP: false_alarms += 1; streak = 0
        elif alarm_at is None: alarm_at = k
base = np.nanmean(zs[:30]); jitter = np.nanstd(zs[:T_BUMP]) / base * 100

md = ["### 과제 6 심화 — 거리 자기 감시: 기준 벽 거리로 '보정 필요' 판단 (가상 시험)\n",
      "| 오른쪽 카메라 회전 | 측정 벽 거리 (m) | 정답 3.0m 대비 |", "|---|---|---|"]
md += [f"| {d}° | {z:.3f} | {p:+.1f}% |" for d, z, p in rows]
md.append(f"\n시간 흐름 시험: 기준값 {base:.3f}m(처음 30프레임 평균), 평소 흔들림 ±{jitter:.2f}%. "
          f"{T_BUMP}프레임째 오른쪽 카메라 {YAW_BUMP}° 틀어짐 → "
          + (f"{alarm_at}프레임째 '보정 필요'(틀어진 뒤 {alarm_at - T_BUMP + 1}프레임). " if alarm_at else "알람 없음. ")
          + f"틀어지기 전 헛알람 {false_alarms}회. 규칙: 기준 대비 {TH*100:.0f}% 넘는 프레임이 {N}번 연속.")
md.append(f"\n가상 시험(합성 영상): 정렬형 카메라 2대 1280x720 초점거리 {F:.0f}px, 기준선 {B}m, 높이 1.2m, "
          f"무늬 있는 벽 {Z_WALL}m, 화면 노이즈 σ=4, StereoSGBM(64 시차, 블록 7), 벽 가운데 영역 시차 중앙값.")
open("runs/vwall.md", "w").write("\n".join(md)); print("\n".join(md))
