"""가상 시험(합성 영상): 과제 5 심화 — 기준선 6cm / 12cm 비교, ΔZ ∝ 1/B 확인
- 평판에 나란히 놓은(정렬형) 카메라 2대: 높이 1.2m, 기울기 0°, 간격 B = 0.06m 또는 0.12m
- 사람(키 1.7m)을 거리 Z = 1, 2, 3, 4, 5m에 세우고 좌우 위치·사람을 바꿔 TRIALS번 찍는다.
- 두 화면에 각각 독립 센서 노이즈(σ = NOISE)를 넣는다 → 실제 카메라처럼 키포인트가 화면마다 조금씩 다르게 잡힘
- 거리 = 두 화면 어깨 중점의 시차 d로 Z = f·B/d. 오차와 시차의 흔들림(Δd)을 기록하고
  이론값 ΔZ ≈ Z²·Δd/(f·B) 와 비교한다.
사용법: python -u vbaseline.py   (결과 runs/vbaseline.md)
"""
import numpy as np, cv2
from vscene import make_cam, cut_people, render, pose_all, shoulder, F

TRIALS, NOISE, KC = 12, 6.0, 0.5
ZS = [1.0, 2.0, 3.0, 4.0, 5.0]
PEOPLE = cut_people(3)
rng = np.random.default_rng(0)

def noisy(img):
    return np.clip(img.astype(np.float32) + rng.normal(0, NOISE, img.shape), 0, 255).astype(np.uint8)

def best(P, u_hint):
    """화면에서 어깨가 보이는 사람 중 예상 위치에 가장 가까운 사람"""
    c = [(abs(s[0] - u_hint), s) for p in P if (s := shoulder(p, KC)) is not None]
    return min(c, key=lambda t: t[0])[1] if c else None

res = {}
for B in (0.06, 0.12):
    cl = make_cam(0, +B / 2, 1.2, tilt_deg=0); cr = make_cam(0, -B / 2, 1.2, tilt_deg=0)
    for Z in ZS:
        errs, disps = [], []
        for k in range(TRIALS):
            y = rng.uniform(-0.25, 0.25) * Z                 # 화면 안쪽에서 좌우 위치를 바꿈
            actors = [(Z, y, k % len(PEOPLE))]
            il, ir = noisy(render(cl, actors, PEOPLE)), noisy(render(cr, actors, PEOPLE))
            u_hint = 640 - F * y / Z
            sl, sr = best(pose_all(il), u_hint), best(pose_all(ir), u_hint)
            if sl is None or sr is None: continue
            d = sl[0] - sr[0]                                 # 정렬형이므로 가로 차이 = 시차
            if d <= 0: continue
            errs.append(F * B / d - Z); disps.append(d)
        res[(B, Z)] = (np.array(errs), np.array(disps))
        e = res[(B, Z)][0]
        print(f"B={B*100:.0f}cm Z={Z}m 성공 {len(e)}/{TRIALS} 평균오차 {np.abs(e).mean() if len(e) else float('nan'):.3f}m", flush=True)

md = ["### 과제 5 심화 — 기준선 6cm vs 12cm (가상 시험, 정렬형 스테레오, 어깨 시차)\n",
      "| 거리 Z (m) | 6cm: 성공 · 평균 |오차| · 오차율 | 12cm: 성공 · 평균 |오차| · 오차율 | 오차 비 (6cm ÷ 12cm) | 이론 ΔZ 6cm / 12cm (Δd 실측) |",
      "|---|---|---|---|---|"]
for Z in ZS:
    cells, mae = [], {}
    for B in (0.06, 0.12):
        e, d = res[(B, Z)]
        if len(e) == 0: cells.append("0 · — · —"); mae[B] = np.nan; continue
        mae[B] = np.abs(e).mean()
        cells.append(f"{len(e)}/{TRIALS} · {mae[B]:.3f}m · {mae[B]/Z*100:.1f}%")
    th = []
    for B in (0.06, 0.12):
        e, d = res[(B, Z)]
        dd = d.std() if len(d) > 1 else np.nan
        th.append(f"{Z**2*dd/(F*B):.3f}m (Δd {dd:.2f}px)")
    ratio = mae[0.06] / mae[0.12] if mae[0.12] and not np.isnan(mae[0.12]) else np.nan
    md.append(f"| {Z:.0f} | {cells[0]} | {cells[1]} | {ratio:.1f} | {th[0]} / {th[1]} |")
md.append(f"\n가상 시험(합성 영상): 카메라 2대 1280x720 초점거리 {F:.0f}px, 높이 1.2m, 기울기 0°, 평행 배치. "
          f"bus.jpg에서 오린 사람 3명(키 1.7m 세운 판), 화면마다 독립 노이즈 σ={NOISE:.0f}, yolo11n-pose, 키포인트 신뢰도 {KC}, "
          f"거리마다 {TRIALS}번(좌우 위치 무작위). 오차 = Z_측정 − Z_정답.")
open("runs/vbaseline.md", "w").write("\n".join(md)); print("\n".join(md))
