"""과제 2-2 (합성 시험): 조명 × 거리별 사람 검출 신뢰도가 어디서 무너지는지
- 사람 사진(bus.jpg, 서 있는 사람 4명)을 1280x720 화면에 '거리별 크기'로 축소·확대해 배치
  키 1.7m, 맥 카메라 렌즈 fy≈962 → 사람 높이(px) = 962 × 1.7 / 거리  (2m 818, 5m 327, 8m 204 ...)
- 조명: 밝기 배율(1.0 / 0.3 / 0.1 / 0.05) + 어두울수록 커지는 센서 노이즈
- 조건마다 노이즈·위치를 바꾼 20장 → 검출률(IoU≥0.5, conf≥0.4), 평균 신뢰도
사용법: python -u light2.py   (결과 runs/light2.md, 예시 runs/light2_<거리>_<밝기>.jpg)
"""
import numpy as np, cv2, os
from ultralytics import YOLO
from ultralytics.utils import ASSETS

W, H, FY, HUMAN = 1280, 720, 962, 1.7
DISTS = [2, 5, 8, 12]
LIGHTS = [("밝음 100%", 1.0), ("어둑 30%", 0.3), ("어두움 10%", 0.1), ("아주 어두움 5%", 0.05)]
src = cv2.imread(str(ASSETS / "bus.jpg"))
m = YOLO("engines/yolo11n_fp16.engine", task="detect")
ref = m.predict(src, classes=[0], conf=0.5, verbose=False)[0].boxes.xyxy.cpu().numpy()
ref = ref[(ref[:, 3] - ref[:, 1]) > 300]                      # 크게 보이는 서 있는 사람만 정답으로
mean_h = float((ref[:, 3] - ref[:, 1]).mean()); print(f"정답 사람 {len(ref)}명, 원본 평균 키 {mean_h:.0f}px")

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

def scene(dist, gain, seed):
    rng = np.random.default_rng(seed)
    s = FY * HUMAN / dist / mean_h                               # 축척
    img = cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((H, W, 3), 90, np.uint8)
    ox = int((W - img.shape[1]) / 2 + rng.integers(-40, 41)); oy = int(H * 0.95 - img.shape[0] * 0.93)  # 발이 화면 아래쪽에 오게
    x0, y0 = max(0, ox), max(0, oy); x1, y1 = min(W, ox + img.shape[1]), min(H, oy + img.shape[0])
    canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    g = canvas.astype(np.float32) * gain + rng.normal(0, 2 + 6 * (1 - gain), canvas.shape)   # 어두울수록 노이즈
    boxes = []
    for b in ref * s + [ox, oy, ox, oy]:
        c = np.clip(b, [0, 0, 0, 0], [W, H, W, H])
        if (c[2] - c[0]) * (c[3] - c[1]) > 0.4 * (b[2] - b[0]) * (b[3] - b[1]): boxes.append(c)   # 화면 안에 40% 이상 보이는 사람만
    return np.clip(g, 0, 255).astype(np.uint8), boxes, s

rows = []
for dist in DISTS:
    for lname, gain in LIGHTS:
        hit = tot = 0; confs = []
        for seed in range(20):
            img, gts, s = scene(dist, gain, seed)
            r = m.predict(img, classes=[0], conf=0.1, verbose=False)[0]
            pb, pc = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            for g in gts:
                tot += 1; best = max([(iou(g, p), c) for p, c in zip(pb, pc)], default=(0, 0))
                if best[0] >= 0.5:
                    confs.append(best[1]); hit += best[1] >= 0.4
            if seed == 0: cv2.imwrite(f"runs/light2_{dist}m_{int(gain*100)}.jpg", img)
        rows.append((dist, round(FY * HUMAN / dist), lname, f"{hit}/{tot}", round(100 * hit / max(tot, 1)), round(float(np.mean(confs)), 3) if confs else 0))
        print(rows[-1], flush=True)

md = "| 거리 (m) | 사람 키 (px) | 조명 | 검출 | 검출률 (%) | 평균 신뢰도 |\n|---|---|---|---|---|---|\n"
md += "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
md += "\n\n합성 시험: bus.jpg 사람을 거리별 크기로 1280x720에 배치(fy 962, 키 1.7m), 밝기 배율 + 노이즈, 조건당 20장. yolo11n FP16, 검출 = IoU≥0.5 & conf≥0.4. 2m는 사람이 화면보다 커서 다리가 잘림.\n"
open("runs/light2.md", "w").write(md); print("\n" + md)
