"""10단계(카메라 2대 거리 측정)를 '가상 카메라 2대 + 합성 영상'으로 재현 — 사람 인식(YOLO 포즈)까지 실제로 돌림
- 사람: bus.jpg에서 YOLOE로 사람을 오려 내(마스크) 키 1.7m의 세운 판(빌보드)으로 3D 공간에 배치
- 가상 카메라 2대: 1280x720, f=960px, 높이 1.2m, 간격 1.0m, 10° 아래로 기울임, 둘 다 +x(앞)를 봄
- 정답 위치를 알므로 방법별 오차를 바로 채점
  ankle1(10-3): cam0 발목 → 바닥(z=0)과 만나는 점
  stereo(10-4): 두 카메라 어깨 중심 삼각측량
  fused (10-5): 두 카메라 다 보이면 stereo, 한쪽만 보이면 어깨 시선 ∩ 높이 1.2m 평면
- 시나리오: S1 트인 곳 걷기 / S2 책상(높이 0.8m)에 다리 가림 / S3 옆으로 비켜 한 카메라에만 보임
사용법: python -u vstereo.py   (결과 runs/vstereo.md, 영상 runs/vstereo_S*.mp4)
"""
import numpy as np, cv2, json
from ultralytics import YOLO
from ultralytics.utils import ASSETS

W, H, F = 1280, 720, 960.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
TILT = np.deg2rad(10)
def cam(y):
    # 월드: x 앞, y 왼쪽, z 위  →  카메라: x 오른쪽, y 아래, z 앞
    Rw2c = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    Rt = np.array([[1, 0, 0], [0, np.cos(TILT), -np.sin(TILT)], [0, np.sin(TILT), np.cos(TILT)]])  # 아래로 기울임
    R = Rt @ Rw2c; C = np.array([0, y, 1.2]); t = -R @ C
    return {"R": R, "t": t, "C": C, "P": K @ np.hstack([R, t[:, None]])}
CAMS = [cam(+0.5), cam(-0.5)]          # cam0 왼쪽, cam1 오른쪽 (간격 1.0m)

def proj(c, X):
    x = c["P"] @ np.append(X, 1); return x[:2] / x[2], x[2]

# ---- 사람 오리기 ----
src = cv2.imread(str(ASSETS / "bus.jpg"))
seg = YOLO("yoloe-11s-seg.pt"); seg.set_classes(["person"], seg.get_text_pe(["person"]))
r = seg.predict(src, conf=0.3, verbose=False)[0]
bx = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
ok = [(cf[j], j) for j, b in enumerate(bx) if (b[3] - b[1]) > 300 and (b[2] - b[0]) / (b[3] - b[1]) > 0.25 and b[0] > 5 and b[2] < src.shape[1] - 5]
i = max(ok)[1]                                       # 화면 끝에 잘리지 않은, 몸 전체가 보이는 사람 중 가장 확실한 사람
x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
mask = np.zeros(src.shape[:2], np.uint8); cv2.fillPoly(mask, [r.masks.xy[i].astype(np.int32)], 1)   # 원본 좌표 다각형
person = src[y1:y2, x1:x2]; pmask = mask[y1:y2, x1:x2]
ph, pw = person.shape[:2]; HGT = 1.7; WID = HGT * pw / ph
print(f"사람 오림 {pw}x{ph}px → 키 {HGT}m, 폭 {WID:.2f}m")

def background(c):
    img = np.full((H, W, 3), (200, 205, 210), np.uint8)
    hz = proj(c, np.array([1000, 0, 1.2]))[0][1]                    # 지평선
    img[int(max(0, hz)):] = (150, 140, 130)
    for gx in np.arange(1, 15, 1.0):                                 # 바닥 격자
        a, za = proj(c, np.array([gx, -6, 0])); b, zb = proj(c, np.array([gx, 6, 0]))
        if za > 0 and zb > 0: cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (120, 112, 104), 1)
    for gy in np.arange(-6, 7, 1.0):
        a, za = proj(c, np.array([0.5, gy, 0])); b, zb = proj(c, np.array([15, gy, 0]))
        if za > 0 and zb > 0: cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (120, 112, 104), 1)
    return img

def put_plane(img, c, corners, tex, tmask=None):
    pts, zs = zip(*[proj(c, X) for X in corners])
    if min(zs) <= 0.1: return
    th, tw = tex.shape[:2]
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), np.float32(pts))
    warped = cv2.warpPerspective(tex, Hm, (W, H))
    m = cv2.warpPerspective(tmask if tmask is not None else np.ones((th, tw), np.uint8), Hm, (W, H))
    img[m > 0] = warped[m > 0]

def render(c, P, desk):
    img = background(c)
    X, Y = P; hw = WID / 2
    put_plane(img, c, [np.array([X, Y + hw, HGT]), np.array([X, Y - hw, HGT]), np.array([X, Y - hw, 0]), np.array([X, Y + hw, 0])], person, pmask)
    if desk:                                                           # 사람 앞 책상(가로 2.4m, 높이 0.8m)
        dx, dy = desk
        put_plane(img, c, [np.array([dx, dy + 1.2, 0.8]), np.array([dx, dy - 1.2, 0.8]), np.array([dx, dy - 1.2, 0]), np.array([dx, dy + 1.2, 0])],
                  np.full((40, 120, 3), (60, 80, 110), np.uint8))
    return img

pose = YOLO("yolo11n-pose.pt")
def kp(img):
    r = pose.predict(img, conf=0.3, verbose=False)[0]
    if r.keypoints is None or len(r.boxes) == 0: return None
    j = int(np.argmax(r.boxes.conf.cpu().numpy()))
    return r.keypoints.xy[j].cpu().numpy(), r.keypoints.conf[j].cpu().numpy()

def ray(c, uv):
    d = c["R"].T @ np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]); return c["C"], d / np.linalg.norm(d)
def on_plane(c, uv, z):
    o, d = ray(c, uv)
    if abs(d[2]) < 1e-6: return None
    s = (z - o[2]) / d[2]
    return (o + s * d)[:2] if s > 0 else None
def tri(uv0, uv1):
    X = cv2.triangulatePoints(CAMS[0]["P"], CAMS[1]["P"], np.float64(uv0).reshape(2, 1), np.float64(uv1).reshape(2, 1))
    return (X[:3] / X[3]).ravel()

L_SH, R_SH, L_AN, R_AN, KC = 5, 6, 15, 16, 0.5
def estimate(k0, k1):
    out = {"ankle1": None, "stereo": None, "fused": None}
    def mid(k, a, b): return (k[0][a] + k[0][b]) / 2 if k is not None and k[1][a] > KC and k[1][b] > KC else None
    a0 = mid(k0, L_AN, R_AN)
    if a0 is not None: out["ankle1"] = on_plane(CAMS[0], a0, 0.0)
    s0, s1 = mid(k0, L_SH, R_SH), mid(k1, L_SH, R_SH)
    if s0 is not None and s1 is not None:
        out["stereo"] = tri(s0, s1)[:2]; out["fused"] = out["stereo"]
    elif s0 is not None: out["fused"] = on_plane(CAMS[0], s0, 1.2)
    elif s1 is not None: out["fused"] = on_plane(CAMS[1], s1, 1.2)
    return out

SCEN = {"S1 트인 곳 걷기 (8m → 1.5m)": ([(x, 0.2) for x in np.linspace(8, 1.5, 40)], None),
        "S2 책상 뒤 걷기 (다리 가림)": ([(x, 0.2) for x in np.linspace(6, 3.2, 30)], (2.6, 0.2)),
        "S3 옆으로 비켜 감 (한 카메라에만)": ([(4.0, y) for y in np.linspace(0.2, -3.0, 32)], None)}
rows = []
for name, (path, desk) in SCEN.items():
    vw = cv2.VideoWriter(f"runs/vstereo_{name[:2]}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 8, (W, H // 2))
    errs = {m: [] for m in ("ankle1", "stereo", "fused")}; miss = {m: 0 for m in errs}
    for P in path:
        im0, im1 = render(CAMS[0], P, desk), render(CAMS[1], P, desk)
        k0, k1 = kp(im0), kp(im1); est = estimate(k0, k1)
        for m, e in est.items():
            if e is None: miss[m] += 1
            else: errs[m].append(float(np.linalg.norm(np.array(e) - np.array(P))))
        view = np.hstack([cv2.resize(im0, (W // 2, H // 2)), cv2.resize(im1, (W // 2, H // 2))])
        txt = " | ".join(f"{m}:{'-' if e is None else f'{e[0]:.2f},{e[1]:.2f}'}" for m, e in est.items())
        cv2.putText(view, f"GT {P[0]:.2f},{P[1]:.2f}  {txt}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2); vw.write(view)
    vw.release()
    for m in errs:
        n = len(path); e = np.array(errs[m]) if errs[m] else np.array([np.nan])
        rows.append({"시나리오": name, "방법": m, "위치 나옴": f"{n - miss[m]}/{n}", "평균 오차 (m)": round(float(np.nanmean(e)), 2),
                     "최대 오차 (m)": round(float(np.nanmax(e)), 2)})
        print(rows[-1], flush=True)
keys = list(rows[0])
md = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n" + "\n".join("| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows)
md += "\n\n가상 시험: 카메라 2대(1280x720, f 960px, 높이 1.2m, 간격 1.0m, 10° 아래), 사람 = bus.jpg에서 오린 키 1.7m 세운 판, yolo11n-pose, 키포인트 신뢰도 0.5, fused 한쪽 추정 높이 1.2m. 오차 = 바닥 위치(x, y) 거리.\n"
open("runs/vstereo.md", "w").write(md); print("\n" + md)
