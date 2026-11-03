"""가상 시험(합성 영상) 공통 모듈: 가상 카메라, 사람 오리기, 장면 그리기, 포즈, 삼각측량
좌표: 월드 x = 앞, y = 왼쪽, z = 위 (m)
"""
import numpy as np, cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS

W, H, F = 1280, 720, 960.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])

def make_cam(x, y, z, tilt_deg=10):
    Rw2c = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    t_ = np.deg2rad(tilt_deg)
    Rt = np.array([[1, 0, 0], [0, np.cos(t_), -np.sin(t_)], [0, np.sin(t_), np.cos(t_)]])
    R = Rt @ Rw2c; C = np.array([x, y, z], float); t = -R @ C
    return {"R": R, "t": t, "C": C, "P": K @ np.hstack([R, t[:, None]])}

def proj(c, X):
    x = c["P"] @ np.append(np.asarray(X, float), 1); return x[:2] / x[2], x[2]

def cut_people(n=3):
    """bus.jpg에서 몸 전체가 보이는 사람을 오려 (이미지, 마스크) 목록으로"""
    src = cv2.imread(str(ASSETS / "bus.jpg"))
    seg = YOLO("yoloe-11s-seg.pt"); seg.set_classes(["person"], seg.get_text_pe(["person"]))
    r = seg.predict(src, conf=0.3, verbose=False)[0]
    out = []
    for j in np.argsort(-r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy().astype(int)
        if (y2 - y1) < 300 or (x2 - x1) / (y2 - y1) < 0.25 or x1 <= 5 or x2 >= src.shape[1] - 5: continue
        m = np.zeros(src.shape[:2], np.uint8); cv2.fillPoly(m, [r.masks.xy[j].astype(np.int32)], 1)
        out.append((src[y1:y2, x1:x2].copy(), m[y1:y2, x1:x2].copy()))
    base = list(out)
    while len(out) < n:                                   # 모자라면 좌우 반전으로 늘림
        im, mk = base[len(out) % len(base)]; out.append((cv2.flip(im, 1), cv2.flip(mk, 1)))
    return out[:n]

def background(c, wall_x=12.0):
    img = np.full((H, W, 3), (200, 205, 210), np.uint8)
    hz, z = proj(c, [1000, c["C"][1], c["C"][2]])
    if z > 0: img[int(np.clip(hz[1], 0, H)):] = (150, 140, 130)
    for gx in np.arange(1, 15, 1.0):
        a, za = proj(c, [gx, -6, 0]); b, zb = proj(c, [gx, 6, 0])
        if za > 0 and zb > 0: cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (120, 112, 104), 1)
    for gy in np.arange(-6, 7, 1.0):
        a, za = proj(c, [0.3, gy, 0]); b, zb = proj(c, [15, gy, 0])
        if za > 0 and zb > 0: cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), (120, 112, 104), 1)
    return img

def put_plane(img, c, corners, tex, tmask=None):
    res = [proj(c, X) for X in corners]
    if min(z for _, z in res) <= 0.1: return
    th, tw = tex.shape[:2]
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), np.float32([p for p, _ in res]))
    warped = cv2.warpPerspective(tex, Hm, (W, H))
    m = cv2.warpPerspective(tmask if tmask is not None else np.ones((th, tw), np.uint8), Hm, (W, H))
    img[m > 0] = warped[m > 0]

def render(c, actors, people, boxes=(), hgt=1.7):
    """actors: [(x, y, 사람번호)], boxes: [(x, y, 폭, 높이, 색)] 세운 판(책상 등). 먼 것부터 그림"""
    img = background(c); items = []
    for x, y, k in actors:
        im, mk = people[k]; wid = hgt * im.shape[1] / im.shape[0]
        items.append((np.linalg.norm(np.array([x, y, 0.9]) - c["C"]), "p", (x, y, wid, hgt, im, mk)))
    for x, y, w, h, col in boxes:
        items.append((np.linalg.norm(np.array([x, y, h / 2]) - c["C"]) - 0.01, "b", (x, y, w, h, np.full((40, 120, 3), col, np.uint8), None)))
    for _, _, (x, y, w, h, tex, mk) in sorted(items, key=lambda t: -t[0]):
        put_plane(img, c, [[x, y + w / 2, h], [x, y - w / 2, h], [x, y - w / 2, 0], [x, y + w / 2, 0]], tex, mk)
    return img

_pose = None
def pose_all(img, conf=0.3):
    global _pose
    if _pose is None: _pose = YOLO("yolo11n-pose.pt")
    r = _pose.predict(img, conf=conf, verbose=False)[0]
    if r.keypoints is None or len(r.boxes) == 0: return []
    return [(r.keypoints.xy[j].cpu().numpy(), r.keypoints.conf[j].cpu().numpy(), r.boxes.xyxy[j].cpu().numpy()) for j in range(len(r.boxes))]

def shoulder(p, kc=0.5):
    k, c, _ = p
    return (k[5] + k[6]) / 2 if c[5] > kc and c[6] > kc else None

def ray(c, uv):
    d = c["R"].T @ np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]); return c["C"], d / np.linalg.norm(d)

def on_plane(c, uv, z):
    o, d = ray(c, uv)
    if abs(d[2]) < 1e-6: return None
    s = (z - o[2]) / d[2]
    return (o + s * d)[:2] if s > 0 else None

def tri(c0, c1, uv0, uv1):
    X = cv2.triangulatePoints(c0["P"], c1["P"], np.float64(uv0).reshape(2, 1), np.float64(uv1).reshape(2, 1))
    X = (X[:3] / X[3]).ravel()
    e = np.linalg.norm(proj(c0, X)[0] - uv0) + np.linalg.norm(proj(c1, X)[0] - uv1)
    return X, e

def stereo_people(c0, c1, P0, P1, max_err=15):
    """두 카메라의 사람들을 재투영 오차가 가장 작은 짝끼리 맞춰 3D 어깨 위치 목록으로"""
    s0 = [shoulder(p) for p in P0]; s1 = [shoulder(p) for p in P1]
    cand = []
    for i, a in enumerate(s0):
        for j, b in enumerate(s1):
            if a is None or b is None: continue
            X, e = tri(c0, c1, a, b)
            if e < max_err and X[0] > 0.3: cand.append((e, i, j, X))
    used0, used1, out = set(), set(), []
    for e, i, j, X in sorted(cand, key=lambda t: t[0]):
        if i in used0 or j in used1: continue
        used0.add(i); used1.add(j); out.append(X)
    return out
