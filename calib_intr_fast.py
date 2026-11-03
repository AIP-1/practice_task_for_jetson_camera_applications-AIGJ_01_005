"""10-1 렌즈 보정 (IP캠/RTSP용 빠른 버전)
- 최신 프레임만 쓰는 스레드: 처리하는 동안 쌓인 옛 프레임을 버려서 화면이 밀리지 않음
- 체커보드 찾기는 절반 크기로 빠르게, 코너 위치는 원본 크기에서 정밀하게
- 진행 상황 + 마지막 화면을 http://<젯슨>:8002 에서 볼 수 있음
사용법: python -u calib_intr_fast.py --src rtsp://localhost:8554/tab1 --name tab1
결과: calib/<name>_intr.npz (K, dist, res, rms)
"""
import argparse, os, time, threading, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square", type=float, default=0.025)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--gap", type=float, default=1.0, help="저장 사이 최소 간격(초)")
ap.add_argument("--port", type=int, default=8002)
a = ap.parse_args()
os.makedirs("calib", exist_ok=True)

cap = cv2.VideoCapture(a.src, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit(f"입력을 열 수 없음: {a.src}")
latest = {"f": None}; lock = threading.Lock(); run = True
def reader():
    while run:
        ok, f = cap.read()
        if ok:
            with lock: latest["f"] = f
        else:
            time.sleep(0.05)
threading.Thread(target=reader, daemon=True).start()

state = {"got": 0, "found": False, "jpg": b""}
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        if self.path.startswith("/img"):
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers()
            self.wfile.write(state["jpg"]); return
        if self.path.startswith("/st"):
            b = json.dumps({"got": state["got"], "n": a.n, "found": state["found"], "done": state.get("done", "")}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b); return
        html = """<body style="background:#111;color:#fff;font:20px sans-serif;margin:8px">
        <div id=t>연결 중...</div><img id=i style="max-width:100%;max-height:88vh;display:block">
        <script>
        const i=document.getElementById('i'),t=document.getElementById('t');
        function img(){const n=new Image();n.onload=()=>{i.src=n.src;setTimeout(img,150)};n.onerror=()=>setTimeout(img,1000);n.src='/img?'+Date.now()}
        async function st(){try{const r=await (await fetch('/st')).json();
          t.innerHTML='<b>NAME</b> 모은 장수 '+r.got+'/'+r.n+' · '+(r.found?'<span style=color:#4f4>검출됨</span>':'<span style=color:#f66>못 찾음</span>')+(r.done?' · <b style=color:#ff4>'+r.done+'</b>':'')}
          catch(e){t.innerHTML='<span style=color:#f66>젯슨 프로그램이 꺼졌거나 계산 중 (터미널 확인)</span>'} setTimeout(st,500)}
        img();st();</script></body>""".replace("NAME", a.name)
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
        self.wfile.write(html.encode())
threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever(), daemon=True).start()

objp = np.zeros((a.cols * a.rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
obj_pts, img_pts, last_t, last_c, prev = [], [], 0, None, None
cells = {}
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print(f"체커보드({a.cols}x{a.rows})를 화면 곳곳·여러 각도로 천천히 움직이세요. 목표 {a.n}장")
print(f"진행 화면: http://<보드 주소>:{a.port}  (맥 브라우저)")
while len(img_pts) < a.n:
    with lock:
        f = None if latest["f"] is None else latest["f"].copy()
    if f is None:
        time.sleep(0.05); continue
    H_, W_ = f.shape[:2]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (W_ // 2, H_ // 2))
    ok, c = cv2.findChessboardCorners(small, (a.cols, a.rows),
                                      cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
    state["found"] = bool(ok)
    if not ok: prev = None
    view = f.copy()
    if ok:
        c = cv2.cornerSubPix(g, (c * 2).astype(np.float32), (11, 11), (-1, -1), crit).reshape(-1, 1, 2)
        cv2.drawChessboardCorners(view, (a.cols, a.rows), c, ok)
        moved = last_c is None or np.linalg.norm(c.mean(0) - last_c.mean(0)) > 40 or \
                abs(cv2.contourArea(cv2.convexHull(c)) - cv2.contourArea(cv2.convexHull(last_c))) > 0.15 * cv2.contourArea(cv2.convexHull(last_c))
        still = prev is not None and np.abs(c - prev).max() < 1.5   # 직전 프레임과 거의 같은 위치 = 멈춰 있음 (흔들림 방지)
        prev = c
        ctr = c.reshape(-1, 2).mean(0); cell = (min(2, int(ctr[0] * 3 // W_)), min(2, int(ctr[1] * 3 // H_)))
        full = cells.get(cell, 0) >= 3
        if time.time() - last_t > a.gap and moved and still and not full:
            cells[cell] = cells.get(cell, 0) + 1
            obj_pts.append(objp); img_pts.append(c); last_t = time.time(); last_c = c
            state["got"] = len(img_pts)
            cv2.imwrite(f"calib/{a.name}_last.jpg", view)
            print(f"  {len(img_pts)}/{a.n}  (위치 x={c[:,0,0].mean():.0f}, y={c[:,0,1].mean():.0f})")
    for q in img_pts:
        cv2.circle(view, tuple(int(v) for v in q.reshape(-1, 2).mean(0)), 12, (255, 200, 0), -1)
    for cx in range(3):
        for cy in range(3):
            k = cells.get((cx, cy), 0)
            cv2.putText(view, f"{k}/3", (W_ * cx // 3 + 15, H_ * cy // 3 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (0, 220, 0) if k >= 2 else ((0, 200, 255) if k == 1 else (0, 0, 255)), 3)
    for gx in (1, 2): cv2.line(view, (W_ * gx // 3, 0), (W_ * gx // 3, H_), (80, 80, 80), 1)
    for gy in (1, 2): cv2.line(view, (0, H_ * gy // 3), (W_, H_ * gy // 3), (80, 80, 80), 1)
    cv2.putText(view, f"{len(img_pts)}/{a.n}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0) if ok else (0, 0, 255), 3)
    state["jpg"] = cv2.imencode(".jpg", cv2.resize(view, (W_ // 2, H_ // 2)))[1].tobytes()

run = False
state["done"] = "20장 완료, 계산 중 (터미널에 RMS 나옴)"
print("계산 중...")
flags = cv2.CALIB_FIX_K3
def calib(op, ip):
    r, K, d, rv, tv = cv2.calibrateCamera(op, ip, (W_, H_), None, None, flags=flags)
    errs = [cv2.norm(ip[i], cv2.projectPoints(op[i], rv[i], tv[i], K, d)[0], cv2.NORM_L2) / np.sqrt(len(op[i])) for i in range(len(op))]
    return r, K, d, np.array(errs)
rms, K, dist, errs = calib(obj_pts, img_pts)
print(f"1차 RMS {rms:.3f}px, 사진별 오차: {errs.round(2).tolist()}")
keep = errs < max(2 * np.median(errs), 0.6)
if keep.sum() >= 10 and (~keep).any():
    obj_pts = [o for o, k in zip(obj_pts, keep) if k]; img_pts = [q for q, k in zip(img_pts, keep) if k]
    rms, K, dist, errs = calib(obj_pts, img_pts)
    print(f"오차 큰 사진 {(~keep).sum()}장 제외 → {len(img_pts)}장으로 다시 계산")
np.savez(f"calib/{a.name}_intr.npz", K=K, dist=dist, res=np.array([W_, H_]), rms=rms)
print(f"\n재투영 RMS {rms:.3f}px  (0.5 이하 양호, 1.0 이상이면 다시 촬영)")
print(f"K=\n{K.round(1)}\ndist={dist.ravel().round(3)}\n저장: calib/{a.name}_intr.npz  (해상도 {W_}x{H_})")
