"""과제 2-3: 추적(ByteTrack) + 가상 선 통과 IN/OUT 세기 → MQTT 발행 + 웹 화면
- 박스 하단 중앙이 선의 어느 쪽인지 외적 부호로 판정, 부호가 바뀔 때만 1회 카운트 (같은 ID 중복 방지)
- 웹: http://<젯슨>:8086  (선, 박스, ID, IN/OUT 숫자)
사용법: python -u line2.py --src rtsp://localhost:8554/mac --line 0.5,0.1,0.5,0.9 --model engines/yolo11n_fp16.engine
  --line x1,y1,x2,y2 (화면 비율 0~1). 위→아래로 그은 세로선이면 왼→오 이동 = IN
"""
import argparse, json, os, subprocess, threading, time
import cv2, numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="rtsp://localhost:8554/mac")
ap.add_argument("--line", default="0.5,0.1,0.5,0.9")
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--port", type=int, default=8096)
ap.add_argument("--margin", type=float, default=0.06, help="선 양쪽 여유 폭(화면 너비 비율). 이 안에서는 판정 안 함(히스테리시스)")
ap.add_argument("--topic", default="aiot/space/lab1/line")
a = ap.parse_args()

cap = cv2.VideoCapture(a.src if not a.src.isdigit() else int(a.src))
latest = {"f": None}; lk = threading.Lock()
def reader():
    while True:
        ok, f = cap.read()
        if ok:
            with lk: latest["f"] = f
        else: time.sleep(0.02)
threading.Thread(target=reader, daemon=True).start()

def pub(obj):
    try: subprocess.Popen(["mosquitto_pub", "-t", a.topic, "-q", "1", "-m", json.dumps(obj, ensure_ascii=False)])
    except Exception as e: print("MQTT 실패:", e)

state = {"jpg": b"", "in": 0, "out": 0, "fps": 0}
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        if self.path.startswith("/img"):
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers(); self.wfile.write(state["jpg"]); return
        if self.path.startswith("/st"):
            b = json.dumps({k: state[k] for k in ("in", "out", "fps")}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b); return
        html = """<body style="background:#111;color:#fff;font:22px sans-serif;margin:8px"><div id=t></div>
<img id=i style="max-width:100%;max-height:85vh"><script>
function img(){const n=new Image();n.onload=()=>{i.src=n.src;setTimeout(img,80)};n.onerror=()=>setTimeout(img,1000);n.src='/img?'+Date.now()}
async function st(){try{const r=await (await fetch('/st')).json();t.innerHTML='IN <b style=color:#4f4>'+r.in+'</b> · OUT <b style=color:#f84>'+r.out+'</b> · '+r.fps+' fps'}catch(e){t.textContent='꺼짐'}setTimeout(st,300)}
img();st();</script></body>"""
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(html.encode())
threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever(), daemon=True).start()

m = YOLO(a.model, task="detect")
lx1, ly1, lx2, ly2 = map(float, a.line.split(","))
side, log = {}, open("runs/line2_log.jsonl", "a")
print(f"웹 화면: http://<보드 주소>:{a.port}   MQTT: {a.topic}   여유 폭 {a.margin}")
n, t0 = 0, time.time()
while True:
    with lk: f = None if latest["f"] is None else latest["f"].copy()
    if f is None: time.sleep(0.05); continue
    H_, W_ = f.shape[:2]; P1 = np.array([lx1 * W_, ly1 * H_]); P2 = np.array([lx2 * W_, ly2 * H_])
    r = m.track(f, persist=True, classes=[0], conf=a.conf, tracker="bytetrack.yaml", verbose=False)[0]
    if r.boxes.id is not None:
        for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
            foot = np.array([(x1 + x2) / 2, y2])                       # 박스 하단 중앙
            d = P2 - P1; cr = d[0] * (foot[1] - P1[1]) - d[1] * (foot[0] - P1[0])   # 외적
            dist = cr / (np.linalg.norm(d) + 1e-9)                                 # 선까지 거리(px, 부호 포함)
            s = np.sign(cr) if abs(dist) > a.margin * W_ else 0                   # 여유 폭 안이면 판정 보류
            prev = side.get(tid)
            if prev is not None and s != 0 and s != prev:
                kind = "IN" if s < prev else "OUT"
                state["in" if kind == "IN" else "out"] += 1
                ev = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "id": tid, "dir": kind, "in": state["in"], "out": state["out"]}
                print("통과:", ev); log.write(json.dumps(ev, ensure_ascii=False) + "\n"); log.flush(); pub(ev)
            if s != 0: side[tid] = s
            cv2.rectangle(f, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(f, f"#{tid}", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.circle(f, tuple(foot.astype(int)), 6, (0, 200, 255), -1)
    cv2.line(f, tuple(P1.astype(int)), tuple(P2.astype(int)), (255, 0, 255), 3)
    if a.margin > 0:
        off = np.array([-(P2 - P1)[1], (P2 - P1)[0]]); off = off / (np.linalg.norm(off) + 1e-9) * a.margin * W_
        for sg in (1, -1):
            cv2.line(f, tuple((P1 + sg * off).astype(int)), tuple((P2 + sg * off).astype(int)), (180, 0, 180), 1)
    cv2.putText(f, f"IN {state['in']}  OUT {state['out']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    state["jpg"] = cv2.imencode(".jpg", cv2.resize(f, (W_ // 2, H_ // 2)))[1].tobytes()
    n += 1
    if time.time() - t0 > 2: state["fps"] = round(n / (time.time() - t0), 1); n, t0 = 0, time.time()
