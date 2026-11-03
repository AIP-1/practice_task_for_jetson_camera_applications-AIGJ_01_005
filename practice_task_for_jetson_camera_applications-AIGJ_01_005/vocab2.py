"""과제 2-4: COCO에 없는 물품 5종을 세 방식으로 인식
 ① YOLO 라벨 그대로 (COCO 80종)
 ② YOLO 박스를 이미지 임베딩(EmbeddingGemma 2)으로 재라벨 (유사도 < 0.20 이면 YOLO 라벨 유지)
 ③ 개방 어휘 검출 YOLOE (텍스트로 클래스 지정)
물품 하나씩: 준비 15초 → 측정 8초 동안 매 프레임 판정, 가장 많이 나온 답을 기록
사용법: python -u vocab2.py --item 빗      (웹 http://100.92.224.43:8097 에서 화면 확인)
결과: runs/vocab2_r2.csv, 스냅샷 runs/vocab2_<물품>.jpg
"""
import argparse, csv, os, threading, time, json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np
from PIL import Image
from ultralytics import YOLO

ITEMS = {"빗": "hair brush", "약통": "pill bottle", "보조배터리": "power bank", "파우치": "cosmetic pouch", "충전기": "phone charger"}
ap = argparse.ArgumentParser()
ap.add_argument("--item", nargs="+", required=True, choices=list(ITEMS), help="여러 개 가능: --item 약통 보조배터리 파우치 충전기")
ap.add_argument("--src", default="rtsp://localhost:8554/mac")
ap.add_argument("--port", type=int, default=8097)
ap.add_argument("--thr", type=float, default=0.20)
ap.add_argument("--prep", type=int, default=15)
ap.add_argument("--yconf", type=float, default=0.15, help="YOLOE 신뢰도 임계값")
ap.add_argument("--prompt", nargs="*", default=[], help="프롬프트 바꾸기: 파우치=small zipper pouch")
ap.add_argument("--meas", type=int, default=8)
a = ap.parse_args()
for kv in a.prompt:
    k, v = kv.split("=", 1); ITEMS[k] = v
names_ko = list(ITEMS); names_en = [ITEMS[k] for k in names_ko]

cap = cv2.VideoCapture(a.src)
latest = {"f": None}; lk = threading.Lock()
def reader():
    while True:
        ok, f = cap.read()
        if ok:
            with lk: latest["f"] = f
        else: time.sleep(0.02)
threading.Thread(target=reader, daemon=True).start()
state = {"jpg": b"", "msg": "모델 불러오는 중..."}
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_GET(self):
        if self.path.startswith("/img"):
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers(); self.wfile.write(state["jpg"]); return
        if self.path.startswith("/st"):
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(json.dumps(state["msg"]).encode()); return
        html = """<body style="background:#111;color:#fff;font:22px sans-serif;margin:8px"><div id=t></div><img id=i style="max-width:100%;max-height:85vh"><script>
function img(){const n=new Image();n.onload=()=>{i.src=n.src;setTimeout(img,120)};n.onerror=()=>setTimeout(img,1000);n.src='/img?'+Date.now()}
async function st(){try{t.textContent=await (await fetch('/st')).json()}catch(e){t.textContent='꺼짐'}setTimeout(st,300)}img();st();</script></body>"""
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(html.encode())
threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever(), daemon=True).start()
print(f"웹 화면: http://100.92.224.43:{a.port}")

det = YOLO("engines/yolo11n_fp16.engine" if os.path.exists("engines/yolo11n_fp16.engine") else "yolo11n.pt", task="detect")
yoloe = YOLO("yoloe-11s-seg.pt"); yoloe.set_classes(names_en, yoloe.get_text_pe(names_en))
emb = None; emb_err = ""
try:
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("google/embeddinggemma-2", device="cuda")
    tvec = emb.encode([f"a photo of a {n}" for n in names_en], normalize_embeddings=True)
except Exception as e:
    emb_err = f"{type(e).__name__}: {e}"[:200]; print("임베딩 모델을 못 불러와 ② 생략:", emb_err)

def biggest(boxes, skip_person=True):
    best = None
    for b in boxes:
        c = int(b.cls); 
        if skip_person and c == 0: continue
        x1, y1, x2, y2 = b.xyxy[0].tolist(); H_, W_ = b.orig_shape
        cx, cy = (x1 + x2) / 2 / W_ - 0.5, (y1 + y2) / 2 / H_ - 0.5
        sc = (x2 - x1) * (y2 - y1) / (W_ * H_) * (1 - min(1, 2 * (cx * cx + cy * cy) ** 0.5))   # 크고 가운데일수록 높음
        if best is None or sc > best[0]: best = (sc, c, float(b.conf), (int(x1), int(y1), int(x2), int(y2)))
    return best

def run_item(item):
    res = {1: Counter(), 2: Counter(), 3: Counter()}; sims = []
    t_start = time.time(); PREP, MEAS = a.prep, a.meas; snap = None
    while True:
        with lk: f = None if latest["f"] is None else latest["f"].copy()
        if f is None: time.sleep(0.05); continue
        el = time.time() - t_start; measuring = PREP <= el < PREP + MEAS
        if el >= PREP + MEAS: break
        view = f.copy(); lab = {}
        r1 = det.predict(f, conf=0.25, verbose=False)[0]; b1 = biggest(r1.boxes)
        if b1:
            lab[1] = f"{det.names[b1[1]]} {b1[2]:.2f}"; x1, y1, x2, y2 = b1[3]
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if emb is not None:
                crop = Image.fromarray(cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                v = emb.encode([crop], normalize_embeddings=True)[0]; s = tvec @ v; k = int(np.argmax(s))
                lab[2] = f"{names_ko[k]} {s[k]:.2f}" if s[k] >= a.thr else f"(유지) {det.names[b1[1]]} {s[k]:.2f}"
                if measuring: sims.append(float(s[k]))
        else:
            lab[1] = "없음"; lab[2] = "없음"
        r3 = yoloe.predict(f, conf=a.yconf, verbose=False)[0]; b3 = biggest(r3.boxes, skip_person=False)
        if b3:
            lab[3] = f"{names_ko[b3[1]]} {b3[2]:.2f}"; x1, y1, x2, y2 = b3[3]
            cv2.rectangle(view, (x1, y1), (x2, y2), (255, 160, 0), 2)
        else: lab[3] = "없음"
        if measuring:
            for k in (1, 2, 3): res[k][lab.get(k, "없음").rsplit(" ", 1)[0]] += 1
            snap = view
        phase = f"측정 중 {PREP + MEAS - el:.0f}초" if measuring else f"준비 {PREP - el:.0f}초 — '{item}'를 카메라에 비추세요"
        state["msg"] = f"[{item}] {phase} | ①YOLO: {lab.get(1)} | ②재라벨: {lab.get(2, '생략')} | ③YOLOE: {lab.get(3)}"
        for i, k in enumerate((1, 2, 3)):
            cv2.putText(view, f"{'①②③'[i] if False else k}: {lab.get(k, '-')}", (15, 40 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        state["jpg"] = cv2.imencode(".jpg", cv2.resize(view, (view.shape[1] // 2, view.shape[0] // 2)))[1].tobytes()

    top = {k: (res[k].most_common(1)[0] if res[k] else ("없음", 0)) for k in res}
    tot = max(1, sum(res[1].values()))
    row = {"시각": time.strftime("%Y-%m-%d %H:%M:%S"), "물품": item, "프롬프트": ITEMS[item], "YOLOE임계값": a.yconf,
           "①YOLO": top[1][0], "①정답": "X", "②재라벨": top[2][0] if emb else "생략", "②정답": "O" if top[2][0] == item else "X",
           "②평균유사도": round(float(np.mean(sims)), 3) if sims else "", "③YOLOE": top[3][0], "③정답": "O" if top[3][0] == item else "X",
           "프레임": tot, "③비율": f"{top[3][1]}/{tot}", "①전체분포": dict(res[1].most_common(3)), "③전체분포": dict(res[3].most_common(3)), "②오류": emb_err}
    new = not os.path.exists("runs/vocab2_r2.csv")
    with open("runs/vocab2_r2.csv", "a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
    if snap is not None: cv2.imwrite(f"runs/vocab2_r2_{item}.jpg", snap)
    print("기록:", json.dumps(row, ensure_ascii=False))
    state["msg"] = "끝: " + json.dumps(row, ensure_ascii=False); time.sleep(3)

for item in a.item:
    run_item(item)
