#!/usr/bin/env python3
"""과제 6 심화: 재실 인원 서비스 자기 감시 (서비스 코드는 건드리지 않는 별도 감시 프로세스)

하는 일
  ① 사람이 센 값과 자동 비교 — 하루 2회(기본 10:00, 15:00) 점검 시각이 되면 '점검 필요'를 띄운다.
     웹 페이지(:8097)에 사람이 센 인원을 넣으면, 그 순간 서비스 값(최근 30초 중앙값)과 비교해 기록한다.
     |오차| ≥ 2명 한 번, 또는 |오차| ≥ 1명이 두 번 연속이면 '보정 필요'.
     점검 시각에서 1시간이 지나도록 입력이 없으면 '점검 누락'.
  ② 사람 없이 하는 자동 점검 — 새벽(02:00~05:00)에는 방이 비어 있어야 한다.
     이 시간에 1명 이상이 10분 넘게 이어지면 '야간 오검출 의심'(조명·반사·포스터 등).
  ③ 카메라 상태가 ok가 아니거나 서비스가 응답하지 않으면 '서비스 이상'.
  상태가 바뀔 때마다 MQTT aiot/service/occupancy/monitor (retain) 로 알리고 runs/occ_monitor.jsonl 에 남긴다.

사용법:  python occ_monitor.py                     (서비스 http://localhost:8095)
         브라우저: http://<젯슨 IP>:8097
옵션:    --checks 10:00,15:00  --service http://localhost:8095  --port 8097  --mqtt localhost:1883
"""
import argparse, json, shutil, statistics, subprocess, threading, time, urllib.request
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ap = argparse.ArgumentParser()
ap.add_argument("--service", default="http://localhost:8095")
ap.add_argument("--port", type=int, default=8097)
ap.add_argument("--checks", default="10:00,15:00")
ap.add_argument("--night", default="02:00-05:00")
ap.add_argument("--mqtt", default="localhost:1883")
ap.add_argument("--topic", default="aiot/service/occupancy/monitor")
a = ap.parse_args()

LOG = Path("runs/occ_monitor.jsonl"); LOG.parent.mkdir(exist_ok=True)
POLL = 10
hist = {}                      # 공간 → deque[(t, count)]
checks = []                    # 사람 비교 기록
alerts = {}                    # 종류 → 내용 (지금 켜져 있는 알림)
last_ok = {"t": None}
lock = threading.Lock()
T_START = datetime.now()       # 감시를 켜기 전에 지난 점검 시각은 누락으로 치지 않음

def now_iso(): return datetime.now().isoformat(timespec="seconds")

def log(kind, **kw):
    row = {"ts": now_iso(), "kind": kind, **kw}
    with LOG.open("a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[감시]", json.dumps(row, ensure_ascii=False), flush=True)

def mqtt(payload):
    if not shutil.which("mosquitto_pub"): return
    h, p = a.mqtt.split(":")
    subprocess.run(["mosquitto_pub", "-h", h, "-p", p, "-q", "1", "-r", "-t", a.topic, "-m",
                    json.dumps(payload, ensure_ascii=False)], capture_output=True, timeout=5)

def set_alert(kind, on, detail=""):
    with lock:
        was = kind in alerts
        if on: alerts[kind] = detail
        else: alerts.pop(kind, None)
    if on != was:
        log("alert_on" if on else "alert_off", alert=kind, detail=detail)
        mqtt({"ts": now_iso(), "status": "보정 필요" if "보정 필요" in alerts else ("주의" if alerts else "정상"),
              "alerts": dict(alerts)})

def service_count(space, secs=30):
    h = hist.get(space)
    if not h: return None
    vals = [c for t, c in h if time.time() - t <= secs]
    return statistics.median(vals) if vals else None

def in_window(rng, dt):
    s, e = [datetime.strptime(x, "%H:%M").time() for x in rng.split("-")]
    return s <= dt.time() < e

def due_checks(dt):
    """오늘 점검 시각 중 이미 지난 것 → (시각, 입력됐는지, 1시간 넘었는지)"""
    out = []
    for hm in a.checks.split(","):
        t = datetime.combine(dt.date(), datetime.strptime(hm, "%H:%M").time())
        if dt < t or t < T_START: continue
        done = any(t <= datetime.fromisoformat(c["ts"]) < t + timedelta(hours=3) for c in checks)
        out.append((hm, done, dt > t + timedelta(hours=1)))
    return out

def poller():
    night_since = {}
    while True:
        try:
            with urllib.request.urlopen(a.service + "/occupancy", timeout=5) as r: snap = json.load(r)
            last_ok["t"] = time.time()
            set_alert("서비스 이상", False)
            bad = []
            for space, s in snap.items():
                hist.setdefault(space, deque(maxlen=400)).append((time.time(), s["count"]))
                if s.get("camera") not in ("정상", "ok"): bad.append(f"{space} 카메라 {s.get('camera')}")
                # ② 야간 자동 점검
                if in_window(a.night, datetime.now()) and s["count"] >= 1:
                    night_since.setdefault(space, time.time())
                    if time.time() - night_since[space] > 600:
                        set_alert("야간 오검출 의심", True, f"{space}: 새벽에 {s['count']}명이 10분 넘게 이어짐")
                else:
                    night_since.pop(space, None)
            set_alert("카메라 이상", bool(bad), ", ".join(bad))
        except Exception as e:
            if last_ok["t"] is None or time.time() - last_ok["t"] > 60:
                set_alert("서비스 이상", True, f"{a.service} 응답 없음: {e.__class__.__name__}")
        # ① 점검 누락
        miss = [hm for hm, done, late in due_checks(datetime.now()) if not done and late]
        set_alert("점검 누락", bool(miss), ", ".join(miss))
        time.sleep(POLL)

def add_check(space, human):
    svc = service_count(space)
    if svc is None: return {"error": f"{space} 서비스 값 없음"}
    err = svc - human
    row = {"ts": now_iso(), "space": space, "human": human, "service": svc, "error": err}
    checks.append(row); log("check", **row)
    prev = [c for c in checks if c["space"] == space][-2:]
    need = abs(err) >= 2 or (len(prev) == 2 and all(abs(c["error"]) >= 1 for c in prev))
    set_alert("보정 필요", need, f"{space}: 사람 {human}명 vs 서비스 {svc:g}명 (오차 {err:+g})" if need else "")
    return row

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>재실 서비스 감시</title><style>body{font-family:sans-serif;max-width:640px;margin:20px auto;padding:0 16px}
.ok{color:#1a7f37}.bad{color:#cf222e}td,th{padding:4px 8px;border-bottom:1px solid #ddd}</style>
<h2>재실 인원 서비스 자기 감시</h2><div id=s></div>
<form onsubmit="go(event)">공간 <select id=sp></select> 사람이 센 인원 <input id=n type=number min=0 style=width:60px required>
<button>비교 기록</button></form><h3>점검 기록</h3><table id=t></table>
<script>
async function load(){const r=await (await fetch('/status')).json();
document.getElementById('s').innerHTML=`<p>상태: <b class=${r.status=='정상'?'ok':'bad'}>${r.status}</b></p>`+
Object.entries(r.alerts).map(([k,v])=>`<p class=bad>⚠ ${k}: ${v}</p>`).join('')+
`<p>서비스 값: ${Object.entries(r.now).map(([k,v])=>k+' '+v+'명').join(', ')||'없음'} · 오늘 점검: ${r.due.map(d=>d[0]+(d[1]?' ✅':' ⏳')).join(', ')||'아직 시각 전'}</p>`;
const sp=document.getElementById('sp'); if(!sp.options.length) Object.keys(r.now).forEach(k=>sp.add(new Option(k)));
document.getElementById('t').innerHTML='<tr><th>시각<th>공간<th>사람<th>서비스<th>오차</tr>'+
r.checks.slice(-20).reverse().map(c=>`<tr><td>${c.ts.slice(5,16)}<td>${c.space}<td>${c.human}<td>${c.service}<td>${c.error>0?'+':''}${c.error}</tr>`).join('')}
async function go(e){e.preventDefault();await fetch(`/check?space=${sp.value}&human=${n.value}`,{method:'POST'});n.value='';load()}
load();setInterval(load,5000)</script>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _send(self, body, ctype="application/json; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(b)
    def status(self):
        with lock: al = dict(alerts)
        return {"status": "보정 필요" if "보정 필요" in al else ("주의" if al else "정상"), "alerts": al,
                "now": {k: service_count(k) for k in hist}, "due": due_checks(datetime.now()), "checks": checks}
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/": self._send(PAGE, "text/html; charset=utf-8")
        elif u.path == "/status": self._send(json.dumps(self.status(), ensure_ascii=False))
        else: self.send_error(404)
    def do_POST(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/check":
            self._send(json.dumps(add_check(q["space"][0], int(q["human"][0])), ensure_ascii=False))
        else: self.send_error(404)

# 다시 켜도 이전 점검 기록을 이어받음
if LOG.exists():
    for line in LOG.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("kind") == "check": checks.append({k: r[k] for k in ("ts", "space", "human", "service", "error")})
threading.Thread(target=poller, daemon=True).start()
log("start", service=a.service, checks=a.checks, night=a.night)
print(f"감시 시작: http://<젯슨 IP>:{a.port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
