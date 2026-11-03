"""과제 6: 12시간 가동 기록 요약 (logs/service_*.jsonl, events.jsonl, tegrastats.log) → runs/occ12h.md
사용법: cd ~/eg2-lab/occupancy && python ~/eg2-lab/occ_summary.py [--start "2026-11-02T18:00"] [--hours 12]
"""
import argparse, glob, json, re, subprocess, statistics as st
from datetime import datetime, timedelta
ap = argparse.ArgumentParser(); ap.add_argument("--start", default=None); ap.add_argument("--hours", type=float, default=12)
a = ap.parse_args()
P = lambda s: datetime.fromisoformat(s[:19])
rows = []
for f in sorted(glob.glob("logs/service_*.jsonl")):
    rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
t0 = P(a.start) if a.start else P(rows[0]["ts"]); t1 = t0 + timedelta(hours=a.hours)
R = [r for r in rows if t0 <= P(r["ts"]) <= t1]
ev = [json.loads(l) for l in open("logs/events.jsonl", encoding="utf-8") if l.strip()]
E = [e for e in ev if t0 <= P(e["ts"]) <= t1]
def col(k): return [r[k] for r in R if isinstance(r.get(k), (int, float))]
def s(v, u=""): return f"평균 {st.mean(v):.1f}{u} / 최대 {max(v):.1f}{u}" if v else "기록 없음"
gaps = [(P(R[i]["ts"]) - P(R[i-1]["ts"])).total_seconds() for i in range(1, len(R))]
up_resets = sum(1 for i in range(1, len(R)) if R[i]["uptime_s"] < R[i-1]["uptime_s"])
cams = {}
for r in R:
    for n, c in (r.get("cams") or {}).items():
        d = cams.setdefault(n, {"ok": 0, "n": 0, "rec": 0, "cnt": []}); d["n"] += 1
        d["ok"] += c.get("camera") == "정상"; d["rec"] = max(d["rec"], c.get("reconnects") or 0)
        if c.get("count") is not None: d["cnt"].append(c["count"])
pw, gr = [], []
try:
    for l in open("logs/tegrastats.log"):
        m = re.search(r"VDD_IN (\d+)mW", l); g = re.search(r"GR3D_FREQ (\d+)%", l)
        if m: pw.append(int(m.group(1)) / 1000)
        if g: gr.append(int(g.group(1)))
except FileNotFoundError: pass
try: nres = subprocess.run(["systemctl", "show", "occupancy", "-p", "NRestarts", "--value"], capture_output=True, text=True).stdout.strip()
except Exception: nres = "?"
md = [f"### 12시간 가동 기록 ({t0:%m-%d %H:%M} ~ {min(t1, P(R[-1]['ts'])):%m-%d %H:%M})\n", "| 항목 | 값 |", "|---|---|",
      f"| 기록 행 (1분마다) | {len(R)}행 / 기대 {int(a.hours*60)}행 |",
      f"| 기록 공백 최대 | {max(gaps) if gaps else 0:.0f}초 |",
      f"| 서비스 재시작 (systemd NRestarts) | {nres}회 (기록상 uptime 초기화 {up_resets}회) |",
      f"| MQTT 연결 유지 | {sum(1 for r in R if r.get('mqtt'))}/{len(R)}행 |",
      f"| 추론 시간 | {s(col('infer_ms_avg'), 'ms')} |",
      f"| CPU | {s(col('cpu_pct'), '%')} |", f"| 메모리 사용 | {s(col('mem_used_mb'), 'MB')} |",
      f"| GPU | {s(col('gpu_pct'), '%')} |", f"| 전력 (tegrastats VDD_IN) | {s(pw, 'W')} |", f"| GPU 클럭 사용률 (GR3D) | {s(gr, '%')} |"]
for n, d in cams.items():
    md.append(f"| 카메라 {n} | 정상 {d['ok']}/{d['n']}행, 재연결 {d['rec']}회, 인원 {s(d['cnt'], '명')} |")
md.append("\n| 시각 | 이벤트 |\n|---|---|")
for e in E: md.append(f"| {e['ts'][5:19]} | {e['event']} {e.get('cam', '')} {e.get('why', '')} {e.get('after_s', '')} |")
out = "\n".join(md); open("/home/team2-2/eg2-lab/runs/occ12h.md", "w").write(out); print(out)
