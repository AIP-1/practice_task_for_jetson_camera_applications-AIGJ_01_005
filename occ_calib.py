"""과제 3 보정: 꾸준히 덜 세는 것을 직선 보정식(정답 ≈ a × 센 값 + b)으로 바로잡기
앞 1,000장으로 a, b를 구하고(학습), 뒤 1,000장에서만 채점(시험) — 같은 장면으로 채점하지 않음
사용법: python occ_calib.py   (runs/occ_eval_frames.csv 사용, 결과 runs/occ_calib.md)
"""
import csv, numpy as np
rows = list(csv.reader(open("runs/occ_eval_frames.csv"))); head, data = rows[0], rows[1:]
gt = np.array([int(r[1]) for r in data]); n = len(gt); h = n // 2
out = ["| 조건 | 보정식 (앞 1,000장) | 시험 MAE 보정 전 | 시험 MAE 보정 후 | 시험 편향 보정 후 | ±2명 이내 보정 후 |", "|---|---|---|---|---|---|"]
for j in range(2, len(head)):
    c = np.array([float(r[j]) for r in data])
    a, b = np.polyfit(c[:h], gt[:h], 1)
    pred = np.round(a * c[h:] + b); e0 = c[h:] - gt[h:]; e1 = pred - gt[h:]
    out.append(f"| {head[j]} | 정답 = {a:.2f} × 센 값 + {b:.1f} | {np.abs(e0).mean():.2f} | {np.abs(e1).mean():.2f} | {e1.mean():+.2f} | {(np.abs(e1) <= 2).mean()*100:.1f}% |")
out.append(f"\n조건: 학습 = 앞 {h}장, 시험 = 뒤 {n-h}장(정답 평균 {gt[h:].mean():.1f}명). 반올림해서 정수 인원으로 채점.")
md = "\n".join(out); open("runs/occ_calib.md", "w").write(md); print(md)
