"""과제 3 재실 인원 정확도: Mall Dataset(2,000장, 프레임마다 사람 수 정답)로 YOLO 사람 세기 평가
- 서비스(occupancy_service.py)와 같은 방식: class 0, conf, imgsz → 프레임별 인원, 이어서 창 중앙값 평활
- 조건(모델 × 입력 크기 × conf)마다 MAE, RMSE, 평균 편향(센 값 − 정답), ±2명 이내 비율
사용법: python -u occ_eval.py --root mall_dataset          (결과 runs/occ_eval.md, runs/occ_eval_frames.csv)
"""
import argparse, glob, os, json, time
import numpy as np, cv2
from scipy.io import loadmat
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="mall_dataset")
ap.add_argument("--n", type=int, default=2000, help="앞에서부터 몇 장 (빠른 시험은 400)")
ap.add_argument("--win", type=int, default=5, help="평활 창(프레임 수) — Mall은 약 2초 간격이라 5장 ≈ 10초")
a = ap.parse_args()

gt_path = glob.glob(os.path.join(a.root, "**", "mall_gt.mat"), recursive=True)[0]
gt = loadmat(gt_path)["count"].ravel().astype(int)
frames = sorted(glob.glob(os.path.join(a.root, "**", "seq_*.jpg"), recursive=True))[: a.n]
gt = gt[: len(frames)]
print(f"프레임 {len(frames)}장, 정답 인원 {gt.min()}~{gt.max()}명 (평균 {gt.mean():.1f})")

CONDS = [("yolo11n FP16", "engines/yolo11n_fp16.engine", 640, 0.4),   # 서비스 기본값
         ("yolo11n FP16", "engines/yolo11n_fp16.engine", 640, 0.25),
         ("yolo11s FP16", "engines/yolo11s_fp16.engine", 640, 0.4),
         ("yolo11s FP16", "engines/yolo11s_fp16.engine", 640, 0.25),
         ("yolo11s PyTorch", "yolo11s.pt", 960, 0.25)]                # 큰 입력: 작은(먼) 사람
def med(x, w):
    return np.array([np.median(x[max(0, i - w + 1): i + 1]) for i in range(len(x))])

rows, per = [], {"frame": [os.path.basename(f) for f in frames], "gt": gt.tolist()}
for name, path, sz, conf in CONDS:
    if not os.path.exists(path): print("없음:", path); continue
    m = YOLO(path, task="detect"); t = time.time(); cnt = []
    for f in frames:
        im = cv2.imread(f)
        r = m.predict(im, classes=[0], conf=conf, imgsz=sz, verbose=False)[0]; cnt.append(len(r.boxes))
    dt = (time.time() - t) / len(frames) * 1000
    cnt = np.array(cnt); sm = med(cnt, a.win)
    for tag, c in (("프레임", cnt), (f"중앙값{a.win}장", sm)):
        e = c - gt
        rows.append({"모델": name, "입력": sz, "conf": conf, "평활": tag, "MAE": round(float(np.abs(e).mean()), 2),
                     "RMSE": round(float(np.sqrt((e ** 2).mean())), 2), "편향": round(float(e.mean()), 2),
                     "±2명이내_%": round(float((np.abs(e) <= 2).mean() * 100), 1), "ms/장": round(dt, 1)})
        print(rows[-1], flush=True)
    per[f"{name}_{sz}_{conf}"] = cnt.tolist()
    if (name, sz, conf) == ("yolo11n FP16", 640, 0.4):            # 서비스 기본값에서 가장 크게 틀린 3장 저장
        for i in np.argsort(-np.abs(cnt - gt))[:3]:
            im = cv2.imread(frames[i]); r = m.predict(im, classes=[0], conf=conf, imgsz=sz, verbose=False)[0]
            for b in r.boxes.xyxy.cpu().numpy().astype(int): cv2.rectangle(im, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 1)
            cv2.putText(im, f"GT {gt[i]} / YOLO {cnt[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imwrite(f"runs/occ_worst_{os.path.basename(frames[i])}", im)

import csv
with open("runs/occ_eval_frames.csv", "w", newline="") as fp:
    w = csv.writer(fp); keys = list(per); w.writerow(keys); w.writerows(zip(*[per[k] for k in keys]))
keys = list(rows[0])
md = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n" + "\n".join("| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows)
md += f"\n\n조건: Mall Dataset 앞 {len(frames)}장(640x480, 정답 {gt.min()}~{gt.max()}명, 평균 {gt.mean():.1f}명), 사람(class 0)만, 편향 = 센 값 − 정답(음수 = 덜 셈)\n"
open("runs/occ_eval.md", "w").write(md); print("\n" + md)
