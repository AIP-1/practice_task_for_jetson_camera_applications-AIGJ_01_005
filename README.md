# practice_task_for_jetson_camera_applications — Jetson 카메라 응용 실습 (2팀)

Jetson Orin Nano Super와 교육장 카메라(VIGI C540-W IP 카메라, Logitech C920 USB)로 과제 1~6을 수행했다. 확인 기준이 **실측**을 요구하는 항목은 실제 카메라로 재서 판정했고, 실측으로 조건을 만들기 어려운 방식 비교·설계 검증만 **가상 시험(합성 영상)**으로 보충했다.

## 결과 보고서

📄 **[결과 보고서 (report.md)](reports/report.md)** — 과제 1~6 결과표와 측정 조건, 원인 분석, 심화(B), 확인 문제 1~6 답안을 정리한 결과 보고서다.

- PDF: [결과 보고서](reports/report.pdf) · 결과 형식 문서: [result_format.md](docs/result_format.md) ([PDF](docs/result_format.pdf)) · [기능 시험 기록](runs/test_log.md)

## 결과 한눈에 (실측)

| 과제 | 확인 기준 | 결과 | 충족 |
| --- | --- | --- | --- |
| 1 카메라 연동 | (a)/(b) 2분 뒤 지연, 재연결 | VIGI (a) 96.2초 계속 증가 vs (b) 증가 없음 / 랜선 분리 후 재시작 없이 재연결 / glass-to-glass C920 1초 이하 | ✅ |
| 2 물체 인식 | FP16 ≤ 10ms, 선 통과 오차 ≤ 1 | 추론 5.1ms(전체 87fps) / 선 통과 10/10, MQTT 10건 / INT8 mAP −0.017 / 조명×거리 검출률 95% 이상 | ✅ |
| | 개방 어휘 5종 중 4종 | yolo 1/5, 임베딩 재라벨 5/5, YOLOE 2/5 | ❌ (YOLOE) |
| 3 재실 인원 | MAE ≤ 1.0, ±1명 ≥ 80%, 빈 방 오판 ≤ 10% | 0~2명 MAE 0.00~0.40, ±1명 100%, 밤 빈 방 오판 0% | ✅ |
| 4 행동 패턴 | 동작 9/10, 행동 10/12, 혼동 사례 5건 | 만세 10/10, 행동 11/12, 혼동 사례 5건, 안정화 깜빡임 12 → 0 (0.3초 지연) | ✅ |
| 5 거리 측정 | 실측 오차 1~2m ≤ 5% | 의자 1.4m → 1.35m (−3.6%), C920 1대를 8cm 옮겨 찍는 대체 방식 | ✅ (일부 ⚠️) |
| 6 응용 서비스 | 12시간, 분리, 재부팅 3분, 영상 저장 0 | 서비스 2개 12시간 넘게 재시작 0 / 자동 재연결 / 1~2분 복귀 / 저장 0 | ✅ |

**심화(B)**: 1080p 하드웨어 디코딩(CPU 65~94% → 12%), 세 프로그램 합치기(프로세스 메모리 3,356 → 1,693MB), 서비스 자기 감시, GRU 처음 본 사람 87% — 실측. 혼합 보정·기준선 6/12cm·거리 자기 감시 — 가상 시험.

## 폴더 구성

| 경로 | 내용 |
| --- | --- |
| `cam1.py` · `dec_compare.py` | 과제 1: 입력 측정, 리더 (a)·(b) 지연 비교, 브라우저 보기·자동 재연결 / 심화: 1080p 하드웨어 vs CPU 디코딩 |
| `bench2.py` · `line2.py` · `vocab2.py` · `light2.py` | 과제 2: TensorRT 벤치마크 / 선 통과 IN/OUT + MQTT / 물품 세 방식 비교 / 조명×거리(가상 보조) |
| `occ_eval.py` · `occ_calib.py` · `vcount.py` · `vhybrid.py` | 과제 3: Mall Dataset 정확도 / 보정식 / 출입 vs 직접(가상) / 심화 혼합 보정(가상) |
| `pose_app.py` · `pose_model.py` · `train_pose.py` | 과제 4: 동작·행동 인식, GRU 학습 |
| `calib_intr_fast.py` · `vscene.py` · `vstereo.py` · `vprox.py` · `vbaseline.py` | 과제 5: 렌즈 보정 / 가상 장면 공통 모듈 / 세 방식 비교·알람·TTC·기준선 6/12cm(가상) |
| `occupancy_service.py` · `occupancy.service` · `occ_summary.py` · `occ_monitor.py` · `merge_bench.py` · `vwall.py` | 과제 6: 재실 서비스, systemd 유닛, 가동 기록 요약 / 심화: 자기 감시(웹 :8097), 세 프로그램 합치기, 거리 자기 감시(가상) |
| `mac/` · `win/` | 맥 / Windows 노트북 카메라를 RTSP로 송출하고 젯슨으로 터널을 여는 스크립트 |
| `data/` · `results/` · `runs/` | 체커보드 / 과제별 결과(얼굴 모자이크) / 기능 시험 기록 |
| `reports/` · `docs/` | 결과 보고서(md, pdf) / 결과 형식 문서, 과제 안내문 |

교육장 실측에 쓴 팀 공용 도구(`line_count.py`, `light_dist.py`, `shift_stereo.py`, 안내판 서비스 A3 등)는 팀 보드에서 실행했고 이 저장소에는 넣지 않았다. 결과는 보고서에 '팀'으로 표시했다.

## 실행 예

```bash
source ~/eg2/bin/activate
python cam1.py probe --src 0                                   # USB 웹캠 입력 측정
python cam1.py lag --src rtsp://localhost:8554/mac --reader b --work 0.2 --secs 120
python bench2.py                                               # TensorRT FP16/INT8 변환·벤치마크
python dec_compare.py                                          # 1080p 하드웨어 vs CPU 디코딩
sudo cp occupancy.service /etc/systemd/system/ && sudo systemctl enable --now occupancy
python occ_monitor.py                                          # 자기 감시, 브라우저 http://<보드 주소>:8097
JETSON=계정@<보드 주소> zsh mac/start_mac.sh                   # 맥 카메라 → 젯슨 RTSP
```

## 보안·개인정보

- 카메라 비밀번호는 코드에 쓰지 않고 환경 변수 `CAM_PW`(systemd는 chmod 600 EnvironmentFile)로 읽는다.
- Tailscale 주소와 카메라 MAC 주소는 저장소에 넣지 않는다. 문서와 코드에는 `<보드 주소>`로 적고, 실행할 때 환경 변수(`JETSON`, `VIGI_MAC`)로 준다.
- 서비스는 영상·스냅샷을 저장하지 않는다(`/health`의 `saved_images` = 0). `results/`의 사진은 시험용 스냅샷이며 사람 얼굴은 모자이크 처리했다.

## 환경

Jetson Orin Nano Super 8GB, JetPack 7.2.1, Ubuntu 24.04, CUDA 13.2, TensorRT 10.16, Ultralytics 8.4, Python 3.12
