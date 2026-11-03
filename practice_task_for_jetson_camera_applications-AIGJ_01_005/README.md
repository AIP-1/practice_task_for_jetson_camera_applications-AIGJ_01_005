# practice_task_for_jetson_camera_applications — Jetson 카메라 응용 실습 (2팀)

Jetson Orin Nano Super와 USB 웹캠(C920)·IP 카메라(VIGI)·가짜 IP 카메라(맥 카메라 RTSP)로 과제 1~6을 수행했다. 재실 인원 서비스는 12시간 무중단(재시작 0회)·카메라 분리·재부팅(72초 복귀) 시험을 통과했다.

## 결과 보고서

📄 **[결과 보고서 (report.md)](reports/report.md)** — 과제 1~6 결과표, 측정 조건, 원인 분석, 확인 문제 1~6 답안을 정리한 결과 보고서다.

- PDF: [결과 보고서](reports/Jetson_카메라_응용_실습_결과보고서.pdf) · 다른 팀용 인터페이스: [결과 형식 문서](docs/결과형식.md) ([PDF](docs/재실인원_서비스_v1_결과형식.pdf)) · [기능 시험 기록](runs/test_log.md)

## 결과 한눈에

| 과제 | 확인 기준 | 결과 | 충족 |
| --- | --- | --- | --- |
| 1 카메라 연동 | (a)/(b) 2분 뒤 지연, 재연결 30초 이내 | RTSP (a) 약 96초 vs (b) 일정, USB (a) 0.83초 vs (b) 0.03초 / 재연결 4.6초 / glass-to-glass(RTSP) 408ms | ✅ |
| 2 물체 인식 | FP16 ≤ 10ms, 선 통과 오차 ≤ 1, 벤치마크 4행 | 5.09ms·60.6fps / 여유 폭 추가 후 오차 0 / INT8 mAP −0.0245 | ✅ |
| | 개방 어휘 5종 중 4종 | YOLOE 최고 3/5 (임베딩 재라벨 4/5) | ❌ |
| 3 재실 인원 | MAE ≤ 1.0명 | Mall Dataset(평균 31명) 16.8 → 3.1명 / 가상 방 0.14명 / 야간 빈 방 오판 0% | ⚠️ |
| 4 행동 패턴 | 동작·행동 정확도, 혼동 사례 | smooth 3/3·깜빡임 제거, 행동 규칙 7/12, GRU 처음 본 사람 87% | ⚠️ |
| 5 거리 측정 | 가상 영상 세 지점 ≤ 5%, 재투영 ≤ 0.5px | 0.1 / 0.2 / 0.0%, 0.15 / 0.19px | ✅ |
| 6 응용 서비스 | 12시간, 카메라 분리, 재부팅 3분, 형식 문서, 영상 저장 0 | 재시작 0회 / 나머지 정상·자동 재연결 / 72초 / 작성 / saved_images 0 | ✅ |

과제 3(출입 vs 직접 계수)·5·2(조명×거리)의 일부는 가상 시험(합성 영상)으로 진행했고 보고서에 그렇게 표시했다.

## 폴더 구성

| 경로 | 내용 |
| --- | --- |
| `cam1.py` | 과제 1: 입력 측정(probe), 리더 (a)·(b) 지연 비교(lag), 브라우저 보기·밀리초 시계·자동 재연결(view) |
| `bench2.py` · `line2.py` · `vocab2.py` · `light2.py` | 과제 2: TensorRT 벤치마크 / ByteTrack 선 통과 IN/OUT + MQTT / 물품 5종 세 방식 비교 / 조명×거리 합성 시험 |
| `occ_eval.py` · `occ_calib.py` · `vcount.py` | 과제 3: Mall Dataset 정확도 / 보정식 / 출입 vs 직접 계수(가상) |
| `pose_app.py` · `pose_model.py` · `train_pose.py` | 과제 4: 동작·행동 인식, GRU 학습 (embedding_gemma2 9단계 코드) |
| `calib_intr_fast.py` · `vstereo.py` · `vprox.py` · `vscene.py` | 과제 5: 렌즈 보정 / 가상 카메라 2대 시험 (`vscene.py`는 가상 장면 공통 모듈) |
| `occupancy_service.py` · `occupancy.service` · `occ_summary.py` | 과제 6: 재실 인원 서비스 v1, systemd 유닛, 12시간 기록 요약 |
| `mac/` | 맥 카메라를 RTSP로 송출하고 젯슨으로 터널을 여는 스크립트 |
| `data/` | 렌즈 보정용 체커보드(9x6) |
| `results/` | 과제별 결과 표·로그·사진 (사람 얼굴은 모자이크) |
| `reports/` | 결과 보고서 `report.md`, PDF |
| `docs/` | 결과 형식 문서, 과제 안내문 |
| `runs/` | 기능 시험 기록 |

## 실행 예

```bash
source ~/eg2/bin/activate
python cam1.py probe --src 0                                    # USB 웹캠 입력 측정
python cam1.py lag --src rtsp://localhost:8554/mac --reader b --work 0.2 --secs 120
python bench2.py                                                # TensorRT FP16/INT8 변환·벤치마크
python line2.py --src 0 --model engines/yolo11n_fp16.engine --port 8096
sudo cp occupancy.service /etc/systemd/system/ && sudo systemctl enable --now occupancy
```

## 보안·개인정보

- 카메라 비밀번호는 코드에 쓰지 않고 환경 변수 `CAM_PW`(systemd는 chmod 600 EnvironmentFile)로 읽는다.
- 서비스는 영상·스냅샷을 저장하지 않는다(`/health`의 `saved_images` = 0). `results/`의 사진은 시험용 스냅샷이며 사람 얼굴은 모자이크 처리했다.

## 환경

Jetson Orin Nano Super, Ubuntu 24.04, TensorRT 10.16, Ultralytics 8.4, Python 3.12
