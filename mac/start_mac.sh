#!/bin/zsh
# 맥 카메라(가짜 IP 카메라) → mediamtx(RTSP) → ssh 역방향 터널 → 젯슨 rtsp://localhost:8554/mac
# 끄기: zsh stop_cams.sh   (젯슨 주소는 JETSON 변수로 바꾼다)
JETSON=${JETSON:-team2-2@100.92.224.43}
LOG=~/Downloads/cam_logs; mkdir -p $LOG
pkill mediamtx 2>/dev/null; pkill -f "rtsp://localhost:8554/" 2>/dev/null; sleep 1
/opt/homebrew/opt/mediamtx/bin/mediamtx /opt/homebrew/etc/mediamtx/mediamtx.yml > $LOG/mediamtx.log 2>&1 &
sleep 2
( while true; do
    ffmpeg -loglevel error -f avfoundation -framerate 30 -video_size 1280x720 -i "0" \
      -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -pix_fmt yuv420p \
      -f rtsp -rtsp_transport tcp rtsp://localhost:8554/mac
    echo "$(date +%T) mac 끊김 → 2초 뒤 재시작"; sleep 2
  done ) > $LOG/mac.log 2>&1 &
while true; do
  ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -R 8554:localhost:8554 $JETSON
  echo "$(date +%T) 터널 끊김 → 3초 뒤 다시 연결"; sleep 3
done
