#!/bin/zsh
pkill -f start_mac.sh; pkill -f start_cams.sh; pkill -f "8554:localhost:8554"; pkill -f "rtsp://localhost:8554/"; pkill mediamtx
echo "송출 전부 끔"
