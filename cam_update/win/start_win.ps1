# 과제 1 심화: Windows 노트북 웹캠(가짜 IP 카메라) -> mediamtx(RTSP) -> ssh 역방향 터널 -> 젯슨 rtsp://localhost:8554/win
# macOS용 mac/start_mac.sh 를 Windows PowerShell로 옮긴 것.
# 준비(한 번만):
#   1) winget install Gyan.FFmpeg            (새 PowerShell 창을 열어야 ffmpeg 명령이 잡힌다)
#   2) mediamtx Windows zip을 받아 압축을 풀고 mediamtx.exe 를 $HOME\mediamtx\ 에 둔다
#      https://github.com/bluenviron/mediamtx/releases  (mediamtx_vX.Y.Z_windows_amd64.zip)
# 실행:  powershell -ExecutionPolicy Bypass -File .\start_win.ps1
#        (카메라 이름을 직접 줄 때) ... -File .\start_win.ps1 -Cam "Integrated Camera"
# 끄기:  이 창에서 Ctrl+C 후  powershell -ExecutionPolicy Bypass -File .\stop_win.ps1
param(
  [string]$Jetson   = "team2-2@100.92.224.43",
  [string]$Cam      = "",
  [string]$MediaMtx = "$HOME\mediamtx\mediamtx.exe",
  [string]$Size     = "1280x720",
  [int]$Fps         = 30,
  [string]$StreamPath = "win"
)
$ErrorActionPreference = "Continue"   # ffmpeg가 stderr에 쓰는 정상 메시지로 멈추지 않게
$Log = "$HOME\Downloads\cam_logs"; New-Item -ItemType Directory -Force -Path $Log | Out-Null

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Write-Host "ffmpeg 없음 -> winget install Gyan.FFmpeg 후 새 창에서 다시 실행"; exit 1 }
if (-not (Test-Path $MediaMtx)) { Write-Host "mediamtx.exe 없음: $MediaMtx"; exit 1 }

# 1) 웹캠 이름 찾기 (DirectShow)
if ($Cam -eq "") {
  $list = (& ffmpeg -hide_banner -list_devices true -f dshow -i dummy 2>&1 | Out-String)
  $m = [regex]::Matches($list, '"([^"]+)"\s+\(video\)')
  if ($m.Count -eq 0) { Write-Host "웹캠을 못 찾음. 아래 목록에서 이름을 -Cam 으로 넘겨라:"; Write-Host $list; exit 1 }
  $Cam = $m[0].Groups[1].Value
}
Write-Host "웹캠: $Cam  ($Size, $Fps fps)  ->  젯슨 rtsp://localhost:8554/$StreamPath"

# 2) 이전에 켜 둔 것 정리 후 mediamtx 시작
Get-Process mediamtx, ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep 1
$mtxDir = Split-Path $MediaMtx
Start-Process -FilePath $MediaMtx -WorkingDirectory $mtxDir -WindowStyle Hidden `
  -RedirectStandardOutput "$Log\mediamtx.log" -RedirectStandardError "$Log\mediamtx.err"
Start-Sleep 2

# 3) 웹캠 -> RTSP 송출 (끊기면 2초 뒤 다시) : 백그라운드 작업
$sizeArgs = if ($Size -ne "") { @("-video_size", $Size) } else { @() }
Start-Job -Name wincam -ArgumentList $Cam, $Fps, $StreamPath, $Log, $sizeArgs -ScriptBlock {
  param($Cam, $Fps, $StreamPath, $Log, $sizeArgs)
  while ($true) {
    & ffmpeg -loglevel error -f dshow -rtbufsize 100M -framerate $Fps @sizeArgs -i "video=$Cam" `
      -c:v libx264 -preset ultrafast -tune zerolatency -g $Fps -pix_fmt yuv420p `
      -f rtsp -rtsp_transport tcp "rtsp://localhost:8554/$StreamPath" 2>> "$Log\win.log"
    Add-Content "$Log\win.log" "$(Get-Date -Format T) 웹캠 송출 끊김 -> 2초 뒤 재시작"
    Start-Sleep 2
  }
} | Out-Null
Start-Sleep 3

# 4) 젯슨으로 역방향 터널 (끊기면 3초 뒤 다시). 이 창은 열어 둔다.
Write-Host "터널 연결 중... (젯슨 비밀번호를 물으면 입력)  끄려면 Ctrl+C"
while ($true) {
  ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -R 8554:localhost:8554 $Jetson
  Write-Host "$(Get-Date -Format T) 터널 끊김 -> 3초 뒤 다시 연결"
  Start-Sleep 3
}
