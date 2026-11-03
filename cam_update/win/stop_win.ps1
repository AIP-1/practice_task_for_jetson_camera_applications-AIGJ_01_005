# start_win.ps1 로 켠 웹캠 송출·mediamtx 끄기
Get-Job -Name wincam -ErrorAction SilentlyContinue | Stop-Job -PassThru | Remove-Job
Get-Process mediamtx, ffmpeg -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "꺼짐"
