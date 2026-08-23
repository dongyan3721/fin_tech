# FastAPI 数据服务控制：start | stop | restart | status
# 用法: powershell -NoProfile -File server\api_ctl.ps1 start [-Port 8000]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('start', 'stop', 'restart', 'status')]
    [string]$Action,
    [int]$Port = 8000
)

$ROOT = 'D:\Data\CodeRepository\py\GraghRiskEvaluate'
$PY = "$ROOT\.venv\Scripts\python.exe"
$MAIN = "$ROOT\server\main.py"
$LOGOUT = "$ROOT\logs\api_out.log"
$LOGERR = "$ROOT\logs\api_err.log"

function Get-PortPid {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return ($conn | Select-Object -First 1).OwningProcess
}

function Start-Api {
    $existing = Get-PortPid
    if ($existing) {
        Write-Output "already running (pid=$existing, port=$Port)"
        return
    }
    New-Item -ItemType Directory -Force -Path "$ROOT\logs" | Out-Null
    $proc = Start-Process -FilePath $PY -ArgumentList "`"$MAIN`"", '--port', "$Port" `
        -WorkingDirectory $ROOT -WindowStyle Hidden `
        -RedirectStandardOutput $LOGOUT -RedirectStandardError $LOGERR -PassThru
    Write-Output "started pid=$($proc.Id)"
}

switch ($Action) {
    'status' {
        $p = Get-PortPid
        if ($p) { Write-Output "RUNNING (pid=$p, port=$Port)" }
        else { Write-Output "STOPPED" }
    }
    'stop' {
        $p = Get-PortPid
        if ($p) { Stop-Process -Id $p -Force; Write-Output "stopped (pid=$p)" }
        else { Write-Output "already stopped" }
    }
    'start' { Start-Api }
    'restart' {
        $p = Get-PortPid
        if ($p) { Stop-Process -Id $p -Force; Start-Sleep -Milliseconds 600 }
        Start-Api
    }
}
