#!/usr/bin/env bash
# API 服务控制：bash server/api.sh start|stop|restart|status [port]
# 纯 Git bash 实现（netstat/taskkill 是 Windows 原生 exe，非 PowerShell）。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${2:-8000}"
PY="$ROOT/.venv/Scripts/python.exe"

port_pid() {
  netstat -ano 2>/dev/null | grep ":$PORT" | grep -i LISTENING | awk '{print $NF}' | head -n1
}

case "$1" in
  status)
    p="$(port_pid)"
    if [ -n "$p" ]; then echo "RUNNING (pid=$p, port=$PORT)"; else echo "STOPPED"; fi
    ;;
  stop)
    p="$(port_pid)"
    if [ -n "$p" ]; then taskkill //F //PID "$p" >/dev/null 2>&1; echo "stopped pid=$p"; else echo "already stopped"; fi
    ;;
  start)
    if [ -n "$(port_pid)" ]; then echo "already running"; exit 0; fi
    mkdir -p "$ROOT/logs"
    ( "$PY" "$ROOT/server/main.py" --port "$PORT" >"$ROOT/logs/api_out.log" 2>"$ROOT/logs/api_err.log" < /dev/null & )
    echo "started"
    ;;
  restart)
    p="$(port_pid)"
    if [ -n "$p" ]; then taskkill //F //PID "$p" >/dev/null 2>&1; sleep 1; fi
    mkdir -p "$ROOT/logs"
    ( "$PY" "$ROOT/server/main.py" --port "$PORT" >"$ROOT/logs/api_out.log" 2>"$ROOT/logs/api_err.log" < /dev/null & )
    echo "started"
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status} [port]"
    exit 1
    ;;
esac
