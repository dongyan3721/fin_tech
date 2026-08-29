#!/usr/bin/env bash
pid=$(netstat -ano | grep ':8000' | grep -i LISTENING | awk '{print $NF}' | head -n1)
if [ -n "$pid" ]; then
  taskkill //F //PID "$pid" && echo "killed $pid"
else
  echo "no listener on 8000"
fi
