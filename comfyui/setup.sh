#!/usr/bin/env bash
# The one command for a new pod (web terminal or SSH), with the network volume at /workspace:
#     bash /workspace/comfy-kit/setup.sh
# Starts the service in the background and follows it until every workflow's models are downloaded.
# Ctrl+C only stops the following; the service keeps running. Re-running it restarts the service.
#     bash /workspace/comfy-kit/setup.sh --background   # start and return at once (Container Start Command)
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP=${APP:-/opt/comfy}
mkdir -p "$APP"
rm -f "$APP/restore.log"
setsid nohup bash "$KIT/start.sh" > "$APP/manager.log" 2>&1 < /dev/null &
svc=$!
echo "started; logs: $APP/manager.log (service), $APP/restore.log (workflows)"
[ "$1" = "--background" ] && exit 0

key=$(tr -d '\r' < "$KIT/env" 2>/dev/null | sed -n 's/^MANAGER_API_KEY=//p')
pod=${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')}
fail() { echo "$1"; tail -n 30 "$APP/manager.log"; exit 1; }

echo "installing / starting (first start on a pod: a few minutes) ..."
last=
until grep -qs 'restore done' "$APP/restore.log"; do
  kill -0 "$svc" 2> /dev/null || fail "the service stopped:"
  sleep 10
  line=$(tail -n 1 "$APP/manager.log" | cut -c1-150)
  [ "$line" != "$last" ] && echo "  $line" && last=$line
done
cat "$APP/restore.log"

echo "downloading models ..."
while curl -s -H "X-API-Key: $key" http://127.0.0.1:8000/jobs | grep -qE '"status":"(queued|running)"'; do
  kill -0 "$svc" 2> /dev/null || fail "the service stopped:"
  sleep 10
  line=$(grep -a '^\[setup:' "$APP/manager.log" | tail -n 1 | cut -c1-150)
  [ "$line" != "$last" ] && echo "  $line" && last=$line
done
echo
curl -s -H "X-API-Key: $key" http://127.0.0.1:8000/workflows | tr '{' '\n' | sed -n 's/.*"name":"\([^"]*\)","ready":\([a-z]*\).*/  \1  ready=\2/p'
echo
echo "API: https://$pod-8000.proxy.runpod.net   (key: MANAGER_API_KEY in $KIT/env)"
