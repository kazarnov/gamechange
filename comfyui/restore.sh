#!/usr/bin/env bash
# Started by start.sh in the background. Registers every bundle in workflows/<name>.json that the
# manager doesn't have yet (always the case on a new pod); the manager then downloads its models.
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
API="http://127.0.0.1:${MANAGER_PORT:-8000}"
KEY="X-API-Key: ${MANAGER_API_KEY:-}"

for _ in $(seq 360); do
  curl -sf "$API/health" | grep -q '"comfyui_running":true' && break
  sleep 5
done

for f in "$KIT"/workflows/*.json; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .json)
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "$KEY" "$API/workflows/$name")
  if [ "$code" = 200 ]; then
    echo "$name: already registered"
    continue
  fi
  echo "$name: $(curl -s -X POST -H "$KEY" -H 'Content-Type: application/json' --data-binary @"$f" "$API/workflows/$name")"
done
echo "restore done (model downloads continue in the manager: GET /jobs)"
