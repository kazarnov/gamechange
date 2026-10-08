#!/usr/bin/env bash
# Runs the manager (which runs ComfyUI) in the foreground; setup.sh starts it in the background.
# Safe to re-run: it stops a running instance first.
# The kit (this folder: scripts, env, workflows/) lives on the network volume. ComfyUI, its venv and
# the models go on the container disk ($APP), which every new pod starts without: the first start
# installs ComfyUI, then restore.sh re-registers workflows/*.json and the manager downloads their models.
set -e
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
APP=${APP:-/opt/comfy}
export WORKSPACE=$APP BUNDLE_DIR=$KIT/workflows  # read by manager.py
VENV=$APP/venv
COMFY_DIR=$APP/ComfyUI
mkdir -p "$APP" "$BUNDLE_DIR"

# Secrets (MANAGER_API_KEY, HF_TOKEN, CIVITAI_TOKEN): KEY=value lines; tr copes with a Windows-edited file
if [ -f "$KIT/env" ]; then set -a; . <(tr -d '\r' < "$KIT/env"); set +a; fi

# Stop a previous instance. The [x] patterns keep pkill from matching the shell that runs it.
RUNNING='[m]anager.py|[m]ain.py --listen 0.0.0.0'
pkill -f "$RUNNING" || true
for _ in $(seq 30); do pgrep -f "$RUNNING" > /dev/null || break; sleep 1; done

# --system-site-packages reuses the image's PyTorch (needs CUDA >= 12.8, see RUNPOD_SETUP.md)
if [ ! -d "$VENV" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"

pip install -q --upgrade comfy-cli fastapi "uvicorn[standard]" requests
comfy --skip-prompt tracking disable || true

# First start on this pod: install ComfyUI + ComfyUI-Manager
if [ ! -f "$COMFY_DIR/main.py" ]; then
  comfy --skip-prompt --workspace="$COMFY_DIR" install --nvidia
fi

# Server-side "Export (API)" conversion endpoint (/workflow/convert)
CONV="$COMFY_DIR/custom_nodes/comfyui-workflow-to-api-converter-endpoint"
if [ ! -d "$CONV" ]; then
  git clone --depth 1 https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint "$CONV"
fi

# Re-register the bundles in workflows/ that this pod doesn't have yet
setsid bash "$KIT/restore.sh" > "$APP/restore.log" 2>&1 < /dev/null &

cd "$APP"
exec python "$KIT/manager.py"
