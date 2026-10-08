# Setting up a new RunPod pod

The setup kit lives on your **network volume** in `/workspace/comfy-kit`. On any new pod that has
the volume attached, this one command brings the whole service back:

```bash
bash /workspace/comfy-kit/setup.sh
```

## How it's split

| Where | What | Survives a new pod |
|---|---|---|
| Network volume, `/workspace/comfy-kit/` | `setup.sh`, `start.sh`, `restore.sh`, `manager.py`, `env` (secrets), `workflows/*.json` (one bundle per workflow: API JSON, model URLs, input mapping) | yes |
| Container disk, `/opt/comfy/` | ComfyUI, its Python venv, custom nodes, **models**, generated files, logs | no, rebuilt by `setup.sh` |

The kit is tiny (kilobytes). Models are deliberately not kept on the volume: they're downloaded again on
each new pod (about 40 GB for today's 4 workflows, about 13 min at the ~50 MB/s measured on this pod). The
volume is S3-backed and slow with large or many files, so keeping models there wouldn't make things faster anyway.

**New workflows follow automatically.** When a workflow you add through the API (template install,
POST, editor save) becomes ready, or you change its mapping, the manager saves its bundle into
`/workspace/comfy-kit/workflows/`. `DELETE /workflows/{name}` removes it there too.

## Every new pod

### 1. Create the pod

Console → **Pods → Deploy**:

1. **Network volume**: select yours first. It lives in one datacenter (yours: **EU-SE-1**), so only GPUs
   available there are listed.
2. **GPU**: see the last section. Under *Additional Filters*, set **CUDA version 12.8 or newer**.
3. **Template**: **Runpod PyTorch 2.8** (the current pod: PyTorch 2.8, CUDA 12.8, Ubuntu 24.04, Python 3.12).
   Any RunPod PyTorch image with CUDA ≥ 12.8 works: `start.sh` reuses the image's PyTorch.
4. **Edit Template**:
   - **Container Disk**: **200 GB** (models live here). Big video models need ~40–50 GB each, so add that per model.
   - **Expose HTTP Ports**: `8000` (the API, required), `8188` (ComfyUI editor, optional: it has no
     password, so leave it out for production), `8888` (Jupyter, optional).
   - **Expose TCP Ports**: `22`. Needed only for `deploy.ps1` (real SSH with file copy). The
     `ssh.runpod.io` connection works without it but can't copy files.
   - Optional, to start everything automatically each time the pod starts, set the **Container Start Command** to
     ```
     bash -c "bash /workspace/comfy-kit/setup.sh --background; /start.sh"
     ```
     Keep `; /start.sh`: it's RunPod's own startup (SSH, Jupyter). Without the volume the first part just fails and the pod starts normally.
5. **SSH key**: your `~/.ssh/id_ed25519.pub` in RunPod → *Settings → SSH Public Keys* (already set).
6. Deploy. Tip: after step 4 use **Save as template**, so next time it's one click.

### 2. Run the setup

Open a terminal on the pod: **Connect → Web Terminal**, or SSH
(`ssh <pod-id>-...@ssh.runpod.io -i ~/.ssh/id_ed25519`, shown under Connect). Then:

```bash
bash /workspace/comfy-kit/setup.sh
```

It installs ComfyUI, starts the manager, registers every bundle in `workflows/`, follows the model
downloads, and finally prints the workflows and the API URL:
```
  wan_i2v  ready=true
  ...
API: https://<pod-id>-8000.proxy.runpod.net   (key: MANAGER_API_KEY in /workspace/comfy-kit/env)
```
**Ctrl+C or closing the terminal only stops the progress display.** The service keeps running. Running it
again restarts the service (about 1 min).

If you set the Container Start Command, all of this already happens when the pod starts; follow it with
`tail -f /opt/comfy/manager.log`.

### 3. Point your integration at the new pod

**The URL changes with every new pod; the API key doesn't** (it comes from `env` on the volume). Keep the base
URL in your platform's config. Until a workflow's models are downloaded, `/run` answers `202 setting_up`.

Quick check from the PC:
```powershell
.\z_image_test.ps1 -Prompt "a lighthouse at dusk" -Base https://<pod-id>-8000.proxy.runpod.net
```

## Changing the kit from the PC

| File here | Role |
|---|---|
| `manager.py`, `start.sh`, `restore.sh`, `setup.sh` | The service, copied to the volume |
| `pod.env` | Secrets (API key, HF token, optional CivitAI token), copied to the volume as `env`. **Keep it private.** |
| `workflows\*.json` | Your local copy of the bundles |
| `deploy.ps1` | Copies all of the above to the volume through a running pod, then runs `setup.sh` |
| `backup.ps1` | Optional: copies every workflow's bundle to `workflows\` over the API (no SSH needed) |
| `z_image_test.ps1` | Quick image test |

After editing `manager.py`, the scripts or `pod.env`, start any pod with the volume and run:
```powershell
.\deploy.ps1 -SshHost 194.68.245.57 -SshPort 22044
```
(IP and port: pod page → Connect → "SSH over exposed TCP".) For workflows, the volume's copy wins: only
bundles the volume doesn't have yet are uploaded, and at the end the volume's bundles are copied back to
`workflows\`. To replace a workflow, POST it to the API under the same name.

Small changes can also be made directly on the pod, e.g. `nano /workspace/comfy-kit/env` and then
`bash /workspace/comfy-kit/setup.sh`.

## Day to day

- **Logs**: `/opt/comfy/manager.log` (manager + ComfyUI), `/opt/comfy/restore.log` (workflow registration).
- **Change the API key**: edit `MANAGER_API_KEY` in `pod.env` and run `deploy.ps1` (or edit `/workspace/comfy-kit/env`
  on the pod and re-run `setup.sh`), then update your platform.
- **Disk**: outputs accumulate in `/opt/comfy/ComfyUI/output` and are lost with the pod; your platform should
  download what it needs. Check usage with `df -h /`.
- **Stop vs terminate**: either way the container disk starts empty next time, so `setup.sh` reinstalls
  (stopping can also leave you without a free GPU on that machine). Terminating doesn't touch the volume.
- **New pod, same IP and port** → ssh refuses with "REMOTE HOST IDENTIFICATION HAS CHANGED". Clear the old entry
  with `ssh-keygen -R "[194.68.245.57]:22044"` (use your IP and port).

## Bigger GPU, better video models

| VRAM | GPUs on RunPod | Good for |
|---|---|---|
| 24–32 GB | RTX 4090, RTX 5090, L4 | the image workflows; Wan 5B with shorter clips |
| 48 GB | A40 (current), RTX A6000, L40S | everything installed today; Wan 2.2 14B fits, but it's slow |
| 80 GB | A100 80GB, **H100** | Wan 2.2 14B, LTX-2.5 and HunyuanVideo 1.5 at 720p. An H100 is several times faster than an A40. |
| 96–141 GB | **RTX PRO 6000** (96 GB), H200 (141 GB) | the same with room for longer/larger clips and several models loaded at once |

The volume ties you to its datacenter, so check which of these EU-SE-1 offers; for another datacenter, create a
new volume there and run `deploy.ps1` once against a pod using it. Also check the pod's **system RAM (64 GB
or more)**: big video models are moved between RAM and VRAM.

Recommended upgrades. All are built-in ComfyUI templates, installed with one API call
(`POST /templates/<template>/install {"as": "<your name>"}`, see API.md Recipe A). Once ready, they're saved to the
volume like any other workflow:

| Model | Templates | Download | Notes |
|---|---|---|---|
| **Wan 2.2 14B** | `video_wan2_2_14B_t2v`, `video_wan2_2_14B_i2v` (also `_flf2v` first→last frame, `_s2v` audio-driven) | ~38 GB each; they share the 7 GB text encoder | The big brother of today's Wan 5B: clearly better motion and detail. Ships with the "lightx2v" 4-step LoRAs for fast generation. |
| **LTX-2.5** (22B) | `video_ltx2_5_t2v`, `video_ltx2_5_i2v`, `video_ltx2_5_flf2v` | ~40 GB (estimate) | Generates **video with synchronized audio**. **Gated**: first accept the license at huggingface.co/Lightricks/LTX-2.5 while logged in to the account that owns the HF token, or downloads fail with 403. Its main node is a subgraph, so expect to add a mapping for `duration`/size. |
| **HunyuanVideo 1.5** | `video_hunyuan_video_1.5_720p_t2v`, `..._i2v` | ~46 GB each; they share ~30 GB | 720p generation with a 1080p upscaler stage. |

After installing one, look at `GET /workflows/<name>`. Big templates often contain two pipelines, or have
the image loader bypassed. API.md (Recipe A checklist) explains how to fix that. Then test with `"wait": false`.
Every workflow on the volume is downloaded on each new pod, so `DELETE` the ones you no longer use.

Other video templates worth a look: `video_wan_animate2` (motion transfer from a reference video),
`video_wan2_1_infinitetalk` (talking avatar from audio), `video_wan_vace_*` (video editing/inpainting).
List them all with `GET /templates?q=video`.
