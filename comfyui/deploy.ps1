# Copies the kit in this folder to the network volume of a running pod (/workspace/comfy-kit) and
# (re)starts the service there. You only need it after changing files here: a new pod that has the
# volume just runs   bash /workspace/comfy-kit/setup.sh   (web terminal or SSH).
#   .\deploy.ps1 -SshHost 194.68.245.57 -SshPort 22044
# Host and port: pod page -> Connect -> "SSH over exposed TCP" (ssh root@<ip> -p <port>).
# Workflows: the copy on the volume wins (the manager keeps it up to date). Only bundles the volume
# doesn't have are uploaded, and at the end the volume's bundles are copied back into workflows\.
param(
    [Parameter(Mandatory = $true)][string]$SshHost,
    [Parameter(Mandatory = $true)][int]$SshPort,
    [string]$KeyFile = "$HOME\.ssh\id_ed25519"
)
$ErrorActionPreference = "Stop"
$opts = @("-i", $KeyFile, "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20")
$target = "root@$SshHost"
$kit = "/workspace/comfy-kit"

function Remote([string]$cmd) {
    ssh @opts -p $SshPort $target $cmd
    if ($LASTEXITCODE -ne 0) { throw "ssh failed: $cmd" }
}
function Copy-To([string[]]$files, [string]$dest) {
    scp @opts -P $SshPort -q @files "${target}:$dest"
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $files" }
}

Push-Location $PSScriptRoot
try {
    foreach ($f in "manager.py", "start.sh", "restore.sh", "setup.sh", "pod.env") {
        if (-not (Test-Path $f)) { throw "missing $f next to deploy.ps1" }
    }
    Write-Host "copying the kit to ${target}:$kit ..."
    Remote "mkdir -p $kit/workflows"
    Copy-To @("manager.py", "start.sh", "restore.sh", "setup.sh") "$kit/"
    Copy-To @("pod.env") "$kit/env"
    $onVolume = @(ssh @opts -p $SshPort $target "ls $kit/workflows")
    $new = @(Get-ChildItem workflows\*.json -ErrorAction SilentlyContinue |
        Where-Object { $onVolume -notcontains $_.Name } | ForEach-Object { "workflows/$($_.Name)" })
    if ($new.Count) { Write-Host "new workflows: $($new -join ', ')"; Copy-To $new "$kit/workflows/" }
    # strip Windows line endings in place (sed -i fails on the volume: it can't set permissions)
    Remote "cd $kit && for f in env *.sh; do tr -d '\r' < `$f > /tmp/lf && cat /tmp/lf > `$f; done"

    Write-Host "running setup.sh on the pod (Ctrl+C stops watching; the pod keeps going) ..."
    ssh @opts -p $SshPort $target "bash $kit/setup.sh"

    New-Item -ItemType Directory -Force workflows | Out-Null
    scp @opts -P $SshPort -q "${target}:$kit/workflows/*.json" workflows\
    if ($LASTEXITCODE -eq 0) { Write-Host "copied the volume's workflow bundles to workflows\" }
}
finally {
    Pop-Location
}
