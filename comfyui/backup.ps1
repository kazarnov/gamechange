# Saves every workflow registered on the pod as workflows\<name>.json, the bundle format deploy.ps1
# re-creates on a new pod. Run it after adding workflows, the pod's disk is not permanent.
#   .\backup.ps1 -Base https://<pod-id>-8000.proxy.runpod.net -ApiKey <key>
param(
    [Parameter(Mandatory = $true)][string]$Base,
    [Parameter(Mandatory = $true)][string]$ApiKey
)
$ErrorActionPreference = "Stop"
$headers = @{ "X-API-Key" = $ApiKey }
$out = Join-Path $PSScriptRoot "workflows"
New-Item -ItemType Directory -Force $out | Out-Null

$list = Invoke-RestMethod "$Base/workflows" -Headers $headers  # assign first: PS 5.1 won't unroll it inline
foreach ($wf in $list) {
    if (-not $wf.ready) { Write-Warning "$($wf.name): not ready, skipped"; continue }
    $res = Invoke-WebRequest -UseBasicParsing "$Base/workflows/$($wf.name)/bundle" -Headers $headers
    $bytes = $res.RawContentStream.ToArray()  # raw UTF-8, prompts may contain non-Latin text
    [IO.File]::WriteAllBytes((Join-Path $out "$($wf.name).json"), $bytes)
    $bundle = [Text.Encoding]::UTF8.GetString($bytes) | ConvertFrom-Json
    Write-Host "saved $($wf.name) ($(@($bundle.models).Count) models)"
    if ($bundle.models_without_url) {
        Write-Warning "$($wf.name): no download URL for $($bundle.models_without_url -join ', '). Add them to `"models`" in the file, or a new pod won't have them."
    }
}
