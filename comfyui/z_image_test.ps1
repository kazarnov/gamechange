# Quick Z-Image Turbo test against the RunPod manager. Images are saved to .\outputs
# Example:
#   .\z_image_test.ps1 -Prompt "a red fox in the snow, telephoto photo" -Width 1216 -Height 832
param(
    [Parameter(Mandatory = $true)][string]$Prompt,
    [int]$Width = 1024,
    [int]$Height = 1024,
    [int]$Steps = 8,
    [Nullable[long]]$Seed = $null,
    [string]$Base = "https://sdcycobzv3miui-8000.proxy.runpod.net",
    [string]$ApiKey = "34cfe8f16596fb3d4b0da8a963832bfce4ee18391bbbb0a8"
)
$headers = @{ "X-API-Key" = $ApiKey }
$body = @{ name = "z_image_turbo"; prompt = $Prompt; width = $Width; height = $Height; steps = $Steps }
if ($null -ne $Seed) { $body.seed = $Seed }
$json = [Text.Encoding]::UTF8.GetBytes(($body | ConvertTo-Json))
$res = Invoke-RestMethod -Method Post -Uri "$Base/run" -Headers $headers -ContentType "application/json; charset=utf-8" -Body $json
if ($res.status -ne "done") { $res | ConvertTo-Json -Depth 5; exit 1 }

$outDir = Join-Path $PSScriptRoot "outputs"
New-Item -ItemType Directory -Force $outDir | Out-Null
foreach ($f in $res.files) {
    $dest = Join-Path $outDir $f.filename
    Invoke-WebRequest -UseBasicParsing -Uri "$Base$($f.url)" -Headers $headers -OutFile $dest
    Write-Host "saved $dest (seed $($res.seed))"
}
