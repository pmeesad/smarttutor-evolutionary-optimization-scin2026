param(
    [Parameter(Mandatory=$true)]
    [string]$EvidencePartsDirectory,

    [string]$Tag = "v1.0.0-scin2026-camera-ready"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is not installed."
}
gh auth status
if ($LASTEXITCODE -ne 0) {
    throw "Authenticate first with: gh auth login"
}

$partsPath = Resolve-Path -LiteralPath $EvidencePartsDirectory
$assets = Get-ChildItem -LiteralPath $partsPath -File |
    Where-Object {
        $_.Name -match '^SCIN2026_Experiments_Part\d+.*\.zip$' -or
        $_.Name -in @("README_SPLIT_DOWNLOAD.txt", "SHA256SUMS.txt")
    } |
    Sort-Object Name

if ($assets.Count -lt 9) {
    throw "Expected the split evidence ZIPs in '$partsPath'; found only $($assets.Count) matching files."
}

Write-Host "Release assets:" -ForegroundColor Cyan
$assets | Select-Object Name,
    @{Name="SizeMiB";Expression={[math]::Round($_.Length / 1MB, 2)}} |
    Format-Table -AutoSize

$existing = gh release view $Tag 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Release '$Tag' already exists; uploading assets with --clobber." -ForegroundColor Yellow
    foreach ($asset in $assets) {
        gh release upload $Tag $asset.FullName --clobber
    }
} else {
    $assetPaths = @($assets | ForEach-Object { $_.FullName })
    & gh release create $Tag @assetPaths `
        --title "SCIN-2026 SmartTutor Research Evidence v1.0.0" `
        --notes-file "RELEASE_NOTES.md"
}

gh release view $Tag --web
