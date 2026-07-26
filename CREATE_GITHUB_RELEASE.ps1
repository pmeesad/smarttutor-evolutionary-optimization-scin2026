param(
    [Parameter(Mandatory=$true)]
    [string]$EvidencePartsDirectory,

    [string]$Tag = "v4.0.0-scin2026-camera-ready"
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

$zipParts = Get-ChildItem -LiteralPath $partsPath -File |
    Where-Object {
        $_.Name -match '^SCIN2026_Experiments_Part\d{2}.*\.zip$'
    } |
    Sort-Object Name

if ($zipParts.Count -ne 10) {
    throw "Expected exactly 10 split ZIP files (Parts 00-09), but found $($zipParts.Count) in '$partsPath'."
}

$requiredMetadata = @(
    "README_SPLIT_DOWNLOAD.txt",
    "SHA256SUMS.txt"
)

$metadataFiles = foreach ($name in $requiredMetadata) {
    $candidate = Join-Path $partsPath $name
    if (-not (Test-Path -LiteralPath $candidate)) {
        throw "Missing required release asset: $candidate"
    }
    Get-Item -LiteralPath $candidate
}

$assets = @($metadataFiles + $zipParts) | Sort-Object Name

Write-Host "Release assets:" -ForegroundColor Cyan
$assets |
    Select-Object Name,
        @{Name="SizeMiB";Expression={[math]::Round($_.Length / 1MB, 2)}} |
    Format-Table -AutoSize

# A missing release is normal on the first run. Windows PowerShell converts
# native stderr into an error record, so temporarily suppress that expected
# diagnostic while checking whether the release already exists.
$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& gh release view $Tag --json tagName *> $null
$releaseExists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $oldErrorActionPreference

if ($releaseExists) {
    Write-Host "Release '$Tag' already exists; uploading assets with --clobber." -ForegroundColor Yellow

    foreach ($asset in $assets) {
        & gh release upload $Tag $asset.FullName --clobber
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to upload release asset: $($asset.FullName)"
        }
    }
}
else {
    Write-Host "Creating release '$Tag'..." -ForegroundColor Cyan

    $notesFile = Join-Path $PSScriptRoot "RELEASE_NOTES.md"
    if (-not (Test-Path -LiteralPath $notesFile)) {
        throw "Release notes file was not found: $notesFile"
    }

    $assetPaths = @($assets | ForEach-Object { $_.FullName })

    & gh release create $Tag @assetPaths `
        --title "SCIN-2026 SmartTutor Research Evidence v4.0.0" `
        --notes-file $notesFile

    if ($LASTEXITCODE -ne 0) {
        throw "GitHub release creation failed."
    }
}

Write-Host "`nRelease verification:" -ForegroundColor Green
gh release view $Tag

Write-Host "`nOpening release in browser..." -ForegroundColor Cyan
gh release view $Tag --web
