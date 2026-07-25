param(
    [string]$GitHubOwner = "pmeesad",
    [string]$RepositoryName = "smarttutor-evolutionary-optimization-scin2026",
    [ValidateSet("public","private")]
    [string]$Visibility = "public"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "Repository directory: $PWD" -ForegroundColor Cyan

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. Install it before continuing."
    }
}

Require-Command git
Require-Command gh

Write-Host "`nChecking GitHub authentication..." -ForegroundColor Cyan
gh auth status
if ($LASTEXITCODE -ne 0) {
    Write-Host "Run: gh auth login" -ForegroundColor Yellow
    throw "GitHub CLI authentication is required."
}

# Configure author identity only if it is not already defined.
if (-not (git config user.name)) {
    git config user.name "Phayung Meesad"
}
if (-not (git config user.email)) {
    git config user.email "phayung.m@itd.kmutnb.ac.th"
}

if (-not (Test-Path ".git")) {
    git init
}
git branch -M main

Write-Host "`nChecking for files at or above 95 MiB..." -ForegroundColor Cyan
$largeFiles = Get-ChildItem -Recurse -File -Force |
    Where-Object {
        $_.FullName -notmatch '\\.git\\' -and
        $_.Length -ge 95MB
    } |
    Sort-Object Length -Descending

if ($largeFiles) {
    $largeFiles |
        Select-Object FullName,
            @{Name="SizeMiB";Expression={[math]::Round($_.Length / 1MB, 2)}} |
        Format-Table -AutoSize

    Write-Warning "Large files are present. Confirm that they are ignored or publish them as release assets."
}

Write-Host "`nCredential-pattern audit..." -ForegroundColor Cyan

# This expression intentionally avoids generic scientific names such as
# token_cost, token_count, retrieval_token, and generation_tokens.
$secretPattern = '(?i)(\bapi[_-]?key\b\s*[:=]|\bsecret[_-]?key\b\s*[:=]|\bpassword\b\s*[:=]|\b(?:access|auth|github|private)[_-]?token\b\s*[:=]|BEGIN [A-Z ]*PRIVATE KEY)'

$secretHits = Get-ChildItem -Recurse -File -Force |
    Where-Object {
        $_.FullName -notmatch '\\.git\\' -and
        $_.Extension -match '\.(py|ps1|sh|toml|yaml|yml|json|md|txt|env|ini|cfg)$'
    } |
    Select-String -Pattern $secretPattern -ErrorAction SilentlyContinue

if ($secretHits) {
    $secretHits |
        Select-Object Path, LineNumber,
            @{Name="MatchedText";Expression={$_.Matches.Value}},
            Line |
        Format-List

    throw "Credential-like content was found. Review or remove it before publishing."
}

Write-Host "No credential-like assignments were detected." -ForegroundColor Green

git add .

# Validate staged file sizes after .gitignore has been applied.
$staged = git diff --cached --name-only --diff-filter=ACM
$oversizedStaged = @()

foreach ($relativePath in $staged) {
    if (Test-Path -LiteralPath $relativePath) {
        $item = Get-Item -LiteralPath $relativePath
        if (-not $item.PSIsContainer -and $item.Length -ge 95MB) {
            $oversizedStaged += $item
        }
    }
}

if ($oversizedStaged.Count -gt 0) {
    $oversizedStaged |
        Select-Object FullName,
            @{Name="SizeMiB";Expression={[math]::Round($_.Length / 1MB, 2)}} |
        Format-Table -AutoSize

    throw "At least one staged file is too large for ordinary Git. Unstage and ignore it."
}

Write-Host "`nFiles staged for the first commit:" -ForegroundColor Cyan
git status --short

git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "Publish SCIN-2026 SmartTutor experiment package"
}
else {
    Write-Host "No new staged changes to commit." -ForegroundColor Yellow
}

$fullName = "$GitHubOwner/$RepositoryName"
$remoteExists = git remote | Select-String -SimpleMatch "origin"

if (-not $remoteExists) {
    gh repo view $fullName *> $null

    if ($LASTEXITCODE -eq 0) {
        git remote add origin "https://github.com/$fullName.git"
    }
    else {
        gh repo create $fullName `
            --$Visibility `
            --source . `
            --remote origin `
            --description "Code, synthetic data summaries, and reproducibility evidence for the SCIN-2026 SmartTutor multi-objective optimization study."
    }
}

Write-Host "`nConfigured remotes:" -ForegroundColor Cyan
git remote -v

git push -u origin main

$tag = "v1.0.0-scin2026-camera-ready"
if (-not (git tag --list $tag)) {
    git tag -a $tag -m "SCIN-2026 camera-ready experiment release"
}
git push origin $tag

$commit = git rev-parse HEAD
$url = gh repo view $fullName --json url --jq '.url'

Write-Host "`nPush completed." -ForegroundColor Green
Write-Host "Repository: $url"
Write-Host "Commit:     $commit"
Write-Host "Tag:        $tag"
Write-Host "`nNext: run CREATE_GITHUB_RELEASE.ps1 with the folder containing Parts 00-09." -ForegroundColor Cyan
