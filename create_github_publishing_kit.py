from pathlib import Path
import zipfile, hashlib, textwrap, os

out = Path("/mnt/data/SCIN2026_GitHub_Publishing_Kit")
out.mkdir(parents=True, exist_ok=True)

gitignore = r"""# Python
__pycache__/
*.py[cod]
*.pyd
*.so
.venv/
venv/
env/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Editors and operating systems
.vscode/
.idea/
.DS_Store
Thumbs.db
desktop.ini

# Secrets and local configuration
.env
.env.*
!.env.example
*.pem
*.key
*.p12
*.pfx

# Build and temporary files
build/
dist/
*.egg-info/
*.tmp
*.temp
*.log

# Large candidate-level evidence is distributed as GitHub release assets
# and archived in Zenodo. Summary outputs remain tracked in the repository.
outputs_v4/runs/
outputs_v4/references/

# Do not commit packaged evidence archives into normal Git history
*.zip
*.7z
*.tar
*.tar.gz
"""

release_notes = r"""# SCIN-2026 SmartTutor Research Evidence — v1.0.0

This release provides the reproducibility evidence for the paper:

**Evolutionary Multi-Objective Configuration Search for a Retrieval-Augmented SmartTutor: A Multi-Scenario Simulation Study**

## Contents

- Python source code and experiment scripts
- Exact feasible-space decoder
- Twelve fixed benchmark realizations
- Matched multi-seed algorithm experiments
- D-NSGA-II and D-SPEA2 implementations
- LMS+shift-scrambled Sobol and random baselines
- Independent reference-front evidence
- Hypervolume, IGD+, and additive-epsilon results
- Decision-rule and query-class analyses
- Sixty-four simulator scenarios
- Figures, tables, statistical outputs, and manifests
- Reproduction instructions and software environment metadata

## Data statement

All requests and performance values are synthetic. The release contains no learner records,
identifiable personal information, production prompts, private credentials, or live service logs.

## Licenses

- Source code: MIT License
- Synthetic data and documentation: CC BY 4.0

## Version identifiers

- Release tag: `v1.0.0-scin2026-camera-ready`
- Git commit: recorded by the GitHub release
- Evidence checksums: see `SHA256SUMS.txt` and `MANIFEST.sha256`
"""

push_ps1 = r'''param(
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
    $largeFiles | Select-Object FullName,
        @{Name="SizeMiB";Expression={[math]::Round($_.Length / 1MB, 2)}} |
        Format-Table -AutoSize
    Write-Warning "Large files are present. Confirm that they are ignored or use release assets."
}

Write-Host "`nBasic secret-pattern audit..." -ForegroundColor Cyan
$secretPattern = '(?i)(api[_-]?key|secret[_-]?key|password\s*=|token\s*=|BEGIN [A-Z ]*PRIVATE KEY)'
$secretHits = Get-ChildItem -Recurse -File -Force |
    Where-Object {
        $_.FullName -notmatch '\\.git\\' -and
        $_.Extension -match '\.(py|ps1|sh|toml|yaml|yml|json|md|txt|env|ini|cfg)$'
    } |
    Select-String -Pattern $secretPattern -ErrorAction SilentlyContinue

if ($secretHits) {
    $secretHits | Select-Object Path, LineNumber, Line | Format-Table -Wrap
    throw "Potential secret-like content was found. Review it before publishing."
}

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
    $oversizedStaged | Select-Object FullName,
        @{Name="SizeMiB";Expression={[math]::Round($_.Length / 1MB, 2)}} |
        Format-Table -AutoSize
    throw "At least one staged file is too large for ordinary Git. Unstage and ignore it."
}

Write-Host "`nFiles staged for the first commit:" -ForegroundColor Cyan
git status --short

$hasStagedChanges = git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "Publish SCIN-2026 SmartTutor experiment package"
} else {
    Write-Host "No new staged changes to commit." -ForegroundColor Yellow
}

$fullName = "$GitHubOwner/$RepositoryName"
$remoteExists = git remote | Select-String -SimpleMatch "origin"
if (-not $remoteExists) {
    gh repo view $fullName *> $null
    if ($LASTEXITCODE -eq 0) {
        git remote add origin "https://github.com/$fullName.git"
    } else {
        gh repo create $fullName `
            --$Visibility `
            --source . `
            --remote origin `
            --description "Code, synthetic data summaries, and reproducibility evidence for the SCIN-2026 SmartTutor multi-objective optimization study."
    }
}

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
'''

release_ps1 = r'''param(
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
'''

guide = r"""# GitHub Publication Guide

## Recommended publication structure

The normal Git repository contains:

- `src/`, `scripts/`, `tests/`, and `figures/`
- project metadata and licenses
- summary CSV/JSON outputs
- scenario summaries and compact Pareto evidence

The following large folders are intentionally excluded from ordinary Git history:

- `outputs_v4/runs/`
- `outputs_v4/references/`

The complete candidate-level and independent-reference evidence is attached to the
fixed GitHub release as split ZIP assets and can also be deposited in Zenodo.

## 1. Copy this kit into the experiment root

Place these files beside `README.md`:

- `.gitignore`
- `PREPARE_AND_PUSH.ps1`
- `CREATE_GITHUB_RELEASE.ps1`
- `RELEASE_NOTES.md`

## 2. Install and authenticate

Open PowerShell in the experiment directory:

```powershell
git --version
gh --version
gh auth login