# GitHub Publication Guide

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
```

Choose:

- GitHub.com
- HTTPS
- Login with a web browser
- Authenticate Git with GitHub credentials: Yes

## 3. Run the push script

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\PREPARE_AND_PUSH.ps1
```

To use another owner or repository name:

```powershell
.\PREPARE_AND_PUSH.ps1 `
  -GitHubOwner "pmeesad" `
  -RepositoryName "smarttutor-evolutionary-optimization-scin2026" `
  -Visibility public
```

## 4. Publish the evidence release

```powershell
.\CREATE_GITHUB_RELEASE.ps1 `
  -EvidencePartsDirectory "D:\PATH\TO\SCIN2026_SmartTutor_Experiments_Split"
```

## 5. Record identifiers

```powershell
git rev-parse HEAD
git tag --list
gh repo view --web
gh release view v1.0.0-scin2026-camera-ready --web
```

Insert the resulting repository URL, full commit SHA, release tag, and later the
Zenodo DOI into the paper and `CITATION.cff`.

## 6. Subsequent corrections

```powershell
git add .
git commit -m "Describe the correction"
git push
```

Do not move the existing release tag after publication. Create `v1.0.1` for any
change that alters the archived evidence.
