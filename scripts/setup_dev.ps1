# Creates .venv at the repo root, installs -e .[dev], and runs pytest.
# Requires Python 3.11+. Prefers the Windows py launcher so a 3.10 `python`
# on PATH does not win.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

function Get-PythonVersion {
    param([Parameter(Mandatory = $true)][string]$Exe)
    $output = & $Exe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($output)) {
        return $null
    }
    return $output.Trim()
}

function Test-PythonAtLeast311 {
    param([Parameter(Mandatory = $true)][string]$Version)
    $parts = $Version.Split(".")
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    return ($major -gt 3) -or (($major -eq 3) -and ($minor -ge 11))
}

function Find-Python {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        # Prefer 3.12/3.11/3.13: the pinned wheels in pyproject.toml match these.
        foreach ($tag in @("3.12", "3.11", "3.13")) {
            $probe = & py "-$tag" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($probe)) {
                return $probe.Trim()
            }
        }
    }

    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -eq $cmd) {
            continue
        }
        $ver = Get-PythonVersion -Exe $cmd.Source
        if ($null -ne $ver -and (Test-PythonAtLeast311 -Version $ver)) {
            return $cmd.Source
        }
    }

    if ($null -ne $py) {
        $probe = & py -3 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($probe)) {
            $exe = $probe.Trim()
            $ver = Get-PythonVersion -Exe $exe
            if ($null -ne $ver -and (Test-PythonAtLeast311 -Version $ver)) {
                return $exe
            }
        }
    }

    return $null
}

$pythonExe = Find-Python
if ($null -eq $pythonExe) {
    Write-Error "Python 3.11+ not found. Install Python 3.11, 3.12, or 3.13 and retry."
}

$foundVersion = Get-PythonVersion -Exe $pythonExe
Write-Host "Using Python $foundVersion at $pythonExe"

$venvDir = Join-Path $RepoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating virtualenv at $venvDir"
    & $pythonExe -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtualenv (exit $LASTEXITCODE)."
    }
}

Write-Host "Installing scout (editable) with [dev] extras"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip upgrade failed (exit $LASTEXITCODE)."
}

& $venvPython -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install -e .[dev] failed (exit $LASTEXITCODE)."
}

Write-Host "Running pytest"
& $venvPython -m pytest
$pytestExit = $LASTEXITCODE
# pytest 5 == no tests collected (expected until M0.2).
if ($pytestExit -ne 0 -and $pytestExit -ne 5) {
    Write-Error "pytest failed (exit $pytestExit)."
}

Write-Host "setup_dev.ps1 completed successfully."
