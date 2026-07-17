param(
    [string]$PythonVersion = "3.12",
    [switch]$RecreateVenv,
    [switch]$SkipModelSmoke,
    [switch]$LocalFilesOnly,
    [switch]$RunSmokeExperiments
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    Write-Host "+ $FilePath $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

Write-Step "Checking Python launcher"
$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $PyLauncher) {
    throw "Python launcher 'py' was not found. Install Python $PythonVersion from python.org or run: winget install Python.Python.$PythonVersion"
}

$PythonPath = (& py "-$PythonVersion" -c "import sys; print(sys.executable)").Trim()
if ($LASTEXITCODE -ne 0 -or -not $PythonPath) {
    throw "Could not find Python $PythonVersion. Run 'py -0p' to list installed versions."
}
if ($PythonPath -like "*\WindowsApps\*") {
    throw "Python resolves to the Microsoft Store shim: $PythonPath. Install Python from python.org, reopen PowerShell, then rerun this script."
}
Write-Host "Using Python: $PythonPath"

$VenvPath = Join-Path $RepoRoot ".venv"
if ($RecreateVenv -and (Test-Path $VenvPath)) {
    Write-Step "Removing existing .venv"
    $ResolvedVenv = Resolve-Path -LiteralPath $VenvPath
    if (-not $ResolvedVenv.Path.StartsWith($RepoRoot.Path)) {
        throw "Refusing to remove venv outside repo: $($ResolvedVenv.Path)"
    }
    Remove-Item -LiteralPath $ResolvedVenv.Path -Recurse -Force
}

if (-not (Test-Path $VenvPath)) {
    Write-Step "Creating virtual environment"
    Invoke-Checked "py" @("-$PythonVersion", "-m", "venv", ".venv")
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python not found at $VenvPython"
}

$VenvPythonPath = (& $VenvPython -c "import sys; print(sys.executable)").Trim()
if ($VenvPythonPath -like "*\WindowsApps\*") {
    throw "The virtual environment points to the Microsoft Store shim. Delete .venv and install Python from python.org."
}

Write-Step "Installing project dependencies"
Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $VenvPython @("-m", "pip", "install", "-e", ".[dev]")

Write-Step "Running fast checks"
Invoke-Checked $VenvPython @("-m", "pytest")
Invoke-Checked $VenvPython @("-m", "lightning_decoding.cli", "--help")

if (-not $SkipModelSmoke) {
    Write-Step "Running Pythia smoke test"
    $SmokeArgs = @(
        "-m",
        "lightning_decoding.cli",
        "smoke",
        "--model",
        "EleutherAI/pythia-160m"
    )
    if ($LocalFilesOnly) {
        $SmokeArgs += "--local-files-only"
    }
    Invoke-Checked $VenvPython $SmokeArgs

    Write-Step "Benchmarking Pythia single forward pass"
    $BenchmarkArgs = @(
        "-m",
        "lightning_decoding.cli",
        "benchmark-forward",
        "--model",
        "EleutherAI/pythia-160m",
        "--runs",
        "20",
        "--prompt",
        "Q: Name one animal that commonly appears in children's books. A: One animal is the"
    )
    if ($LocalFilesOnly) {
        $BenchmarkArgs += "--local-files-only"
    }
    Invoke-Checked $VenvPython $BenchmarkArgs
}

if ($RunSmokeExperiments) {
    Write-Step "Filtering single-token category answer spaces"
    $FilterArgs = @(
        "-m",
        "lightning_decoding.cli",
        "filter-token-space",
        "configs\base.yaml"
    )
    if ($LocalFilesOnly) {
        $FilterArgs += "--local-files-only"
    }
    Invoke-Checked $VenvPython $FilterArgs

    Write-Step "Running phase 1 smoke experiment"
    Invoke-Checked $VenvPython @(
        "-m",
        "lightning_decoding.cli",
        "run",
        "configs\phase1_baselines.yaml"
    )

    Write-Step "Running phase 2 smoke experiment"
    Invoke-Checked $VenvPython @(
        "-m",
        "lightning_decoding.cli",
        "run",
        "configs\phase2_ensemble.yaml"
    )
}

Write-Step "Setup complete"
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
