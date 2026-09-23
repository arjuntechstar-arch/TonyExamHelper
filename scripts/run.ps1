[CmdletBinding()]
param(
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$BackendPath = Join-Path $Root 'backend'
$FrontendPath = Join-Path $Root 'frontend'
$PythonPath = Join-Path $Root '.venv\Scripts\python.exe'
$Processes = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()

function Show-Usage {
    Write-Host 'Usage: .\scripts\run.ps1'
    Write-Host ''
    Write-Host 'Starts the FastAPI backend and Angular frontend.'
    Write-Host 'Backend:  http://localhost:8000/api/health'
    Write-Host 'Frontend: http://localhost:4200'
}

function Stop-ProcessTree([int]$ProcessId) {
    if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        & taskkill.exe /PID $ProcessId /T /F | Out-Null
    }
}

function Assert-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

if ($Help) {
    Show-Usage
    exit 0
}

Assert-Command 'npm'

if (-not (Test-Path $PythonPath)) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonPath = (Get-Command python).Source
    }
    else {
        throw "Python was not found. Create the project's .venv or install Python."
    }
}

if (-not (Test-Path (Join-Path $FrontendPath 'node_modules'))) {
    Write-Host 'Frontend dependencies are missing. Installing them...'
    Push-Location $FrontendPath
    try {
        & npm install
        if ($LASTEXITCODE -ne 0) {
            throw 'npm install failed.'
        }
    }
    finally {
        Pop-Location
    }
}

try {
    Write-Host 'Starting FastAPI backend on http://localhost:8000 ...'
    $BackendProcess = Start-Process `
        -FilePath $PythonPath `
        -ArgumentList '-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--reload' `
        -WorkingDirectory $Root `
        -PassThru `
        -NoNewWindow
    $Processes.Add($BackendProcess)

    Write-Host 'Starting Angular frontend on http://localhost:4200 ...'
    $FrontendProcess = Start-Process `
        -FilePath 'npm.cmd' `
        -ArgumentList 'start' `
        -WorkingDirectory $FrontendPath `
        -PassThru `
        -NoNewWindow
    $Processes.Add($FrontendProcess)

    Write-Host ''
    Write-Host 'Application is running. Press Ctrl+C to stop both services.'
    Write-Host 'Health:  http://localhost:8000/api/health'
    Write-Host 'Frontend: http://localhost:4200'

    while ($true) {
        foreach ($Process in $Processes) {
            if ($Process.HasExited) {
                throw "A service exited with code $($Process.ExitCode)."
            }
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    foreach ($Process in $Processes) {
        if (-not $Process.HasExited) {
            Write-Host "Stopping process tree $($Process.Id)..."
            Stop-ProcessTree $Process.Id
        }
    }
}
