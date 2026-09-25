[CmdletBinding()]
param(
    [switch]$Help,
    [switch]$StopInfrastructure
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$BackendPath = Join-Path $Root 'backend'
$FrontendPath = Join-Path $Root 'frontend'
$LogPath = Join-Path $Root 'logs'
$PythonPath = Join-Path $Root '.venv\Scripts\python.exe'
$BackendPort = 8765
$Processes = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()
$MongoStartedByScript = $false

if (-not (Test-Path $LogPath)) {
    New-Item -ItemType Directory -Path $LogPath | Out-Null
}

function Show-Usage {
    Write-Host 'Usage: .\scripts\run.ps1 [-StopInfrastructure]'
    Write-Host ''
    Write-Host 'Starts the FastAPI backend and Angular frontend, using a local MongoDB when available.'
    Write-Host "Backend:  http://localhost:$BackendPort/api/health"
    Write-Host 'Frontend: http://localhost:4200'
    Write-Host ''
    Write-Host 'If MongoDB is not already running, Docker Compose is used when Docker is available.'
    Write-Host '-StopInfrastructure stops MongoDB only when this script started it with Docker Compose.'
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

function Test-TcpPort([string]$HostName, [int]$Port, [int]$TimeoutMilliseconds = 1000) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $connected = $client.BeginConnect($HostName, $Port, $null, $null)
        return $connected.AsyncWaitHandle.WaitOne($TimeoutMilliseconds) -and $client.Connected
    }
    finally {
        $client.Dispose()
    }
}

function Wait-TcpPort([string]$HostName, [int]$Port, [int]$TimeoutSeconds = 30) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if (Test-TcpPort $HostName $Port) {
            return
        }
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)

    throw "Timed out waiting for MongoDB at $HostName`:$Port."
}

function Assert-PythonDependencies([string]$Executable) {
    & $Executable -c 'import fastapi, uvicorn, pymongo' 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Backend dependencies are missing. Installing them...'
        & $Executable -m pip install -r (Join-Path $BackendPath 'requirements.txt')
        if ($LASTEXITCODE -ne 0) {
            throw 'Backend dependency installation failed.'
        }
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

Assert-PythonDependencies $PythonPath

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
    if (Test-TcpPort 'localhost' 27017) {
        Write-Host 'Using the MongoDB instance already running on localhost:27017.'
    }
    elseif (Get-Command 'docker' -ErrorAction SilentlyContinue) {
        Write-Host 'Starting MongoDB with Docker Compose...'
        & docker compose up -d mongodb
        if ($LASTEXITCODE -ne 0) {
            throw 'MongoDB failed to start. Ensure Docker Desktop is running and retry.'
        }
        $MongoStartedByScript = $true
        Wait-TcpPort 'localhost' 27017
    }
    else {
        throw "MongoDB is not running on localhost:27017 and Docker is unavailable. Start a local MongoDB service, or install Docker Desktop and run this script again."
    }

    $BackendProcess = $null
    if (Test-TcpPort 'localhost' $BackendPort) {
        try {
            $Health = Invoke-RestMethod -Uri "http://localhost:$BackendPort/api/health" -TimeoutSec 3
            if ($Health.status -ne 'ok') {
                throw 'unexpected health response'
            }
            Write-Host "Reusing the Aug27 FastAPI backend already running on http://localhost:$BackendPort."
        }
        catch {
            throw "Port $BackendPort is already in use by another application. Stop that application, or configure it to use a different port before running Aug27 Exam Studio."
        }
    }
    else {
        Write-Host "Starting FastAPI backend on http://localhost:$BackendPort ..."
        $BackendProcess = Start-Process `
            -FilePath $PythonPath `
            -ArgumentList '-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', $BackendPort `
            -WorkingDirectory $Root `
            -PassThru `
            -NoNewWindow
        $Processes.Add($BackendProcess)
    }

    Write-Host 'Starting Angular frontend on http://localhost:4200 ...'
    $FrontendProcess = Start-Process `
        -FilePath 'npm.cmd' `
        -ArgumentList 'start' `
        -WorkingDirectory $FrontendPath `
        -PassThru `
        -NoNewWindow
    $Processes.Add($FrontendProcess)

    Write-Host ''
    Write-Host 'Application is running. Press Ctrl+C to stop the backend and frontend.'
    Write-Host "Health:  http://localhost:$BackendPort/api/health"
    Write-Host 'Frontend: http://localhost:4200'

    while ($true) {
        foreach ($Process in $Processes) {
            if ($Process.HasExited) {
                $ServiceName = if ($BackendProcess -and $Process.Id -eq $BackendProcess.Id) { 'FastAPI backend' } else { 'Angular frontend' }
                $ExitCode = $Process.ExitCode
                throw "$ServiceName exited with code $ExitCode. Check the service output above for details."
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
    if ($StopInfrastructure -and $MongoStartedByScript) {
        Write-Host 'Stopping MongoDB Compose service...'
        & docker compose stop mongodb | Out-Null
    }
}
