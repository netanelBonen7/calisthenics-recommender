$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$configDir = Join-Path $repoRoot "config"
$cacheDir = Join-Path $repoRoot "data\cache"
$exampleConfig = Join-Path $configDir "docker-runtime.toml.example"
$runtimeConfig = Join-Path $configDir "docker-runtime.toml"

Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force $configDir | Out-Null
    New-Item -ItemType Directory -Force $cacheDir | Out-Null

    if (-not (Test-Path $exampleConfig)) {
        throw "Missing example config: $exampleConfig"
    }

    if (-not (Test-Path $runtimeConfig)) {
        Copy-Item $exampleConfig $runtimeConfig
        Write-Host "Created config\docker-runtime.toml from the tracked example."
    } else {
        Write-Host "Using existing config\docker-runtime.toml."
    }

    & uv run build-exercise-cache --config ".\config\docker-runtime.toml"
    if ($LASTEXITCODE -ne 0) {
        throw "uv run build-exercise-cache failed with exit code $LASTEXITCODE"
    }

    Write-Host "Prepared data\cache\docker_smoke_embeddings.sqlite."
}
finally {
    Pop-Location
}
