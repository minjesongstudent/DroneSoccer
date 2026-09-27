param(
    [string]$Name = 'unified-local',
    [int]$Steps = 1000000,
    [int]$Port = 8773,
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        throw 'Create the local .venv using README.md or supply -Python with your interpreter path.'
    }
}
$launchArgs = @('-m', 'src.live', '--steps', "$Steps", '--name', $Name, '--port', "$Port")
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot "runs\$Name\resume.json")) {
    $launchArgs += '--resume'
}
& $Python @launchArgs
if ($LASTEXITCODE -ne 0) { throw "Trainer exited with code $LASTEXITCODE" }
