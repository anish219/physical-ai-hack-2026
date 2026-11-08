$ErrorActionPreference = 'Stop'
& "$PSScriptRoot/.venv/Scripts/python.exe" "$PSScriptRoot/server.py" @args
exit $LASTEXITCODE
