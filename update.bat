@echo off
cd /d "%~dp0"
echo Este script hará un reset --hard del repositorio.
set /p confirm="¿Deseas continuar? (s/n): "
if /i "%confirm%"=="s" (
    echo Haciendo reset hard del repositorio...
    git fetch origin && git reset --hard origin/main
) else (
    echo Operación cancelada.
)