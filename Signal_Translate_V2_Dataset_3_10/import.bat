cd /d "%~dp0"
echo Activando entorno virtual...
call .venv\Scripts\activate
echo Instalando dependencias desde requirements.txt...
pip install -r requirements.txt

echo.
echo ✅ Instalación completada.