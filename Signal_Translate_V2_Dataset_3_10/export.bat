cd /d "%~dp0"
echo Activando entorno virtual...
call venv\Scripts\activate

echo Exportando dependencias a requirements.txt...
pip freeze > requirements.txt

echo.
echo ✅ Archivo requirements.txt generado correctamente.