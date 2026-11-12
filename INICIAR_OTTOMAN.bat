@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creando entorno e instalando dependencias, puede tardar unos minutos...
  py -3.11 -m venv .venv || python -m venv .venv
  .venv\Scripts\python.exe -m pip install -q --upgrade pip
  .venv\Scripts\python.exe -m pip install -q -r requirements.txt
)
if not exist .env copy .env.example .env >nul
start "" http://127.0.0.1:8765
.venv\Scripts\python.exe -m uvicorn coach.servidor:app --host 127.0.0.1 --port 8765 --reload --reload-dir coach
pause
