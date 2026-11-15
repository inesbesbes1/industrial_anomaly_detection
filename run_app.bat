@echo off
REM Lance l'interface Streamlit avec l'environnement virtuel du projet (.venv)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERREUR] .venv introuvable. Creez-le avec : py -3.11 -m venv .venv  puis  .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
pause
