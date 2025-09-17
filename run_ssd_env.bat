@echo off
chcp 65001 >NUL
call ".venv_ssd\Scripts\activate.bat"
echo [OK] venv ready. run python / streamlit / uvicorn here.
cmd /k

streamlit run app.py --server.port 8501