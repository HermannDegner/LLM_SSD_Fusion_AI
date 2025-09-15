@echo off
setlocal enableextensions
rem =============================================================
rem  SSD-LLM_Manager (Fixed)
rem  Safe, defensive Windows batch for SSD/LLM projects
rem  - Avoids IF/FOR paren pitfalls
rem  - Quotes all paths
rem  - Uses simple IF comparisons (no multi-line compound blocks)
rem =============================================================

rem remember script root (folder where this .bat resides)
set "ROOT=%~dp0"
set "VENV_DIR=%ROOT%ssd_llm_env"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PIP_EXE=%VENV_DIR%\Scripts\pip.exe"
set "REQ_FILE=%ROOT%requirements.txt"

:menu
echo.
echo ===================[ SSD-LLM Manager ]===================
echo   1) Setup Environment (create venv, install deps)
echo   2) Activate venv (start a shell with venv active)
echo   3) Run main (python main.py)
echo   4) Update dependencies (pip install -r requirements.txt)
echo   5) Clean venv (delete ssd_llm_env)
echo   0) Exit
echo =========================================================
set /p "OPT=Select: "
if "%OPT%"=="1" goto setup_env
if "%OPT%"=="2" goto activate
if "%OPT%"=="3" goto run_main
if "%OPT%"=="4" goto update_deps
if "%OPT%"=="5" goto clean_venv
if "%OPT%"=="0" goto end
echo Invalid option.
goto menu

:setup_env
echo.
echo ===================[ 1. Setup Environment ]===================
echo ROOT       = "%ROOT%"
echo VENV_DIR   = "%VENV_DIR%"
echo PYTHON_EXE = "%PYTHON_EXE%"
echo PIP_EXE    = "%PIP_EXE%"
echo.

rem 1) Find Python (py launcher preferred, then python)
where py >nul 2>&1
if errorlevel 1 (
  where python >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10-3.12 from python.org.
    goto menu
  )
)

rem 2) Create venv if missing
if exist "%PYTHON_EXE%" (
  echo venv already exists.
) else (
  echo Creating venv...
  py -3 -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create venv. Try running as Admin or check AV.
    goto menu
  )
)

rem 3) Upgrade pip/setuptools/wheel
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [WARN] Failed to upgrade pip/setuptools/wheel.
)

rem 4) Install requirements if present
if exist "%REQ_FILE%" (
  echo Installing requirements.txt ...
  "%PIP_EXE%" install -r "%REQ_FILE%"
  if errorlevel 1 (
    echo [ERROR] Failed installing dependencies.
    goto menu
  )
) else (
  echo requirements.txt not found. Skipping deps install.
)

echo.
echo [OK] Setup complete.
goto menu

:activate
echo.
echo ===================[ 2. Activate venv ]===================
if not exist "%PYTHON_EXE%" (
  echo venv not found. Run "1) Setup Environment" first.
  goto menu
)
echo Starting a new shell with venv activated...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate venv.
  goto menu
)
echo (venv) To exit this shell later, type:  deactivate
cmd /k
goto menu

:run_main
echo.
echo ===================[ 3. Run main.py ]===================
if not exist "%PYTHON_EXE%" (
  echo venv not found. Run "1) Setup Environment" first.
  goto menu
)
if exist "%ROOT%main.py" (
  "%PYTHON_EXE%" "%ROOT%main.py"
) else (
  echo main.py not found at "%ROOT%".
)
goto menu

:update_deps
echo.
echo ===================[ 4. Update dependencies ]===================
if not exist "%PYTHON_EXE%" (
  echo venv not found. Run "1) Setup Environment" first.
  goto menu
)
if exist "%REQ_FILE%" (
  "%PIP_EXE%" install -r "%REQ_FILE%"
) else (
  echo requirements.txt not found at "%REQ_FILE%".
)
goto menu

:clean_venv
echo.
echo ===================[ 5. Clean venv ]===================
if exist "%VENV_DIR%" (
  echo Deleting "%VENV_DIR%" ...
  rem Use rmdir /s /q for directories; if it fails, fallback to PowerShell
  rmdir /s /q "%VENV_DIR%" >nul 2>&1
  if exist "%VENV_DIR%" (
    powershell -NoProfile -Command "Remove-Item -Recurse -Force '%VENV_DIR%'"
  )
  if exist "%VENV_DIR%" (
    echo [ERROR] Failed to delete venv. Close any processes using it.
  ) else (
    echo venv deleted.
  )
) else (
  echo venv not found.
)
goto menu

:end
echo Bye.
exit /b 0
