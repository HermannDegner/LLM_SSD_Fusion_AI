@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
chcp 65001 >NUL
title SSD-LLM Setup (Win11 / RTX 4090) - v2

echo.
echo ====== PRECHECKS ======

:: PowerShell (ZIP展開等に使用)
where powershell >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [!] PowerShell not found. Please enable or install it.
  pause & exit /b 1
)

:: Git チェック（括弧ブロックを使わず直列で）
where git >NUL 2>&1
if %ERRORLEVEL% EQU 0 (
  for /f "delims=" %%i in ('git --version') do set "GIT_VER=%%i"
  echo [OK] !GIT_VER!
  set "HAVE_GIT=1"
) else (
  echo [!] Git not found.
  set "HAVE_GIT=0"
)

:: 変数
set "REPO_URL=https://github.com/HermannDegner/LLM_SSD_Fusion_AI"
set "REPO_DIR=LLM_SSD_Fusion_AI"
set "VENV_DIR=.venv_ssd"
set "PYTHON_EXE=python"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"

echo.
echo ====== FETCH REPO ======
if "%HAVE_GIT%"=="1" (
  if not exist "%REPO_DIR%" (
    echo [>] git clone ...
    git clone "%REPO_URL%"
    if %ERRORLEVEL% NEQ 0 echo [x] clone failed.& pause & exit /b 1
  ) else (
    echo [=] repo exists. git pull ...
    pushd "%REPO_DIR%" && git pull && popd
  )
) else (
  :: Gitなし → ZIP 取得
  set "REPO_ZIP_URL=https://github.com/HermannDegner/LLM_SSD_Fusion_AI/archive/refs/heads/main.zip"
  set "ZIP_PATH=%TEMP%\LLM_SSD_Fusion_AI.zip"
  set "EXTRACT_DIR=%TEMP%\LLM_SSD_Fusion_AI-main"

  echo [>] download ZIP ...
  powershell -NoProfile -Command "Invoke-WebRequest -Uri '%REPO_ZIP_URL%' -OutFile '%ZIP_PATH%'" || (echo [x] download failed.& pause & exit /b 1)

  echo [>] expand ZIP ...
  if exist "%EXTRACT_DIR%" rmdir /S /Q "%EXTRACT_DIR%"
  powershell -NoProfile -Command "Expand-Archive -Path '%ZIP_PATH%' -DestinationPath '%TEMP%' -Force" || (echo [x] expand failed.& pause & exit /b 1)

  if exist "%REPO_DIR%" rmdir /S /Q "%REPO_DIR%"
  ren "%EXTRACT_DIR%" "%REPO_DIR%" || (echo [x] rename failed.& pause & exit /b 1)
)

echo.
echo ====== PYTHON / VENV ======
where "%PYTHON_EXE%" >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [!] Python 3.10-3.11 required. Install and add to PATH.
  echo     https://www.python.org/downloads/windows/
  pause & exit /b 1
)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
  echo [>] create venv ...
  "%PYTHON_EXE%" -m venv "%VENV_DIR%" || (echo [x] venv failed.& pause & exit /b 1)
) else (
  echo [=] venv exists.
)

call "%VENV_DIR%\Scripts\activate.bat" || (echo [x] activate failed.& pause & exit /b 1)

echo.
echo ====== PIP / PACKAGES ======
python -m pip install --upgrade pip setuptools wheel

echo [>] PyTorch (CUDA 12.1) ...
pip install --index-url "%TORCH_INDEX%" torch torchvision torchaudio
if %ERRORLEVEL% NEQ 0 (
  echo [x] PyTorch install failed. Check network/proxy or CUDA build.
  pause & exit /b 1
)

echo [>] Core packages ...
pip install ^
  transformers accelerate sentencepiece ^
  numpy scipy pandas networkx ^
  fastapi uvicorn websockets ^
  streamlit plotly ^
  peft datasets huggingface_hub optimum

set /p INSTALL_BNB=[?] Install bitsandbytes for 4/8-bit quantization (y/N): 
if /I "!INSTALL_BNB!"=="Y" (
  pip install bitsandbytes || (
    echo [!] bitsandbytes optional install failed ^(safe to skip^).
  )
)

echo.
echo ====== QUICK GPU TEST ======
set "TEST_PY=%TEMP%\ssd_llm_quick_test.py"
> "!TEST_PY!" (
  echo import torch, platform
  echo print("=== SSD-LLM Quick Test ===")
  echo print("Python:", platform.python_version())
  echo print("Torch:", torch.__version__)
  echo print("CUDA available:", torch.cuda.is_available())
  echo print("CUDA device count:", torch.cuda.device_count())
  echo "GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print("No CUDA")
)
python "!TEST_PY!"

echo.
echo ====== LAUNCHER ======
> "run_ssd_env.bat" (
  echo @echo off
  echo chcp 65001 ^>NUL
  echo call "%VENV_DIR%\Scripts\activate.bat"
  echo echo [OK] venv ready. run python / streamlit / uvicorn here.
  echo cmd /k
)

echo.
echo ====== DONE ======
echo [OK] venv: %CD%\%VENV_DIR%
echo [OK] launcher: %CD%\run_ssd_env.bat
echo next:
echo   cd %REPO_DIR%
echo   ..\run_ssd_env.bat
echo   ^(ex^) streamlit run web_dashboard.py
echo   ^(ex^) python examples\ssd_llm_demo.py
echo.
pause
endlocal
