@echo off
title SSD-LLM Minimal Setup

echo Starting SSD-LLM setup...

set VENV_DIR=%~dp0ssd_env

if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

echo Activating environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo Installing basic libraries...
"%VENV_DIR%\Scripts\pip.exe" install torch transformers --quiet

echo Creating demo file...
echo import random > demo.py
echo. >> demo.py
echo class SSD: >> demo.py
echo     def __init__(self): >> demo.py
echo         self.heat = 0 >> demo.py
echo     def chat(self, msg): >> demo.py
echo         self.heat += len(msg) * 0.01 >> demo.py
echo         if self.heat ^> 0.5: >> demo.py
echo             mode = "LEAP" >> demo.py
echo             self.heat *= 0.7 >> demo.py
echo         else: >> demo.py
echo             mode = "ALIGN" >> demo.py
echo         return f"[{mode}] I understand. Heat: {self.heat:.2f}" >> demo.py
echo. >> demo.py
echo ssd = SSD() >> demo.py
echo while True: >> demo.py
echo     msg = input("You: ") >> demo.py
echo     if msg == "exit": break >> demo.py
echo     print(ssd.chat(msg)) >> demo.py

echo Starting demo...
"%VENV_DIR%\Scripts\python.exe" demo.py

pause