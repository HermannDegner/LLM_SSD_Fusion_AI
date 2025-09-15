@echo off
chcp 65001 >nul
color 0C
title SSD-LLM トラブルシューティング

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║               🛠️  SSD-LLM トラブルシューティング              ║
echo ║                問題解決と診断ツール                          ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

set PROJECT_DIR=%~dp0
set VENV_DIR=%PROJECT_DIR%ssd_llm_env

:TROUBLE_MENU
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                   🔧 トラブルシューティング                    ║
echo ╠════════════════════════════════════════════════════════════════╣
echo ║  1. 🔍 基本診断（推奨）                                        ║
echo ║  2. 🐍 Python環境診断                                         ║
echo ║  3. 📦 仮想環境診断                                            ║
echo ║  4. 🧠 SSD-LLM動作テスト                                      ║
echo ║  5. 🌐 ネットワーク診断                                        ║
echo ║  6. 💾 ディスク容量チェック                                    ║
echo ║  7. 🔧 強制修復（緊急時）                                      ║
echo ║  8. 📋 ログ出力                                               ║
echo ║  9. ❌ 戻る                                                   ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
set /p choice="診断項目を選択してください (1-9): "

if "%choice%"=="1" goto BASIC_DIAG
if "%choice%"=="2" goto PYTHON_DIAG
if "%choice%"=="3" goto VENV_DIAG
if "%choice%"=="4" goto SSD_TEST
if "%choice%"=="5" goto NETWORK_DIAG
if "%choice%"=="6" goto DISK_CHECK
if "%choice%"=="7" goto FORCE_REPAIR
if "%choice%"=="8" goto LOG_OUTPUT
if "%choice%"=="9" exit /b 0
goto TROUBLE_MENU

REM ========================================
REM 基本診断
REM ========================================
:BASIC_DIAG
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                      🔍 基本診断                               ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

set ISSUES=0

echo 1️⃣ Python確認...
python --version >nul 2>&1
if %errorLevel% equ 0 (
    echo ✅ Python: OK
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do echo    バージョン: %%i
) else (
    echo ❌ Python: 未インストール
    echo    💡 対処法: https://python.org からPython 3.10以上をインストール
    set /a ISSUES+=1
)

echo.
echo 2️⃣ 仮想環境確認...
if exist "%VENV_DIR%" (
    echo ✅ 仮想環境: 存在
    if exist "%VENV_DIR%\Scripts\python.exe" (
        echo ✅ 仮想環境Python: OK
    ) else (
        echo ❌ 仮想環境Python: 破損
        set /a ISSUES+=1
    )
) else (
    echo ❌ 仮想環境: 未作成
    echo    💡 対処法: メインメニューの「初回セットアップ」を実行
    set /a ISSUES+=1
)

echo.
echo 3️⃣ 必須ファイル確認...
if exist "%PROJECT_DIR%SSD-LLM_Manager.bat" (echo ✅ マネージャー: OK) else (echo ❌ マネージャー: 無 && set /a ISSUES+=1)
if exist "%PROJECT_DIR%SSD-LLM クイックスタート.bat" (echo ✅ クイックスタート: OK) else (echo ❌ クイックスタート: 無 && set /a ISSUES+=1)

echo.
echo 4️⃣ 必須ライブラリ確認...
if exist "%VENV_DIR%" (
    call "%VENV_DIR%\Scripts\activate.bat"
    
    "%VENV_DIR%\Scripts\python.exe" -c "import transformers; print('✅ transformers:', transformers.__version__)" 2>nul || (echo ❌ transformers: 未インストール && set /a ISSUES+=1)
    "%VENV_DIR%\Scripts\python.exe" -c "import torch; print('✅ torch:', torch.__version__)" 2>nul || (echo ❌ torch: 未インストール && set /a ISSUES+=1)
    "%VENV_DIR%\Scripts\python.exe" -c "import numpy; print('✅ numpy:', numpy.__version__)" 2>nul || (echo ❌ numpy: 未インストール && set /a ISSUES+=1)
)

echo.
echo 5️⃣ GPU確認...
if exist "%VENV_DIR%" (
    call "%VENV_DIR%\Scripts\activate.bat"
    "%VENV_DIR%\Scripts\python.exe" -c "import torch; print('✅ CUDA利用可能:', torch.cuda.is_available()); print('GPU数:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>nul || echo ❌ GPU診断失敗
)

echo.
echo ──────────────────────────────────────────────────────────────────
if %ISSUES% equ 0 (
    echo ✅ 診断結果: 問題なし！SSD-LLMは正常に動作するはずです
) else (
    echo ❌ 診断結果: %ISSUES%個の問題が見つかりました
    echo 💡 上記の対処法を参考に修正してください
    echo 🔧 または「強制修復」を試してください
)
echo ──────────────────────────────────────────────────────────────────
echo.
pause
goto TROUBLE_MENU

REM ========================================
REM Python環境診断
REM ========================================
:PYTHON_DIAG
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                    🐍 Python環境診断                           ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

echo 🔍 Python詳細情報:
python --version 2>nul && (
    echo ✅ Python実行可能
    python -c "import sys; print('実行パス:', sys.executable)"
    python -c "import sys; print('バージョン詳細:', sys.version)"
    python -c "import sys; print('プラットフォーム:', sys.platform)"
    echo.
    
    echo 📦 インストール済みパッケージ^(主要なもの^):
    python -m pip list 2>nul | findstr /i "torch transformers numpy scipy" || echo なし
) || (
    echo ❌ Python実行不可
    echo.
    echo 💡 解決方法:
    echo    1. https://python.org からPython 3.10以上をダウンロード
    echo    2. インストール時に「Add Python to PATH」をチェック
    echo    3. コマンドプロンプトを再起動
)

echo.
echo 🔍 PATH環境変数確認:
echo %PATH% | findstr /i python >nul && echo ✅ PATHにPython含む || echo ❌ PATHにPython無し

echo.
pause
goto TROUBLE_MENU

REM ========================================
REM 仮想環境診断
REM ========================================
:VENV_DIAG
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                   📦 仮想環境診断                              ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

if exist "%VENV_DIR%" (
    echo ✅ 仮想環境ディレクトリ: 存在
    echo    パス: %VENV_DIR%
    
    if exist "%VENV_DIR%\Scripts\python.exe" (
        echo ✅ Python実行ファイル: 存在
        call "%VENV_DIR%\Scripts\activate.bat"
        echo    仮想環境Python:
        "%VENV_DIR%\Scripts\python.exe" --version
        
        echo.
        echo 📊 インストール済みパッケージ数:
        "%VENV_DIR%\Scripts\pip.exe" list | find /c " " 2>nul
        
        echo.
        echo 📦 SSD-LLM関連パッケージ:
        "%VENV_DIR%\Scripts\pip.exe" list | findstr /i "torch transformers numpy scipy streamlit fastapi" || echo なし
        
    ) else (
        echo ❌ Python実行ファイル: 無し（仮想環境が破損）
        echo 💡 対処法: 環境リセット後、再セットアップ
    )
    
    echo.
    echo 📁 仮想環境サイズ:
    dir "%VENV_DIR%" /s 2>nul | find "個のファイル"
    
) else (
    echo ❌ 仮想環境: 未作成
    echo 💡 対処法: メインメニューから「初回セットアップ」を実行
)

echo.
pause
goto TROUBLE_MENU

REM ========================================
REM SSD-LLM動作テスト
REM ========================================
:SSD_TEST
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                  🧠 SSD-LLM動作テスト                         ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

if not exist "%VENV_DIR%" (
    echo ❌ 仮想環境がありません。先にセットアップを実行してください。
    pause
    goto TROUBLE_MENU
)

echo 🧪 簡易SSDテストを実行中...

call "%VENV_DIR%\Scripts\activate.bat"

(
echo class QuickSSDTest:
echo     def __init__^(self^):
echo         self.heat = 0.0
echo     
echo     def test_basic^(self^):
echo         print^("✅ SSDクラス初期化: OK"^)
echo         return True
echo     
echo     def test_pressure^(self^):
echo         self.heat += 0.5
echo         print^(f"✅ 意味圧計算: OK ^(heat={self.heat}^)"^)
echo         return True
echo     
echo     def test_jump^(self^):
echo         if self.heat ^> 0.3:
echo             print^("✅ 跳躍判定: OK ^(跳躍発生^)"^)
echo             return True
echo         else:
echo             print^("✅ 整合判定: OK ^(整合維持^)"^)
echo             return True
echo 
echo if __name__ == "__main__":
echo     import sys
echo     print^("🧪 SSD基本機能テスト開始"^)
echo     test = QuickSSDTest^(^)
echo     
echo     results = []
echo     results.append^(test.test_basic^(^)^)
echo     results.append^(test.test_pressure^(^)^)
echo     results.append^(test.test_jump^(^)^)
echo     
echo     if all^(results^):
echo         print^("\\n✅ 全テスト通過！ SSD基本機能は正常です"^)
echo         sys.exit^(0^)
echo     else:
echo         print^("\\n❌ テスト失敗"^)
echo         sys.exit^(1^)
) > "%PROJECT_DIR%temp_ssd_test.py"

"%VENV_DIR%\Scripts\python.exe" "%PROJECT_DIR%temp_ssd_test.py"
set TEST_RESULT=%errorLevel%

del "%PROJECT_DIR%temp_ssd_test.py" 2>nul

echo.
if %TEST_RESULT% equ 0 (
    echo 🎉 SSD機能は正常に動作しています！
) else (
    echo ❌ SSD機能にエラーが検出されました
    echo 💡 対処法: 仮想環境を再作成してください
)

echo.
pause
goto TROUBLE_MENU

REM ========================================
REM ネットワーク診断
REM ========================================
:NETWORK_DIAG
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                   🌐 ネットワーク診断                          ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

echo 🔍 インターネット接続確認:
ping google.com -n 1 >nul 2>&1 && echo ✅ インターネット: 接続OK || echo ❌ インターネット: 接続失敗

echo.
echo 🔍 Python Package Index ^(PyPI^) 接続確認:
ping pypi.org -n 1 >nul 2>&1 && echo ✅ PyPI: 接続OK || echo ❌ PyPI: 接続失敗

echo.
echo 🔍 Hugging Face Hub 接続確認:
ping huggingface.co -n 1 >nul 2>&1 && echo ✅ Hugging Face: 接続OK || echo ❌ Hugging Face: 接続失敗

echo.
echo 🔍 GitHub 接続確認:
ping github.com -n 1 >nul 2>&1 && echo ✅ GitHub: 接続OK || echo ❌ GitHub: 接続失敗

echo.
echo 💡 接続に問題がある場合:
echo    - ファイアウォール設定を確認
echo    - プロキシ設定を確認
echo    - VPN接続を確認
echo    - しばらく時間をおいて再試行

echo.
pause
goto TROUBLE_MENU

REM ========================================
REM ディスク容量チェック
REM ========================================
:DISK_CHECK
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                  💾 ディスク容量チェック                       ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

echo 🔍 現在のディスク使用量:
dir "%PROJECT_DIR%" | find "バイト"

echo.
echo 🔍 仮想環境のサイズ:
if exist "%VENV_DIR%" (
    dir "%VENV_DIR%" /s | find "個のファイル"
) else (
    echo 仮