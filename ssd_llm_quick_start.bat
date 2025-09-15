@echo off
chcp 65001 >nul
color 0B
title SSD-LLM クイックスタート

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    🚀 SSD-LLM クイックスタート                ║
echo ║              今すぐ体験！ワンクリックでAI会話                  ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM 環境チェック
set PROJECT_DIR=%~dp0
set VENV_DIR=%PROJECT_DIR%ssd_llm_env

echo 🔍 環境チェック中...

REM Python存在チェック
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Pythonがインストールされていません
    echo 📥 以下からPython 3.10以上をダウンロードしてください:
    echo    https://www.python.org/downloads/
    echo.
    echo 📋 インストール時の注意:
    echo    ☑️ "Add Python to PATH" にチェックを入れてください
    echo    ☑️ "Install for all users" を選択してください
    pause
    exit /b 1
)

echo ✅ Python確認完了

REM 仮想環境チェック
if not exist "%VENV_DIR%" (
    echo 🔧 初回セットアップが必要です
    echo 📦 仮想環境を作成中... ^(数分かかります^)
    python -m venv "%VENV_DIR%"
    
    if %errorLevel% neq 0 (
        echo ❌ 仮想環境作成に失敗しました
        pause
        exit /b 1
    )
    
    echo ✅ 仮想環境作成完了
    
    REM 最小限のライブラリインストール
    echo 📚 必要なライブラリをインストール中...
    call "%VENV_DIR%\Scripts\activate.bat"
    
    echo    - transformers インストール中...
    "%VENV_DIR%\Scripts\pip.exe" install transformers torch --quiet
    
    if %errorLevel% neq 0 (
        echo ❌ ライブラリインストールに失敗しました
        echo 🌐 ネットワーク接続を確認してください
        pause
        exit /b 1
    )
    
    echo ✅ ライブラリインストール完了
)

REM 超軽量デモファイル作成
if not exist "%PROJECT_DIR%quick_demo.py" (
echo 🛠️  デモファイルを作成中...
(
echo import random
echo import time
echo.
echo class UltraLightSSD:
echo     def __init__^(self^):
echo         self.heat = 0.0
echo         self.responses = {
echo             'greeting': ['こんにちは！', 'お疲れ様です！', 'よろしくお願いします！'],
echo             'creative': ['面白いアイデアですね！', '新しい視点で考えてみましょう', '創造的な発想が必要ですね'],
echo             'analytical': ['詳しく分析してみましょう', '論理的に整理すると...', 'データを基に考えると'],
echo             'empathetic': ['お気持ちわかります', 'それは大変でしたね', '一緒に考えましょう'],
echo             'default': ['なるほど', 'そうですね', 'もう少し詳しく教えてください']
echo         }
echo         print^("🧠 Ultra-Light SSD初期化完了！"^)
echo.
echo     def analyze^(self, text^):
echo         pressure = len^(text^) / 50.0
echo         
echo         if any^(w in text for w in ['こんにちは', 'おはよう', 'こんばんは']^):
echo             return 'greeting', pressure
echo         elif any^(w in text for w in ['創造', 'アイデア', '新しい', '斬新']^):
echo             return 'creative', pressure + 0.3
echo         elif any^(w in text for w in ['分析', '理由', 'なぜ', '説明']^):
echo             return 'analytical', pressure + 0.2
echo         elif any^(w in text for w in ['悲しい', '困っ', '心配', '不安']^):
echo             return 'empathetic', pressure + 0.4
echo         else:
echo             return 'default', pressure
echo.
echo     def respond^(self, user_input^):
echo         category, pressure = self.analyze^(user_input^)
echo         self.heat += pressure * 0.3
echo         
echo         # 跳躍判定
echo         if self.heat ^> 0.8:
echo             mode = "🚀 LEAP"
echo             # より創造的な応答
echo             response = random.choice^(self.responses['creative']^) + " " + random.choice^(self.responses['default']^)
echo             self.heat *= 0.6  # 放熱
echo         else:
echo             mode = "⚖️ ALIGN"
echo             response = random.choice^(self.responses.get^(category, self.responses['default']^)^)
echo         
echo         return response, mode, self.heat
echo.
echo def main^(^):
echo     print^("╔══════════════════════════════════════════════════════════════╗"^)
echo     print^("║              🚀 Ultra-Light SSD Demo                        ║"^)
echo     print^("║         構造主観力学 超軽量デモンストレーション                 ║"^)
echo     print^("╚══════════════════════════════════════════════════════════════╝"^)
echo     print^(^)
echo     
echo     ssd = UltraLightSSD^(^)
echo     
echo     print^("💡 使い方:"^)
echo     print^("   - 普通に話しかけてください"^)
echo     print^("   - '創造'や'アイデア'を含むと創造モードに"^)
echo     print^("   - '分析'や'なぜ'を含むと分析モードに"^)
echo     print^("   - 感情的な言葉を使うと共感モードに"^)
echo     print^("   - Heat Level が高いと跳躍^(LEAP^)しやすくなります"^)
echo     print^("   - 'exit'で終了"^)
echo     print^("─" * 60^)
echo     
echo     while True:
echo         try:
echo             user_input = input^("あなた: "^).strip^(^)
echo             
echo             if user_input.lower^(^) in ['exit', 'quit', '終了']:
echo                 print^("👋 ありがとうございました！"^)
echo                 break
echo             
echo             if not user_input:
echo                 continue
echo             
echo             response, mode, heat = ssd.respond^(user_input^)
echo             print^(f"AI [{mode}]: {response}"^)
echo             print^(f"   📊 Heat Level: {heat:.2f}"^)
echo             print^("─" * 60^)
echo             
echo         except KeyboardInterrupt:
echo             print^("\\n👋 終了します"^)
echo             break
echo         except Exception as e:
echo             print^(f"❌ エラー: {e}"^)
echo.
echo if __name__ == "__main__":
echo     main^(^)
) > "%PROJECT_DIR%quick_demo.py"
echo ✅ デモファイル作成完了
)

echo.
echo 🎉 準備完了！SSD-LLMデモを開始します...
echo.

REM 仮想環境アクティベート
call "%VENV_DIR%\Scripts\activate.bat"

REM デモ実行
"%VENV_DIR%\Scripts\python.exe" "%PROJECT_DIR%quick_demo.py"

echo.
echo 👋 デモ終了
echo 📖 より詳細な機能を試すには 'SSD-LLM_Manager.bat' を実行してください
echo.
pause