@echo off
chcp 65001 >nul
color 0A
title SSD-LLM 自動セットアップ・実行システム

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              SSD-LLM 自動セットアップシステム                 ║
echo ║          Structural Subjectivity Dynamics + LLM             ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM 管理者権限チェック
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ 管理者権限で実行中
) else (
    echo ⚠️  管理者権限が必要です。右クリック→「管理者として実行」してください。
    pause
    exit /b 1
)

REM 環境変数設定
set PROJECT_DIR=%~dp0
set VENV_DIR=%PROJECT_DIR%ssd_llm_env
set PYTHON_EXE=python
set PIP_EXE=pip

echo 📂 プロジェクトディレクトリ: %PROJECT_DIR%
echo.

REM メニュー表示
:MENU
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                    SSD-LLM メインメニュー                      ║
echo ╠════════════════════════════════════════════════════════════════╣
echo ║  1. 🔧 初回セットアップ（環境構築）                             ║
echo ║  2. 🚀 SSD-LLM デモ実行                                        ║
echo ║  3. 🌐 Webダッシュボード起動                                   ║
echo ║  4. 🧪 DLLテスト実行                                           ║
echo ║  5. 📊 システム情報表示                                        ║
echo ║  6. 🔄 環境リセット                                            ║
echo ║  7. ❌ 終了                                                    ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
set /p choice="選択してください (1-7): "

if "%choice%"=="1" goto SETUP
if "%choice%"=="2" goto RUN_DEMO
if "%choice%"=="3" goto WEB_DASHBOARD
if "%choice%"=="4" goto DLL_TEST
if "%choice%"=="5" goto SYSTEM_INFO
if "%choice%"=="6" goto RESET_ENV
if "%choice%"=="7" goto EXIT
goto MENU

REM ========================================
REM 初回セットアップ
REM ========================================
:SETUP
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                      🔧 初回セットアップ                       ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Python存在チェック
echo 🔍 Pythonインストールチェック...
%PYTHON_EXE% --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ Pythonが見つかりません。
    echo 📥 Python 3.10以上をダウンロードしてインストールしてください:
    echo    https://www.python.org/downloads/
    pause
    goto MENU
)

for /f "tokens=2" %%i in ('%PYTHON_EXE% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% が見つかりました

REM 仮想環境作成
echo.
echo 📦 仮想環境を作成中...
if exist "%VENV_DIR%" (
    echo ⚠️  既存の仮想環境を削除中...
    rmdir /s /q "%VENV_DIR%"
)

%PYTHON_EXE% -m venv "%VENV_DIR%"
if %errorLevel% neq 0 (
    echo ❌ 仮想環境の作成に失敗しました
    pause
    goto MENU
)

echo ✅ 仮想環境を作成しました

REM 仮想環境アクティベート
call "%VENV_DIR%\Scripts\activate.bat"

REM Pythonパス更新
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe
set PIP_EXE=%VENV_DIR%\Scripts\pip.exe

REM pip更新
echo.
echo 🔄 pipを最新版に更新中...
"%PIP_EXE%" install --upgrade pip

REM GPU対応チェック
echo.
echo 🔍 GPU対応チェック中...
"%PYTHON_EXE%" -c "import torch; print('✅ PyTorch:', torch.__version__); print('🔥 CUDA利用可能:', torch.cuda.is_available())" 2>nul
if %errorLevel% neq 0 (
    echo 📥 PyTorchをインストール中（CPU版）...
    "%PIP_EXE%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
    echo 📥 PyTorchをインストール中（GPU版）...
    "%PIP_EXE%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

REM 必須ライブラリインストール
echo.
echo 📚 必須ライブラリをインストール中...
"%PIP_EXE%" install transformers accelerate
"%PIP_EXE%" install numpy scipy networkx scikit-learn
"%PIP_EXE%" install fastapi uvicorn websockets
"%PIP_EXE%" install streamlit plotly
"%PIP_EXE%" install datasets peft
"%PIP_EXE%" install bitsandbytes optimum

REM プロジェクトファイル作成
echo.
echo 📝 プロジェクトファイルを作成中...

REM requirements.txt作成
(
echo torch^>=2.0.0
echo transformers^>=4.35.0
echo accelerate^>=0.20.0
echo numpy^>=1.24.0
echo scipy^>=1.10.0
echo networkx^>=3.0
echo scikit-learn^>=1.3.0
echo fastapi^>=0.100.0
echo uvicorn^>=0.20.0
echo websockets^>=11.0
echo streamlit^>=1.28.0
echo plotly^>=5.15.0
echo datasets^>=2.14.0
echo peft^>=0.6.0
echo bitsandbytes^>=0.41.0
echo optimum^>=1.14.0
) > "%PROJECT_DIR%requirements.txt"

REM 軽量デモ版作成
(
echo import os
echo import sys
echo import time
echo import random
echo from typing import Dict, Any, Optional
echo.
echo try:
echo     from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
echo     import torch
echo     import numpy as np
echo except ImportError as e:
echo     print^(f"❌ 必要なライブラリが不足しています: {e}"^)
echo     print^("セットアップを再実行してください"^)
echo     sys.exit^(1^)
echo.
echo class SimpleSSDLLM:
echo     def __init__^(self^):
echo         print^("🔧 SSD-LLM初期化中..."^)
echo         self.heat_level = 0.0
echo         self.memory = {}
echo         self.conversation_count = 0
echo         
echo         # 軽量モデルで初期化
echo         try:
echo             self.tokenizer = AutoTokenizer.from_pretrained^("microsoft/DialoGPT-medium"^)
echo             self.model = AutoModelForCausalLM.from_pretrained^("microsoft/DialoGPT-medium"^)
echo             if self.tokenizer.pad_token is None:
echo                 self.tokenizer.pad_token = self.tokenizer.eos_token
echo             print^("✅ モデル読み込み完了"^)
echo         except Exception as e:
echo             print^(f"⚠️  軽量モデルで代替: {e}"^)
echo             self.model = pipeline^("text-generation", model="distilgpt2"^)
echo             self.tokenizer = None
echo     
echo     def analyze_meaning_pressure^(self, user_input: str^) -^> float:
echo         """意味圧の簡易計算"""
echo         pressure = len^(user_input^) / 100.0
echo         
echo         # 感情的要素
echo         emotional_words = ['嬉しい', '悲しい', '怒', '困', '悩', '不安', '心配', '楽しい']
echo         if any^(word in user_input for word in emotional_words^):
echo             pressure += 0.3
echo         
echo         # 質問要素
echo         if any^(q in user_input for q in ["?", "？", "教えて", "どう", "なぜ", "何"]^):
echo             pressure += 0.2
echo         
echo         # 複雑性要素
echo         complex_words = ['複雑', '難しい', '詳しく', '分析', '比較', '検討']
echo         if any^(word in user_input for word in complex_words^):
echo             pressure += 0.4
echo         
echo         # 緊急性要素
echo         urgent_words = ['急', 'すぐ', '至急', '緊急', 'ASAP']
echo         if any^(word in user_input for word in urgent_words^):
echo             pressure += 0.5
echo         
echo         return min^(pressure, 2.0^)
echo     
echo     def generate_response^(self, user_input: str^) -^> Dict[str, Any]:
echo         """SSD統合応答生成"""
echo         start_time = time.time^(^)
echo         
echo         # 意味圧計算
echo         meaning_pressure = self.analyze_meaning_pressure^(user_input^)
echo         self.heat_level += meaning_pressure * 0.5
echo         
echo         # 整合/跳躍判定
echo         jump_threshold = 0.6 + random.uniform^(-0.1, 0.1^)
echo         did_jump = self.heat_level ^> jump_threshold
echo         
echo         if did_jump:
echo             # 跳躍モード: 創造的応答
echo             mode = "leap"
echo             temperature = min^(0.9, 0.4 + self.heat_level * 0.3^)
echo             self.heat_level *= 0.7  # 放熱
echo         else:
echo             # 整合モード: 安定応答
echo             mode = "alignment"
echo             temperature = 0.3
echo         
echo         # 応答生成
echo         try:
echo             if self.tokenizer:
echo                 # Transformers使用
echo                 inputs = self.tokenizer.encode^(user_input + self.tokenizer.eos_token, return_tensors="pt"^)
echo                 with torch.no_grad^(^):
echo                     outputs = self.model.generate^(
echo                         inputs,
echo                         max_length=inputs.shape[1] + 50,
echo                         temperature=temperature,
echo                         do_sample=True,
echo                         pad_token_id=self.tokenizer.eos_token_id
echo                     ^)
echo                 response = self.tokenizer.decode^(outputs[0][inputs.shape[1]:], skip_special_tokens=True^).strip^(^)
echo             else:
echo                 # Pipeline使用
echo                 result = self.model^(user_input, max_length=50, temperature=temperature^)
echo                 response = result[0]['generated_text'].replace^(user_input, ""^).strip^(^)
echo         except Exception as e:
echo             response = f"申し訳ありません。技術的な問題が発生しました: {str^(e^)[:50]}"
echo         
echo         generation_time = time.time^(^) - start_time
echo         self.conversation_count += 1
echo         
echo         return {
echo             'response': response,
echo             'ssd_metadata': {
echo                 'mode_used': mode,
echo                 'heat_level': self.heat_level,
echo                 'meaning_pressure': meaning_pressure,
echo                 'did_jump': did_jump,
echo                 'temperature_used': temperature,
echo                 'conversation_count': self.conversation_count
echo             },
echo             'generation_time': generation_time
echo         }
echo     
echo     def get_status^(self^) -^> Dict[str, Any]:
echo         """システム状態取得"""
echo         return {
echo             'heat_level': self.heat_level,
echo             'conversation_count': self.conversation_count,
echo             'memory_size': len^(self.memory^),
echo             'system_ready': True
echo         }
echo.
echo def main^(^):
echo     print^("╔══════════════════════════════════════════════════════════════╗"^)
echo     print^("║              SSD-LLM デモンストレーション                    ║"^)
echo     print^("║          Structural Subjectivity Dynamics + LLM             ║"^)
echo     print^("╚══════════════════════════════════════════════════════════════╝"^)
echo     print^(^)
echo     
echo     # システム初期化
echo     ssd_llm = SimpleSSDLLM^(^)
echo     
echo     print^("🎮 デモモード開始！ ^('exit' で終了^)"^)
echo     print^("💡 ヒント: 感情的な言葉や複雑な質問をすると跳躍モードになりやすくなります"^)
echo     print^("─" * 60^)
echo     
echo     while True:
echo         try:
echo             user_input = input^("あなた: "^).strip^(^)
echo             
echo             if user_input.lower^(^) in ['exit', 'quit', '終了', 'やめる']:
echo                 print^("👋 ありがとうございました！"^)
echo                 break
echo             
echo             if not user_input:
echo                 continue
echo             
echo             # 応答生成
echo             result = ssd_llm.generate_response^(user_input^)
echo             meta = result['ssd_metadata']
echo             
echo             # 結果表示
echo             mode_emoji = "🚀" if meta['did_jump'] else "⚖️"
echo             print^(f"AI [{mode_emoji} {meta['mode_used']}]: {result['response']}"^)
echo             
echo             # SSD情報表示
echo             print^(f"   📊 Heat: {meta['heat_level']:.2f} ^| "
echo                   f"Pressure: {meta['meaning_pressure']:.2f} ^| "
echo                   f"Time: {result['generation_time']:.2f}s"^)
echo             print^("─" * 60^)
echo             
echo         except KeyboardInterrupt:
echo             print^("\\n👋 中断されました"^)
echo             break
echo         except Exception as e:
echo             print^(f"❌ エラーが発生しました: {e}"^)
echo             continue
echo.
echo if __name__ == "__main__":
echo     main^(^)
) > "%PROJECT_DIR%ssd_llm_demo.py"

REM Webダッシュボード作成
(
echo import streamlit as st
echo import plotly.graph_objects as go
echo import plotly.express as px
echo import pandas as pd
echo import numpy as np
echo import sys
echo import os
echo.
echo # プロジェクトパスを追加
echo sys.path.append^(os.path.dirname^(os.path.abspath^(__file__^)^)^)
echo.
echo try:
echo     from ssd_llm_demo import SimpleSSDLLM
echo except ImportError:
echo     st.error^("SSD-LLMモジュールの読み込みに失敗しました"^)
echo     st.stop^(^)
echo.
echo st.set_page_config^(
echo     page_title="SSD-LLM Dashboard",
echo     page_icon="🧠",
echo     layout="wide"
echo ^)
echo.
echo st.title^("🧠 SSD-LLM リアルタイムダッシュボード"^)
echo.
echo # セッション状態初期化
echo if 'ssd_llm' not in st.session_state:
echo     with st.spinner^("SSD-LLM初期化中..."^):
echo         st.session_state.ssd_llm = SimpleSSDLLM^(^)
echo         st.session_state.conversation_history = []
echo.
echo # サイドバー: システム情報
echo with st.sidebar:
echo     st.header^("📊 システム状態"^)
echo     status = st.session_state.ssd_llm.get_status^(^)
echo     
echo     st.metric^("未処理圧レベル", f"{status['heat_level']:.2f}"^)
echo     st.metric^("会話回数", status['conversation_count']^)
echo     st.metric^("メモリサイズ", status['memory_size']^)
echo     
echo     if st.button^("🔄 システムリセット"^):
echo         st.session_state.ssd_llm = SimpleSSDLLM^(^)
echo         st.session_state.conversation_history = []
echo         st.success^("リセット完了！"^)
echo.
echo # メインエリア
echo col1, col2 = st.columns^([2, 1]^)
echo.
echo with col1:
echo     st.header^("💬 対話インターフェース"^)
echo     
echo     # チャット履歴表示
echo     chat_container = st.container^(^)
echo     
echo     # 入力フォーム
echo     with st.form^("chat_form"^):
echo         user_input = st.text_input^("メッセージを入力:", placeholder="例: 創造的なアイデアを教えて"^)
echo         submitted = st.form_submit_button^("送信"^)
echo     
echo     if submitted and user_input:
echo         # 応答生成
echo         with st.spinner^("応答生成中..."^):
echo             result = st.session_state.ssd_llm.generate_response^(user_input^)
echo         
echo         # 履歴追加
echo         st.session_state.conversation_history.append^({
echo             'user': user_input,
echo             'ai': result['response'],
echo             'metadata': result['ssd_metadata']
echo         }^)
echo     
echo     # チャット履歴表示
echo     with chat_container:
echo         for i, conv in enumerate^(reversed^(st.session_state.conversation_history[-10:]^)^):
echo             with st.chat_message^("user"^):
echo                 st.write^(conv['user']^)
echo             
echo             mode_emoji = "🚀" if conv['metadata']['did_jump'] else "⚖️"
echo             with st.chat_message^("assistant"^):
echo                 st.write^(f"{mode_emoji} {conv['ai']}"^)
echo                 st.caption^(f"Mode: {conv['metadata']['mode_used']} ^| "
echo                           f"Heat: {conv['metadata']['heat_level']:.2f}"^)
echo.
echo with col2:
echo     st.header^("📈 SSD分析"^)
echo     
echo     if st.session_state.conversation_history:
echo         # Heat Level推移
echo         heat_data = [conv['metadata']['heat_level'] for conv in st.session_state.conversation_history]
echo         fig_heat = go.Figure^(^)
echo         fig_heat.add_trace^(go.Scatter^(
echo             y=heat_data,
echo             mode='lines+markers',
echo             name='Heat Level',
echo             line=dict^(color='red', width=2^)
echo         ^)^)
echo         fig_heat.update_layout^(
echo             title="未処理圧レベル推移",
echo             yaxis_title="Heat Level",
echo             height=300
echo         ^)
echo         st.plotly_chart^(fig_heat, use_container_width=True^)
echo         
echo         # Mode分布
echo         modes = [conv['metadata']['mode_used'] for conv in st.session_state.conversation_history]
echo         mode_counts = pd.Series^(modes^).value_counts^(^)
echo         
echo         fig_pie = px.pie^(
echo             values=mode_counts.values,
echo             names=mode_counts.index,
echo             title="応答モード分布"
echo         ^)
echo         st.plotly_chart^(fig_pie, use_container_width=True^)
echo         
echo         # 最新の詳細情報
echo         if st.session_state.conversation_history:
echo             latest = st.session_state.conversation_history[-1]['metadata']
echo             st.subheader^("📋 最新応答の詳細"^)
echo             st.json^(latest^)
echo.
echo # フッター
echo st.markdown^("---"^)
echo st.markdown^("🏗️ **SSD-LLM** - Structural Subjectivity Dynamics + Large Language Model"^)
) > "%PROJECT_DIR%web_dashboard.py"

REM DLLテスト用ファイル作成
(
echo import ctypes
echo import os
echo.
echo def test_dll_basic^(^):
echo     """基本的なDLL動作テスト"""
echo     dll_path = "./ssd_align_leap.dll"
echo     
echo     if not os.path.exists^(dll_path^):
echo         print^("❌ DLLファイルが見つかりません:", dll_path^)
echo         print^("🔧 C++ソースからビルドが必要です"^)
echo         return False
echo     
echo     try:
echo         dll = ctypes.CDLL^(dll_path^)
echo         print^("✅ DLL読み込み成功"^)
echo         return True
echo     except Exception as e:
echo         print^(f"❌ DLL読み込み失敗: {e}"^)
echo         return False
echo.
echo def test_mock_ssd^(^):
echo     """モックSSDテスト"""
echo     print^("🧪 モックSSDテスト開始"^)
echo     
echo     class MockSSD:
echo         def __init__^(self^):
echo             self.heat = 0.0
echo             self.node = 0
echo         
echo         def step^(self, pressure^):
echo             self.heat += pressure * 0.5
echo             if self.heat ^> 1.0:
echo                 self.node = ^(self.node + 1^) % 8
echo                 self.heat *= 0.7
echo                 return True  # jumped
echo             return False  # aligned
echo     
echo     ssd = MockSSD^(^)
echo     
echo     test_pressures = [0.1, 0.3, 0.8, 1.2, 0.5]
echo     for i, p in enumerate^(test_pressures^):
echo         jumped = ssd.step^(p^)
echo         status = "JUMP" if jumped else "ALIGN"
echo         print^(f"Step {i+1}: pressure={p:.1f}, heat={ssd.heat:.2f}, node={ssd.node}, status={status}"^)
echo     
echo     print^("✅ モックSSDテスト完了"^)
echo     return True
echo.
echo if __name__ == "__main__":
echo     print^("🔍 SSD-LLM DLLテスト"^)
echo     print^("=" * 40^)
echo     
echo     dll_ok = test_dll_basic^(^)
echo     print^(^)
echo     mock_ok = test_mock_ssd^(^)
echo     
echo     print^("\\n📊 テスト結果:"^)
echo     print^(f"DLL: {'✅' if dll_ok else '❌'}"^)
echo     print^(f"Mock SSD: {'✅' if mock_ok else '❌'}"^)
) > "%PROJECT_DIR%dll_test.py"

echo.
echo ✅ セットアップ完了！
echo.
echo 📁 作成されたファイル:
echo    - requirements.txt
echo    - ssd_llm_demo.py
echo    - web_dashboard.py  
echo    - dll_test.py
echo.
echo 🎉 これでSSD-LLMを使用する準備が整いました！
echo    メニューから各機能を試してみてください。
echo.
pause
goto MENU

REM ========================================
REM デモ実行
REM ========================================
:RUN_DEMO
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                     🚀 SSD-LLM デモ実行                        ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

if not exist "%VENV_DIR%" (
    echo ❌ 仮想環境が見つかりません。先にセットアップを実行してください。
    pause
    goto MENU
)

echo 🔄 仮想環境をアクティベート中...
call "%VENV_DIR%\Scripts\activate.bat"

echo 🚀 SSD-LLMデモを起動中...
"%VENV_DIR%\Scripts\python.exe" "%PROJECT_DIR%ssd_llm_demo.py"

echo.
echo デモが終了しました。
pause
goto MENU

REM ========================================
REM Webダッシュボード
REM ========================================
:WEB_DASHBOARD
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                   🌐 Webダッシュボード起動                     ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

if not exist "%VENV_DIR%" (
    echo ❌ 仮想環境が見つかりません。先にセットアップを実行してください。
    pause
    goto MENU
)

echo 🔄 仮想環境をアクティベート中...
call "%VENV_DIR%\Scripts\activate.bat"

echo 🌐 Webダッシュボードを起動中...
echo 📱 ブラウザでhttp://localhost:8501にアクセスしてください
echo ⏹️  終了するにはCtrl+Cを押してください
echo.

"%VENV_DIR%\Scripts\streamlit.exe" run "%PROJECT_DIR%web_dashboard.py"

pause
goto MENU

REM ========================================
REM DLLテスト
REM ========================================
:DLL_TEST
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                      🧪 DLLテスト実行                          ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

if not exist "%VENV_DIR%" (
    echo ❌ 仮想環境が見つかりません。先にセットアップを実行してください。
    pause
    goto MENU
)

echo 🔄 仮想環境をアクティベート中...
call "%VENV_DIR%\Scripts\activate.bat"

echo 🧪 DLLテストを実行中...
"%VENV_DIR%\Scripts\python.exe" "%PROJECT_DIR%dll_test.py"

echo.
pause
goto MENU

REM ========================================
REM システム情報表示
REM ========================================
:SYSTEM_INFO
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                     📊 システム情報表示                        ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

echo 💻 システム情報:
echo    OS: %OS%
echo    アーキテクチャ: %PROCESSOR_ARCHITECTURE%
echo    プロセッサ: %PROCESSOR_IDENTIFIER%
echo.

echo 📂 プロジェクト情報:
echo    プロジェクトディレクトリ: %PROJECT_DIR%
echo    仮想環境: %VENV_DIR%
echo.

if exist "%VENV_DIR%" (
    echo ✅ 仮想環境: 存在
    call "%VENV_DIR%\Scripts\activate.bat"
    echo    Python: 
    "%VENV_DIR%\Scripts\python.exe" --version
    echo    pip: 
    "%VENV_DIR%\Scripts\pip.exe" --version
) else (
    echo ❌ 仮想環境: 未作成
)

echo.
echo 📁 プロジェクトファイル:
if exist "%PROJECT_DIR%ssd_llm_demo.py" (echo    ✅ ssd_llm_demo.py) else (echo    ❌ ssd_llm_demo.py)
if exist "%PROJECT_DIR%web_dashboard.py" (echo    ✅ web_dashboard.py) else (echo    ❌ web_dashboard.py)
if exist "%PROJECT_DIR%dll_test.py" (echo    ✅ dll_test.py) else (echo    ❌ dll_test.py)
if exist "%PROJECT_DIR%requirements.txt" (echo    ✅ requirements.txt) else (echo    ❌ requirements.txt)
if exist "%PROJECT_DIR%ssd_align_leap.dll" (echo    ✅ ssd_align_leap.dll) else (echo    ❌ ssd_align_leap.dll)

echo.
echo 🔍 GPU情報:
if exist "%VENV_DIR%" (
    call "%VENV_DIR%\Scripts\activate.bat"
    "%VENV_DIR%\Scripts\python.exe" -c "import torch; print('PyTorch CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>nul
    if %errorLevel% neq 0 echo    ❌ PyTorch未インストール
) else (
    echo    ❌ 仮想環境未作成
)

echo.
echo 💾 ディスク使用量:
dir "%PROJECT_DIR%" | find "個のファイル" 2>nul
if exist "%VENV_DIR%" (
    echo    仮想環境サイズ: 
    dir "%VENV_DIR%" /s | find "個のファイル" 2>nul
)

echo.
pause
goto MENU

REM ========================================
REM 環境リセット
REM ========================================
:RESET_ENV
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                     🔄 環境リセット                            ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

set /p confirm="⚠️  仮想環境とプロジェクトファイルを削除しますか？ (y/N): "
if /i not "%confirm%"=="y" (
    echo キャンセルしました。
    pause
    goto MENU
)

echo.
echo 🗑️  環境をリセット中...

REM 仮想環境削除
if exist "%VENV_DIR%" (
    echo    仮想環境を削除中...
    rmdir /s /q "%VENV_DIR%"
    echo    ✅ 仮想環境削除完了
)

REM プロジェクトファイル削除
if exist "%PROJECT_DIR%requirements.txt" del "%PROJECT_DIR%requirements.txt"
if exist "%PROJECT_DIR%ssd_llm_demo.py" del "%PROJECT_DIR%ssd_llm_demo.py"
if exist "%PROJECT_DIR%web_dashboard.py" del "%PROJECT_DIR%web_dashboard.py"
if exist "%PROJECT_DIR%dll_test.py" del "%PROJECT_DIR%dll_test.py"

REM キャッシュ削除
if exist "%PROJECT_DIR%__pycache__" rmdir /s /q "%PROJECT_DIR%__pycache__"
if exist "%PROJECT_DIR%.streamlit" rmdir /s /q "%PROJECT_DIR%.streamlit"

echo ✅ 環境リセット完了！
echo 📝 再度セットアップから始めてください。
echo.
pause
goto MENU

REM ========================================
REM 終了
REM ========================================
:EXIT
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║                      👋 ありがとうございました                  ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo 🏗️  SSD-LLM Structural Subjectivity Dynamics + LLM
echo 📧  サポート: https://github.com/your-repo/ssd-llm
echo.
echo プログラムを終了します...
timeout /t 3 >nul
exit /b 0