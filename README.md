# 構造主観力学-LLM融合AIシステム (SSD-LLM Fusion AI)

## 概要
SSD-LLM (Structural Subjectivity Dynamics Enhanced Large Language Model) は、構造主観力学（SSD）の理論をオープンソースの大規模言語モデル（LLM）に統合し、従来の「質問応答」パターンを超えた、人間らしい学習・創造・適応能力を持つAIシステムの構築を目指すプロジェクトです。

## 核心コンセプト
このシステムの動作は、SSD理論の2つの基本原理に基づいています。

**整合 (Alignment)**: 安定した知識や過去の成功パターンに基づき、効率的で一貫性のある応答を生成します。

**跳躍 (Leap)**: ユーザーからの入力（意味圧）が既存の知識の限界を超えたと判断した場合、創造的で新しいアイデアや視点を生成します。

## ✨ 主な特徴
- **動的な応答モード切替**: ユーザー入力の「意味圧」を分析し、安定した「整合モード」と創造的な「跳躍モード」を自動で切り替えます。

- **多様な応答戦略**: 「分析的」「創造的」「共感的」「実用的」など、複数の応答戦略を持ち、状況に応じて最適なものを選択します。

- **継続的な自己学習**: ユーザーとの対話やフィードバックを通じて、応答戦略を継続的に改善します。高品質な対話データを用いて、LoRAによるファインチューニングや強化学習を行います。

- **ハイブリッドアーキテクチャ**: 計算負荷の高いSSDコアロジックをC++のDLLで実装し、Pythonから呼び出すことで高速な処理を実現しています。

- **透明性の高い内部状態**: AIの内部状態（熱レベル、整合効率、跳躍確率など）をリアルタイムで監視・分析することが可能です。

## 🛠️ アーキテクチャ
システムは、ユーザー入力をSSD理論に基づいて分析し、LLMの応答生成を制御するパイプラインで構成されています。

```
graph TB
    Input[ユーザー入力] --> SPA[SSD前処理アナライザー]
    SPA --> MPD[意味圧検知器]
    MPD --> AJD[整合/跳躍判定器]
    
    AJD -->|整合モード| APG[整合経路生成器]
    AJD -->|跳躍モード| CPG[創造的パス生成器]
    
    APG --> LLM[オープンソースLLM]
    CPG --> LLM
    
    LLM --> SPP[SSD後処理プロセッサ]
    SPP --> SMU[SSD記憶更新器]
    SPP --> Output[出力 + メタ情報]
    
    SMU --> MM[記憶マトリックス]
    MM --> SPA
```

## 🚀 セットアップ手順

### 1. 前提条件
- **Python 3.10以上**
- **Git**
- **NVIDIA GPU (VRAM 16GB以上を推奨)**
- **CUDA Toolkit 12.1以上**

### 2. 簡単インストール（推奨）
```bash
# 1. ファイルをダウンロード
git clone https://github.com/hermanndegner/llm_ssd_fusion_ai.git
cd llm_ssd_fusion_ai

# 2. ワンクリックセットアップ（Windows）
# 管理者権限でコマンドプロンプトを開き、以下を実行
SSD-LLM_Manager.bat
```

### 3. 手動セットアップ
```bash
# 仮想環境作成
python -m venv ssd_llm_env
source ssd_llm_env/bin/activate  # Windows: ssd_llm_env\Scripts\activate

# 必須ライブラリインストール
pip install torch transformers accelerate
pip install numpy scipy networkx scikit-learn
pip install fastapi uvicorn websockets
pip install streamlit plotly peft datasets
pip install bitsandbytes optimum
```

### 4. C++コアの準備
このプロジェクトは、C++で実装されたコアロジック (`ssd_align_leap.dll`) を使用します。提供されているDLLファイルがご自身の環境で動作しない場合は、C++ソースコードからビルドし直す必要があります。

### 5. LLMモデルの準備
Hugging Face Hubから、使用するオープンソースLLMをダウンロードします。事前にHugging Faceアカウントで認証が必要です。

```bash
pip install huggingface_hub
huggingface-cli login
```

## 💡 使い方

### クイックスタート
```bash
# 超軽量版ですぐに試す
SSD-LLM クイックスタート.bat
```

### フル機能版
```python
from ssd_llm_cpp_integration import SSDLLMConfig, SSDEnhancedLLM

# 軽量なモデルでテストする場合
config = SSDLLMConfig(
    model_name="microsoft/DialoGPT-medium",
    ssd_nodes=8,
    max_tokens=128,
    ssd_dll_path="./ssd_align_leap.dll"
)

# SSD-LLMを初期化
ssd_llm = SSDEnhancedLLM(config)

print("SSD-LLMとの対話を開始します。（'exit'で終了）")
while True:
    user_input = input("あなた: ")
    if user_input.lower() == 'exit':
        break

    # 応答を生成
    result = ssd_llm.generate_response(user_input)
    
    # メタデータと共に応答を表示
    meta = result['ssd_metadata']
    print(f"AI [{meta['strategy_type']} / {meta['mode_used']}]: {result['response']}")
    if meta['did_jump']:
        print(f"  -> 🚀 跳躍が発生しました！")
```

### Webダッシュボード
```bash
# ブラウザベースの対話・監視インターフェース
streamlit run web_dashboard.py
```

## 🧠 学習システム
このシステムは、`ssd_llm_learning_strategies.py` に定義された継続的学習機能を備えています。

- **対話データの収集**: 全ての対話とフィードバックはSQLiteデータベースに保存されます。

- **自己改善**: 収集されたデータを用いて、定期的にLoRAファインチューニングや強化学習が実行され、AIの応答品質が向上します。

## 📊 監視・デバッグ

### リアルタイム監視指標
- **未処理圧レベル**: システムの「ストレス」状態
- **探索温度**: 創造性の度合い
- **整合効率**: 安定応答の品質
- **跳躍頻度**: 創造的応答の発生率
- **創造歩留まり**: 新規接続の成功率

### トラブルシューティング
```bash
# 問題が発生した場合
SSD-LLM トラブルシューティング.bat
```

## 🔧 設定とカスタマイズ

### SSDパラメータの調整
```python
# 創造性を高める設定
config.ssd_params.T0 = 0.5        # 探索温度上昇
config.ssd_params.h0 = 0.3        # 跳躍頻度上昇
config.ssd_params.c1 = 0.7        # 熱感度上昇

# 安定性を高める設定  
config.ssd_params.beta_E = 0.25   # 熱減衰強化
config.ssd_params.Theta0 = 1.5    # 跳躍閾値上昇
config.ssd_params.a1 = 0.8        # 慣性重視
```

### 応答戦略のカスタマイズ
```python
# 新しい戦略の追加
custom_strategy = {
    'type': 'scientific',
    'temperature': 0.1,
    'top_p': 0.7,
    'system_prompt': "あなたは厳密な科学的根拠に基づいて回答するAIです。"
}
ssd_llm.response_strategies[new_node] = custom_strategy
```

## 📚 対応モデル

### 推奨オープンソースLLM
| モデル | 用途 | メモリ要件 | 特徴 |
|--------|------|------------|------|
| **Llama 3.1 8B** | メイン | 16GB | バランス型、汎用性高 |
| **Mistral 7B** | 論理推論 | 14GB | 高効率、推論速度速 |
| **Qwen2.5 7B** | 多言語・創造 | 14GB | 創造性、日本語対応 |
| **Code Llama 7B** | コード生成 | 14GB | プログラミング特化 |

## 🤝 貢献
このプロジェクトへの貢献を歓迎します。バグ報告、機能提案、プルリクエストなど、お気軽にどうぞ。

1. リポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを作成

## 📄 ライセンス
このプロジェクトはMITライセンスの下で公開されています。詳細は`LICENSE`ファイルをご覧ください。

## 🆘 サポート

### よくある問題と解決法

**Q: DLLが見つからないエラーが出る**
A: `ssd_align_leap.dll`をプロジェクトルートに配置するか、C++ソースからビルドしてください。

**Q: GPU使用時にメモリ不足エラーが出る**
A: より軽量なモデル（DialoGPT-medium、DistilGPT-2など）を使用するか、`config.dtype_preference="fp32"`でCPUモードに切り替えてください。

**Q: 応答が単調で創造性がない**
A: SSDパラメータの`T0`（探索温度）や`h0`（跳躍強度）を上げてみてください。

**Q: 応答が不安定すぎる**
A: `beta_E`（熱減衰）を上げるか、`Theta0`（跳躍閾値）を上げて安定性を向上させてください。

### 技術サポート
- **GitHub Issues**: バグ報告・機能要望
- **Discord**: リアルタイムサポート
- **Email**: technical-support@ssd-llm.org

## 🔮 ロードマップ

### 短期目標 (3ヶ月)
- [ ] より多くのオープンソースLLMとの統合
- [ ] GPU最適化の改善
- [ ] Web UI の機能拡張

### 中期目標 (6ヶ月)
- [ ] マルチエージェントシステムの実装
- [ ] 音声・画像入力への対応
- [ ] モバイルアプリ版の開発

### 長期目標 (1年)
- [ ] エッジデバイス対応
- [ ] 企業向けAPIサービス
- [ ] 学術研究との連携

## 📖 参考文献
- 構造主観力学（SSD）理論文書
- Nano-SSD実装仕様
- LLM統合設計書

---

**🏗️ SSD-LLM** - 人工知能に人間らしい創造性と適応性をもたらす革新的プロジェクト