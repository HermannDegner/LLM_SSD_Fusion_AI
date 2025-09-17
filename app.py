# app.py — SSD Chat UI (Streamlit)
# 依存: ssd_llm_cpp_integration.py（SSDLLMConfig / SSDEnhancedLLM）
import os
from pathlib import Path
from typing import Any, Dict

import streamlit as st

# ─────────────────────────────────────────────────────────
# 0) DLLの検索パスと絶対パス設定
# ─────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DLL_PATH = BASE / "ssd_align_leap.dll"  # 同じフォルダに置くのが推奨

# Windows 3.8+ なら依存DLL解決のためにフォルダを追加できる
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(BASE))

# (任意) HFのシンボリックリンク警告を消したい場合
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

st.set_page_config(page_title="SSD Chat", page_icon="🧠", layout="wide")

# ─────────────────────────────────────────────────────────
# 1) エンジンのロード（初回のみ）
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_engine():
    from ssd_llm_cpp_integration import SSDEnhancedLLM, SSDLLMConfig

    if not DLL_PATH.exists():
        raise FileNotFoundError(
            f"ssd_align_leap.dll が見つかりません: {DLL_PATH}\n"
            "同じフォルダにDLLを置くか、DLL_PATHを修正してください。"
        )

    cfg = SSDLLMConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        ssd_nodes=8,
        max_tokens=256,          # 余裕あるので増やせる
        device="cuda",
        ssd_dll_path=str(DLL_PATH),
        dtype_preference="bf16"  # 4090ならbf16が安定＆高速
    )
    eng = SSDEnhancedLLM(cfg)

    # 初期の実験値（必要に応じて調整）
    try:
        eng.update_ssd_params(T0=0.6, c1=0.9, c2=0.3, h0=0.35, gamma=0.9, Theta0=0.9)
    except Exception:
        pass
    return eng

try:
    engine = load_engine()
except Exception as e:
    st.error(f"エンジン初期化に失敗: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────
# 2) サイドバー：基本操作
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 設定")
    max_tokens = st.slider("max_new_tokens（長さ）", 32, 256, engine.config.max_tokens, 8)
    temp_boost = st.slider("温度ブースト（生成の多様性）", 0.0, 0.8, 0.30, 0.05)

    st.markdown("---")
    st.markdown("### SSD パラメータ（実験）")
    col1, col2 = st.columns(2)
    with col1:
        T0 = st.number_input("T0 (探索温度の下限)", value=0.60, step=0.05)
        h0 = st.number_input("h0 (ベース跳躍率)", value=0.35, step=0.05)
        Theta0 = st.number_input("Θ0 (基本閾値)", value=0.90, step=0.05)
    with col2:
        c1 = st.number_input("c1 (熱→温度利得)", value=0.90, step=0.05)
        c2 = st.number_input("c2 (硬直補正)", value=0.30, step=0.05)
        gamma = st.number_input("γ (発火曲線の鋭さ)", value=0.90, step=0.05)

    if st.button("SSDに反映"):
        try:
            engine.update_ssd_params(T0=T0, h0=h0, Theta0=Theta0, c1=c1, c2=c2, gamma=gamma)
            st.success("反映しました。")
        except Exception as e:
            st.error(f"反映に失敗: {e}")

    st.markdown("---")
    if st.button("状態リセット（会話履歴のみ）"):
        st.session_state["messages"] = []
        st.session_state["last_meta"] = None
        st.success("チャット履歴をクリアしました。")

# ─────────────────────────────────────────────────────────
# 3) チャットUI
# ─────────────────────────────────────────────────────────
st.title("🧠 SSD Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_meta" not in st.session_state:
    st.session_state["last_meta"] = None

# 既存メッセージを描画
for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(content)

# 入力欄（/say で直出し）
user = st.chat_input("メッセージを入力（/say で直出し可）")
if user:
    st.session_state["messages"].append(("user", user))
    with st.chat_message("user"):
        st.markdown(user)

    # 生成設定を反映
    engine.config.max_tokens = int(max_tokens)

    if user.startswith("/say "):
        reply_text = user[5:]  # 任意文字をそのまま返す
        meta: Dict[str, Any] = {"mode_used": "verbatim_bypass"}
    else:
        # 簡易に探索温度を底上げ（必要な場合のみ）
        if temp_boost > 0:
            try:
                engine.update_ssd_params(T0=T0 + float(temp_boost))
            except Exception:
                pass

        out = engine.generate_response(user)
        reply_text = out["response"]
        meta = out["ssd_metadata"]

    with st.chat_message("assistant"):
        st.markdown(reply_text)
    st.session_state["messages"].append(("assistant", reply_text))
    st.session_state["last_meta"] = meta

# ─────────────────────────────────────────────────────────
# 4) テレメトリ表示
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 SSD テレメトリ")

status = engine.get_ssd_status()
if status.get("status") == "no_data":
    st.info("まだテレメトリがありません。メッセージを送ってください。")
else:
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Heat (E)", f"{status['current_state']['heat_level']:.3f}")
        st.metric("Jump Prob.", f"{status['current_state']['jump_probability']:.3f}")
    with colB:
        st.metric("Alignment Eff.", f"{status['current_state']['alignment_efficiency']:.3f}")
        st.metric("kappa_mean", f"{status['current_state']['kappa_mean']:.3f}")
    with colC:
        st.metric("Current Node", status['current_state']['current_node'])
        st.metric("Total Jumps", status['statistics']['total_jumps'])
    st.caption("Strategy dist.")
    st.json(status["statistics"]["strategy_distribution"])
