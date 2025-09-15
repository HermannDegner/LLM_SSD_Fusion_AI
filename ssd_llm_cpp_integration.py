"""
SSD-LLM Integration (Improved Version)

変更点:
- DLL の追加関数 (ssd_get_params / ssd_set_params / ssd_get_N) バインディング
- softmax/指数安定化に対応する C++ パッチ前提
- GPU 未使用時は float32 / CPU へ移行
- ノード範囲安全化
- ジャンプ確率 p_jump を Python メタデータに追加 (p_jump = 1 - exp(-h*dt))
- fetch_ssd_params / apply_ssd_params を追加
- エラーハンドリング・ログ改善
"""

import ctypes
import os
import time
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# ctypes Structures (C++ と 1:1 対応)
# =============================================================================

class SSDParams(ctypes.Structure):
    _fields_ = [
        ("G0", ctypes.c_double),
        ("g", ctypes.c_double),
        ("eps_noise", ctypes.c_double),
        ("eta", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("lam", ctypes.c_double),
        ("kappa_min", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("beta_E", ctypes.c_double),
        ("Theta0", ctypes.c_double),
        ("a1", ctypes.c_double),
        ("a2", ctypes.c_double),
        ("h0", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("T0", ctypes.c_double),
        ("c1", ctypes.c_double),
        ("c2", ctypes.c_double),
        ("sigma", ctypes.c_double),
        ("delta_w", ctypes.c_double),
        ("delta_kappa", ctypes.c_double),
        ("c0_cool", ctypes.c_double),
        ("q_relax", ctypes.c_double),
        ("eps_relax", ctypes.c_double),
        ("eps0", ctypes.c_double),
        ("d1", ctypes.c_double),
        ("d2", ctypes.c_double),
        ("b_path", ctypes.c_double),
    ]


class SSDTelemetry(ctypes.Structure):
    _fields_ = [
        ("E", ctypes.c_double),
        ("Theta", ctypes.c_double),
        ("h", ctypes.c_double),
        ("T", ctypes.c_double),
        ("H", ctypes.c_double),
        ("J_norm", ctypes.c_double),
        ("align_eff", ctypes.c_double),
        ("kappa_mean", ctypes.c_double),
        ("current", ctypes.c_int32),
        ("did_jump", ctypes.c_int32),
        ("rewired_to", ctypes.c_int32),
    ]


# =============================================================================
# DLL Wrapper
# =============================================================================

class SSDCoreDLL:
    def __init__(self, dll_path: str = "./ssd_align_leap.dll"):
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL not found: {dll_path}")
        self.dll = ctypes.CDLL(dll_path)

        # Bind core functions
        self.dll.ssd_create.argtypes = [ctypes.c_int32, ctypes.POINTER(SSDParams), ctypes.c_uint64]
        self.dll.ssd_create.restype = ctypes.c_void_p

        self.dll.ssd_destroy.argtypes = [ctypes.c_void_p]
        self.dll.ssd_destroy.restype = None

        self.dll.ssd_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(SSDTelemetry),
        ]
        self.dll.ssd_step.restype = None

        self.dll.ssd_get_kappa_row.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int32,
        ]
        self.dll.ssd_get_kappa_row.restype = ctypes.c_int32

        # New utility bindings
        self.dll.ssd_get_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(SSDParams)]
        self.dll.ssd_get_params.restype = None
        self.dll.ssd_set_params.argtypes = [ctypes.c_void_p, ctypes.POINTER(SSDParams)]
        self.dll.ssd_set_params.restype = None
        self.dll.ssd_get_N.argtypes = [ctypes.c_void_p]
        self.dll.ssd_get_N.restype = ctypes.c_int32

    def create_default_params(self) -> SSDParams:
        p = SSDParams()
        p.G0 = 0.5
        p.g = 0.7
        p.eps_noise = 0.0
        p.eta = 0.3
        p.rho = 0.3
        p.lam = 0.02
        p.kappa_min = 0.0
        p.alpha = 0.6
        p.beta_E = 0.15
        p.Theta0 = 1.0
        p.a1 = 0.5
        p.a2 = 0.4
        p.h0 = 0.2
        p.gamma = 0.8
        p.T0 = 0.3
        p.c1 = 0.5
        p.c2 = 0.6
        p.sigma = 0.2
        p.delta_w = 0.2
        p.delta_kappa = 0.2
        p.c0_cool = 0.6
        p.q_relax = 0.1
        p.eps_relax = 0.01
        p.eps0 = 0.02
        p.d1 = 0.2
        p.d2 = 0.2
        p.b_path = 0.5
        return p


# =============================================================================
# Config Dataclass
# =============================================================================

@dataclass
class SSDLLMConfig:
    model_name: str = "microsoft/DialoGPT-medium"  # 軽量モデルでテスト
    ssd_nodes: int = 8
    max_tokens: int = 128
    device: str = "auto"          # 'cuda' / 'cpu' / 'mps'
    ssd_dll_path: str = "./ssd_align_leap.dll"
    dtype_preference: str = "auto"  # "auto" / "fp16" / "fp32"


# =============================================================================
# SSD-Enhanced LLM
# =============================================================================

class SSDEnhancedLLM:
    def __init__(self, config: SSDLLMConfig):
        self.config = config

        print(f"[SSD-LLM] Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            # 一部モデル（DialoGPTなど）は pad_token 未定義
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_cuda = torch.cuda.is_available()
        if config.dtype_preference == "fp32":
            torch_dtype = torch.float32
        elif config.dtype_preference == "fp16":
            torch_dtype = torch.float16 if use_cuda else torch.float32
        else:  # auto
            torch_dtype = torch.float16 if use_cuda else torch.float32

        device_map = "auto" if use_cuda and config.device in ("auto", "cuda") else None

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=False
        )
        if device_map is None:
            # CPU / 単一デバイス
            target_device = "cuda" if (use_cuda and config.device == "cuda") else "cpu"
            self.model.to(target_device)

        print("[SSD-LLM] Initializing SSD core...")
        self.ssd_dll = SSDCoreDLL(config.ssd_dll_path)
        self.ssd_params = self.ssd_dll.create_default_params()
        self.ssd_handle = self.ssd_dll.dll.ssd_create(
            config.ssd_nodes,
            ctypes.byref(self.ssd_params),
            123456789
        )
        if not self.ssd_handle:
            raise RuntimeError("Failed to create SSD instance (ssd_create returned null).")

        self.response_strategies = self._initialize_response_strategies()
        self.conversation_history: List[Dict[str, Any]] = []
        self.ssd_metrics_history: List[Dict[str, Any]] = []

    # -------------------------------
    # Strategy Initialization
    # -------------------------------
    def _initialize_response_strategies(self) -> Dict[int, Dict[str, Any]]:
        strategies: Dict[int, Dict[str, Any]] = {}
        N = self.config.ssd_nodes
        for i in range(N):
            if i < N // 4:
                strategies[i] = {
                    'type': 'analytical',
                    'temperature': 0.2,
                    'top_p': 0.8,
                    'system_prompt': "あなたは論理的で分析的な思考を重視するAIです。事実に基づいて明確に説明してください。"
                }
            elif i < N // 2:
                strategies[i] = {
                    'type': 'creative',
                    'temperature': 0.9,
                    'top_p': 0.95,
                    'system_prompt': "あなたは創造的で斬新なアイデアを提案するAIです。多角的に発想してください。"
                }
            elif i < 3 * N // 4:
                strategies[i] = {
                    'type': 'empathetic',
                    'temperature': 0.6,
                    'top_p': 0.9,
                    'system_prompt': "あなたは共感的で温かいAIです。相手の感情に寄り添いながら回答してください。"
                }
            else:
                strategies[i] = {
                    'type': 'practical',
                    'temperature': 0.4,
                    'top_p': 0.85,
                    'system_prompt': "あなたは実用的で効率的な助言を行うAIです。具体的で実行可能な提案をしてください。"
                }
        return strategies

    # -------------------------------
    # Meaning Pressure Estimation
    # -------------------------------
    def analyze_meaning_pressure(self, user_input: str, context: Optional[str] = None) -> float:
        pressure = len(user_input) / 1000.0
        if any(q in user_input for q in ("?", "？")):
            pressure += 0.3
        if any(w in user_input for w in ['困っ', '悩ん', '嬉しい', '悲しい', '怒', '不安', '心配']):
            pressure += 0.5
        if any(w in user_input for w in ['複雑', '難しい', '詳しく', '詳細', '分析', '比較']):
            pressure += 0.4
        if any(w in user_input for w in ['急', 'すぐ', '至急', '緊急', 'ASAP']):
            pressure += 0.6
        return min(pressure, 2.0)

    # -------------------------------
    # Core Generation
    # -------------------------------
    def generate_response(self, user_input: str, context: Optional[str] = None, dt: float = 1.0) -> Dict[str, Any]:
        if not self.ssd_handle:
            raise RuntimeError("SSD handle not initialized.")

        start_time = time.time()

        meaning_pressure = self.analyze_meaning_pressure(user_input, context)
        telemetry = SSDTelemetry()
        self.ssd_dll.dll.ssd_step(
            self.ssd_handle,
            ctypes.c_double(meaning_pressure),
            ctypes.c_double(dt),
            ctypes.byref(telemetry)
        )

        # Node guard
        node = int(telemetry.current)
        if node < 0 or node >= self.config.ssd_nodes:
            node = node % self.config.ssd_nodes
        strategy = self.response_strategies.get(node, self.response_strategies[0])

        # Dynamic temperature
        dynamic_temp = strategy['temperature']
        if telemetry.did_jump:
            dynamic_temp = min(dynamic_temp + 0.3, 1.0)
        if telemetry.align_eff > 0.8:
            dynamic_temp = max(dynamic_temp - 0.2, 0.1)

        # Prompt
        system_prompt = strategy['system_prompt']
        full_prompt = f"{system_prompt}\n\nユーザー: {user_input}\nAI:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=dynamic_temp,
                top_p=strategy['top_p'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        reply = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        generation_time = time.time() - start_time

        # Jump probability (reconstruct)
        p_jump = 1.0 - float(np.exp(-telemetry.h * dt)) if telemetry.h >= 0 else 0.0
        if p_jump > 1.0:
            p_jump = 1.0
        elif p_jump < 0.0:
            p_jump = 0.0

        ssd_meta = {
            'mode_used': 'leap' if telemetry.did_jump else 'alignment',
            'strategy_type': strategy['type'],
            'current_node': node,
            'heat_level': telemetry.E,
            'theta': telemetry.Theta,
            'jump_rate': telemetry.h,
            'jump_probability': p_jump,
            'exploration_temp': telemetry.T,
            'policy_entropy': telemetry.H,
            'alignment_efficiency': telemetry.align_eff,
            'kappa_mean': telemetry.kappa_mean,
            'did_jump': bool(telemetry.did_jump),
            'rewired_to': telemetry.rewired_to,
            'meaning_pressure': meaning_pressure
        }

        gen_meta = {
            'temperature_used': dynamic_temp,
            'top_p_used': strategy['top_p'],
            'generation_time_sec': generation_time,
            'model_used': self.config.model_name
        }

        self.conversation_history.append({
            'user_input': user_input,
            'response': reply,
            'timestamp': time.time()
        })
        self.ssd_metrics_history.append(ssd_meta)

        return {
            'response': reply,
            'ssd_metadata': ssd_meta,
            'generation_metadata': gen_meta
        }

    # -------------------------------
    # Kappa Matrix
    # -------------------------------
    def get_kappa_matrix(self) -> np.ndarray:
        if not self.ssd_handle:
            raise RuntimeError("SSD not initialized.")
        N = self.ssd_dll.dll.ssd_get_N(self.ssd_handle)
        mat = np.zeros((N, N))
        row_buf = (ctypes.c_double * N)()
        for i in range(N):
            count = self.ssd_dll.dll.ssd_get_kappa_row(
                self.ssd_handle, i, row_buf, N
            )
            if count > 0:
                mat[i, :count] = [row_buf[j] for j in range(count)]
        return mat

    # -------------------------------
    # Parameter Ops
    # -------------------------------
    def fetch_ssd_params(self) -> SSDParams:
        if not self.ssd_handle:
            raise RuntimeError("SSD not initialized.")
        params_copy = SSDParams()
        self.ssd_dll.dll.ssd_get_params(self.ssd_handle, ctypes.byref(params_copy))
        return params_copy

    def apply_ssd_params(self):
        if not self.ssd_handle:
            raise RuntimeError("SSD not initialized.")
        self.ssd_dll.dll.ssd_set_params(self.ssd_handle, ctypes.byref(self.ssd_params))

    def update_ssd_params(self, **kwargs):
        updated = []
        for k, v in kwargs.items():
            if hasattr(self.ssd_params, k):
                setattr(self.ssd_params, k, v)
                updated.append(k)
        if updated:
            self.apply_ssd_params()
        return updated

    # -------------------------------
    # Status / Stats
    # -------------------------------
    def _calculate_strategy_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for m in self.ssd_metrics_history:
            stype = m['strategy_type']
            dist[stype] = dist.get(stype, 0) + 1
        return dist

    def get_ssd_status(self) -> Dict[str, Any]:
        if not self.ssd_metrics_history:
            return {"status": "no_data"}

        latest = self.ssd_metrics_history[-1]
        kappa_matrix = self.get_kappa_matrix()
        jumps = sum(1 for m in self.ssd_metrics_history if m['did_jump'])
        total = len(self.ssd_metrics_history)

        return {
            "current_state": {
                "heat_level": latest['heat_level'],
                "theta": latest['theta'],
                "exploration_temp": latest['exploration_temp'],
                "current_node": latest['current_node'],
                "alignment_efficiency": latest['alignment_efficiency'],
                "kappa_mean": latest['kappa_mean'],
                "jump_probability": latest['jump_probability']
            },
            "statistics": {
                "total_interactions": total,
                "total_jumps": jumps,
                "jump_rate": jumps / total if total else 0.0,
                "avg_heat_level": sum(m['heat_level'] for m in self.ssd_metrics_history) / total,
                "strategy_distribution": self._calculate_strategy_distribution()
            },
            "kappa_matrix_shape": kappa_matrix.shape,
            "kappa_matrix_stats": {
                "mean": float(np.mean(kappa_matrix)),
                "std": float(np.std(kappa_matrix)),
                "max": float(np.max(kappa_matrix)),
                "min": float(np.min(kappa_matrix))
            }
        }

    # -------------------------------
    # Cleanup
    # -------------------------------
    def close(self):
        if getattr(self, 'ssd_handle', None):
            try:
                self.ssd_dll.dll.ssd_destroy(self.ssd_handle)
            finally:
                self.ssd_handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# Example Main
# =============================================================================

def main():
    config = SSDLLMConfig(
        model_name="microsoft/DialoGPT-medium",
        ssd_nodes=8,
        max_tokens=96,
        device="auto"
    )
    engine = SSDEnhancedLLM(config)

    test_inputs = [
        "こんにちは！今日はどんな日？",
        "空が青い理由を物理現象として説明してください。",
        "急ぎで創造的なプロダクトアイデアを3つください。",
        "最近仕事で悩んでいます。どう整理すればいいでしょう？"
    ]

    for i, text in enumerate(test_inputs, 1):
        result = engine.generate_response(text)
        meta = result['ssd_metadata']
        print(f"\n--- Turn {i} ---")
        print("User:", text)
        print(f"AI [{meta['strategy_type']}]: {result['response']}")
        print(f"  Node={meta['current_node']} Mode={meta['mode_used']} Heat={meta['heat_level']:.3f} "
              f"JumpProb={meta['jump_probability']:.3f} Jump={meta['did_jump']}")

    print("\n[Final SSD Status]")
    print(json.dumps(engine.get_ssd_status(), ensure_ascii=False, indent=2))

    km = engine.get_kappa_matrix()
    print("\nKappa Matrix (excerpt):")
    with np.printoptions(precision=3, suppress=True):
        print(km)

    engine.close()


if __name__ == "__main__":
    main()