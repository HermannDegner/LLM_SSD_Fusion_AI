# SSD-LLM統合システム（C++ DLL活用版）

import ctypes
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json

# =============================================================================
# C++ DLL バインディング
# =============================================================================

class SSDParams(ctypes.Structure):
    """C++のSSDParams構造体にマッピング"""
    _fields_ = [
        # Alignment (deterministic)
        ("G0", ctypes.c_double),
        ("g", ctypes.c_double), 
        ("eps_noise", ctypes.c_double),
        ("eta", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("lam", ctypes.c_double),
        ("kappa_min", ctypes.c_double),
        
        # Heat
        ("alpha", ctypes.c_double),
        ("beta_E", ctypes.c_double),
        
        # Threshold / jump  
        ("Theta0", ctypes.c_double),
        ("a1", ctypes.c_double),
        ("a2", ctypes.c_double),
        ("h0", ctypes.c_double),
        ("gamma", ctypes.c_double),
        
        # Temperature
        ("T0", ctypes.c_double),
        ("c1", ctypes.c_double),
        ("c2", ctypes.c_double),
        
        # Policy
        ("sigma", ctypes.c_double),
        
        # Rewire
        ("delta_w", ctypes.c_double),
        ("delta_kappa", ctypes.c_double),
        ("c0_cool", ctypes.c_double),
        ("q_relax", ctypes.c_double),
        ("eps_relax", ctypes.c_double),
        
        # Epsilon-random
        ("eps0", ctypes.c_double),
        ("d1", ctypes.c_double),
        ("d2", ctypes.c_double),
        
        # Action
        ("b_path", ctypes.c_double),
    ]

class SSDTelemetry(ctypes.Structure):
    """C++のSSDTelemetry構造体にマッピング"""
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

class SSDCoreDLL:
    """C++ SSD DLLのPythonラッパー"""
    
    def __init__(self, dll_path: str = "./ssd_align_leap.dll"):
        self.dll = ctypes.CDLL(dll_path)
        
        # 関数シグネチャの定義
        self.dll.ssd_create.argtypes = [ctypes.c_int32, ctypes.POINTER(SSDParams), ctypes.c_uint64]
        self.dll.ssd_create.restype = ctypes.c_void_p
        
        self.dll.ssd_destroy.argtypes = [ctypes.c_void_p]
        self.dll.ssd_destroy.restype = None
        
        self.dll.ssd_step.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.POINTER(SSDTelemetry)]
        self.dll.ssd_step.restype = None
        
        self.dll.ssd_get_kappa_row.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
        self.dll.ssd_get_kappa_row.restype = ctypes.c_int32
        
    def create_default_params(self) -> SSDParams:
        """デフォルトパラメータでSSDParams構造体を作成"""
        params = SSDParams()
        params.G0 = 0.5
        params.g = 0.7
        params.eps_noise = 0.0
        params.eta = 0.3
        params.rho = 0.3
        params.lam = 0.02
        params.kappa_min = 0.0
        params.alpha = 0.6
        params.beta_E = 0.15
        params.Theta0 = 1.0
        params.a1 = 0.5
        params.a2 = 0.4
        params.h0 = 0.2
        params.gamma = 0.8
        params.T0 = 0.3
        params.c1 = 0.5
        params.c2 = 0.6
        params.sigma = 0.2
        params.delta_w = 0.2
        params.delta_kappa = 0.2
        params.c0_cool = 0.6
        params.q_relax = 0.1
        params.eps_relax = 0.01
        params.eps0 = 0.02
        params.d1 = 0.2
        params.d2 = 0.2
        params.b_path = 0.5
        return params

# =============================================================================
# SSD強化LLMクラス
# =============================================================================

@dataclass
class SSDLLMConfig:
    """SSD-LLM設定"""
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    ssd_nodes: int = 16  # SSDノード数（応答戦略の数）
    max_tokens: int = 512
    device: str = "auto"
    ssd_dll_path: str = "./ssd_align_leap.dll"

class SSDEnhancedLLM:
    """C++ DLLを使ったSSD強化LLM"""
    
    def __init__(self, config: SSDLLMConfig):
        self.config = config
        
        # LLMの初期化
        print(f"Loading LLM: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map=config.device,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # SSD DLLの初期化
        print("Initializing SSD Core...")
        self.ssd_dll = SSDCoreDLL(config.ssd_dll_path)
        self.ssd_params = self.ssd_dll.create_default_params()
        self.ssd_handle = self.ssd_dll.dll.ssd_create(
            config.ssd_nodes, 
            ctypes.byref(self.ssd_params), 
            123456789  # seed
        )
        
        # 応答戦略の定義（各ノードに対応）
        self.response_strategies = self._initialize_response_strategies()
        
        # 会話履歴とメトリクス
        self.conversation_history = []
        self.ssd_metrics_history = []
        
    def _initialize_response_strategies(self) -> Dict[int, Dict[str, Any]]:
        """各SSDノードに応答戦略を割り当て"""
        strategies = {}
        N = self.config.ssd_nodes
        
        for i in range(N):
            if i < N // 4:
                # 論理的・分析的戦略
                strategies[i] = {
                    'type': 'analytical',
                    'temperature': 0.2,
                    'top_p': 0.8,
                    'system_prompt': "あなたは論理的で分析的な思考を重視するAIです。事実に基づいて詳細に説明してください。"
                }
            elif i < N // 2:
                # 創造的・発想的戦略  
                strategies[i] = {
                    'type': 'creative',
                    'temperature': 0.9,
                    'top_p': 0.95,
                    'system_prompt': "あなたは創造的で斬新なアイデアを提案するAIです。型にはまらない発想を大切にしてください。"
                }
            elif i < 3 * N // 4:
                # 共感的・対話的戦略
                strategies[i] = {
                    'type': 'empathetic', 
                    'temperature': 0.6,
                    'top_p': 0.9,
                    'system_prompt': "あなたは共感的で温かいAIです。相手の気持ちに寄り添った応答をしてください。"
                }
            else:
                # 実用的・問題解決戦略
                strategies[i] = {
                    'type': 'practical',
                    'temperature': 0.4,
                    'top_p': 0.85,
                    'system_prompt': "あなたは実用的で効率的な解決策を提案するAIです。具体的で実行可能な助言をしてください。"
                }
        
        return strategies
    
    def analyze_meaning_pressure(self, user_input: str, context: Optional[str] = None) -> float:
        """ユーザー入力から意味圧を計算"""
        # 簡易的な意味圧計算（実際はもっと sophisticated な分析が必要）
        pressure = 0.0
        
        # 文字数による基本圧力
        pressure += len(user_input) / 1000.0
        
        # 質問符の検出
        if '?' in user_input or '？' in user_input:
            pressure += 0.3
        
        # 感情的表現の検出
        emotional_words = ['困っ', '悩ん', '嬉しい', '悲しい', '怒', '不安', '心配']
        for word in emotional_words:
            if word in user_input:
                pressure += 0.5
                break
        
        # 複雑な要求の検出
        complex_words = ['複雑', '難しい', '詳しく', '詳細', '分析', '比較']
        for word in complex_words:
            if word in user_input:
                pressure += 0.4
                break
        
        # 緊急性の検出
        urgent_words = ['急', 'すぐ', '至急', '緊急', 'ASAP']
        for word in urgent_words:
            if word in user_input:
                pressure += 0.6
                break
                
        return min(pressure, 2.0)  # 上限設定
    
    def generate_response(self, user_input: str, context: Optional[str] = None) -> Dict[str, Any]:
        """SSD強化された応答生成"""
        start_time = time.time()
        
        # 1. 意味圧の分析
        meaning_pressure = self.analyze_meaning_pressure(user_input, context)
        
        # 2. SSDステップの実行
        telemetry = SSDTelemetry()
        self.ssd_dll.dll.ssd_step(
            self.ssd_handle, 
            ctypes.c_double(meaning_pressure),
            ctypes.c_double(1.0),  # dt
            ctypes.byref(telemetry)
        )
        
        # 3. 現在のノード（戦略）の取得
        current_node = telemetry.current
        strategy = self.response_strategies[current_node]
        
        # 4. SSD状態に基づく動的パラメータ調整
        dynamic_temp = strategy['temperature']
        if telemetry.did_jump:
            # 跳躍時は創造性を高める
            dynamic_temp = min(dynamic_temp + 0.3, 1.0)
        
        if telemetry.align_eff > 0.8:
            # 高い整合効率時は安定性を重視
            dynamic_temp = max(dynamic_temp - 0.2, 0.1)
        
        # 5. プロンプトの構築
        system_prompt = strategy['system_prompt']
        full_prompt = f"{system_prompt}\n\nユーザー: {user_input}\nAI:"
        
        # 6. LLM推論の実行
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to(self.model.device),
                max_new_tokens=self.config.max_tokens,
                temperature=dynamic_temp,
                top_p=strategy['top_p'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 7. 応答のデコード
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # 8. メトリクスの収集
        generation_time = time.time() - start_time
        
        result = {
            'response': response.strip(),
            'ssd_metadata': {
                'mode_used': 'leap' if telemetry.did_jump else 'alignment',
                'strategy_type': strategy['type'],
                'current_node': current_node,
                'heat_level': telemetry.E,
                'exploration_temp': telemetry.T,
                'alignment_efficiency': telemetry.align_eff,
                'kappa_mean': telemetry.kappa_mean,
                'jump_probability': telemetry.h,
                'did_jump': bool(telemetry.did_jump),
                'rewired_to': telemetry.rewired_to,
                'meaning_pressure': meaning_pressure
            },
            'generation_metadata': {
                'temperature_used': dynamic_temp,
                'top_p_used': strategy['top_p'],
                'generation_time': generation_time,
                'model_used': self.config.model_name.split('/')[-1]
            }
        }
        
        # 9. 履歴の保存
        self.conversation_history.append({
            'user_input': user_input,
            'response': response,
            'timestamp': time.time()
        })
        self.ssd_metrics_history.append(result['ssd_metadata'])
        
        return result
    
    def get_kappa_matrix(self) -> np.ndarray:
        """整合慣性マトリックスの取得"""
        N = self.config.ssd_nodes
        kappa_matrix = np.zeros((N, N))
        
        buffer = (ctypes.c_double * N)()
        for i in range(N):
            count = self.ssd_dll.dll.ssd_get_kappa_row(
                self.ssd_handle, i, buffer, N
            )
            if count > 0:
                kappa_matrix[i, :count] = [buffer[j] for j in range(count)]
        
        return kappa_matrix
    
    def update_ssd_params(self, **kwargs):
        """SSDパラメータの動的更新"""
        for key, value in kwargs.items():
            if hasattr(self.ssd_params, key):
                setattr(self.ssd_params, key, value)
        
        # DLLにパラメータを更新
        self.ssd_dll.dll.ssd_set_params(self.ssd_handle, ctypes.byref(self.ssd_params))
    
    def get_ssd_status(self) -> Dict[str, Any]:
        """現在のSSD状態の取得"""
        if not self.ssd_metrics_history:
            return {"status": "no_data"}
        
        latest = self.ssd_metrics_history[-1]
        kappa_matrix = self.get_kappa_matrix()
        
        return {
            "current_state": {
                "heat_level": latest['heat_level'],
                "exploration_temp": latest['exploration_temp'],
                "current_node": latest['current_node'],
                "alignment_efficiency": latest['alignment_efficiency'],
                "kappa_mean": latest['kappa_mean']
            },
            "statistics": {
                "total_interactions": len(self.conversation_history),
                "total_jumps": sum(1 for m in self.ssd_metrics_history if m['did_jump']),
                "jump_rate": sum(1 for m in self.ssd_metrics_history if m['did_jump']) / max(len(self.ssd_metrics_history), 1),
                "avg_heat_level": sum(m['heat_level'] for m in self.ssd_metrics_history) / max(len(self.ssd_metrics_history), 1),
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
    
    def _calculate_strategy_distribution(self) -> Dict[str, int]:
        """戦略使用分布の計算"""
        distribution = {}
        for metrics in self.ssd_metrics_history:
            node = metrics['current_node']
            strategy_type = self.response_strategies[node]['type']
            distribution[strategy_type] = distribution.get(strategy_type, 0) + 1
        return distribution
    
    def __del__(self):
        """デストラクタでSSDハンドルの解放"""
        if hasattr(self, 'ssd_handle') and self.ssd_handle:
            self.ssd_dll.dll.ssd_destroy(self.ssd_handle)

# =============================================================================
# 使用例とテストコード
# =============================================================================

def main():
    """SSD強化LLMのテスト実行"""
    
    # 設定
    config = SSDLLMConfig(
        model_name="microsoft/DialoGPT-medium",  # 軽量テスト用
        ssd_nodes=8,
        max_tokens=128,
        ssd_dll_path="./ssd_align_leap.dll"
    )
    
    try:
        # SSD-LLM初期化
        print("Initializing SSD-Enhanced LLM...")
        ssd_llm = SSDEnhancedLLM(config)
        
        # テスト会話
        test_inputs = [
            "こんにちは！今日はいい天気ですね。",
            "人工知能の未来について教えてください。", 
            "なぜ空は青いのですか？物理的に詳しく説明してください。",
            "最近仕事で悩んでいます。どうしたらいいでしょうか？",
            "創造的なアイデアを出すコツを教えてください。"
        ]
        
        print("\n" + "="*60)
        print("SSD-Enhanced LLM Test Conversation")
        print("="*60)
        
        for i, user_input in enumerate(test_inputs):
            print(f"\n--- Turn {i+1} ---")
            print(f"User: {user_input}")
            
            # 応答生成
            result = ssd_llm.generate_response(user_input)
            
            print(f"AI [{result['ssd_metadata']['strategy_type']}]: {result['response']}")
            
            # SSDメタデータの表示
            meta = result['ssd_metadata']
            print(f"  └─ Mode: {meta['mode_used']}, Heat: {meta['heat_level']:.2f}, "
                  f"Efficiency: {meta['alignment_efficiency']:.2f}, "
                  f"Node: {meta['current_node']}")
            
            if meta['did_jump']:
                print(f"  └─ 🚀 JUMP occurred! {meta['current_node']} → {meta['rewired_to']}")
        
        # 最終状態の表示
        print("\n" + "="*60)
        print("Final SSD Status")
        print("="*60)
        
        status = ssd_llm.get_ssd_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        
        # 整合慣性マトリックスの可視化
        kappa_matrix = ssd_llm.get_kappa_matrix()
        print(f"\nKappa Matrix:\n{kappa_matrix}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
