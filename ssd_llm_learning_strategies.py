# SSD-LLM学習戦略統合システム

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import sqlite3
from datetime import datetime
import asyncio

# =============================================================================
# 学習戦略の定義
# =============================================================================

@dataclass
class LearningConfig:
    """学習設定"""
    # LoRA設定
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 強化学習設定  
    rl_learning_rate: float = 1e-4
    rl_buffer_size: int = 10000
    
    # SSD学習設定
    ssd_adaptation_rate: float = 0.01
    feedback_aggregation_window: int = 10
    
    # データ収集設定
    min_feedback_score: float = 0.3
    max_training_samples: int = 1000

class SSDLearningManager:
    """SSD強化LLMの学習マネージャー"""
    
    def __init__(self, ssd_llm, learning_config: LearningConfig):
        self.ssd_llm = ssd_llm
        self.config = learning_config
        
        # 学習データ収集
        self.interaction_database = InteractionDB()
        self.feedback_buffer = []
        
        # LoRA アダプター
        self.lora_adapters = {}
        self._setup_lora_adapters()
        
        # 強化学習コンポーネント
        self.rl_optimizer = torch.optim.Adam(
            self.ssd_llm.model.parameters(), 
            lr=learning_config.rl_learning_rate
        )
        
        # SSD適応システム
        self.ssd_adapter = SSDParameterAdapter(ssd_llm.ssd_handle, learning_config)
        
    def _setup_lora_adapters(self):
        """戦略別LoRAアダプターのセットアップ"""
        strategy_types = ['analytical', 'creative', 'empathetic', 'practical']
        
        for strategy_type in strategy_types:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            # 戦略特化アダプター
            adapter_model = get_peft_model(self.ssd_llm.model, lora_config)
            self.lora_adapters[strategy_type] = adapter_model

# =============================================================================
# 1. インタラクション型学習（最も重要）
# =============================================================================

class InteractionLearning:
    """ユーザーとの対話から継続学習"""
    
    def __init__(self, learning_manager):
        self.learning_manager = learning_manager
        
    async def process_user_feedback(self, 
                                  conversation_id: str,
                                  user_feedback: Dict[str, Any]):
        """ユーザーフィードバックの処理と学習"""
        
        # フィードバックの解析
        feedback_score = self._parse_feedback(user_feedback)
        
        # 会話コンテキストの取得
        conversation = self.learning_manager.interaction_database.get_conversation(conversation_id)
        
        # SSDパラメータの適応
        await self._adapt_ssd_parameters(conversation, feedback_score)
        
        # LoRAファインチューニング
        if feedback_score > 0.7:  # 高評価時
            await self._positive_reinforcement_learning(conversation)
        elif feedback_score < 0.3:  # 低評価時
            await self._negative_feedback_learning(conversation)
    
    def _parse_feedback(self, feedback: Dict[str, Any]) -> float:
        """フィードバックスコアの算出"""
        score = 0.0
        
        # 明示的評価
        if 'rating' in feedback:
            score += feedback['rating'] / 5.0 * 0.5
        
        # 行動による評価
        if feedback.get('continued_conversation', False):
            score += 0.3
        if feedback.get('positive_reaction', False):
            score += 0.2
        if feedback.get('negative_reaction', False):
            score -= 0.4
        
        # テキスト感情分析
        if 'follow_up_message' in feedback:
            sentiment = self._analyze_sentiment(feedback['follow_up_message'])
            score += sentiment * 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _adapt_ssd_parameters(self, conversation, feedback_score):
        """SSDパラメータの適応的調整"""
        last_ssd_state = conversation['ssd_states'][-1]
        
        if feedback_score > 0.7:
            # 成功した戦略の強化
            strategy_node = last_ssd_state['current_node']
            self.learning_manager.ssd_adapter.strengthen_pathway(
                strategy_node, feedback_score
            )
        elif feedback_score < 0.3:
            # 失敗した戦略の弱化と探索促進
            self.learning_manager.ssd_adapter.weaken_current_pathway()
            self.learning_manager.ssd_adapter.increase_exploration_temp(0.1)

# =============================================================================
# 2. LoRAファインチューニング学習
# =============================================================================

class LoRAFineTuner:
    """戦略特化LoRAアダプターのファインチューニング"""
    
    def __init__(self, learning_manager):
        self.learning_manager = learning_manager
        
    async def fine_tune_strategy_adapter(self, 
                                       strategy_type: str, 
                                       training_examples: List[Dict]):
        """特定戦略のLoRAアダプターをファインチューニング"""
        
        # データセットの準備
        dataset = self._prepare_strategy_dataset(training_examples, strategy_type)
        
        # 戦略特化アダプターの取得
        adapter_model = self.learning_manager.lora_adapters[strategy_type]
        
        # トレーニング設定
        training_args = TrainingArguments(
            output_dir=f"./lora_{strategy_type}",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="no",
            warmup_steps=10,
            dataloader_num_workers=0,
        )
        
        # カスタムトレーナー
        trainer = SSDStrategyTrainer(
            model=adapter_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.learning_manager.ssd_llm.tokenizer,
            strategy_type=strategy_type
        )
        
        # ファインチューニング実行
        trainer.train()
        
        # アダプターの保存
        adapter_model.save_pretrained(f"./lora_{strategy_type}")
        
    def _prepare_strategy_dataset(self, examples: List[Dict], strategy_type: str) -> Dataset:
        """戦略特化データセットの準備"""
        
        # 戦略に応じたプロンプト構築
        strategy_prompts = {
            'analytical': "論理的で分析的な観点から回答してください：",
            'creative': "創造的で斬新なアイデアを提案してください：", 
            'empathetic': "共感的で温かい気持ちで応答してください：",
            'practical': "実用的で具体的な解決策を提案してください："
        }
        
        formatted_examples = []
        for example in examples:
            if example['feedback_score'] > 0.7:  # 高品質例のみ
                prompt = strategy_prompts[strategy_type] + example['user_input']
                formatted_examples.append({
                    'input_ids': prompt,
                    'labels': example['ai_response']
                })
        
        return Dataset.from_list(formatted_examples)

class SSDStrategyTrainer(Trainer):
    """SSD戦略特化カスタムトレーナー"""
    
    def __init__(self, strategy_type: str, **kwargs):
        super().__init__(**kwargs)
        self.strategy_type = strategy_type
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """SSD適応型損失関数"""
        
        # 基本言語モデル損失
        outputs = model(**inputs)
        base_loss = outputs.loss
        
        # 戦略一貫性損失の追加
        strategy_consistency_loss = self._compute_strategy_consistency_loss(outputs)
        
        # 総損失
        total_loss = base_loss + 0.1 * strategy_consistency_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _compute_strategy_consistency_loss(self, outputs):
        """戦略一貫性損失の計算"""
        # 戦略特化語彙の重み付け
        strategy_vocab = {
            'analytical': ['分析', '論理', '根拠', '証拠', '結論'],
            'creative': ['アイデア', '発想', '創造', '新しい', '斬新'],
            'empathetic': ['気持ち', '感情', '共感', '理解', '優しい'],
            'practical': ['具体的', '実用', '方法', '手順', '効果的']
        }
        
        target_words = strategy_vocab.get(self.strategy_type, [])
        
        # 出力に戦略語彙がどの程度含まれているかを評価
        # 実際の実装では、vocabularyの確率分布を分析
        consistency_score = 0.0  # 簡略化
        
        return consistency_score

# =============================================================================
# 3. 強化学習（RLHF風）
# =============================================================================

class SSDReinforcementLearning:
    """SSD統合強化学習"""
    
    def __init__(self, learning_manager):
        self.learning_manager = learning_manager
        self.experience_buffer = []
        
    def collect_experience(self, 
                         state: Dict, 
                         action: str, 
                         reward: float, 
                         next_state: Dict):
        """経験の収集"""
        experience = {
            'state': state,
            'action': action, 
            'reward': reward,
            'next_state': next_state,
            'timestamp': datetime.now()
        }
        
        self.experience_buffer.append(experience)
        
        # バッファサイズ制限
        if len(self.experience_buffer) > self.learning_manager.config.rl_buffer_size:
            self.experience_buffer.pop(0)
    
    async def policy_gradient_update(self):
        """ポリシー勾配による更新"""
        
        if len(self.experience_buffer) < 32:  # 最小バッチサイズ
            return
        
        # 経験のバッチサンプリング
        batch = np.random.choice(self.experience_buffer, 32, replace=False)
        
        # 状態価値の計算
        states = [exp['state'] for exp in batch]
        actions = [exp['action'] for exp in batch]
        rewards = [exp['reward'] for exp in batch]
        
        # SSD状態エンコーディング
        state_features = self._encode_ssd_states(states)
        
        # ポリシー勾配の計算
        policy_loss = self._compute_policy_gradient_loss(
            state_features, actions, rewards
        )
        
        # 勾配更新
        self.learning_manager.rl_optimizer.zero_grad()
        policy_loss.backward()
        self.learning_manager.rl_optimizer.step()
    
    def _encode_ssd_states(self, states: List[Dict]) -> torch.Tensor:
        """SSD状態のエンコーディング"""
        features = []
        for state in states:
            feature_vector = [
                state['heat_level'],
                state['exploration_temp'],
                state['alignment_efficiency'],
                state['kappa_mean'],
                float(state['current_node']) / 16.0,  # 正規化
                1.0 if state['did_jump'] else 0.0
            ]
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _compute_policy_gradient_loss(self, states, actions, rewards):
        """ポリシー勾配損失の計算"""
        # 簡略化された実装
        # 実際は、行動選択確率と報酬の関係をモデル化
        
        # アクション価値の推定
        action_values = self._estimate_action_values(states, actions)
        
        # 報酬との比較
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # ポリシー損失
        policy_loss = -torch.mean(action_values * reward_tensor)
        
        return policy_loss

# =============================================================================
# 4. SSDパラメータ適応学習
# =============================================================================

class SSDParameterAdapter:
    """SSDパラメータの動的適応"""
    
    def __init__(self, ssd_handle, learning_config):
        self.ssd_handle = ssd_handle
        self.config = learning_config
        self.adaptation_history = []
        
    def strengthen_pathway(self, node_id: int, feedback_score: float):
        """成功した経路の強化"""
        # DLLパラメータの動的調整
        current_params = self._get_current_params()
        
        # 学習率の一時的増加
        current_params.eta *= (1.0 + feedback_score * self.config.ssd_adaptation_rate)
        
        # 忘却率の一時的減少
        current_params.lam *= (1.0 - feedback_score * self.config.ssd_adaptation_rate * 0.5)
        
        self._update_ssd_params(current_params)
        
        # 適応履歴の記録
        self.adaptation_history.append({
            'action': 'strengthen',
            'node_id': node_id,
            'feedback_score': feedback_score,
            'timestamp': datetime.now()
        })
    
    def weaken_current_pathway(self):
        """現在の経路の弱化"""
        current_params = self._get_current_params()
        
        # 忘却率の一時的増加
        current_params.lam *= 1.2
        
        self._update_ssd_params(current_params)
    
    def increase_exploration_temp(self, delta: float):
        """探索温度の増加"""
        current_params = self._get_current_params()
        current_params.T0 += delta
        self._update_ssd_params(current_params)
    
    def _get_current_params(self):
        """現在のSSDパラメータ取得"""
        # DLLから現在のパラメータを取得
        # 実装は既存のSSDクラスのメソッドを利用
        pass
    
    def _update_ssd_params(self, params):
        """SSDパラメータの更新"""
        # DLLのパラメータ更新関数を呼び出し
        pass

# =============================================================================
# 5. データ収集・管理システム
# =============================================================================

class InteractionDB:
    """対話データの収集・管理"""
    
    def __init__(self, db_path: str = "ssd_interactions.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """データベースの初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_input TEXT,
                ai_response TEXT,
                ssd_state TEXT,
                strategy_type TEXT,
                timestamp DATETIME,
                feedback_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_type TEXT,
                training_data TEXT,
                performance_metrics TEXT,
                timestamp DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_conversation(self, conversation_data: Dict):
        """会話データの保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (id, user_input, ai_response, ssd_state, strategy_type, timestamp, feedback_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            conversation_data['id'],
            conversation_data['user_input'],
            conversation_data['ai_response'],
            json.dumps(conversation_data['ssd_state']),
            conversation_data['strategy_type'],
            datetime.now(),
            conversation_data.get('feedback_score', None)
        ))
        
        conn.commit()
        conn.close()
    
    def get_high_quality_examples(self, 
                                strategy_type: str = None, 
                                min_feedback: float = 0.7,
                                limit: int = 100) -> List[Dict]:
        """高品質な会話例の取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT user_input, ai_response, ssd_state, feedback_score
            FROM conversations 
            WHERE feedback_score >= ?
        '''
        params = [min_feedback]
        
        if strategy_type:
            query += ' AND strategy_type = ?'
            params.append(strategy_type)
        
        query += ' ORDER BY feedback_score DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'user_input': row[0],
                'ai_response': row[1], 
                'ssd_state': json.loads(row[2]),
                'feedback_score': row[3]
            }
            for row in results
        ]

# =============================================================================
# 6. 統合学習オーケストレーター
# =============================================================================

class SSDLearningOrchestrator:
    """全学習プロセスの統合管理"""
    
    def __init__(self, ssd_llm, learning_config: LearningConfig):
        self.ssd_llm = ssd_llm
        self.config = learning_config
        
        # 各学習コンポーネント
        self.learning_manager = SSDLearningManager(ssd_llm, learning_config)
        self.interaction_learning = InteractionLearning(self.learning_manager)
        self.lora_fine_tuner = LoRAFineTuner(self.learning_manager)
        self.rl_learning = SSDReinforcementLearning(self.learning_manager)
        
        # 学習スケジューラー
        self.learning_scheduler = LearningScheduler()
    
    async def continuous_learning_loop(self):
        """継続的学習ループ"""
        while True:
            # 1. 新しいフィードバックの処理
            await self._process_pending_feedback()
            
            # 2. 定期的なLoRAファインチューニング
            if self.learning_scheduler.should_run_lora_training():
                await self._run_lora_training_cycle()
            
            # 3. 強化学習の更新
            if self.learning_scheduler.should_run_rl_update():
                await self.rl_learning.policy_gradient_update()
            
            # 4. SSDパラメータの適応
            await self._adaptive_ssd_tuning()
            
            # 5. パフォーマンス評価
            await self._evaluate_learning_progress()
            
            # 次のサイクルまで待機
            await asyncio.sleep(60)  # 1分間隔
    
    async def _process_pending_feedback(self):
        """保留中フィードバックの処理"""
        pending_feedback = self.learning_manager.interaction_database.get_pending_feedback()
        
        for feedback in pending_feedback:
            await self.interaction_learning.process_user_feedback(
                feedback['conversation_id'],
                feedback['feedback_data']
            )
    
    async def _run_lora_training_cycle(self):
        """LoRAトレーニングサイクル"""
        for strategy_type in ['analytical', 'creative', 'empathetic', 'practical']:
            # 高品質データの取得
            training_examples = self.learning_manager.interaction_database.get_high_quality_examples(
                strategy_type=strategy_type,
                min_feedback=0.7,
                limit=100
            )
            
            if len(training_examples) >= 20:  # 最小訓練データ数
                await self.lora_fine_tuner.fine_tune_strategy_adapter(
                    strategy_type, training_examples
                )

class LearningScheduler:
    """学習スケジューリング"""
    
    def __init__(self):
        self.last_lora_training = datetime.now()
        self.last_rl_update = datetime.now()
    
    def should_run_lora_training(self) -> bool:
        """LoRAトレーニングの実行判定"""
        return (datetime.now() - self.last_lora_training).hours >= 24
    
    def should_run_rl_update(self) -> bool:
        """強化学習更新の実行判定"""
        return (datetime.now() - self.last_rl_update).minutes >= 30

# =============================================================================
# 使用例
# =============================================================================

async def main():
    """学習システムの使用例"""
    
    # 設定
    learning_config = LearningConfig(
        lora_r=16,
        rl_learning_rate=1e-4,
        ssd_adaptation_rate=0.01
    )
    
    # SSD-LLMの初期化（既存のSSDEnhancedLLMを使用）
    ssd_llm = SSDEnhancedLLM(config)
    
    # 学習オーケストレーターの初期化
    learning_orchestrator = SSDLearningOrchestrator(ssd_llm, learning_config)
    
    # 継続的学習の開始
    await learning_orchestrator.continuous_learning_loop()

if __name__ == "__main__":
    asyncio.run(main())
