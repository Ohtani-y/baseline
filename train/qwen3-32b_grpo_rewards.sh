#!/bin/bash
#SBATCH --job-name=grpo_qwen32b_rewards
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu[85]
#SBATCH --cpus-per-task=80
#SBATCH --time=50:00:00
#SBATCH --output=/home/Competition2025/P12/P12U022/baseline/train/logs/output_rewards.out
#SBATCH --error=/home/Competition2025/P12/P12U022/baseline/train/logs/error_rewards.out

################################################################################
# スクリプト名: qwen3-32b_grpo_rewards.sh
# 概要:
#   Qwen-3 32Bモデルに対し、Kaggle Fast-Math-R1報酬関数を使用した
#   強化学習手法GRPO（Generalized Rank-based Policy Optimization）
#   を用いたファインチューニングを行うためのSBATCHジョブスクリプト。
#
# 目的:
#   Kaggle Fast-Math-R1データセットにおけるモデルの推論能力を向上させる。
#   Kaggleの報酬関数システム（format2、cosine、length）により、
#   効率的で高速な推論を実現しながら精度を維持する。
#
# 前提条件:
#   - 環境構築が終わっていること
#   - 学習したいモデルのチェックポイントがきちんと指定されたパスに存在すること。
#   - SBATCHを修正していること
#   - Hugging Face Hubへのアップロードには、`HF_TOKEN`環境変数が設定されていること。
#   - 必要な依存関係: math-verify, latex2sympy2_extended, python-levenshtein
#
# 実行方法:
#   sbatch qwen3-32b_grpo_rewards.sh
#
# 作成者: Metokiさんを参考にさせていただきました。
# 修正者: Claude Code (Kaggle報酬関数移植)
# 作成日: 2025-08-10
# 修正日: 2025-08-15
################################################################################

# 現在のモジュール環境をリセットする（読み込まれている全てのモジュールをアンロード）
module reset

# NCCL（NVIDIA Collective Communications Library）バージョン2.22.3を読み込む
module load nccl/2.22.3

# HPC-X（高性能通信ライブラリ）バージョン2.18.1をCUDA 12およびGCCに対応する構成で読み込む
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

module load miniconda/24.7.1-py311

source /home/appli/miniconda3/24.7.1-py311/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
which conda && echo "====" && conda --version

#step0 でインストールした conda のディレクトリ
export CONDA_PATH="~/conda_env"

source ~/.bashrc

conda init

conda config --set auto_activate_base false

# 念のため既に有効化されているPython仮想環境がある場合に備えてリセットのために無効化する。
conda deactivate
conda deactivate

# 作成したPython仮想環境を有効化。
conda activate $CONDA_PATH

# Hugging Face 認証
#export HF_TOKEN=<Huggingfaceのトークン>
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

# エラー時に停止
set -e

# Step 1-4: 強化学習（GRPO）の実行 - Kaggle報酬関数版
echo "=== Step 1-4: GRPO Training with Kaggle Rewards ==="


# ディレクトリ作成（パス統一）
mkdir -p ~/training/sft_grpo_rewards_001
mkdir -p ~/training/sft_grpo_rewards_001/checkpoints
mkdir -p ~/training/sft_grpo_rewards_001/data
cd ~/training/sft_grpo_rewards_001

# 既存のparquetデータを使用する
DATA_DIR="/home/user/work/baseline/dataset/data"
echo "Using parquet dataset at $DATA_DIR"
if [ ! -f "$DATA_DIR/fast_math_r1_sft_train.parquet" ] || [ ! -f "$DATA_DIR/fast_math_r1_sft_test.parquet" ]; then
  echo "Parquet files not found in $DATA_DIR"; exit 1
fi

# 基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited
ulimit -m unlimited

# Ray クラスターの起動
echo "Starting Ray cluster..."
ray stop  # 既存のRayプロセスを停止
# Rayのヘッドノードを起動
# --num-cpusはノードのCPU数に合わせて調整
# --num-gpusは使用するGPUの数に合わせて調整
ray start --head --port=6379 --num-cpus=240 --num-gpus=8
echo "Ray cluster started"

# 名前は自分のものに修正してください
export WANDB_ENTITY="yohtani"
export WANDB_PROJECT_NAME="Qwen3_32B_SFT+GRPO_Rewards"
export WANDB_RUN_NAME="Qwen3_32B_SFT_MATH_Kaggle_Rewards"

echo "Starting GRPO training with Kaggle reward functions..."

# Fast-Math-R1 SFTチェックポイントの最新を自動検出
LATEST_SFT=$(find $HOME/training/sft_Qwen3_fast_math_r1/checkpoints -name "global_step_*" -type d | sort -V | tail -1)
if [ -z "$LATEST_SFT" ]; then
  echo "No Fast-Math-R1 SFT checkpoint found at $HOME/training/sft_Qwen3_fast_math_r1/checkpoints"
  echo "Please run qwen3-32b_sft_fast_math.sh first or set actor_rollout_ref.model.path manually."
  exit 1
fi
echo "Using SFT checkpoint: $LATEST_SFT/huggingface_fast_math_r1"

# GRPO学習実行 - Kaggle報酬関数版
# actor_rollout_ref.model.pathを学習したいモデルに変更してください
#
# Kaggle Fast-Math-R1報酬関数の設定:
#   - format2: </think>タグ後に答えを提示することを促進
#             (推論完了後に明確な答えを提示、構造化された出力形式)
#   - cosine: 生成長に基づく連続的な報酬スケーリング
#            正解時: 短いほど高報酬 (max=1.0 → min=0.1)
#            不正解時: 長いほどペナルティ軽減 (max=-0.1 → min=-1.0)
#   - length: トークン効率を促進する長さベースの報酬
#            (バッチ内の相対的な長さで計算、過度な熟考を抑制)
#   - weights: 各報酬関数の重み（デフォルト: [1.0, 1.0, 1.0]）
#
# 報酬合成方法:
#   1. 加重和: R_{i,g} = Σ_k w_k · r_{i,g}^{(k)}
#   2. プロンプト単位z正規化: A_{i,g} = (R_{i,g} - mean(R_{i,*})) / std(R_{i,*})
#
# 注意: trainer.total_epochs=15とrollout.n=4は既存設定を維持
#       期待効果: 推論速度30%向上、トークン使用量20-30%削減

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/fast_math_r1_sft_train.parquet \
  data.val_files=$DATA_DIR/fast_math_r1_sft_test.parquet \
  data.prompt_key=problem \
  data.response_key=solution \
  data.train_batch_size=128 \
  data.max_prompt_length=256 \
  data.max_response_length=1024 \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path=$LATEST_SFT/huggingface_fast_math_r1 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  +reward.funcs=['format2','cosine','length'] \
  +reward.weights=[1.0,1.0,1.0] \
  +reward.cosine_max_value_correct=1.0 \
  +reward.cosine_min_value_correct=0.1 \
  +reward.cosine_max_value_wrong=-0.1 \
  +reward.cosine_min_value_wrong=-1.0 \
  +reward.cosine_max_len=30000 \
  +reward.cosine_clip_len=True \
  trainer.logger=['console'] \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.default_local_dir=$HOME/training/sft_grpo_rewards_001/checkpoints \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT_NAME \
  trainer.experiment_name=$WANDB_RUN_NAME \
  trainer.total_epochs=15 2>&1 | tee verl_grpo_rewards.log

echo "GRPO training with Kaggle rewards completed"

# Step 1-5: チェックポイントの変換
echo "=== Step 1-5: Converting checkpoint to HuggingFace format ==="

# 最新のチェックポイントを探す
LATEST_CHECKPOINT=$(find $HOME/training/sft_grpo_rewards_001/checkpoints -name "global_step_*" -type d | sort -V | tail -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found!"
    exit 1
fi

echo "Converting checkpoint: $LATEST_CHECKPOINT"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LATEST_CHECKPOINT/actor \
    --target_dir $LATEST_CHECKPOINT/actor/huggingface

echo "Checkpoint conversion completed"

# Step 1-6: モデルのアップロード（オプション）
echo "=== Step 1-6: Model upload (optional) ==="

# HF_TOKENが設定されている場合は自動アップロード
if [ -n "$HF_TOKEN" ]; then
    echo "Uploading model to HuggingFace Hub..."
    huggingface-cli upload \
        y-ohtani/Qwen3-32B-SFT-GRPO-Rewards \
        $LATEST_CHECKPOINT/actor/huggingface \
        --token $HF_TOKEN
    echo "Model upload completed"
else
    echo "HF_TOKEN not set. Upload manually if needed:"
    echo "huggingface-cli upload y-ohtani/Qwen3-32B-SFT-GRPO-Rewards $LATEST_CHECKPOINT/actor/huggingface --token YOUR_TOKEN"
fi

echo "=== GRPO Full Pipeline with Kaggle Rewards Completed ==="
echo "End time: $(date)"
echo "Checkpoint location: $LATEST_CHECKPOINT/actor/huggingface"
echo ""
echo "Reward functions used:"
echo "  - format2: Ensures answer appears after </think> tag"
echo "  - cosine: Length-based continuous reward scaling"
echo "  - length: Token efficiency promotion"
echo "Expected improvements: 30% faster inference, 20-30% token reduction"

# クリーンアップ: GPU監視とRayを停止
kill $pid_nvsmi || true
ray stop || true