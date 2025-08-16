#!/bin/bash
#SBATCH --job-name=sft_qwen32b_fast_math
#SBATCH --partition=P12
#SBATCH --nodes=1
#SBATCH --gres=gpu:8 # GPUが必要な場合
#SBATCH --nodelist=osk-gpu85
#SBATCH --cpus-per-task=40
#SBATCH --time=50:00:00
#SBATCH --output=/home/Competition2025/P12/P12U022/baseline/train/logs/output_fast_math_sft.out
#SBATCH --error=/home/Competition2025/P12/P12U022/baseline/train/logs/error_fast_math_sft.out

################################################################################
# スクリプト名: qwen3-32b_sft_fast_math.sh
# 概要:
#   Qwen-3 32Bモデルに対し、Kaggle Fast-Math-R1 SFTデータセットを使用した
#   教師ありファインチューニング（SFT）を行うためのSBATCHジョブスクリプト。
#
# 目的:
#   Kaggle Fast-Math-R1の高難度数学問題データセット（fast_math_r1_sft.json）を使用して
#   モデルの数学的推論能力を向上させる。このデータセットには、OpenR1 Math、
#   openr1_hard、Light-R1-SFTDataから選別された7900件の「問題・R1トレース・解答」
#   セットが含まれている。
#
# データセット詳細:
#   - OpenR1 Math: R1推論トレースが12,800トークン超かつ正答率50%超のサンプル3000件
#   - openr1_hard: r1-distill-32bが4回試行でも解けなかった難問約2.5k件
#   - Light-R1-SFTData: Light-R1の第2段階データ
#   - 合計: 7900件の重複削除済み高難度データセット
#
# 前提条件:
#   - torchrun、verlなどの必要なライブラリが環境にインストールされていること
#   - Qwen3-32Bモデルのベースモデルが$HOME/model/Qwen3-32Bに存在すること
#   - fast_math_r1_sft.jsonが現在のディレクトリに存在すること
#   - Hugging Face Hubへのアップロードには、HF_TOKEN環境変数が設定されていること
#
# 実行方法:
#   sbatch qwen3-32b_sft_fast_math.sh
#
# 作成者: オリジナルはMetokiさんを参考
# 修正者: Claude Code (Kaggle Fast-Math-R1データセット対応)
# 作成日: 2025-08-10
# 修正日: 2025-08-15
################################################################################

# GPUのID一覧と枚数
echo "CVD=$CUDA_VISIBLE_DEVICES"
NUM_GPUS=$(tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | sed '/^$/d' | wc -l)
echo "NUM_GPUS=$NUM_GPUS"   # ← ここが 8 になるはず

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
export CONDA_PATH="~/conda_env"
conda activate $CONDA_PATH

# Hugging Face 認証
#export HF_TOKEN=<Huggingfaceのトークン>
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" -l 3 > nvidia-smi_fast_math_sft.log &
pid_nvsmi=$!

# エラー時に停止
set -e

# Step 1: Fast-Math-R1 SFT（教師ありファインチューニング）の実行
echo "=== Step 1: Kaggle Fast-Math-R1 Supervised Fine-Tuning (SFT) ==="
echo "Dataset: fast_math_r1_sft.json (7900 high-difficulty math problems)"
echo "Training approach: Full parameter fine-tuning with high-quality R1 reasoning traces"

# ディレクトリ作成（パス統一）
# SFTのチェックポイントを保存するためのディレクトリ（オリジナルと重複しないよう命名）
mkdir -p ~/training/sft_Qwen3_fast_math_r1
mkdir -p ~/training/sft_Qwen3_fast_math_r1/checkpoints
cd ~/training/sft_Qwen3_fast_math_r1

# Use prebuilt parquet dataset
DATA_DIR="/home/user/work/baseline/dataset/data"
echo "Using parquet dataset at $DATA_DIR"
if [ ! -f "$DATA_DIR/fast_math_r1_sft_train.parquet" ] || [ ! -f "$DATA_DIR/fast_math_r1_sft_test.parquet" ]; then
  echo "Parquet files not found in $DATA_DIR"; exit 1
fi

# 基本的なネットワーク設定
export NCCL_SOCKET_IFNAME=enp25s0np0
export NVTE_FUSED_ATTN=0

# Slurm が割り当てた GPU を使う。手動固定はしない
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

unset ROCR_VISIBLE_DEVICES
ulimit -v unlimited
ulimit -m unlimited

# WandBのプロジェクト設定（オリジナルと区別するため命名変更）
export WANDB_ENTITY="yohtanisan"
export WANDB_PROJECT_NAME="Qwen3_32B_Fast_Math_R1_SFT"
export WANDB_RUN_NAME="Qwen3_32B_Fast_Math_R1_SFT"

echo "Starting Fast-Math-R1 SFT training..."
echo "Expected training time: ~5-6 hours (10 epochs on 8 GPUs)"

# hydraエラー回避
export PYTHONPATH=$HOME/conda_env/lib/python3.11/site-packages:$PYTHONPATH

# Fast-Math-R1 SFT学習実行
# Kaggleの第1段階設定を参考にパラメータ調整
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# verlのfsdp_sft_trainerを使用してKaggle Fast-Math-R1データで学習
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=$DATA_DIR/fast_math_r1_sft_train.parquet \
  data.val_files=$DATA_DIR/fast_math_r1_sft_test.parquet \
  data.prompt_key=problem \
  data.response_key=solution \
  data.prompt_dict_keys=[] \
  +data.response_dict_keys=[] \
  data.train_batch_size=32 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=24000 \
  +data.dataloader_num_workers=16 \
  data.truncation=right \
  ++data.filter_overlong_prompts=True \
  model.fsdp_config.model_dtype=bf16 \
  model.lora_rank=64 \
  model.lora_alpha=128 \
  model.partial_pretrain=$HOME/model/Qwen3-32B \
  +model.override_config.attn_implementation=flash_attention_2 \
  +model.use_remove_padding=True \
  +model.use_fused_kernels=True \
  model.enable_gradient_checkpointing=True \
  ++model.fsdp_config.forward_prefetch=True \
  trainer.total_epochs=10 \
  trainer.save_freq=50 \
  trainer.default_local_dir=$HOME/training/sft_Qwen3_fast_math_r1/checkpoints \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT_NAME \
  trainer.experiment_name=$WANDB_RUN_NAME 2>&1 | tee verl_fast_math_r1_sft.log

echo "Fast-Math-R1 SFT training completed"

# Step 2: チェックポイントの変換
echo "=== Step 2: Converting checkpoint to HuggingFace format ==="

# 最新のチェックポイントを探す
LATEST_CHECKPOINT=$(find $HOME/training/sft_Qwen3_fast_math_r1/checkpoints -name "global_step_*" -type d | sort -V | tail -1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found!"
    exit 1
fi

echo "Converting checkpoint: $LATEST_CHECKPOINT"

# Fast-Math-R1専用のhuggingfaceフォルダに変換（オリジナルと区別）
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LATEST_CHECKPOINT \
    --target_dir $LATEST_CHECKPOINT/huggingface_fast_math_r1

echo "Checkpoint conversion completed"

# Step 3: モデルのアップロード（オプション）
echo "=== Step 3: Fast-Math-R1 Model upload (optional) ==="

# HF_TOKENが設定されている場合は自動アップロード（名前を変更してオリジナルと区別）
if [ -n "$HF_TOKEN" ]; then
    echo "Uploading Fast-Math-R1 SFT model to HuggingFace Hub..."
    huggingface-cli upload \
        y-ohtani/Qwen3-32B-SFT-Fast-Math-R1 \
        $LATEST_CHECKPOINT/huggingface_fast_math_r1 \
        --token $HF_TOKEN
    echo "Fast-Math-R1 model upload completed"
else
    echo "HF_TOKEN not set. Upload manually if needed:"
    echo "huggingface-cli upload y-ohtani/Qwen3-32B-SFT-Fast-Math-R1 $LATEST_CHECKPOINT/huggingface_fast_math_r1 --token YOUR_TOKEN"
fi

# GPU監視プロセスを終了
kill $pid_nvsmi

echo "=== Fast-Math-R1 SFT Full Pipeline Completed ==="
echo "End time: $(date)"
echo "Checkpoint location: $LATEST_CHECKPOINT/huggingface_fast_math_r1"
echo ""
echo "Dataset used: Kaggle Fast-Math-R1 SFT (7900 high-difficulty problems)"
echo "Key features:"
echo "  - OpenR1 Math: >12.8k tokens, >50% accuracy samples"
echo "  - openr1_hard: Problems that r1-distill-32b failed in 4 attempts"
echo "  - Light-R1-SFTData: Stage 2 data"
echo "  - Deduplication and shortest token generation selection applied"
echo ""
echo "Next step: Use this model as input for GRPO with Kaggle rewards:"
echo "  Update qwen3-32b_grpo_rewards.sh to use: $LATEST_CHECKPOINT/huggingface_fast_math_r1"