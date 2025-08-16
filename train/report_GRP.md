# Qwen3-32B GRPO: Kaggle報酬版とベース版の比較解説

本資料は、Qwen3-32B を対象とした GRPO（Generalized Rank-based Policy Optimization）学習手順の解説です。
ベース版（`qwen3-32b_grpo.sh`）と Kaggle Fast-Math 報酬版（`qwen3-32b_grpo_rewards.sh`）の違い、使用データ、主要ハイパラ、報酬設計をまとめます。

## 対象スクリプト

- `./qwen3-32b_grpo_rewards.sh`（以下「Kaggle報酬版」）
- `./qwen3-32b_grpo.sh`（以下「ベース版」）

注意: W&B・Hugging Face・ログの保存先に関する記述は意図的に省略しています。

## 概要

- __目的の違い__
  - __ベース版__: GSM8K データセットを用いて一般的な数学推論能力を強化。
  - __Kaggle報酬版__: Fast-Math-R1 SFT を前提に、`format2`/`cosine`/`length` の複合報酬で短く正確な回答とトークン効率を両立。
- 未検討　
  - `data.max_response_length` を 2048〜4096 に拡大。
    併せて `data.train_batch_size` や `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` を縮小してVRAM/計算量を調整。
  - 長さペナルティ弱化


## 学習設定（verl 引数）の主な差分

- __シーケンス長とバッチ設定__
  - __ベース版__: `data.train_batch_size=128`、`data.max_prompt_length=256`、`data.max_response_length=1024`、`actor_rollout_ref.actor.ppo_mini_batch_size=64`、`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`
  - __Kaggle報酬版__: 上記と同一（共通）

- __モデル/並列化・効率化オプション__
  - __共通__: `+actor_rollout_ref.actor.fsdp_config.model_dtype=bf16`、`actor_rollout_ref.rollout.name=vllm`、`actor_rollout_ref.rollout.tensor_model_parallel_size=4`、`actor_rollout_ref.rollout.gpu_memory_utilization=0.8`、`trainer.n_gpus_per_node=8`、`trainer.nnodes=1`、`trainer.val_before_train=False`
  - __Kaggle報酬版の追加__: `+reward.funcs=['format2','cosine','length']`、`+reward.weights=[1.0,1.0,1.0]`、`+reward.cosine_max_value_correct=1.0`、`+reward.cosine_min_value_correct=0.1`、`+reward.cosine_max_value_wrong=-0.1`、`+reward.cosine_min_value_wrong=-1.0`、`+reward.cosine_max_len=30000`、`+reward.cosine_clip_len=True`

## データ準備・前処理の違い

- __Kaggle報酬版の追加処理__
  - `fast_math_r1_sft.json` を作業ディレクトリへコピー。
  - `convert_fast_math_data.py` を用いて JSON を Parquet に変換（`--split-validation` で学習/検証に分割）。
  - 直近の Fast-Math-R1 SFT チェックポイントを自動検出し参照（`$LATEST_SFT/huggingface_fast_math_r1`）。

- __学習用データ指定__（verl 引数）
  - __ベース版__: `$HOME/data/gsm8k/train.parquet` と `$HOME/data/gsm8k/test.parquet`。
  - __Kaggle報酬版__: `./data/fast_math_r1_sft_train_split.parquet` と `./data/fast_math_r1_sft_val.parquet`、`data.prompt_key=problem`、`data.response_key=solution`。

## 報酬設計（Kaggle報酬版）

実装: `train/verl_grpo_rewards.py`

- __format2__
  - 目的: `</think>` より前に `\boxed{...}` の最終解答を提示するフォーマットを促進
  - 判定: 正規表現で `\boxed{...}` が `</think>` の前に存在すれば 1.0、なければ 0.0

- __cosine__
  - 目的: 生成長に応じた連続スケーリングで、短い正解を高評価、長い誤答のペナルティを緩和
  - パラメータ（CLI から変更可）:
    - `+reward.cosine_max_value_correct=1.0`
    - `+reward.cosine_min_value_correct=0.1`
    - `+reward.cosine_max_value_wrong=-0.1`
    - `+reward.cosine_min_value_wrong=-1.0`
    - `+reward.cosine_max_len=30000`
    - `+reward.cosine_clip_len=True`

- __length__
  - 目的: バッチ内の相対長さに基づき、過度な冗長化（熟考のしすぎ）を抑制
  - 正解: 短いほど高い報酬、誤答: ペナルティは 0 以下にクリップ

- __合成方法__（`train/verl_grpo_config.py` の `compute_combined_rewards()`）
  - 加重和: `R_{i,g} = Σ_k w_k · r_{i,g}^{(k)}`
  - プロンプト単位 z 正規化（分散 0 回避に `eps` を加算）
  - 既定重み: 指定なければ各 1.0。CLI から `+reward.weights=[1.0,1.0,1.0]` のように指定可能

- __CLI 設定例（Kaggle報酬版の追加フラグ）__

```bash
+reward.funcs=['format2','cosine','length'] \
+reward.weights=[1.0,1.0,1.0] \
+reward.cosine_max_value_correct=1.0 \
+reward.cosine_min_value_correct=0.1 \
+reward.cosine_max_value_wrong=-0.1 \
+reward.cosine_min_value_wrong=-1.0 \
+reward.cosine_max_len=30000 \
+reward.cosine_clip_len=True \
```

## ベース版とKaggle報酬版の主な差分

- __データ__
  - ベース: GSM8K Parquet
  - Kaggle報酬: Fast-Math-R1（JSON→Parquet変換、`problem`/`solution` 列を使用）

- __参照モデル__
  - ベース: 手動指定の SFT HF 形式ディレクトリ
  - Kaggle報酬: `sft_Qwen3_fast_math_r1` の最新チェックポイントを自動検出

- __報酬__
  - ベース: 既定（明示的な複合報酬指定なし）
  - Kaggle報酬: `format2` + `cosine` + `length` の複合 + 重み指定 + z 正規化

- __SLURM/実行周辺__

  - `--cpus-per-task`: ベース 240 / Kaggle報酬 80（環境に応じて調整）
  - 出力パス・W&B 実験名・チェックポイント保存先が別

- __期待効果__（スクリプト注記）

  - 推論速度 ~30% 向上
  - トークン使用量 20–30% 削減

## 実行方法

- ベース版
  - `sbatch train/qwen3-32b_grpo.sh`
  - 実行前に `actor_rollout_ref.model.path` を自分の SFT チェックポイントへ調整

- Kaggle報酬版
  - `sbatch train/qwen3-32b_grpo_rewards.sh`
  - `fast_math_r1_sft.json` 入手・配置後に実行（スクリプト内で Parquet 変換）

## 成果物

- チェックポイント保存先
  - ベース: `$HOME/training/sft_grpo_001/checkpoints`
  - Kaggle報酬: `$HOME/training/sft_grpo_rewards_001/checkpoints`

- Hugging Face 形式への変換
  - `python -m verl.model_merger merge --backend fsdp --local_dir <...>/actor --target_dir <...>/actor/huggingface`

- 任意の Hub アップロード
  - ベース例: `Ta1k1/Qwen3-32B-SFT-GRPO`
  - Kaggle報酬例: `y-ohtani/Qwen3-32B-SFT-GRPO-Rewards`

## 依存関係（Kaggle報酬版で追加）

- `math-verify`
- `latex2sympy2_extended`
- `python-levenshtein`
