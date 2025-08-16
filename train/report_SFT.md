# Qwen3-32B SFT: Fast-Math版とベース版の比較解説

対象スクリプト:
- `./qwen3-32b_sft_fast_math.sh`（以下「Fast-Math版」）
- `./qwen3-32b_sft.sh`（以下「ベース版」）

注意: W&B・Hugging Face・ログの保存先に関する記述は意図的に省略しています。

## 概要
- __目的の違い__
  - __ベース版__: 一般的な MATH データを用いた SFT。
  - __Fast-Math版__: Kaggle Fast-Math-R1（高難度問題＋R1推論トレース）専用設定。長文トレース前提の学習に最適化。

## データ準備・前処理の違い
- __Fast-Math版の追加処理__
  - `fast_math_r1_sft.cleaned.json` を作業ディレクトリへコピー。
  - `convert_fast_math_data.py` を用いて JSON を Parquet に変換（`--split-validation` で学習/検証に分割）。
- __学習用データ指定__（torchrun引数）
  - __ベース版__: `./data/math/train.parquet` と `./data/math/test.parquet`。
  - __Fast-Math版__: `./data/fast_math_r1_sft_train_split.parquet` と `./data/fast_math_r1_sft_val.parquet`。

## 学習設定（torchrun 引数）の主な差分
- __シーケンス長とバッチ設定__
  - __ベース版__: `data.max_length=4096`、`data.micro_batch_size_per_gpu=4`。
  - __Fast-Math版__:
    - `data.max_length=24000`（長文R1トレース対応）。
    - `data.train_batch_size=32`、`data.micro_batch_size_per_gpu=1`。
    - `+data.dataloader_num_workers=16`、`data.truncation=right`、`++data.filter_overlong_prompts=True`。
- __モデル/並列化・効率化オプション__
  - __ベース版__:
    - `'+model.attn_implementation=flash_attention_2'`、`'+model.torch_dtype=bfloat16'`。
    - `model.fsdp_config.model_dtype=bfloat16`、`model.use_liger=True`。
    - `model.enable_gradient_checkpointing=true`。
  - __Fast-Math版__:
    - `+model.override_config.attn_implementation=flash_attention_2`。
    - `model.fsdp_config.model_dtype=bf16`、`model.enable_gradient_checkpointing=True`。
    - 長文・効率化: `+model.use_remove_padding=True`、`+model.use_fused_kernels=True`、`++model.fsdp_config.forward_prefetch=True`。
    - LoRA: `model.lora_rank=64`、`model.lora_alpha=128`。
- __エポック等の制御__
  - __ベース版__: `trainer.total_epochs` が重複指定（`4` と `15`。最終的には `15` が有効）。
  - __Fast-Math版__: `trainer.total_epochs=10`、`trainer.save_freq=50`（保存頻度を明示）。

## Fast-Math版の具体的な挙動（前処理・効率化）
- __前処理（`./preprocess_fast_math_sft.py`）__
  - THINKタグを正規化: `<think>` と `</think>` に統一し、`</think>` の直後は「ちょうど4つの改行」に揃える。
  - `\boxed{...}` ラッパーを除去し、中身のみ残す。
  - トークン長フィルタ: `problem + solution` の合計トークンが上限を超えるサンプルは除外（既定: 24000）。
- __変換（`./convert_fast_math_data.py`）__
  - JSON→Parquet 変換と学習/検証分割を行う。
  - テキスト内容は変更しない。`answer` 列の抽出を試みる（`\boxed{...}` が既に除去されている場合は空になる可能性あり）。
- __学習時の効率化フラグ（`./qwen3-32b_sft_fast_math.sh`）__
  - `+model.use_remove_padding=True`: バッチ内のPADトークンを計算から事前に除去し、実トークンのみで計算して無駄なFLOPs/メモリアクセスを削減。
  - `+model.use_fused_kernels=True`: Bias+Dropout+ResidualやLayerNorm等を融合カーネルで実行し、カーネル起動回数・中間メモリアクセスを削減。
  - `++model.fsdp_config.forward_prefetch=True`: FSDPで次層パラメータのall-gatherを先読みし、フォワード計算と通信を重ねてステップ時間を短縮。

---
本書は、学習内容（データ処理・ハイパーパラメータ・実行時の主要フラグ）の相違点に焦点を当てています。W&B、Hugging Face、ログの保存先に関する事項は記載していません。
