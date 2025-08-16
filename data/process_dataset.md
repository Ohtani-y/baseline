# Fast-Math-R1-SFTデータセット処理ツール

Matsuo Lab Competition 2025向けのFast-Math-R1-SFTデータセット処理ツールです。

## 概要

このツールは、高難度数学問題データセット「Fast-Math-R1-SFT」をVERL（Versatile Reinforcement Learning）フレームワーク対応の形式に変換します。SFT（Supervised Fine-Tuning）とGRPO（Reinforcement Learning from Human Preferences）学習に最適化されたデータセットを生成します。

## 📋 クイックスタート

### 1. 環境準備
```bash
pip install -r requirements.txt
```

### 2. 実行
```bash
python process_dataset.py
```

### 3. 出力確認
```bash
# 生成されるファイル
ls -la data/
├── fast_math_r1_sft_train.parquet      # 訓練用データ
└── fast_math_r1_sft_test.parquet       # テスト用データ
```

## 🔧 設定カスタマイズ

### 主要パラメータ（process_dataset.py:30-36）

```python
MAX_TOKEN_LENGTH = 19000                              # 最大トークン数制限
TEST_RATIO = 0.1                                      # テストデータ比率 (10%)
TRAIN_OUTPUT = "data/fast_math_r1_sft_train.parquet"  # 訓練用出力ファイル
TEST_OUTPUT = "data/fast_math_r1_sft_test.parquet"    # テスト用出力ファイル
TOKENIZER_PATH = "Qwen/Qwen3-0.6B"                    # トークナイザーモデル
RANDOM_SEED = 42                                      # 再現性確保用シード
SHUFFLE_DATA = True                                   # データシャッフル有効化
```





## 🔧 主な機能

### データ変換・正規化
- **HuggingFace Dataset → JSON/Parquet変換**: VERL学習用フォーマット対応
- **`\boxed{}`タグ除去**: LaTeX形式の答えボックスを自動削除
- **`<think>`タグ正規化**: 推論過程の統一フォーマット化
- **フィールド統合**: `generation` + `answer` → `solution`フィールド結合

### 品質管理
- **トークン長フィルタリング**: 設定可能な最大長での除外（デフォルト: 19,000トークン）
- **統計情報表示**: パーセンタイル分析、採用/除外率の詳細レポート
- **エラーハンドリング**: 堅牢なデータ処理とエラー回復

### 学習対応
- **Train/Test分割**: カスタマイズ可能な比率設定
- **データシャッフル**: 再現可能なランダムシード対応
- **Parquet出力**: 高速ロード対応の圧縮フォーマット

## 📊 データセット特徴

### 基本情報
- **サンプル数**: 約45,000問題（高難度数学問題）
- **データサイズ**: 大容量（詳細な推論過程含む）
- **解答特徴**: 複雑な数学問題の段階的思考プロセス

### トークン長分布（推定）
| パーセンタイル | トークン数 | 用途 |
|---------------|-----------|------|
| 50% | ~3,000 | 標準的な問題 |
| 90% | ~9,000 | 複雑な問題 |
| 95% | ~12,000 | 高難度問題 |
| 99% | ~18,000 | 最高難度 |
| 99.5% | ~19,000 | 推奨制限値 |

### 設定指針
| 設定値 | 採用率 | 用途 |
|--------|--------|------|
| 15,000 | ~95% | 高速学習重視 |
| 19,000 | ~99.5% | 品質・量バランス（推奨） |
| 25,000 | ~99.9% | 最大品質重視 |

## 📁 出力データ形式

### JSON構造
```json
{
  "train": [
    {
      "problem": "数学問題文",
      "solution": "<think>\n推論過程\n</think>\n\n\n\nAnswer: 最終答え"
    }
  ],
  "test": [...]
}
```

### Parquet構造
| カラム | 型 | 説明 |
|--------|-----|------|
| problem | string | 問題文 |
| solution | string | 推論過程+答え（`<think>`タグ付き） |

## 🔄 処理フロー

1. **環境チェック**: transformersライブラリの可用性確認
2. **データセットダウンロード**: HuggingFaceから自動取得
3. **データクリーニング**:
   - `\boxed{}`タグ除去（中身は保持）
   - `<think>`タグ正規化（詳細は下記）
   - フィールド統合（generation + answer → solution）
4. **品質フィルタリング**: トークン長制限による除外
5. **分割・シャッフル**: train/test分割とランダム化
6. **出力**: Parquet形式での直接保存

### 🔧 `<think>`タグ正規化の詳細

推論プロセスの統一フォーマット化のため、以下の正規化を実行：

#### 文字変換
```
＜think＞ → <think>     # 全角括弧を半角に変換
＜ /think ＞ → </think> # 全角括弧を半角に変換
```

#### 開始タグの正規化
```
<THINK>     → <think>   # 大文字を小文字に
< think >   → <think>   # スペースを除去
<Think>     → <think>   # 混在ケースを統一
```

#### 終了タグの正規化
```
</THINK>    → </think>  # 大文字を小文字に
< /think >  → </think>  # スペースを除去
< / think > → </think>  # スペースを除去
</Think>    → </think>  # 混在ケースを統一
```

#### 改行の統一化
```
</think>任意の改行 → </think>\n\n\n\n
```
- 終了タグ後に**正確に4つの改行**を強制
- モデルが推論終了を明確に認識できるように統一

### 📝 正規化の実例

#### 変換前（生データ）
```
＜THINK＞
Let me solve this step by step...
< /THINK >

Answer: 42
```

#### 変換後（正規化済み）
```
<think>
Let me solve this step by step...
</think>



Answer: 42
```

### 🎯 正規化が必要な理由

1. **学習効率向上**: 統一されたタグ形式により、モデルが推論構造を効率的に学習
2. **パース精度向上**: 一貫したフォーマットにより、推論開始・終了の認識精度が向上
3. **生成品質向上**: 学習時の一貫性が、推論時の構造化された出力生成に寄与
4. **データ品質管理**: 手動入力による表記揺れを自動修正し、高品質なデータセットを確保

