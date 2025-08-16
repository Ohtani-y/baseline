"""
Fast-Math-R1-SFTデータセット処理ツール（シンプル版）

使用方法:
    python process_dataset.py

すべての設定は先頭の定数で変更可能
"""

import json
import random
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from datasets import load_dataset
from sklearn.model_selection import train_test_split

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformersライブラリが見つかりません。pip install transformersを実行してください。")

# ========================
# 主要設定（ユーザー変更可能）
# ========================
MAX_TOKEN_LENGTH = 19000  # 問題文＋解答の最大トークン数（約0.5%除外、99.5%採用）
TEST_RATIO = 0.1  # テストデータの比率（0.1 = 10%）
TRAIN_OUTPUT = "data/fast_math_r1_sft_train.parquet"  # 訓練用出力ファイル
TEST_OUTPUT = "data/fast_math_r1_sft_test.parquet"   # テスト用出力ファイル
TOKENIZER_PATH = "Qwen/Qwen3-0.6B"  # トークナイザーのパス（超軽量で高性能）
RANDOM_SEED = 42  # train/test分割のシード値
SHUFFLE_DATA = True  # データをシャッフルするかどうか


def find_matching_brace(text: str, start_idx: int) -> int:
    """Given text and the index of the opening '{', return index of matching '}' or -1."""
    depth = 0
    for i in range(start_idx, len(text)):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1


def remove_boxed_wrappers(s: str) -> str:
    """Remove all occurrences of \\boxed{...} (case-insensitive) keeping inner content."""
    pattern = r"\\boxed\s*\{"
    while True:
        m = re.search(pattern, s, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            break
        open_brace_idx = m.end() - 1
        close_brace_idx = find_matching_brace(s, open_brace_idx)
        if close_brace_idx == -1:
            s = s[: m.start()] + s[m.end():]
            continue
        inner = s[open_brace_idx + 1 : close_brace_idx]
        s = s[: m.start()] + inner + s[close_brace_idx + 1 :]
    return s


def normalize_think_tags(s: str) -> str:
    """Normalize THINK tags to lowercase and enforce four newlines after closing tag."""
    # Convert fullwidth brackets to ASCII
    s = s.replace("＜", "<").replace("＞", ">")
    # Normalize opening tag
    s = re.sub(r"<\s*think\s*>", "<think>", s, flags=re.IGNORECASE)
    # Normalize closing tag
    s = re.sub(r"<\s*/\s*think\s*>", "</think>", s, flags=re.IGNORECASE)
    # Ensure exactly four newlines after </think>
    s = re.sub(r"</think>\s*", "</think>\n\n\n\n", s)
    return s


def fix_format(text: str) -> str:
    """テキストを正しいフォーマットに修正"""
    if not isinstance(text, str):
        text = str(text)
    
    if not text.startswith("<think>"):
        text = "<think>" + text
    
    text = re.sub(r"</think>[\n]*$", "", text)
    text = text + "</think>\n\n\n\n"
    
    # 深いクリーニング
    text = remove_boxed_wrappers(text)
    text = normalize_think_tags(text)
    
    return text


def main():
    """メイン処理"""
    print("=== Fast-Math-R1-SFTデータセット処理 ===\n")
    
    # トークナイザーの準備
    tokenizer = None
    if TRANSFORMERS_AVAILABLE:
        try:
            print(f"トークナイザーを読み込み中: {TOKENIZER_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_PATH, 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"警告: トークナイザーの読み込みに失敗しました: {e}")
            print("トークン長フィルタリングをスキップします。\n")
    else:
        print("transformersライブラリが利用できません。トークン長フィルタリングをスキップします。\n")
    
    # データセットをダウンロード
    print("データセットをダウンロード中...")
    dataset = load_dataset("RabotniKuma/Fast-Math-R1-SFT")
    print(f"データセット構造: {dataset}\n")
    
    # データを処理
    processed_data = []
    removed_count = 0
    token_lengths = []
    max_adopted_tokens = 0
    max_rejected_tokens = 0
    total_samples = 0
    
    print("データを処理中...")
    
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        total_samples += len(split_data)
        
        for idx, item in enumerate(split_data):
            # フォーマット修正と統合
            generation = item.get("generation", "")
            answer = item.get("answer", "")
            solution = fix_format(generation) + f"Answer: {answer}"
            
            # トークン長計算
            if tokenizer:
                problem = item.get("problem", "")
                combined = problem + "\n" + solution
                try:
                    tokens = tokenizer(combined, add_special_tokens=False)["input_ids"]
                    token_length = len(tokens)
                    token_lengths.append(token_length)
                    
                    # トークン長チェック
                    if token_length > MAX_TOKEN_LENGTH:
                        removed_count += 1
                        max_rejected_tokens = max(max_rejected_tokens, token_length)
                        continue
                    else:
                        max_adopted_tokens = max(max_adopted_tokens, token_length)
                except Exception as e:
                    print(f"警告: トークン化エラー (サンプル {idx}): {e}")
                    continue
            
            # 最終データ構造
            processed_item = {
                "problem": item.get("problem", ""),
                "solution": solution
            }
            processed_data.append(processed_item)
            
            # 進捗表示
            if (idx + 1) % 500 == 0:
                print(f"  処理済み: {idx + 1}/{len(split_data)}件")
    
    # 統計情報の表示
    print(f"  処理完了: {len(processed_data)}件")
    
    if tokenizer and token_lengths:
        print(f"\n=== トークン長統計 ===")
        print(f"総サンプル数: {total_samples:,}件")
        print(f"採用サンプル数: {len(processed_data):,}件 ({len(processed_data)/total_samples*100:.1f}%)")
        print(f"除外サンプル数: {removed_count:,}件 ({removed_count/total_samples*100:.1f}%)")
        print(f"")
        print(f"トークン長範囲:")
        print(f"  - 全体の最小: {min(token_lengths):,}トークン")
        print(f"  - 全体の最大: {max(token_lengths):,}トークン")
        print(f"  - 採用済み最大: {max_adopted_tokens:,}トークン")
        if max_rejected_tokens > 0:
            print(f"  - 除外済み最大: {max_rejected_tokens:,}トークン")
        print(f"  - 制限値: {MAX_TOKEN_LENGTH:,}トークン")
        
        # パーセンタイル情報
        sorted_lengths = sorted(token_lengths)
        p50 = sorted_lengths[len(sorted_lengths)//2]
        p90 = sorted_lengths[int(len(sorted_lengths)*0.9)]
        p95 = sorted_lengths[int(len(sorted_lengths)*0.95)]
        p99 = sorted_lengths[int(len(sorted_lengths)*0.99)]
        p995 = sorted_lengths[int(len(sorted_lengths)*0.995)]
        
        print(f"")
        print(f"トークン長分布:")
        print(f"  - 50%タイル: {p50:,}トークン")
        print(f"  - 90%タイル: {p90:,}トークン")
        print(f"  - 95%タイル: {p95:,}トークン")
        print(f"  - 99%タイル: {p99:,}トークン")
        print(f"  - 99.5%タイル: {p995:,}トークン ← 0.5%除外する場合の推奨制限値")
        print()
    
    # データのシャッフル
    if SHUFFLE_DATA:
        print("データをシャッフル中...")
        random.seed(RANDOM_SEED)
        random.shuffle(processed_data)
    
    # train/test分割
    print(f"データを分割中... (全{len(processed_data)}件)")
    train_data, test_data = train_test_split(
        processed_data, 
        test_size=TEST_RATIO, 
        random_state=RANDOM_SEED,
        shuffle=False  # すでにシャッフル済み
    )
    
    # dataディレクトリ作成
    Path("data").mkdir(exist_ok=True)
    
    print("\nParquet形式で保存中...")
    try:
        # 訓練データをParquetで保存
        df_train = pd.DataFrame(train_data)
        df_train.to_parquet(TRAIN_OUTPUT, index=False)
        print(f"訓練データ保存完了: {TRAIN_OUTPUT} ({len(df_train)}件)")
        
        # テストデータをParquetで保存
        df_test = pd.DataFrame(test_data)
        df_test.to_parquet(TEST_OUTPUT, index=False)
        print(f"テストデータ保存完了: {TEST_OUTPUT} ({len(df_test)}件)")
        
    except Exception as e:
        print(f"エラー: Parquetファイル保存に失敗しました: {e}")
        return
    
    # 結果表示
    print(f"\n=== 処理完了 ===")
    print(f"訓練ファイル: {TRAIN_OUTPUT}")
    print(f"テストファイル: {TEST_OUTPUT}")
    print(f"trainデータ: {len(train_data):,}件 ({len(train_data)/len(processed_data)*100:.1f}%)")
    print(f"testデータ: {len(test_data):,}件 ({len(test_data)/len(processed_data)*100:.1f}%)")
    
    print(f"\n処理設定:")
    print(f"  - カラム統合（generation + answer → solution）: 有効")
    print(f"  - \\boxed{{}}除去: 有効")
    print(f"  - <think>タグ正規化: 有効")
    print(f"  - データシャッフル: {'有効' if SHUFFLE_DATA else '無効'}")
    if tokenizer:
        print(f"  - トークナイザー: {TOKENIZER_PATH}")
        print(f"  - 最大トークン長制限: {MAX_TOKEN_LENGTH:,}トークン")
    else:
        print(f"  - トークン長制限: 無効（トークナイザー未指定）")
    
    # サンプル表示
    if len(train_data) > 0:
        print(f"\n=== サンプル (最初の2件) ===")
        for i, item in enumerate(train_data[:2], 1):
            print(f"\n{i}. problem: {item['problem'][:100]}...")
            print(f"   solution: {item['solution'][:100]}...")
            print(f"            ...{item['solution'][-50:]}")


if __name__ == "__main__":
    main()



