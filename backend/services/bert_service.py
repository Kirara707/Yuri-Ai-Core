"""
BERT 推理服务

封装原始 script/model_BERT_infer.py 的核心逻辑，提供函数式接口。
支持：
- 滑动窗口分割长文本
- HuggingFace Trainer 批量推理
- 按 book_id / vol_id 聚合分数
"""

import os
import time
import glob
from typing import List, Dict, Optional

import torch
import pandas as pd
from loguru import logger

from backend.utils.config import settings


# ── 全局模型缓存（进程内单例）────────────
_tokenizer = None
_model = None
_device = None


def _ensure_model_loaded():
    """延迟加载 BERT 模型，只在首次调用时加载"""
    global _tokenizer, _model, _device

    if _model is not None:
        return

    from transformers import BertTokenizerFast, BertForSequenceClassification

    ckpt = settings.bert.abs_model_path
    _device = torch.device(
        settings.bert.device if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"加载 BERT 模型: {ckpt} → {_device}")
    t0 = time.time()

    _tokenizer = BertTokenizerFast.from_pretrained(ckpt, local_files_only=True)
    _model = BertForSequenceClassification.from_pretrained(
        ckpt, local_files_only=True
    ).to(_device)
    _model.eval()

    logger.info(f"BERT 模型加载完毕 耗时={time.time()-t0:.2f}s")


# ── 工具函数 ──────────────────────────────


def slide_window(text: str) -> List[List[int]]:
    """
    将长文本用滑动窗口切分为多个 token 序列

    Returns: [[cls, tok, ..., sep], ...]
    """
    _ensure_model_loaded()
    max_len = int(settings.bert.max_len)
    stride = int(settings.bert.stride)

    tokens = _tokenizer.encode(text, add_special_tokens=False)
    windows = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i : i + max_len - 2]
        if len(chunk) < 30:
            continue
        windows.append(
            [_tokenizer.cls_token_id] + chunk + [_tokenizer.sep_token_id]
        )
    return windows


def extract_golden_sentence(text: str, max_chars: int = 100) -> str:
    """截取文本中间的 "金句" 片段"""
    text = text.replace(" ", "").replace("\n", "")
    if len(text) <= max_chars:
        return text
    mid = len(text) // 2
    start = max(0, mid - max_chars // 2)
    end = min(len(text), mid + max_chars // 2)
    return "..." + text[start:end] + "..."


# ── 核心推理接口 ──────────────────────────


def infer_text(text: str) -> Dict:
    """
    对单段文本进行 BERT 推理

    Args:
        text: 原始文本内容

    Returns:
        {
            "bert_score": float,          # 平均概率
            "window_scores": [float, ...], # 每个窗口的概率
            "window_count": int,
            "duration_ms": float,
        }
    """
    from transformers import (
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    from datasets import Dataset

    _ensure_model_loaded()
    t0 = time.time()

    windows = slide_window(text)
    if not windows:
        return {
            "bert_score": 0.0,
            "window_scores": [],
            "window_count": 0,
            "duration_ms": 0.0,
        }

    pool = [{"input_ids": w} for w in windows]
    ds = Dataset.from_list(pool)
    max_len = int(settings.bert.max_len)
    ds = ds.map(lambda x: {"input_ids": x["input_ids"][:max_len]}, num_proc=1)
    ds.set_format(type="torch", columns=["input_ids"])

    args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=int(settings.bert.batch_size),
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=_model,
        args=args,
        tokenizer=_tokenizer,
        data_collator=DataCollatorWithPadding(_tokenizer),
    )

    logits = trainer.predict(ds).predictions
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].tolist()

    duration = (time.time() - t0) * 1000
    avg_score = sum(probs) / len(probs) if probs else 0.0

    logger.info(
        f"BERT 推理完成 | windows={len(probs)} avg_score={avg_score:.4f} "
        f"duration={duration:.0f}ms"
    )

    return {
        "bert_score": avg_score,
        "window_scores": probs,
        "window_count": len(probs),
        "duration_ms": duration,
    }


def infer_text_files(input_dir: str = None) -> pd.DataFrame:
    """
    批量推理某目录下所有 .txt 文件（兼容旧流程 script/model_BERT_infer.py）

    Returns:
        DataFrame: columns=[filename, book_id, vol_id, pred_prob]
    """
    input_dir = input_dir or settings.paths.txt_test_cleaned_dir
    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))

    if not txt_files:
        logger.warning(f"目录为空: {input_dir}")
        return pd.DataFrame(columns=["filename", "book_id", "vol_id", "pred_prob"])

    results = []
    for f in txt_files:
        fname = os.path.basename(f)[:-4]
        text = open(f, encoding="utf-8", errors="ignore").read()
        res = infer_text(text)

        # 解析 book_id / vol_id
        if "_" in fname:
            parts = fname.split("_")
            book_id, vol_id = int(parts[0]), int(parts[1])
        else:
            try:
                book_id = int(fname)
            except ValueError:
                book_id = 0
            vol_id = 1

        results.append(
            {
                "filename": fname,
                "book_id": book_id,
                "vol_id": vol_id,
                "pred_prob": res["bert_score"],
            }
        )

    return pd.DataFrame(results)
