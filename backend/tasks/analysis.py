"""
分析任务定义

提供 3 个原子任务和 1 个聚合工作流:
1. bert_infer_task     — BERT 推理
2. llm_dialogue_task   — LLM 对话分析
3. llm_verb_task       — LLM 动词分析
4. full_analysis_task  — Chord 并行编排 → 聚合

调用方式:
    from backend.tasks.analysis import full_analysis_task
    result = full_analysis_task.delay(text, book_id)
"""

import time
from celery import chord
from loguru import logger

from backend.tasks.celery_app import celery_app
from backend.services.cache_service import CacheService
from backend.services.metrics_service import metrics
from backend.utils.config import settings


# ── 原子任务 ──────────────────────────────

@celery_app.task(
    name="backend.tasks.analysis.bert_infer_task",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
)
def bert_infer_task(self, text: str, book_id: str) -> dict:
    """BERT 推理任务"""
    from backend.services.bert_service import infer_text

    cache = CacheService()
    cache.set_progress(
        self.request.id,
        status="running",
        progress=0.1,
        current_step="BERT 推理中",
    )

    try:
        with metrics.timed("bert_infer"):
            result = infer_text(text)

        result["book_id"] = book_id
        result["task_type"] = "bert"
        logger.info(f"[BERT] book={book_id} score={result['bert_score']:.4f}")
        return result

    except Exception as exc:
        logger.error(f"[BERT] 任务失败: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    name="backend.tasks.analysis.llm_dialogue_task",
    bind=True,
    max_retries=2,
    default_retry_delay=15,
)
def llm_dialogue_task(self, dialogues: list, book_id: str) -> dict:
    """LLM 对话分析任务"""
    from backend.services.llm_service import analyze_dialogue_block

    cache = CacheService()
    cache.set_progress(
        self.request.id,
        status="running",
        progress=0.1,
        current_step="LLM 对话分析中",
    )

    try:
        with metrics.timed("llm_dialogue"):
            yuri_count = analyze_dialogue_block(dialogues)
            total = len(dialogues)
            score = yuri_count / total if total > 0 else 0.0

        result = {
            "book_id": book_id,
            "task_type": "dialogue",
            "yuri_count": yuri_count,
            "total_count": total,
            "dialogue_score": round(score, 4),
        }
        logger.info(
            f"[LLM_Dialogue] book={book_id} "
            f"yuri={yuri_count}/{total} score={score:.4f}"
        )
        return result

    except Exception as exc:
        logger.error(f"[LLM_Dialogue] 任务失败: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    name="backend.tasks.analysis.llm_verb_task",
    bind=True,
    max_retries=2,
    default_retry_delay=15,
)
def llm_verb_task(self, sentences: list, book_id: str) -> dict:
    """LLM 动词分析任务"""
    from backend.services.llm_service import analyze_verb_block

    cache = CacheService()
    cache.set_progress(
        self.request.id,
        status="running",
        progress=0.1,
        current_step="LLM 动词分析中",
    )

    try:
        with metrics.timed("llm_verb"):
            yuri_count = analyze_verb_block(sentences)
            total = len(sentences)
            score = yuri_count / total if total > 0 else 0.0

        result = {
            "book_id": book_id,
            "task_type": "verb",
            "yuri_count": yuri_count,
            "total_count": total,
            "verb_score": round(score, 4),
        }
        logger.info(
            f"[LLM_Verb] book={book_id} "
            f"yuri={yuri_count}/{total} score={score:.4f}"
        )
        return result

    except Exception as exc:
        logger.error(f"[LLM_Verb] 任务失败: {exc}")
        raise self.retry(exc=exc)


# ── 聚合回调 ──────────────────────────────

@celery_app.task(
    name="backend.tasks.analysis.aggregate_results",
    bind=True,
)
def aggregate_results(self, results: list, book_id: str, task_id: str) -> dict:
    """
    聚合 BERT / dialogue / verb 的结果，计算加权分数

    权重: BERT 0.4 + dialogue 0.35 + verb 0.25
    """
    from backend.services.llm_service import generate_comprehensive_analysis

    cache = CacheService()
    cache.set_progress(
        task_id, status="running", progress=0.8, current_step="聚合分析中"
    )

    bert_score = 0.0
    dialogue_score = 0.0
    verb_score = 0.0

    for r in results:
        if r is None:
            continue
        if r.get("task_type") == "bert":
            bert_score = r.get("bert_score", 0.0)
        elif r.get("task_type") == "dialogue":
            dialogue_score = r.get("dialogue_score", 0.0)
        elif r.get("task_type") == "verb":
            verb_score = r.get("verb_score", 0.0)

    # 加权评分
    w_bert = float(settings.llm.weights.get("bert", 0.4))
    w_dialogue = float(settings.llm.weights.get("dialogue", 0.35))
    w_verb = float(settings.llm.weights.get("verb", 0.25))

    final_score = (
        w_bert * bert_score
        + w_dialogue * dialogue_score
        + w_verb * verb_score
    )

    # LLM 综合分析
    with metrics.timed("comprehensive_analysis"):
        analysis = generate_comprehensive_analysis(
            book_id=book_id,
            bert_score=bert_score,
            dialogue_score=dialogue_score,
            verb_score=verb_score,
            final_score=final_score,
        )

    final_result = {
        "book_id": book_id,
        "bert_score": bert_score,
        "dialogue_score": dialogue_score,
        "verb_score": verb_score,
        "final_score": round(final_score, 4),
        "weights": {"bert": w_bert, "dialogue": w_dialogue, "verb": w_verb},
        "analysis": analysis,
    }

    # 写回缓存
    cache.set_task_result(task_id, final_result)
    cache.set_progress(
        task_id, status="completed", progress=1.0, current_step="分析完成"
    )

    logger.info(
        f"[聚合] book={book_id} final_score={final_score:.4f} "
        f"BERT={bert_score:.4f} dialogue={dialogue_score:.4f} verb={verb_score:.4f}"
    )
    return final_result


# ── 编排入口 ──────────────────────────────

@celery_app.task(
    name="backend.tasks.analysis.full_analysis_task",
    bind=True,
)
def full_analysis_task(
    self,
    text: str,
    dialogues: list,
    sentences: list,
    book_id: str,
) -> str:
    """
    完整分析流程：
    1. 并行发起 BERT / dialogue / verb 三个子任务
    2. 使用 Celery chord 在全部完成后触发聚合
    3. 返回编排后的 task_id

    Args:
        text:       全文文本（供 BERT 分析）
        dialogues:  对话列表（供 LLM 对话分析）
        sentences:  动作描写列表（供 LLM 动词分析）
        book_id:    书籍 ID

    Returns:
        聚合任务的 task_id
    """
    task_id = self.request.id
    cache = CacheService()
    cache.set_progress(
        task_id, status="running", progress=0.0, current_step="任务编排中"
    )

    # 构建 chord：三路并行 → 聚合回调
    header = [
        bert_infer_task.s(text, book_id),
        llm_dialogue_task.s(dialogues, book_id),
        llm_verb_task.s(sentences, book_id),
    ]
    callback = aggregate_results.s(book_id=book_id, task_id=task_id)

    workflow = chord(header)(callback)

    cache.set_progress(
        task_id, status="running", progress=0.05, current_step="子任务已分发"
    )

    logger.info(
        f"[编排] book={book_id} task_id={task_id} → 3 子任务已分发"
    )
    return task_id
