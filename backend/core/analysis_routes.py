"""
分析相关路由  /api/v1/analysis/*

POST /submit          — 提交分析任务
POST /batch           — 批量提交
GET  /progress/{id}   — 查询进度
GET  /result/{id}     — 获取结果
"""

from fastapi import APIRouter, HTTPException, status

from backend.models.schemas import (
    AnalysisRequest,
    BatchAnalysisRequest,
    TaskResponse,
    TaskStatus,
    ProgressResponse,
    TaskResultResponse,
    AnalysisResult,
    ScoreDetail,
    YuriLevel,
)
from backend.services.cache_service import CacheService
from backend.tasks.analysis import full_analysis_task
from backend.utils.config import settings

router = APIRouter(prefix="/analysis", tags=["分析"])

cache = CacheService()


def _score_to_level(score: float) -> YuriLevel:
    if score >= 0.9:
        return YuriLevel.SUPER_HEAVY
    elif score >= 0.7:
        return YuriLevel.HEAVY
    elif score >= 0.5:
        return YuriLevel.MEDIUM
    elif score >= 0.3:
        return YuriLevel.LIGHT
    return YuriLevel.MICRO


# ── 提交分析 ──────────────────────────────

@router.post("/submit", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_analysis(req: AnalysisRequest):
    """提交一个分析任务（异步）"""
    # 查缓存
    text_hash = cache.text_hash(req.text)
    cached = cache.get_cached_result(text_hash)
    if cached:
        return TaskResponse(
            task_id=cached.get("task_id", "cached"),
            status=TaskStatus.COMPLETED,
            message="分析结果已缓存",
        )

    # 发起 Celery 任务
    task = full_analysis_task.delay(
        text=req.text,
        dialogues=req.dialogues,
        sentences=req.sentences,
        book_id=req.book_id,
    )
    return TaskResponse(task_id=task.id, status=TaskStatus.PENDING)


@router.post("/batch", response_model=list[TaskResponse], status_code=status.HTTP_202_ACCEPTED)
async def batch_analysis(req: BatchAnalysisRequest):
    """批量提交分析任务"""
    responses = []
    for item in req.items:
        task = full_analysis_task.delay(
            text=item.text,
            dialogues=item.dialogues,
            sentences=item.sentences,
            book_id=item.book_id,
        )
        responses.append(TaskResponse(task_id=task.id, status=TaskStatus.PENDING))
    return responses


# ── 查询进度 ──────────────────────────────

@router.get("/progress/{task_id}", response_model=ProgressResponse)
async def get_progress(task_id: str):
    """查询任务进度"""
    progress = cache.get_progress(task_id)
    if progress:
        return ProgressResponse(
            task_id=task_id,
            status=TaskStatus(progress["status"]),
            progress=progress["progress"],
            current_step=progress.get("current_step", ""),
            updated_at=progress.get("updated_at"),
        )

    if settings.llm.mock_mode:
        return ProgressResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            progress=0.0,
            current_step="mock 模式，状态模拟",
        )

    # 回退到查 Celery AsyncResult
    from backend.tasks.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)
    if result.state == "PENDING":
        return ProgressResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            progress=0.0,
            current_step="等待执行",
        )
    elif result.state == "STARTED":
        return ProgressResponse(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            progress=0.1,
            current_step="正在执行",
        )
    elif result.state == "SUCCESS":
        return ProgressResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            progress=1.0,
            current_step="已完成",
        )
    elif result.state == "FAILURE":
        return ProgressResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            progress=0.0,
            current_step=str(result.info),
        )

    return ProgressResponse(
        task_id=task_id,
        status=TaskStatus.RUNNING,
        progress=0.5,
        current_step=result.state,
    )


# ── 获取结果 ──────────────────────────────

@router.get("/result/{task_id}", response_model=TaskResultResponse)
async def get_result(task_id: str):
    """获取任务最终结果"""
    # 先查 Redis 缓存
    cached_result = cache.get_task_result(task_id)
    if cached_result:
        scores = ScoreDetail(
            bert_score=cached_result.get("bert_score", 0),
            dialogue_score=cached_result.get("dialogue_score", 0),
            verb_score=cached_result.get("verb_score", 0),
            final_score=cached_result.get("final_score", 0),
            weights=cached_result.get("weights", {}),
        )
        return TaskResultResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=AnalysisResult(
                book_id=cached_result.get("book_id", ""),
                scores=scores,
                level=_score_to_level(scores.final_score),
                analysis=cached_result.get("analysis"),
            ),
        )

    # Celery AsyncResult 回退
    from backend.tasks.celery_app import celery_app

    ar = celery_app.AsyncResult(task_id)
    if ar.state == "SUCCESS" and ar.result:
        r = ar.result
        scores = ScoreDetail(
            bert_score=r.get("bert_score", 0),
            dialogue_score=r.get("dialogue_score", 0),
            verb_score=r.get("verb_score", 0),
            final_score=r.get("final_score", 0),
            weights=r.get("weights", {}),
        )
        return TaskResultResponse(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            result=AnalysisResult(
                book_id=r.get("book_id", ""),
                scores=scores,
                level=_score_to_level(scores.final_score),
                analysis=r.get("analysis"),
            ),
        )
    elif ar.state == "FAILURE":
        return TaskResultResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=str(ar.info),
        )
    elif ar.state in ("PENDING", "STARTED"):
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail="任务尚未完成，请稍后再试",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"未找到任务 {task_id}",
    )
