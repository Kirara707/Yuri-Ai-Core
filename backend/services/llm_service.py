"""
LLM 推理服务

封装 Moonshot (Kimi) API 调用，集成:
- 指数退避重试
- 熔断器保护
- Redis 缓存

提供两类分析:
1. 对话分析（dialogue）：统计百合氛围对话占比
2. 动作分析（verb）：统计百合心理/动作描写占比
"""

import re
import time
import json
from typing import Optional

from openai import OpenAI
from loguru import logger

from backend.utils.config import settings
from backend.utils.retry_decorator import exponential_backoff_retry
from backend.utils.circuit_breaker import CircuitBreaker


# ── 全局熔断器实例 ──────────────────────
llm_circuit_breaker = CircuitBreaker(
    name="moonshot_api",
    failure_threshold=settings.circuit_breaker.failure_threshold,
    success_threshold=settings.circuit_breaker.success_threshold,
    recovery_timeout=settings.circuit_breaker.recovery_timeout,
)


def _create_client() -> OpenAI:
    """创建 OpenAI 兼容客户端"""
    return OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
    )


# ── 底层 API 调用（带重试）────────────────


@exponential_backoff_retry(
    max_retries=5,
    base_delay=1.0,
    backoff_factor=2.0,
    max_delay=32.0,
    exceptions=(Exception,),
)
def _call_llm(prompt: str, system_prompt: str = None, temperature: float = 0.0) -> str:
    """
    底层 LLM 调用，已集成指数退避重试

    Args:
        prompt:        用户 prompt
        system_prompt: 系统 prompt
        temperature:   温度参数

    Returns:
        LLM 返回的文本内容

    Raises:
        Exception: 所有重试耗尽后仍失败
    """
    client = _create_client()
    completion = client.chat.completions.create(
        model=settings.llm.model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
                or "你是 Kimi，由 Moonshot AI 提供的人工智能助手，擅长中文和英文对话。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        timeout=settings.llm.timeout,
    )
    return completion.choices[0].message.content


def call_llm_safe(
    prompt: str,
    system_prompt: str = None,
    temperature: float = 0.0,
    fallback_value: str = None,
) -> Optional[str]:
    """
    安全调用 LLM（集成熔断器）

    - 熔断器打开 → 直接返回 fallback
    - 调用成功 → 记录成功
    - 调用失败 → 记录失败，返回 fallback
    """
    if not llm_circuit_breaker.can_execute():
        logger.warning("[LLM] 熔断器 OPEN，使用降级方案")
        return fallback_value

    try:
        result = _call_llm(prompt, system_prompt, temperature)
        llm_circuit_breaker.record_success()
        return result
    except Exception as e:
        llm_circuit_breaker.record_failure()
        logger.error(f"[LLM] 调用失败（重试已耗尽）: {e}")
        return fallback_value


# ── 对话分析 ──────────────────────────────


def analyze_dialogue_block(dialogues: list[str]) -> int:
    """
    分析一组对话中体现百合氛围的数量

    Args:
        dialogues: 对话列表

    Returns:
        百合台词数量（int）
    """
    prompt = (
        f"以下是 {len(dialogues)} 条台词，请统计其中体现"百合氛围""
        f"（暧昧、亲密、暗恋、浪漫等）的数量，只输出数字，不要额外文字：\n"
    )
    for i, line in enumerate(dialogues, start=1):
        prompt += f"{i}. {line}\n"

    response = call_llm_safe(prompt, fallback_value="0")
    if response is None:
        return 0

    match = re.search(r"\d+", response)
    return int(match.group()) if match else 0


def analyze_verb_block(sentences: list[str]) -> int:
    """
    分析一组描写中体现百合心理/动作的数量

    Args:
        sentences: 句子列表

    Returns:
        符合百合心理和动作的描写数量（int）
    """
    prompt = (
        f"以下是 {len(sentences)} 条对话或动作描写，请统计其中体现"
        f"'重百合氛围'的数量，只输出数字：\n"
    )
    for i, line in enumerate(sentences, start=1):
        prompt += f"{i}. {line}\n"

    response = call_llm_safe(prompt, fallback_value="0")
    if response is None:
        return 0

    match = re.search(r"\d+", response)
    return int(match.group()) if match else 0


# ── 综合分析 ──────────────────────────────


def generate_comprehensive_analysis(
    book_id: str,
    bert_score: float,
    dialogue_score: float,
    verb_score: float,
    final_score: float,
    dialogue_sample: list[str] = None,
    verb_sample: list[str] = None,
) -> dict:
    """
    为一本书生成 LLM 综合分析（剧情概括、人物关系图、高光时刻）

    这是原 weighted_fun.py 中 generate_comprehensive_analysis 的服务化版本。
    """
    # 确定轻重度等级
    if final_score >= 0.9:
        level = "超重度百合"
    elif final_score >= 0.7:
        level = "重度百合"
    elif final_score >= 0.5:
        level = "中度百合"
    elif final_score >= 0.3:
        level = "轻度百合"
    else:
        level = "微百合/友情向"

    dialogue_text = "\n".join((dialogue_sample or [])[:5])
    verb_text = ", ".join((verb_sample or [])[:10])

    prompt = f"""
请基于以下数据，为这部百合作品进行深度分析：

书名：书籍{book_id}
轻重度评分：{final_score:.3f} ({level})
BERT评分：{bert_score:.3f}
对话评分：{dialogue_score:.3f}
动词评分：{verb_score:.3f}

对话样本：
{dialogue_text}

动词样本：
{verb_text}

请从以下角度进行分析：
1. **剧情概括与评语**
2. **人物关系图构建**
3. **高光时刻选择**

请以JSON格式返回：
{{
    "plot_summary": "详细的剧情概括",
    "character_relationships": {{
        "characters": [{{"name": "人物名", "role": "角色描述", "importance": "high/medium/low"}}],
        "relationships": [{{"from": "人物A", "to": "人物B", "type": "关系类型", "strength": 0.9}}]
    }},
    "highlights": [
        {{"text": "高光时刻", "reason": "选择理由", "score": 0.95}}
    ]
}}
"""

    response = call_llm_safe(prompt, temperature=0.7, fallback_value=None)

    base_result = {
        "book_id": book_id,
        "final_score": final_score,
        "level": level,
    }

    if not response:
        base_result["plot_summary"] = (
            f"书籍{book_id}的轻重度评分为{final_score:.3f}，属于{level}。"
            "由于API调用失败，无法提供详细分析。"
        )
        base_result["character_relationships"] = {"characters": [], "relationships": []}
        base_result["highlights"] = []
        return base_result

    try:
        analysis = json.loads(response)
        analysis.update(base_result)
        return analysis
    except json.JSONDecodeError:
        base_result["plot_summary"] = (
            f"书籍{book_id}评分{final_score:.3f}({level})。" + response[:200]
        )
        base_result["character_relationships"] = {"characters": [], "relationships": []}
        base_result["highlights"] = []
        return base_result


# ── 熔断器状态查询 ────────────────────────

def get_breaker_status() -> dict:
    """返回 LLM 熔断器当前状态"""
    return llm_circuit_breaker.get_status()
