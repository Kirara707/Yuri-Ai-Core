# Yuri AI Core

百合文学智能分析平台，FastAPI 后端 + Celery 编排 + mock 模式可在无外部服务下跑通所有核心流程。

## 现在可以做什么

- **Mock 模式一键开启**：测试或本地跑一套 API 只需设置 `LLM_MOCK_MODE=true`（`.env` 或环境变量），所有对外模型/缓存/队列都降级到内存模拟，再也不用依赖 Redis、Celery 或 PyTorch。集成测试在这个模式下全部通过。
- **分析入口**：`POST /api/v1/analysis/submit` 接收 `text/dialogues/sentences`，返回 Celery 任务 ID；`GET /analysis/progress/{task_id}` 和 `/analysis/result/{task_id}` 会优先读取缓存，mock 模式下直接返回模拟状态。
- **系统观察**：`/system/health` 展示 mock 下缓存状态、mock breaker、整体状态；`/system/mock-status` 报告 mock 是否激活及调用统计；`/system/metrics`、`/system/breaker` 仍可实时检查。
- **LLM mock 统计**：所有模拟调用由 `MockStatsTracker` 记录，`/system/mock-status` 能看到调用次数与每次活动，方便单元/集成测试验证侧路行为。

## 依赖与安装（开发）

```bash
pip install -r requirements.txt
pip install fastapi uvicorn httpx redis celery pytest pytest-asyncio torch
```

> `torch` 仅在非 mock 模式、需要 BERT 推理时必须；mock 模式下无需，但在 Windows 上安装仍可能触发 DLL 访问冲突，请谨慎使用。Mock 模式下 `cache_service` 会自动切换到内存缓存，`health_check` 跳过 `bert_service` 导入。

## 启动 & 运行

```bash
# 设置 mock 模式
set LLM_MOCK_MODE=true
set MOONSHOT_API_KEY=

# 启动 FastAPI
uvicorn backend.core.api:app --reload
```

在 mock 模式下，Celery worker/Redis/torch 依赖都可以跳过，所有路由仍然返回合理结构。真实模式下请确保 Redis/Celery/torch 均可用，模型 checkpoint 放到 `models/checkpoint-47200`。

## 测试

```bash
pytest tests/unit
pytest tests/integration
```

- 集成测试现在覆盖了根路由、健康/指标、LLM mock 状态、分析进度和 mock 健康点，且 mock 模式下不再依赖 Redis/Celery。请先确保安装 `redis`, `celery`, `pytest-asyncio`, `torch`（真实模式）或仅在 mock 模式下跳过。

## 目录概览

```
backend/            FastAPI + Celery 服务
  ├── core/          FastAPI 路由/中间件
  ├── services/      cache/LLM/BERT/metrics 业务逻辑
  ├── tasks/         Celery 任务（analysis workflow）
  ├── utils/         配置/重试/熔断器/日志
  └── db/            ORM + 会话

tests/              单元 + 集成
assets/             训练辅助资料（stopwords、环境清单）
csv/                示例数据 + 结果
models/             BERT checkpoint（mock 模式无需）
```

## 继续做什么

1. mock 模式下所有 API 可以独立验证；欲切换回生产模式，补 Redis/Celery/torch 环境并清除 `LLM_MOCK_MODE`。
2. 若需扩展更多模型接入，先在 `backend/services/llm_service.py` 补充真实 API 调用，再更新 mock stats 记录。
3. 运行 `docker compose up` 可快速启动真实依赖链（Redis + Postgres + FastAPI + Celery + Flower），适用于全量集成测试。
