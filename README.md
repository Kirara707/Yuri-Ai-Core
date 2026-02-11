# Yuri AI Core

百合文学智能分析平台，分为 **FastAPI 后端** + **Celery 分析管线** + **Redis / PostgreSQL / Docker / CI**。

## 项目亮点

- **BERT + Moonshot LLM 二合一**：BERT滑窗推理、LLM对话/动作分析、加权融合 + 综合分析生成。
- **工业级协程**：FastAPI 提供 REST 接口；Celery 负责任务调度；Redis 缓存分析结果；PostgreSQL 存储历史。
- **稳定 / 可观测**：loguru JSON 日志、重试装饰器、熔断器、指标聚合、健康/指标路由。
- **DevOps 友好**：Dockerfile + docker-compose 启动六个服务；GitHub Actions 实现 lint/test/docker。

## 一键启动（开发）

```bash
cp .env.example .env     # 补充 Moonshot/API 密钥、Redis、Postgres 配置
docker compose up -d      # 启动 redis/postgres/api/worker/beat/flower
```

- FastAPI docs：`http://localhost:8000/docs`
- Celery flower：`http://localhost:5555`
- 结果缓存/进度：`/api/v1/analysis/result/{task_id}`

## REST 接口

| 路径 | 说明 |
| --- | --- |
| `POST /api/v1/analysis/submit` | 提交包含 text/dialogues/sentences 的异步分析任务。
| `GET /api/v1/analysis/progress/{task_id}` | 查询 Redis 中的任务进度。
| `GET /api/v1/analysis/result/{task_id}` | 获取最终结果（缓存或 Celery 结果）。
| `GET /api/v1/system/health` | 健康信息（Redis、数据库、BERT 是否加载、LLM 熔断器）。
| `GET /api/v1/system/metrics` | p50/p95/p99 延迟 & 成功率统计。
| `GET /api/v1/system/breaker` | LLM 熔断器的状态。

## 核心目录

```
backend/            FastAPI + Celery 服务代码
  ├── core/          FastAPI 入口 + 路由 + 中间件
  ├── services/      BERT/LLM/缓存/指标业务逻辑
  ├── tasks/         Celery 任务 + orchestrator
  ├── utils/         配置、日志、retry、circuit breaker
  └── db/            SQLAlchemy ORM + 会话管理

script/             旧版训练/推理脚本（可离线使用）
assets/             训练辅助材料 (stopwords、frozen requirements)
csv/                数据/输出 CSV
models/             BERT/HuggingFace Checkpoint
tests/              pytest 单元&集成测试
```

## 本地开发建议

1. `pip install -r requirements.txt`
2. 准备 `.env`（参考 `.env.example`）填写 Moonshot API 和数据库配置。
3. `uvicorn backend.core.api:app --reload` 运行 FastAPI。
4. `celery -A backend.tasks.celery_app:celery_app worker -Q analysis --loglevel=info`
5. `celery -A backend.tasks.celery_app:celery_app beat --loglevel=info`（可选）。
6. `pytest tests/` 进行单位与集成测试。

## CI/CD / Docker

- `Dockerfile`：多阶段构建镜像，日志目录和模型可挂载。
- `docker-compose.yml`：6 个服务，包括 Flower。
- `.github/workflows/ci-cd.yml`：lint → test → docker build 推送。

## 许可证

MIT
