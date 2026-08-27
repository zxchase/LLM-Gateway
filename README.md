# LLM-Gateway

独立 LLM Gateway 服务：对外统一 HTTP 协议，对内适配 DeepSeek / Anthropic / OpenAI 等供应商，
把 `chat.completions`、`responses`、Anthropic `messages` 三种上游响应统一为同一数据结构。
内置 Prompt 托管、结构化输出校验、指数退避重试、主备降级与调用审计。

## 启动

```powershell
测试命令：
"""网关集成测试：启动真实网关服务 + 模拟上游大模型，发起真实 HTTP 请求。

运行方式：

    python test_gateway_live.py

特性：
1. 在子线程中以 uvicorn 启动真实网关服务（真实 HTTP 栈，非 TestClient）；
2. 网关对上游大模型的调用被 FakeUpstream 替代（可编程的成功/失败/结构化输出，
   不依赖网络与真实 API Key），ProviderDispatcher 按 provider_type 正常路由；
3. 覆盖网关全部功能点：非流式/流式调用、结构化输出校验、Prompt 托管、
   重试与指数退避、主备降级、幂等审计、供应商路由、参数校验与错误码；
4. 每个场景打印请求体与响应（含异常提示），结束后输出通过/失败汇总，
   退出码 0 表示全部通过。
"""

```

## 模块结构

```text
LLM-Gateway/
├── gateway.py              # 启动入口
├── run_tests.py            # 离线测试运行器
├── test_gateway.py         # 离线单元测试（FakeProvider）
├── test_gateway_live.py    # 集成测试（自启服务 + 模拟上游，20 个场景）
└── llm_gateway/
    ├── api/                # FastAPI 路由
    ├── adapters/provider.py # 各供应商适配器 + 分派器
    ├── services/gateway.py # 降级、重试、审计、SSE
    ├── models.py           # Pydantic 模型
    ├── config.py          # 模型白名单、Prompt 模板、价格
    └── errors.py          # 统一错误
```



真实调用需设置环境变量（见 `.env.example`）：

```text
DEEPSEEK_API_KEY=
DEEPSEEK_BACKUP_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
```

## 模型与供应商

| 模型名 | provider_type | 上游协议 |
|--------|---------------|----------|
| `general-primary` | `openai_chat` | `/chat/completions` |
| `general-backup` | `openai_chat` | `/chat/completions` |
| `anthropic-primary` | `anthropic` | `/v1/messages` |
| `openai-responses` | `openai_responses` | `/v1/responses` |

## HTTP 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/llm` | 非流式调用 |
| POST | `/v1/llm/stream` | SSE 流式调用 |
| GET | `/v1/traces` | 调用审计记录 |

## 核心行为

- **重试**：每个模型最多 3 次尝试，指数退避（0.1s → 0.2s → 0.4s）
- **降级**：主模型耗尽后切 `general-backup`；`request_id` 全程不变，审计只记一次
- **结构化输出**：`response_schema` 传入时校验模型返回的 JSON（`jsonschema`）
- **流式幂等**：已下发内容后不再重试/降级，避免内容重复

## 测试

```powershell
python run_tests.py           # 离线单元测试（不依赖网络）
python test_gateway_live.py   # 集成测试（自启服务，上游由 FakeUpstream 模拟）
```
