# LLM-Gateway

## 会自动启动网关服务

python test_gateway_live.py

独立 LLM Gateway 服务：对外统一 HTTP 协议，对内适配 DeepSeek / Anthropic / OpenAI 等供应商，
把 `chat.completions`、`responses`、Anthropic `messages` 三种上游响应统一为同一数据结构。
内置 Prompt 托管、结构化输出校验（原生格式约束）、按模型名分桶限流、指数退避重试、
主备降级、流式 TTFT/Token 统计与调用审计。

真实调用需设置环境变量（完整清单见 `.env.example`，按需设置用到的模型）：

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
$env:DEEPSEEK_BACKUP_API_KEY = "sk-..."
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:OPENAI_API_KEY = "sk-..."
```

限流为可选配置，默认每个模型 60 次 / 60 秒：

```powershell
$env:RATE_LIMIT_MAX_REQUESTS = "60"
$env:RATE_LIMIT_WINDOW_SECONDS = "60"
```

服务默认监听 `http://127.0.0.1:8000`（交互式文档见 `/docs`），
等效于 `uvicorn gateway:app --host 127.0.0.1 --port 8000`。

## curl 示例

以下示例假设服务运行在 `http://127.0.0.1:8000`。
PowerShell 下请使用 `curl.exe`（避免 `curl` 别名指向 `Invoke-WebRequest`）。

非流式调用：

```bash
curl -s http://127.0.0.1:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{"model": "general-primary", "messages": [{"role": "user", "content": "你好"}]}'
```

流式调用（SSE，`-N` 关闭缓冲以实时接收）：

```bash
curl -N http://127.0.0.1:8000/v1/llm/stream \
  -H "Content-Type: application/json" \
  -d '{"model": "general-primary", "messages": [{"role": "user", "content": "讲个笑话"}]}'
```

结构化输出（`response_schema` 校验模型返回的 JSON）：

```bash
curl -s http://127.0.0.1:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{"model": "general-primary", "messages": [{"role": "user", "content": "回答"}], "response_schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}'
```

Prompt 模板渲染（模板与变量见 `config.py`）：

```bash
curl -s http://127.0.0.1:8000/v1/llm \
  -H "Content-Type: application/json" \
  -d '{"model": "general-primary", "messages": [{"role": "user", "content": "hi"}], "prompt": {"name": "knowledge_decision", "version": "v1", "variables": {"product_name": "智答助手"}}}'
```

调用审计查询：

```bash
curl -s http://127.0.0.1:8000/v1/traces
```

## 模块结构

```text
LLM-Gateway/
├── gateway.py                       # 启动入口
├── test_gateway_live.py             # 集成测试（自启服务 + 模拟上游，24 个场景）
└── llm_gateway/
    ├── api/                         # FastAPI 路由与应用
    ├── adapters/provider.py         # 各供应商适配器 + 分派器
    ├── services/gateway.py          # 降级、重试、审计、SSE、TTFT
    ├── services/rate_limiter.py    # 按模型名分桶的滑动窗口限流器
    ├── models.py                    # Pydantic 模型
    ├── config.py                    # 模型白名单、Prompt 模板、价格
    └── errors.py                    # 统一错误
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

- **限流**：按模型名分桶的滑动窗口（`RATE_LIMIT_MAX_REQUESTS`/`RATE_LIMIT_WINDOW_SECONDS`），
  超限返回 429 `rate_limited`，不重试、不降级；模型之间互不影响，
  主模型失败后的降级也不会绕过备用模型的桶
- **重试**：每个模型最多 3 次尝试，指数退避（0.1s → 0.2s → 0.4s）
- **降级**：主模型耗尽后切 `general-backup`；`request_id` 全程不变，审计只记一次
- **结构化输出**：`json_schema` 模式走原生 strict schema（`response_format` / `text.format`）；
  `json_object` 模式设置原生 `response_format: json_object`，schema 通过提示词传达；
  Anthropic 无原生 JSON 模式，通过 system 提示词约束
- **流式观测**：首个非空 delta 记录 TTFT（`response.completed` 携带 `ttft_ms` 与上游 `usage`），
  审计记录流式 Token 并计费（chat 协议需上游支持 `stream_options.include_usage`）
- **流式幂等**：已下发内容后不再重试/降级，避免内容重复

## 测试

```powershell
python test_gateway_live.py
```

集成测试自启真实网关服务，上游由 FakeUpstream 模拟，覆盖非流式/流式、结构化输出、
限流与模型间隔离、TTFT/流式 Token、原生格式约束、重试退避、主备降级、幂等审计、
供应商路由、参数校验与错误码，退出码 0 表示全部通过。
