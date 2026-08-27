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

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import traceback
from typing import Any, Callable

import httpx
import uvicorn

from llm_gateway.adapters.provider import ProviderDispatcher
from llm_gateway.api.app import create_app
from llm_gateway.config import MODEL_CONFIGS
from llm_gateway.errors import GatewayError
from llm_gateway.models import Usage

# ---------------------------------------------------------------------------
# 准备：端口、凭据环境变量
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


PORT = _free_port()
BASE = f"http://127.0.0.1:{PORT}"

# 为所有已注册模型注入测试用假凭据，避免 503 gateway_misconfigured
for _config in MODEL_CONFIGS.values():
    os.environ.setdefault(_config.api_key_env, "test-key")

PRIMARY_MODEL = MODEL_CONFIGS["general-primary"].provider_model
BACKUP_MODEL = MODEL_CONFIGS["general-backup"].provider_model


# ---------------------------------------------------------------------------
# 模拟上游大模型：网关请求大模型时由它代替
# ---------------------------------------------------------------------------


class FakeUpstream:
    """可编程的假上游：按 provider_model 控制失败次数、结构化输出内容等。"""

    def __init__(self) -> None:
        self.failures: dict[str, int] = {}  # provider_model -> 剩余失败次数
        self.always_fail: set[str] = set()  # provider_model 永远失败
        self.structured_content: str | None = None
        self.stream_deltas: tuple[str, ...] = ("Hello", " from", " fake", " upstream")
        self.stream_fail_after: int | None = None
        self.calls: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.failures.clear()
        self.always_fail.clear()
        self.structured_content = None
        self.stream_fail_after = None
        self.calls.clear()

    def _check(self, config: Any) -> None:
        if not os.getenv(config.api_key_env):
            raise GatewayError("gateway_misconfigured", "Gateway 模型凭据未配置", 503)
        key = config.provider_model
        if key in self.always_fail:
            raise httpx.ConnectError(f"simulated upstream failure: {key}")
        if self.failures.get(key, 0) > 0:
            self.failures[key] -= 1
            raise httpx.ConnectError(f"simulated transient failure: {key}")

    async def complete(
        self,
        config: Any,
        messages: list[Any],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, Usage]:
        self.calls.append(
            {"provider_type": config.provider_type, "model": config.provider_model, "stream": False}
        )
        self._check(config)
        usage = Usage(input_tokens=12, output_tokens=34)
        if response_schema is not None:
            text = (
                self.structured_content
                if self.structured_content is not None
                else json.dumps({"answer": "structured"}, ensure_ascii=False)
            )
            return text, usage
        return "fake completion text", usage

    async def stream(
        self,
        config: Any,
        messages: list[Any],
        timeout_seconds: float,
    ) -> Any:
        self.calls.append(
            {"provider_type": config.provider_type, "model": config.provider_model, "stream": True}
        )
        self._check(config)
        for index, delta in enumerate(self.stream_deltas):
            yield delta
            if self.stream_fail_after is not None and index + 1 >= self.stream_fail_after:
                raise httpx.ConnectError("mid-stream failure")


FAKE = FakeUpstream()


# ---------------------------------------------------------------------------
# 报告与断言工具
# ---------------------------------------------------------------------------

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def show(label: str, data: Any) -> None:
    if isinstance(data, (dict, list)):
        text = json.dumps(data, ensure_ascii=False)
    else:
        text = str(data)
    print(f"  {label}: {text}")


def post(path: str, payload: dict[str, Any]) -> httpx.Response:
    with httpx.Client(timeout=60) as client:
        return client.post(f"{BASE}{path}", json=payload)


def get(path: str) -> httpx.Response:
    with httpx.Client(timeout=60) as client:
        return client.get(f"{BASE}{path}")


def sse_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[5:].strip()))
            except json.JSONDecodeError:
                continue
    return events


def show_response(response: httpx.Response) -> dict[str, Any]:
    """打印响应状态与内容（JSON 或原始文本），并返回解析后的 body。"""
    print(f"  响应状态码: {response.status_code}")
    try:
        body = response.json()
        show("响应内容", body)
        return body
    except ValueError:
        print(f"  响应内容(原始): {response.text!r}")
        return {}


def error_code(body: dict[str, Any]) -> str:
    return (body.get("detail") or {}).get("code", "")


def run_scenario(name: str, func: Callable[[], None]) -> None:
    print(f"\n====== 场景: {name} ======")
    FAKE.reset()
    try:
        func()
        PASSED.append(name)
        print("  结果: [PASS]")
    except AssertionError as exc:
        FAILED.append((name, str(exc)))
        print(f"  结果: [FAIL] {exc}")
    except Exception as exc:  # noqa: BLE001
        FAILED.append((name, f"未预期异常: {exc}"))
        print(f"  结果: [FAIL] 未预期异常: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 测试场景
# ---------------------------------------------------------------------------

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def scenario_basic_complete() -> None:
    payload = {"model": "general-primary", "messages": [{"role": "user", "content": "你好"}]}
    show("请求", payload)
    body = show_response(post("/v1/llm", payload))
    check("content" in body, "响应缺少 content 字段")
    check(body.get("content") == "fake completion text", f"content 不符合预期: {body.get('content')}")
    check(body.get("usage", {}).get("input_tokens") == 12, "usage.input_tokens 不符合预期")
    check(body.get("attempts") == 1, f"首次成功 attempts 应为 1: {body.get('attempts')}")
    check(bool(body.get("request_id")), "缺少 request_id")


def scenario_structured_output() -> None:
    payload = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "结构化"}],
        "response_schema": ANSWER_SCHEMA,
    }
    show("请求", payload)
    body = show_response(post("/v1/llm", payload))
    check(body.get("parsed") == {"answer": "structured"}, f"parsed 不符合预期: {body.get('parsed')}")


def scenario_invalid_json() -> None:
    FAKE.structured_content = "这不是 JSON"
    payload = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "x"}],
        "response_schema": ANSWER_SCHEMA,
    }
    show("请求", payload)
    response = post("/v1/llm", payload)
    body = show_response(response)
    check(response.status_code == 502, f"状态码应为 502: {response.status_code}")
    check(error_code(body) == "invalid_json", f"错误码应为 invalid_json: {error_code(body)}")


def scenario_schema_validation_failed() -> None:
    FAKE.structured_content = json.dumps({"wrong": 1})
    payload = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "x"}],
        "response_schema": ANSWER_SCHEMA,
    }
    show("请求", payload)
    response = post("/v1/llm", payload)
    body = show_response(response)
    check(response.status_code == 502, f"状态码应为 502: {response.status_code}")
    check(error_code(body) == "schema_validation_failed", f"错误码应为 schema_validation_failed: {error_code(body)}")


def scenario_unknown_model() -> None:
    payload = {"model": "missing-model", "messages": [{"role": "user", "content": "hi"}]}
    show("请求", payload)
    response = post("/v1/llm", payload)
    body = show_response(response)
    check(response.status_code == 400, f"状态码应为 400: {response.status_code}")
    check(error_code(body) == "unknown_model", f"错误码应为 unknown_model: {error_code(body)}")


def scenario_prompt_render() -> None:
    payload = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "hi"}],
        "prompt": {
            "name": "knowledge_decision",
            "version": "v1",
            "variables": {"product_name": "智答助手"},
        },
    }
    show("请求", payload)
    body = show_response(post("/v1/llm", payload))
    check(body.get("content") == "fake completion text", "Prompt 场景应成功返回")
    traces = get("/v1/traces").json()
    last = traces[-1]
    show("最新审计记录", {k: last[k] for k in ("prompt_name", "prompt_version", "status", "attempts")})
    check(last.get("prompt_name") == "knowledge_decision", "审计应记录 prompt_name")
    check(last.get("prompt_version") == "v1", "审计应记录 prompt_version")


def scenario_unknown_prompt() -> None:
    payload = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "hi"}],
        "prompt": {"name": "missing", "version": "v1", "variables": {}},
    }
    show("请求", payload)
    response = post("/v1/llm", payload)
    body = show_response(response)
    check(response.status_code == 400, f"状态码应为 400: {response.status_code}")
    check(error_code(body) == "unknown_prompt_template", f"错误码应为 unknown_prompt_template: {error_code(body)}")


def scenario_missing_prompt_variable() -> None:
    payload = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "hi"}],
        "prompt": {"name": "knowledge_decision", "version": "v1", "variables": {}},
    }
    show("请求", payload)
    response = post("/v1/llm", payload)
    body = show_response(response)
    check(response.status_code == 400, f"状态码应为 400: {response.status_code}")
    check(error_code(body) == "missing_prompt_variable", f"错误码应为 missing_prompt_variable: {error_code(body)}")


def scenario_stream_flag_rejected() -> None:
    payload = {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    show("请求", payload)
    response = post("/v1/llm", payload)
    body = show_response(response)
    check(response.status_code == 400, f"状态码应为 400: {response.status_code}")
    check(error_code(body) == "use_stream_endpoint", f"错误码应为 use_stream_endpoint: {error_code(body)}")


def scenario_stream_with_schema_rejected() -> None:
    payload = {
        "model": "general-primary",
        "messages": [{"role": "user", "content": "hi"}],
        "response_schema": ANSWER_SCHEMA,
    }
    show("请求", payload)
    response = post("/v1/llm/stream", payload)
    body = show_response(response)
    check(response.status_code == 400, f"状态码应为 400: {response.status_code}")
    check(error_code(body) == "unsupported_combination", f"错误码应为 unsupported_combination: {error_code(body)}")


def scenario_request_validation() -> None:
    cases = [
        ("多余字段被拒绝", {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}], "extra": 1}),
        ("空 content 被拒绝", {"model": "general-primary", "messages": [{"role": "user", "content": ""}]}),
        ("timeout_seconds 超界被拒绝", {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}], "timeout_seconds": 500}),
        ("stream+schema 组合被拒绝", {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}], "stream": True, "response_schema": ANSWER_SCHEMA}),
    ]
    for name, payload in cases:
        print(f"  -- {name}")
        show("请求", payload)
        response = post("/v1/llm", payload)
        body = show_response(response)
        check(response.status_code == 422, f"{name}: 状态码应为 422: {response.status_code}")
        check(bool(body.get("detail")), f"{name}: 应返回校验错误详情")


def scenario_stream_success() -> None:
    payload = {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}]}
    show("请求", payload)
    response = post("/v1/llm/stream", payload)
    print(f"  响应状态码: {response.status_code}")
    print(f"  响应内容(原始 SSE):\n{response.text}")
    events = sse_events(response.text)
    deltas = [event["delta"] for event in events if event.get("type") == "content.delta"]
    completed = [event for event in events if event.get("type") == "response.completed"]
    check(response.status_code == 200, f"状态码应为 200: {response.status_code}")
    check("text/event-stream" in response.headers.get("content-type", ""), "应返回 text/event-stream")
    check("".join(deltas) == "Hello from fake upstream", f"流式拼接结果不符合预期: {''.join(deltas)}")
    check(len(completed) == 1 and completed[0].get("model") == "general-primary", "应包含 response.completed 且 model 正确")


def scenario_stream_mid_failure() -> None:
    FAKE.stream_fail_after = 1
    payload = {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}]}
    show("请求", payload)
    response = post("/v1/llm/stream", payload)
    print(f"  响应状态码: {response.status_code}")
    print(f"  响应内容(原始 SSE):\n{response.text}")
    events = sse_events(response.text)
    check(any(event.get("type") == "content.delta" for event in events), "应已下发部分内容")
    failed = [event for event in events if event.get("type") == "response.failed"]
    check(len(failed) == 1 and failed[0].get("error") == "upstream_stream_failed", "应返回 response.failed/upstream_stream_failed")
    # 幂等降级：已下发内容后不得再切换 backup
    backup_stream_calls = [
        call for call in FAKE.calls if call["stream"] and call["model"] == BACKUP_MODEL
    ]
    check(not backup_stream_calls, "已下发内容后不应再降级调用 backup")


def scenario_retry_with_backoff() -> None:
    FAKE.failures[PRIMARY_MODEL] = 2  # 前两次失败，第三次成功
    payload = {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}]}
    show("请求", payload)
    body = show_response(post("/v1/llm", payload))
    check(body.get("attempts") == 3, f"指数退避重试后 attempts 应为 3: {body.get('attempts')}")
    primary_calls = [call for call in FAKE.calls if not call["stream"] and call["model"] == PRIMARY_MODEL]
    check(len(primary_calls) == 3, f"主模型应被调用 3 次: {len(primary_calls)}")


def scenario_fallback_to_backup() -> None:
    FAKE.always_fail.add(PRIMARY_MODEL)
    payload = {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}]}
    show("请求", payload)
    body = show_response(post("/v1/llm", payload))
    check(body.get("model") == "general-backup", f"应降级到 general-backup: {body.get('model')}")
    check(body.get("attempts") == 4, f"3 次主模型 + 1 次备用 attempts 应为 4: {body.get('attempts')}")


def scenario_all_models_fail() -> None:
    FAKE.always_fail.update({PRIMARY_MODEL, BACKUP_MODEL})
    payload = {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}]}
    show("请求", payload)
    response = post("/v1/llm", payload)
    body = show_response(response)
    check(response.status_code == 502, f"状态码应为 502: {response.status_code}")
    check(error_code(body) == "model_unavailable", f"错误码应为 model_unavailable: {error_code(body)}")


def scenario_stream_fallback_before_first_token() -> None:
    FAKE.always_fail.add(PRIMARY_MODEL)
    payload = {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}]}
    show("请求", payload)
    response = post("/v1/llm/stream", payload)
    print(f"  响应状态码: {response.status_code}")
    print(f"  响应内容(原始 SSE):\n{response.text}")
    events = sse_events(response.text)
    completed = [event for event in events if event.get("type") == "response.completed"]
    check(len(completed) == 1 and completed[0].get("model") == "general-backup", "流式应降级到 general-backup 并完成")
    primary_stream_calls = [call for call in FAKE.calls if call["stream"] and call["model"] == PRIMARY_MODEL]
    check(len(primary_stream_calls) == 3, f"流式主模型应尝试 3 次: {len(primary_stream_calls)}")


def scenario_traces() -> None:
    response = get("/v1/traces")
    body = show_response(response)
    check(response.status_code == 200, f"状态码应为 200: {response.status_code}")
    check(isinstance(body, list) and body, "审计记录应为非空列表")
    last = body[-1]
    for field in ("request_id", "requested_model", "input_tokens", "output_tokens", "latency_ms", "attempts", "status"):
        check(field in last, f"审计记录缺少字段: {field}")


def scenario_provider_routing() -> None:
    for model, expected_type in (("anthropic-primary", "anthropic"), ("openai-responses", "openai_responses")):
        payload = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
        show("请求", payload)
        body = show_response(post("/v1/llm", payload))
        check(body.get("model") == model, f"{model} 应调用成功")
        last_call = FAKE.calls[-1]
        check(last_call["provider_type"] == expected_type, f"{model} 应路由到 {expected_type}: {last_call['provider_type']}")
        show("上游路由记录", last_call)


def scenario_misconfigured_credentials() -> None:
    env_name = MODEL_CONFIGS["general-primary"].api_key_env
    original = os.environ.pop(env_name, None)
    try:
        payload = {"model": "general-primary", "messages": [{"role": "user", "content": "hi"}]}
        show("请求", payload)
        response = post("/v1/llm", payload)
        body = show_response(response)
        check(response.status_code == 503, f"状态码应为 503: {response.status_code}")
        check(error_code(body) == "gateway_misconfigured", f"错误码应为 gateway_misconfigured: {error_code(body)}")
    finally:
        if original is not None:
            os.environ[env_name] = original


SCENARIOS: list[tuple[str, Callable[[], None]]] = [
    ("非流式基础调用", scenario_basic_complete),
    ("结构化输出成功", scenario_structured_output),
    ("模型返回非法 JSON (invalid_json)", scenario_invalid_json),
    ("模型结果不符合 schema (schema_validation_failed)", scenario_schema_validation_failed),
    ("未知模型 (unknown_model)", scenario_unknown_model),
    ("Prompt 模板渲染与审计", scenario_prompt_render),
    ("未知 Prompt 模板 (unknown_prompt_template)", scenario_unknown_prompt),
    ("缺少 Prompt 变量 (missing_prompt_variable)", scenario_missing_prompt_variable),
    ("stream=true 指向流式端点 (use_stream_endpoint)", scenario_stream_flag_rejected),
    ("流式+schema 被拒绝 (unsupported_combination)", scenario_stream_with_schema_rejected),
    ("请求参数校验 (422)", scenario_request_validation),
    ("流式调用成功 (SSE)", scenario_stream_success),
    ("流式中途失败且不重复降级", scenario_stream_mid_failure),
    ("指数退避重试后成功 (attempts=3)", scenario_retry_with_backoff),
    ("主模型降级到备用模型", scenario_fallback_to_backup),
    ("主备模型全部不可用 (model_unavailable)", scenario_all_models_fail),
    ("流式首 token 前降级备用", scenario_stream_fallback_before_first_token),
    ("调用审计查询 (/v1/traces)", scenario_traces),
    ("供应商路由 (anthropic / openai_responses)", scenario_provider_routing),
    ("凭据未配置 (gateway_misconfigured)", scenario_misconfigured_credentials),
]


# ---------------------------------------------------------------------------
# 服务启动与入口
# ---------------------------------------------------------------------------


def start_server() -> uvicorn.Server:
    dispatcher = ProviderDispatcher(
        providers={provider_type: FAKE for provider_type in ("openai_chat", "openai_responses", "anthropic")}
    )
    app = create_app(provider=dispatcher)
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if get("/v1/traces").status_code == 200:
                return server
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError("网关服务启动超时")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    print(f"启动网关服务: {BASE} (上游大模型已由 FakeUpstream 模拟)")
    server = start_server()
    print("网关服务就绪，开始执行测试场景")

    for name, func in SCENARIOS:
        run_scenario(name, func)

    print("\n" + "=" * 60)
    print(f"测试汇总: 通过 {len(PASSED)} / 共 {len(SCENARIOS)}")
    for name, reason in FAILED:
        print(f"  [FAIL] {name}: {reason}")
    print("=" * 60)

    server.should_exit = True
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
