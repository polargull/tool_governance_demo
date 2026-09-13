from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PermissionMode(StrEnum):
    """定义运行时权限模式，控制工具调用是否需要确认或限制副作用。"""

    DEFAULT = "default"
    PLAN = "plan"
    BYPASS_PERMISSIONS = "bypassPermissions"
    DONT_ASK = "dontAsk"


class Effect(StrEnum):
    """描述工具执行可能产生的副作用类型。"""

    READ = "read"
    WRITE = "write"
    SHELL = "shell"


class Risk(StrEnum):
    """标记工具或业务动作的风险等级。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionAction(StrEnum):
    """表示权限引擎对一次工具调用给出的最终动作。"""

    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


Permission = Literal["order:read", "refund:create", "shell:run", "transfer:execute"]


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """承载一次执行所需的身份、租户、权限和审批上下文。"""

    trace_id: str
    user_id: str
    tenant_id: str
    mode: PermissionMode
    permissions: frozenset[Permission]
    allowed_tools: frozenset[str]
    approval_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """声明单个工具的副作用、风险、业务权限与执行约束。"""

    effect: Effect
    risk: Risk
    permission: Permission
    requires_approval: bool
    timeout_seconds: float
    max_retries: int
    idempotent: bool


class StrictArgs(BaseModel):
    """模型只能提交 Schema 允许的业务候选参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)


class GetOrderArgs(StrictArgs):
    """查询订单工具的参数 Schema。"""

    order_id: str = Field(pattern=r"^ord_[0-9]{4}$")


class CreateRefundArgs(StrictArgs):
    """创建退款工具的参数 Schema。"""

    order_id: str = Field(pattern=r"^ord_[0-9]{4}$")
    amount: float = Field(gt=0, le=10_000)
    reason: str = Field(min_length=4, max_length=200)


class RunShellArgs(StrictArgs):
    """模拟 Shell 工具的参数 Schema。"""

    command: str = Field(min_length=1, max_length=200)

class TransferArgs(StrictArgs):
    from_account: str = Field(pattern=r"^ACC-[A-Z]-[0-9]{6}$")
    to_account: str = Field(pattern=r"^ACC-[A-Z]-[0-9]{6}$")
    amount: float = Field(gt=0, le=100000)

ArgsModel = GetOrderArgs | CreateRefundArgs | RunShellArgs
Handler = Callable[[str, ArgsModel, ExecutionContext], Awaitable[Mapping[str, Any]]]
Precheck = Callable[[ArgsModel, ExecutionContext], Awaitable[None]]
CanonicalTarget = Callable[[ArgsModel], str]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """聚合模型可见定义、治理策略、执行函数与业务预检。"""

    name: str
    description: str
    parameters_model: type[StrictArgs]
    policy: ToolPolicy
    handler: Handler
    canonical_target: CanonicalTarget
    precheck: Precheck | None = None

    def to_model_tool(self) -> dict[str, Any]:
        """只投影模型需要的描述和 JSON Schema，不暴露 handler 与治理策略。"""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_model.model_json_schema(),
            },
        }


@dataclass(frozen=True, slots=True)
class PermissionRule:
    """配置针对工具名和目标前缀的 allow / deny 规则。"""

    effect: Literal["allow", "deny"]
    tool_name: str
    target_prefix: str | None = None


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """记录权限状态机的动作、原因码、说明和决策来源。"""

    action: DecisionAction
    code: str
    reason: str
    source: Literal[
        "rule", "mode", "whitelist", "rbac", "business", "approval", "risk", "default"
    ]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """表示模型或调用方提交的一次原始工具调用。"""

    tool_call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """封装工具执行结果，并负责投影成模型可消费的 tool 消息。"""

    tool_call_id: str
    tool_name: str
    ok: bool
    action: DecisionAction
    code: str
    content: Any
    retryable: bool = False

    def to_tool_message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": json.dumps(
                {
                    "ok": self.ok,
                    "code": self.code,
                    "action": self.action,
                    "content": self.content,
                },
                ensure_ascii=False,
            ),
        }


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """记录工具调用在决策或执行阶段产生的审计事件。"""

    trace_id: str
    tool_call_id: str
    tool_name: str
    user_id: str
    tenant_id: str
    phase: Literal["decision", "execution"]
    decision: str
    code: str
    argument_keys: tuple[str, ...]
    latency_ms: int | None = None
    redacted_arguments: dict[str, Any] | None = None


@dataclass(slots=True)
class ApprovalRecord:
    """保存一次参数绑定、租户绑定且可一次性消费的审批凭证。"""

    approval_id: str
    user_id: str
    tenant_id: str
    tool_name: str
    digest: str
    expires_at: float
    used: bool = False


class PolicyDenied(RuntimeError):
    """表示业务规则或权限检查主动拒绝本次调用。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TransientToolError(RuntimeError):
    """表示可重试的临时性工具错误。"""

    pass


def _stable_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _stable_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {key: _stable_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    return value


def _approval_digest(tool_name: str, arguments: ArgsModel | Mapping[str, Any]) -> str:
    canonical = json.dumps(_stable_value(arguments), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(f"{tool_name}:{canonical}".encode()).hexdigest()


class ApprovalStore:
    """以内存方式管理审批记录的创建、校验和一次性消费。"""

    def __init__(self) -> None:
        self._records: dict[str, ApprovalRecord] = {}

    def approve(
        self,
        approval_id: str,
        context: ExecutionContext,
        tool_name: str,
        arguments: ArgsModel | Mapping[str, Any],
        *,
        ttl_seconds: float = 300,
    ) -> None:
        self._records[approval_id] = ApprovalRecord(
            approval_id=approval_id,
            user_id=context.user_id,
            tenant_id=context.tenant_id,
            tool_name=tool_name,
            digest=_approval_digest(tool_name, arguments),
            expires_at=time.time() + ttl_seconds,
        )

    def consume(
        self,
        approval_id: str | None,
        context: ExecutionContext,
        tool_name: str,
        arguments: ArgsModel,
    ) -> bool:
        record = self._records.get(approval_id or "")
        valid = bool(
            record
            and not record.used
            and record.expires_at >= time.time()
            and record.user_id == context.user_id
            and record.tenant_id == context.tenant_id
            and record.tool_name == tool_name
            and record.digest == _approval_digest(tool_name, arguments)
        )
        if valid and record:
            record.used = True
        return valid


class AuditSink:
    """收集权限决策和工具执行阶段的审计记录。"""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def append(self, record: AuditRecord) -> None:
        self.records.append(record)


DANGEROUS_SHELL_PATTERNS = (
    re.compile(r"\brm\s+-rf\b", re.I),
    re.compile(r"\bgit\s+push\s+--force\b", re.I),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.I),
    re.compile(r"\bsudo\b", re.I),
    re.compile(r"\bmkfs\b", re.I),
    re.compile(r">\s*/dev/", re.I),
)


def _is_dangerous_shell(command: str) -> bool:
    return any(pattern.search(command) for pattern in DANGEROUS_SHELL_PATTERNS)


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: "***" if re.search(r"token|secret|password|authorization", key, re.I) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "***@***", value)
        value = re.sub(r"(ACC-[A-Z]-)\d{2}(\d{4})", r"\1****\2", value)
        return value
    return value


def _redact_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """对参数字典进行脱敏处理，用于审计日志记录。"""
    return _redact(dict(arguments))


class PermissionEngine:
    """固定优先级的三态权限状态机。"""

    def __init__(self, rules: Sequence[PermissionRule], approvals: ApprovalStore) -> None:
        self._rules = tuple(rules)
        self._approvals = approvals

    def _rule_matches(self, rule: PermissionRule, tool: ToolDefinition, arguments: ArgsModel) -> bool:
        if rule.tool_name != tool.name:
            return False
        if rule.target_prefix is None:
            return True
        return tool.canonical_target(arguments).startswith(rule.target_prefix)

    async def decide(
        self,
        tool: ToolDefinition,
        arguments: ArgsModel,
        context: ExecutionContext,
    ) -> PermissionDecision:
        # 1. deny-first：硬拒绝不能被 allow 或 bypass 覆盖。
        if any(
            rule.effect == "deny" and self._rule_matches(rule, tool, arguments)
            for rule in self._rules
        ):
            return PermissionDecision(DecisionAction.DENY, "DENY_RULE", "命中 deny 规则", "rule")

        # 2. plan 是执行层只读契约，而不是一句系统提示词。
        if context.mode is PermissionMode.PLAN and tool.policy.effect is not Effect.READ:
            return PermissionDecision(
                DecisionAction.DENY,
                "PLAN_MODE_DENIED",
                "plan 模式禁止写操作和 Shell",
                "mode",
            )

        # 3. 发现阶段过滤后，执行阶段仍然要重新检查白名单。
        if tool.name not in context.allowed_tools:
            return PermissionDecision(
                DecisionAction.DENY,
                "TOOL_NOT_ALLOWED",
                "工具不在本轮执行白名单",
                "whitelist",
            )

        # 4. 只相信认证层生成的 ExecutionContext。
        if tool.policy.permission not in context.permissions:
            return PermissionDecision(
                DecisionAction.DENY,
                "PERMISSION_DENIED",
                f"缺少业务权限 {tool.policy.permission}",
                "rbac",
            )

        # 5. 资源归属、状态和额度在 handler 之前验证。
        try:
            if tool.precheck:
                await tool.precheck(arguments, context)
        except PolicyDenied as error:
            return PermissionDecision(DecisionAction.DENY, error.code, str(error), "business")

        # 6. 高风险业务写操作必须使用一次性、参数绑定审批。
        if tool.policy.requires_approval or tool.policy.risk is Risk.HIGH:
            if self._approvals.consume(context.approval_id, context, tool.name, arguments):
                return PermissionDecision(
                    DecisionAction.ALLOW,
                    "APPROVED",
                    "审批与当前用户、租户、工具和参数完全匹配",
                    "approval",
                )
            if context.mode is PermissionMode.DONT_ASK:
                return PermissionDecision(
                    DecisionAction.DENY,
                    "APPROVAL_REQUIRED",
                    "非交互模式无法完成高风险确认",
                    "approval",
                )
            return PermissionDecision(
                DecisionAction.CONFIRM,
                "APPROVAL_REQUIRED",
                "需要确认本次具体动作",
                "approval",
            )

        # 7. bypass 只能跳过普通确认，不能跳过前面的硬边界。
        if context.mode is PermissionMode.BYPASS_PERMISSIONS:
            return PermissionDecision(
                DecisionAction.ALLOW,
                "BYPASS_ALLOWED",
                "跳过普通确认，但硬边界已经全部通过",
                "mode",
            )

        # 8. allow 规则只在 deny、plan、白名单、RBAC 和审批以后生效。
        if any(
            rule.effect == "allow" and self._rule_matches(rule, tool, arguments)
            for rule in self._rules
        ):
            return PermissionDecision(DecisionAction.ALLOW, "ALLOW_RULE", "命中 allow 规则", "rule")

        # 9. 正则只是教学兜底，生产中必须配合窄工具、AST 与沙箱。
        if tool.policy.effect is Effect.SHELL and _is_dangerous_shell(
            str(getattr(arguments, "command", ""))
        ):
            if context.mode is PermissionMode.DONT_ASK:
                return PermissionDecision(
                    DecisionAction.DENY,
                    "DANGEROUS_OPERATION",
                    "危险 Shell 在非交互模式下被拒绝",
                    "risk",
                )
            return PermissionDecision(
                DecisionAction.CONFIRM,
                "DANGEROUS_OPERATION",
                "危险 Shell 需要用户确认",
                "risk",
            )

        return PermissionDecision(
            DecisionAction.ALLOW,
            "DEFAULT_ALLOWED",
            "所有确定性检查均已通过",
            "default",
        )


class ToolRuntime:
    """模型、CLI、测试与未来 Provider 共用的唯一工具执行入口。"""

    def __init__(
        self,
        tools: Sequence[ToolDefinition],
        permission_engine: PermissionEngine,
        audit_sink: AuditSink,
    ) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._permission_engine = permission_engine
        self._audit = audit_sink

    def model_tools(self, context: ExecutionContext) -> list[dict[str, Any]]:
        """发现期白名单：减少模型可见能力，不把 handler 暴露给模型。"""

        return [
            tool.to_model_tool()
            for tool in self._tools.values()
            if tool.name in context.allowed_tools
        ]

    async def invoke(self, call: ToolCall, context: ExecutionContext) -> ToolResult:
        started = time.perf_counter()
        tool = self._tools.get(call.name)
        if tool is None:
            return self._rejected(call, context, "TOOL_NOT_FOUND", "工具不存在")

        # prepare-1：Pydantic 把不可信字典转换成 handler 可接收的业务对象。
        try:
            arguments = tool.parameters_model.model_validate(call.arguments)
        except ValidationError as error:
            details = [
                {"path": ".".join(map(str, item["loc"])), "message": item["msg"]}
                for item in error.errors(include_url=False)
            ]
            return self._rejected(call, context, "INVALID_ARGUMENT", details)

        # prepare-2：执行期重新授权，返回 allow / deny / confirm。
        decision = await self._permission_engine.decide(tool, arguments, context)
        self._audit.append(
            AuditRecord(
                trace_id=context.trace_id,
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                phase="decision",
                decision=decision.action,
                code=decision.code,
                argument_keys=tuple(sorted(call.arguments)),
                redacted_arguments=_redact_arguments(call.arguments),
            )
        )
        if decision.action is not DecisionAction.ALLOW:
            return ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                ok=False,
                action=decision.action,
                code=decision.code,
                content=decision.reason,
            )

        # execute：只有通过全部确定性检查后，handler 才可能产生副作用。
        try:
            raw = await self._execute_with_recovery(tool, call.tool_call_id, arguments, context)
        except TimeoutError:
            code = "TIMEOUT" if tool.policy.effect is Effect.READ or tool.policy.idempotent else "TIMEOUT_UNKNOWN"
            return self._failed(call, context, started, code, "工具执行超时")
        except PolicyDenied as error:
            return self._failed(call, context, started, error.code, str(error))
        except Exception as error:  # 生产中映射异常类型，不把 traceback 交给模型。
            return self._failed(call, context, started, "TOOL_ERROR", str(error))

        # finalize：先投影与脱敏，再形成模型能看见的 ToolResult。
        safe_content = _redact(dict(raw))
        latency_ms = round((time.perf_counter() - started) * 1_000)
        self._audit.append(
            AuditRecord(
                trace_id=context.trace_id,
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                phase="execution",
                decision="executed",
                code="OK",
                argument_keys=tuple(sorted(call.arguments)),
                latency_ms=latency_ms,
                redacted_arguments=_redact_arguments(call.arguments),
            )
        )
        return ToolResult(call.tool_call_id, call.name, True, DecisionAction.ALLOW, "OK", safe_content)

    async def _execute_with_recovery(
        self,
        tool: ToolDefinition,
        tool_call_id: str,
        arguments: ArgsModel,
        context: ExecutionContext,
    ) -> Mapping[str, Any]:
        retries = tool.policy.max_retries if tool.policy.effect is Effect.READ or tool.policy.idempotent else 0
        for attempt in range(retries + 1):
            try:
                async with asyncio.timeout(tool.policy.timeout_seconds):
                    return await tool.handler(tool_call_id, arguments, context)
            except TransientToolError:
                if attempt == retries:
                    raise
                await asyncio.sleep(min(0.05 * (2**attempt), 0.2))
        raise AssertionError("unreachable")

    def _rejected(
        self,
        call: ToolCall,
        context: ExecutionContext,
        code: str,
        content: Any,
    ) -> ToolResult:
        self._audit.append(
            AuditRecord(
                trace_id=context.trace_id,
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                phase="decision",
                decision="deny",
                code=code,
                argument_keys=tuple(sorted(call.arguments)),
            )
        )
        return ToolResult(call.tool_call_id, call.name, False, DecisionAction.DENY, code, content)

    def _failed(
        self,
        call: ToolCall,
        context: ExecutionContext,
        started: float,
        code: str,
        content: Any,
    ) -> ToolResult:
        self._audit.append(
            AuditRecord(
                trace_id=context.trace_id,
                tool_call_id=call.tool_call_id,
                tool_name=call.name,
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                phase="execution",
                decision="failed",
                code=code,
                argument_keys=tuple(sorted(call.arguments)),
                latency_ms=round((time.perf_counter() - started) * 1_000),
            )
        )
        return ToolResult(call.tool_call_id, call.name, False, DecisionAction.DENY, code, content)


ORDERS = {
    ("tenant_a", "ord_1001"): {
        "status": "paid",
        "refundable": 399.0,
        "customer_email": "alice@example.com",
    }
}
ACCOUNTS = {
    ("tenant_a", "ACC-A-123456"): 100000.0,
    ("tenant_a", "ACC-A-654321"): 5000.0,
    ("tenant_a", "ACC-A-888888"): 20000.0,
    ("tenant_b", "ACC-A-111111"): 50000.0
}
SIDE_EFFECTS = {"refund_executions": 0, "shell_executions": 0, "transfer_executions": 0}


def reset_side_effects() -> None:
    SIDE_EFFECTS.update(refund_executions=0, shell_executions=0, transfer_executions=0)


async def get_order_handler(
    _tool_call_id: str,
    raw_arguments: ArgsModel,
    context: ExecutionContext,
) -> Mapping[str, Any]:
    arguments = raw_arguments
    assert isinstance(arguments, GetOrderArgs)
    order = ORDERS.get((context.tenant_id, arguments.order_id))
    if not order:
        raise PolicyDenied("ORDER_NOT_FOUND", "当前租户下不存在该订单")
    return {**order, "access_token": "tok_demo_should_not_leak"}


async def refund_precheck(raw_arguments: ArgsModel, context: ExecutionContext) -> None:
    arguments = raw_arguments
    assert isinstance(arguments, CreateRefundArgs)
    order = ORDERS.get((context.tenant_id, arguments.order_id))
    if not order or order["status"] != "paid":
        raise PolicyDenied("BUSINESS_RULE_DENIED", "订单不存在或状态不可退款")
    if arguments.amount > float(order["refundable"]):
        raise PolicyDenied("BUSINESS_RULE_DENIED", "退款金额超过可退金额")


async def transfer_precheck(raw_arguments: ArgsModel, context: ExecutionContext) -> None:
    arguments = raw_arguments
    assert isinstance(arguments, TransferArgs)
    if arguments.amount > 50_000:
        raise PolicyDenied("EXCEED_LIMIT", "转账金额超过单笔限额")
    balance = ACCOUNTS.get((context.tenant_id, arguments.from_account), 0.0)
    if balance < arguments.amount:
        raise PolicyDenied("INSUFFICIENT_BALANCE", "转出账户余额不足")


async def transfer_handler(
    tool_call_id: str,
    raw_arguments: ArgsModel,
    context: ExecutionContext,
) -> Mapping[str, Any]:
    arguments = raw_arguments
    assert isinstance(arguments, TransferArgs)
    SIDE_EFFECTS["transfer_executions"] += 1
    if arguments.amount > 80_000:
        await asyncio.sleep(3.0)
    to_key = (context.tenant_id, arguments.to_account)
    if to_key not in ACCOUNTS:
        raise PolicyDenied("ACCOUNT_NOT_FOUND", "转入账户不存在")
    from_key = (context.tenant_id, arguments.from_account)
    ACCOUNTS[from_key] = ACCOUNTS.get(from_key, 0.0) - arguments.amount
    ACCOUNTS[to_key] = ACCOUNTS[to_key] + arguments.amount
    return {
        "txn_id": tool_call_id[-6:],
        "from": arguments.from_account,
        "to": arguments.to_account,
        "amount": arguments.amount,
        "status": "completed",
    }


async def create_refund_handler(
    tool_call_id: str,
    raw_arguments: ArgsModel,
    context: ExecutionContext,
) -> Mapping[str, Any]:
    arguments = raw_arguments
    assert isinstance(arguments, CreateRefundArgs)
    SIDE_EFFECTS["refund_executions"] += 1
    return {
        "refund_id": "ref_9001",
        "idempotency_key": tool_call_id,
        "tenant_id": context.tenant_id,
        "order_id": arguments.order_id,
        "amount": arguments.amount,
        "status": "accepted",
    }


async def simulated_shell_handler(
    _tool_call_id: str,
    raw_arguments: ArgsModel,
    _context: ExecutionContext,
) -> Mapping[str, Any]:
    arguments = raw_arguments
    assert isinstance(arguments, RunShellArgs)
    SIDE_EFFECTS["shell_executions"] += 1
    return {
        "simulated": True,
        "command": arguments.command,
        "stdout": "教学模拟：没有创建真实子进程",
    }


def build_tools() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="get_order",
            description="查询当前租户订单状态和可退金额",
            parameters_model=GetOrderArgs,
            policy=ToolPolicy(Effect.READ, Risk.MEDIUM, "order:read", False, 1.0, 2, True),
            handler=get_order_handler,
            canonical_target=lambda args: str(getattr(args, "order_id")),
        ),
        ToolDefinition(
            name="create_refund",
            description="为当前租户的已支付订单创建退款",
            parameters_model=CreateRefundArgs,
            policy=ToolPolicy(Effect.WRITE, Risk.HIGH, "refund:create", True, 2.0, 0, False),
            handler=create_refund_handler,
            precheck=refund_precheck,
            canonical_target=lambda args: f"{getattr(args, 'order_id')}:{getattr(args, 'amount')}",
        ),
        ToolDefinition(
            name="run_shell",
            description="教学用模拟 Shell，不执行真实系统命令",
            parameters_model=RunShellArgs,
            policy=ToolPolicy(Effect.SHELL, Risk.MEDIUM, "shell:run", False, 1.0, 0, False),
            handler=simulated_shell_handler,
            canonical_target=lambda args: str(getattr(args, "command")),
        ),
        ToolDefinition(
            name="transfer",
            description="为当前租户的账户执行转账操作",
            parameters_model=TransferArgs,
            policy=ToolPolicy(Effect.WRITE, Risk.HIGH, "transfer:execute", True, 1.5, 0, False),
            handler=transfer_handler,
            precheck=transfer_precheck,
            canonical_target=lambda args: f"{args.from_account}->{args.to_account}"
        )
    ]


DEFAULT_RULES = (
    PermissionRule("deny", "run_shell", "rm -rf"),
    PermissionRule("deny", "run_shell", "git push --force"),
    PermissionRule("allow", "run_shell", "pytest"),
)


def base_context(**overrides: Any) -> ExecutionContext:
    context = ExecutionContext(
        trace_id="trace_demo",
        user_id="u_100",
        tenant_id="tenant_a",
        mode=PermissionMode.DEFAULT,
        permissions=frozenset({"order:read", "refund:create", "shell:run"}),
        allowed_tools=frozenset({"get_order", "create_refund", "run_shell"}),
    )
    return replace(context, **overrides)


def build_runtime(
    *,
    approvals: ApprovalStore | None = None,
    audit: AuditSink | None = None,
    rules: Sequence[PermissionRule] = DEFAULT_RULES,
) -> tuple[ToolRuntime, ApprovalStore, AuditSink]:
    approval_store = approvals or ApprovalStore()
    audit_sink = audit or AuditSink()
    engine = PermissionEngine(rules, approval_store)
    return ToolRuntime(build_tools(), engine, audit_sink), approval_store, audit_sink


async def run_offline_demo() -> None:
    reset_side_effects()
    runtime, approvals, audit = build_runtime()
    context = base_context()
    refund_arguments = {"order_id": "ord_1001", "amount": 399.0, "reason": "商品存在质量问题"}

    results = [
        await runtime.invoke(ToolCall("call_01", "get_order", {"order_id": "ord_1001"}), context),
        await runtime.invoke(ToolCall("call_02", "create_refund", refund_arguments), context),
    ]
    approvals.approve("approval_01", context, "create_refund", refund_arguments)
    results.append(
        await runtime.invoke(
            ToolCall("call_03", "create_refund", refund_arguments),
            replace(context, approval_id="approval_01"),
        )
    )
    results.extend(
        [
            await runtime.invoke(
                ToolCall(
                    "call_04",
                    "create_refund",
                    {**refund_arguments, "user_id": "admin", "approved": True},
                ),
                context,
            ),
            await runtime.invoke(
                ToolCall("call_05", "run_shell", {"command": "rm -rf /tmp/demo"}),
                replace(context, mode=PermissionMode.BYPASS_PERMISSIONS),
            ),
            await runtime.invoke(
                ToolCall("call_06", "create_refund", refund_arguments),
                replace(context, mode=PermissionMode.PLAN, approval_id="approval_01"),
            ),
        ]
    )

    for result in results:
        print(json.dumps(result.__dict__ if hasattr(result, "__dict__") else {
            "tool_call_id": result.tool_call_id,
            "tool_name": result.tool_name,
            "ok": result.ok,
            "action": result.action,
            "code": result.code,
            "content": result.content,
        }, ensure_ascii=False, default=str))
    print(json.dumps({"side_effects": SIDE_EFFECTS, "audit_records": len(audit.records)}, ensure_ascii=False))


async def run_deepseek_agent(user_input: str) -> None:
    """可选真实模型闭环；所有工具调用仍经过同一个 ToolRuntime.invoke。"""

    from openai import AsyncOpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")

    runtime, _, _ = build_runtime()
    context = base_context(allowed_tools=frozenset({"get_order"}))
    client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "你是订单助手。只根据工具结果回答，不得伪造订单事实。",
        },
        {"role": "user", "content": user_input},
    ]

    for _round in range(8):
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=runtime.model_tools(context),
            stream=True,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text_parts: list[str] = []
        pending_calls: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text_parts.append(delta.content)
                print(delta.content, end="", flush=True)
            for delta_call in delta.tool_calls or []:
                current = pending_calls.setdefault(
                    delta_call.index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if delta_call.id:
                    current["id"] = delta_call.id
                if delta_call.function:
                    if delta_call.function.name:
                        current["function"]["name"] += delta_call.function.name
                    if delta_call.function.arguments:
                        current["function"]["arguments"] += delta_call.function.arguments

        provider_calls = [pending_calls[index] for index in sorted(pending_calls)]
        assistant_message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
        if provider_calls:
            assistant_message["tool_calls"] = provider_calls
        messages.append(assistant_message)

        if not provider_calls:
            print()
            return

        if text_parts:
            print()
        for provider_call in provider_calls:
            try:
                raw_arguments = json.loads(provider_call["function"]["arguments"])
            except json.JSONDecodeError:
                raw_arguments = {"_invalid_json": provider_call["function"]["arguments"]}
            result = await runtime.invoke(
                ToolCall(provider_call["id"], provider_call["function"]["name"], raw_arguments),
                context,
            )
            print(f"[tool_result] {result.tool_name} {result.code}")
            messages.append(result.to_tool_message())

    raise RuntimeError("Agent Loop 超过最大轮数 8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python 工具治理与权限状态机演示")
    parser.add_argument("--agent", action="store_true", help="使用 DeepSeek 运行真实 Agent Loop")
    parser.add_argument("--input", default="请查询订单 ord_1001 的状态和可退金额")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(run_deepseek_agent(cli_args.input) if cli_args.agent else run_offline_demo())
