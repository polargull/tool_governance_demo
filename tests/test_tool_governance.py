"""第二章作业验收：转账工具（transfer）的 5 个链路测试。

约法三章（作业"不能改动的地方"）：
- 所有调用都经过 `ToolRuntime.invoke`，不直接调 handler。
- 不改 `PermissionEngine.decide` 的优先级顺序，测试只观察它的输出。

未完成对应任务时，测试给的是"哪一步没做"的失败信息，而不是导入错误：
缺少 `ACCOUNTS` 会命中 `accounts()` 的断言，没注册 transfer 工具会拿到 `TOOL_NOT_FOUND`。
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from typing import Any

import pytest

import tool_governance_demo as demo

TRANSFER = "transfer"
APPROVAL_ID = "approval_transfer"
FROM_ACCOUNT = "ACC-A-123456"  # tenant_a，余额 100000.0
TO_ACCOUNT = "ACC-A-654321"  # tenant_a，余额 5000.0
SMALL_ACCOUNT = "ACC-A-888888"  # tenant_a，余额 20000.0
SLEEP_SECONDS = 3.0  # transfer_handler 里刻意制造的慢调用


def async_test(test: Callable[..., Awaitable[None]]) -> Callable[..., None]:
    """不依赖 pytest-asyncio，让 async 测试在任何 pytest 环境下都能跑。"""

    @functools.wraps(test)
    def run(*args: Any, **kwargs: Any) -> None:
        asyncio.run(test(*args, **kwargs))

    return run


def transfer_context(**overrides: Any) -> demo.ExecutionContext:
    """带 transfer:execute 权限和 transfer 执行白名单的上下文。"""

    defaults: dict[str, Any] = {
        "permissions": frozenset({"order:read", "refund:create", "shell:run", "transfer:execute"}),
        "allowed_tools": frozenset({"get_order", "create_refund", "run_shell", "transfer"}),
    }
    return demo.base_context(**{**defaults, **overrides})


def accounts() -> dict[tuple[str, str], float]:
    store = getattr(demo, "ACCOUNTS", None)
    assert isinstance(store, dict), "任务 1 未完成：tool_governance_demo.ACCOUNTS 不存在"
    return store


def balance(account: str, tenant_id: str = "tenant_a") -> float:
    store = accounts()
    assert (tenant_id, account) in store, f"任务 1 未完成：ACCOUNTS 缺少 {(tenant_id, account)}"
    return store[(tenant_id, account)]


def approve(approvals: demo.ApprovalStore, arguments: Mapping[str, Any]) -> None:
    """对同一份参数做一次性审批（摘要绑定，值必须与调用时逐字节一致）。"""

    approvals.approve(APPROVAL_ID, transfer_context(), TRANSFER, dict(arguments))


async def transfer(
    runtime: demo.ToolRuntime,
    tool_call_id: str,
    arguments: Mapping[str, Any],
    **context_overrides: Any,
) -> demo.ToolResult:
    return await runtime.invoke(
        demo.ToolCall(tool_call_id, TRANSFER, arguments),
        transfer_context(**context_overrides),
    )


@pytest.fixture(autouse=True)
def isolated_accounts() -> Iterator[None]:
    """ACCOUNTS 是模块级可变状态，逐个用例还原，保证互相隔离且可重复运行。"""

    store = getattr(demo, "ACCOUNTS", None)
    snapshot = dict(store) if isinstance(store, dict) else None
    demo.reset_side_effects()
    yield
    if snapshot is not None:
        store.clear()
        store.update(snapshot)


@async_test
async def test_transfer_rejects_injected_arguments_and_invalid_amount() -> None:
    """参数校验：extra="forbid" 挡住模型注入，Field 约束挡住非法账号与额度。"""

    runtime, _approvals, _audit = demo.build_runtime()
    valid = {"from_account": FROM_ACCOUNT, "to_account": TO_ACCOUNT, "amount": 100.0}

    injected = await transfer(runtime, "call_tr_inject", {**valid, "approved": True, "user_id": "u_admin"})
    assert injected.ok is False
    assert injected.code == "INVALID_ARGUMENT"

    malformed = await transfer(runtime, "call_tr_format", {**valid, "from_account": "ACC-123456"})
    assert malformed.code == "INVALID_ARGUMENT"

    zero = await transfer(runtime, "call_tr_zero", {**valid, "amount": 0.0})
    assert zero.code == "INVALID_ARGUMENT"

    over_schema = await transfer(runtime, "call_tr_schema", {**valid, "amount": 100_000.1})
    assert over_schema.code == "INVALID_ARGUMENT"

    assert balance(FROM_ACCOUNT) == 100_000.0


@async_test
async def test_transfer_precheck_blocks_over_limit_and_insufficient_balance() -> None:
    """业务预检：单笔限额与余额充足都是 handler 之前的确定性拒绝。"""

    runtime, approvals, _audit = demo.build_runtime()

    over_limit = {"from_account": FROM_ACCOUNT, "to_account": TO_ACCOUNT, "amount": 60_000.0}
    approve(approvals, over_limit)
    rejected = await transfer(runtime, "call_tr_limit", over_limit, approval_id=APPROVAL_ID)
    assert rejected.ok is False
    assert rejected.action is demo.DecisionAction.DENY
    assert rejected.code == "EXCEED_LIMIT"

    short = {"from_account": SMALL_ACCOUNT, "to_account": TO_ACCOUNT, "amount": 30_000.0}
    approve(approvals, short)
    rejected = await transfer(runtime, "call_tr_balance", short, approval_id=APPROVAL_ID)
    assert rejected.action is demo.DecisionAction.DENY
    assert rejected.code == "INSUFFICIENT_BALANCE"

    assert balance(FROM_ACCOUNT) == 100_000.0
    assert balance(SMALL_ACCOUNT) == 20_000.0
    assert balance(TO_ACCOUNT) == 5_000.0


@async_test
async def test_transfer_is_denied_without_permission_or_whitelist_and_in_plan_mode() -> None:
    """权限判断：带上审批也越不过 RBAC、执行白名单和 plan 只读契约。"""

    runtime, approvals, _audit = demo.build_runtime()
    arguments = {"from_account": FROM_ACCOUNT, "to_account": TO_ACCOUNT, "amount": 1_000.0}
    approve(approvals, arguments)
    approved: dict[str, Any] = {"approval_id": APPROVAL_ID}

    no_permission = await transfer(
        runtime,
        "call_tr_rbac",
        arguments,
        permissions=frozenset({"order:read", "refund:create", "shell:run"}),
        **approved,
    )
    assert no_permission.action is demo.DecisionAction.DENY
    assert no_permission.code == "PERMISSION_DENIED"

    not_whitelisted = await transfer(
        runtime,
        "call_tr_whitelist",
        arguments,
        allowed_tools=frozenset({"get_order", "create_refund", "run_shell"}),
        **approved,
    )
    assert not_whitelisted.code == "TOOL_NOT_ALLOWED"

    plan_mode = await transfer(
        runtime,
        "call_tr_plan",
        arguments,
        mode=demo.PermissionMode.PLAN,
        **approved,
    )
    assert plan_mode.code == "PLAN_MODE_DENIED"

    assert balance(FROM_ACCOUNT) == 100_000.0


@async_test
async def test_transfer_requires_approval_bound_to_arguments_and_masks_accounts() -> None:
    """人工审批 + 成功路径 + 结果脱敏 + 审计追踪。"""

    runtime, approvals, audit = demo.build_runtime()
    arguments = {"from_account": FROM_ACCOUNT, "to_account": TO_ACCOUNT, "amount": 1_200.0}

    pending = await transfer(runtime, "call_tr_000001", arguments)
    assert pending.ok is False
    assert pending.action is demo.DecisionAction.CONFIRM
    assert pending.code == "APPROVAL_REQUIRED"
    assert balance(FROM_ACCOUNT) == 100_000.0

    approve(approvals, arguments)
    tampered = await transfer(
        runtime, "call_tr_000002", {**arguments, "amount": 9_000.0}, approval_id=APPROVAL_ID
    )
    assert tampered.action is demo.DecisionAction.CONFIRM
    assert tampered.code == "APPROVAL_REQUIRED"

    executed = await transfer(runtime, "call_tr_000003", arguments, approval_id=APPROVAL_ID)
    assert executed.ok is True
    assert executed.action is demo.DecisionAction.ALLOW
    assert executed.code == "OK"
    assert executed.content["txn_id"] == "000003"
    assert executed.content["amount"] == 1_200.0
    assert executed.content["status"] == "accepted"
    assert executed.content["from"] == "ACC-A-****3456"
    assert executed.content["to"] == "ACC-A-****4321"
    assert balance(FROM_ACCOUNT) == 98_800.0
    assert balance(TO_ACCOUNT) == 6_200.0

    assert [(record.phase, record.code) for record in audit.records] == [
        ("decision", "APPROVAL_REQUIRED"),
        ("decision", "APPROVAL_REQUIRED"),
        ("decision", "APPROVED"),
        ("execution", "OK"),
    ]

    replay = await transfer(runtime, "call_tr_000004", arguments, approval_id=APPROVAL_ID)
    assert replay.action is demo.DecisionAction.CONFIRM


@async_test
async def test_transfer_timeout_is_reported_as_unknown_and_leaves_balances_untouched() -> None:
    """超时处理：非幂等写超时是 TIMEOUT_UNKNOWN，且超时点必须早于扣款。"""

    runtime, approvals, audit = demo.build_runtime()
    # 50000 是 precheck 的边界（spec 是 amount > 50000 才拒绝），也是限额内最大的合法金额：
    # 只要慢调用阈值严格小于 50000，这笔一定会进入 sleep(3.0) 分支。
    arguments = {"from_account": FROM_ACCOUNT, "to_account": TO_ACCOUNT, "amount": 50_000.0}
    approve(approvals, arguments)

    started = time.perf_counter()
    result = await transfer(runtime, "call_tr_timeout", arguments, approval_id=APPROVAL_ID)
    elapsed = time.perf_counter() - started

    assert result.ok is False
    assert result.code == "TIMEOUT_UNKNOWN"
    assert elapsed < SLEEP_SECONDS
    assert balance(FROM_ACCOUNT) == 100_000.0
    assert balance(TO_ACCOUNT) == 5_000.0
    assert audit.records[-1].phase == "execution"
    assert audit.records[-1].code == "TIMEOUT_UNKNOWN"
