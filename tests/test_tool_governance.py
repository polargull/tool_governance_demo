"""四、必须通过的 5 个转账测试"""

import asyncio
import json
import pytest

from tool_governance_demo import (
    ApprovalStore,
    ArgsModel,
    AuditSink,
    Effect,
    ExecutionContext,
    PermissionEngine,
    PolicyDenied,
    Risk,
    SIDE_EFFECTS,
    ToolCall,
    ToolDefinition,
    ToolPolicy,
    ToolRuntime,
    TransferArgs,
    base_context,
    build_runtime,
    build_tools,
    reset_side_effects,
    transfer_handler,
)


@pytest.fixture(autouse=True)
def _reset():
    """每个测试前重置副作用计数。"""
    reset_side_effects()
    yield


def _transfer_context(**overrides):
    """构造包含 transfer 权限与白名单的上下文。"""
    ctx = base_context(
        permissions=frozenset(
            {"order:read", "refund:create", "shell:run", "transfer:execute"}
        ),
        allowed_tools=frozenset(
            {"get_order", "create_refund", "run_shell", "transfer"}
        ),
    )
    from dataclasses import replace

    return replace(ctx, **overrides)


# ─────────────────────────────────────────────
# 审计日志打印辅助函数
# ─────────────────────────────────────────────
def print_audit_logs(audit, label=""):
    """打印审计日志，验证账号脱敏和审批状态。"""
    print(f"\n{'='*60}")
    print(f"审计日志 {label}")
    print(f"{'='*60}")
    for i, record in enumerate(audit.records, 1):
        print(f"\n记录 {i}:")
        print(f"  Phase: {record.phase}")
        print(f"  Decision: {record.decision}")
        print(f"  Code: {record.code}")
        print(f"  Argument keys: {record.argument_keys}")
        if record.redacted_arguments:
            print(f"  Redacted arguments: {json.dumps(record.redacted_arguments, ensure_ascii=False)}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# 1. Schema 校验：格式错误或多传 approved 字段
# ─────────────────────────────────────────────
@pytest.mark.anyio
@pytest.mark.parametrize(
    "arguments, label",
    [
        (
            {
                "from_account": "bad-account",
                "to_account": "ACC-A-654321",
                "amount": 100,
            },
            "bad_from_account",
        ),
        (
            {
                "from_account": "ACC-A-123456",
                "to_account": "ACC-A-654321",
                "amount": 100,
                "approved": True,
            },
            "extra_approved_field",
        ),
    ],
)
async def test_transfer_schema_rejects_extra(arguments, label):
    runtime, _, audit = build_runtime()
    ctx = _transfer_context()
    result = await runtime.invoke(ToolCall("call_schema", "transfer", arguments), ctx)

    # 打印审计日志
    print_audit_logs(audit, f"- Schema 校验测试 ({label})")

    assert result.code == "INVALID_ARGUMENT", (
        f"[{label}] 期望 INVALID_ARGUMENT，实际 {result.code}"
    )
    assert SIDE_EFFECTS["transfer_executions"] == 0, (
        f"[{label}] 副作用应为 0，实际 {SIDE_EFFECTS['transfer_executions']}"
    )


# ─────────────────────────────────────────────
# 2. 预检：余额不足
# ─────────────────────────────────────────────
@pytest.mark.anyio
async def test_transfer_precheck_insufficient():
    """ACC-A-654321 余额 5000，转账 6000 → INSUFFICIENT_BALANCE。"""
    runtime, _, audit = build_runtime()
    ctx = _transfer_context()
    arguments = {
        "from_account": "ACC-A-654321",
        "to_account": "ACC-A-123456",
        "amount": 6000,
    }
    result = await runtime.invoke(ToolCall("call_insufficient", "transfer", arguments), ctx)

    # 打印审计日志
    print_audit_logs(audit, "- 余额不足预检测试")

    assert result.code == "INSUFFICIENT_BALANCE", (
        f"期望 INSUFFICIENT_BALANCE，实际 {result.code}"
    )
    assert SIDE_EFFECTS["transfer_executions"] == 0, (
        f"副作用应为 0，实际 {SIDE_EFFECTS['transfer_executions']}"
    )


# ─────────────────────────────────────────────
# 3. 预检：超出单笔限额
# ─────────────────────────────────────────────
@pytest.mark.anyio
async def test_transfer_precheck_exceed_limit():
    """转账 60000（超 5 万）→ EXCEED_LIMIT。"""
    runtime, _, audit = build_runtime()
    ctx = _transfer_context()
    arguments = {
        "from_account": "ACC-A-123456",
        "to_account": "ACC-A-654321",
        "amount": 60000,
    }
    result = await runtime.invoke(ToolCall("call_exceed", "transfer", arguments), ctx)

    # 打印审计日志
    print_audit_logs(audit, "- 超额预检测试")

    assert result.code == "EXCEED_LIMIT", (
        f"期望 EXCEED_LIMIT，实际 {result.code}"
    )
    assert SIDE_EFFECTS["transfer_executions"] == 0, (
        f"副作用应为 0，实际 {SIDE_EFFECTS['transfer_executions']}"
    )


# ─────────────────────────────────────────────
# 4. 审批绑定：审批金额 100，执行时改为 200
# ─────────────────────────────────────────────
@pytest.mark.anyio
async def test_transfer_approval_binding():
    """审批时金额 100，执行时改为 200 → APPROVAL_REQUIRED（旧审批失效）。"""
    runtime, approvals, audit = build_runtime()
    ctx = _transfer_context()

    approved_args = {
        "from_account": "ACC-A-123456",
        "to_account": "ACC-A-654321",
        "amount": 100,
    }
    approvals.approve("approval_bind", ctx, "transfer", approved_args)

    # 执行时金额改为 200，digest 不匹配
    executed_args = {**approved_args, "amount": 200}
    result = await runtime.invoke(
        ToolCall("call_binding", "transfer", executed_args),
        _transfer_context(approval_id="approval_bind"),
    )

    # 打印审计日志
    print_audit_logs(audit, "- 审批绑定测试")

    assert result.code == "APPROVAL_REQUIRED", (
        f"期望 APPROVAL_REQUIRED，实际 {result.code}"
    )
    assert SIDE_EFFECTS["transfer_executions"] == 0, (
        f"副作用应为 0，实际 {SIDE_EFFECTS['transfer_executions']}"
    )


# ─────────────────────────────────────────────
# 5. 超时不重试：转账 90000 触发超时
# ─────────────────────────────────────────────
@pytest.mark.anyio
async def test_transfer_timeout_no_retry():
    """转账 90000（>80000 会 sleep 3s，超过 1.5s 超时）→ TIMEOUT_UNKNOWN，transfer_executions <= 1。

    由于默认 precheck 限额为 50000，此处构建一个 precheck 限额为 100000 的
    自定义 runtime，使 90000 能进入 handler 触发超时。
    """
    from tool_governance_demo import (
        ACCOUNTS,
        ArgsModel,
        ExecutionContext,
        PolicyDenied,
        TransferArgs,
        transfer_handler,
    )

    async def _lenient_precheck(raw_arguments: ArgsModel, context: ExecutionContext) -> None:
        arguments = raw_arguments
        assert isinstance(arguments, TransferArgs)
        # 限额提高到 100000，允许 90000 进入 handler
        if arguments.amount > 100_000:
            raise PolicyDenied("EXCEED_LIMIT", "转账金额超过单笔限额")
        balance = ACCOUNTS.get((context.tenant_id, arguments.from_account), 0.0)
        if balance < arguments.amount:
            raise PolicyDenied("INSUFFICIENT_BALANCE", "转出账户余额不足")

    from tool_governance_demo import (
        Effect,
        Risk,
        build_tools,
    )

    tools = build_tools()
    # 替换 transfer 工具的 precheck 和 policy（关闭审批要求，使 90000 能进入 handler 触发超时）
    tools = [
        ToolDefinition(
            name=t.name,
            description=t.description,
            parameters_model=t.parameters_model,
            policy=ToolPolicy(
                effect=t.policy.effect,
                risk=Risk.MEDIUM,  # 降低风险等级，避免触发审批检查
                permission=t.policy.permission,
                requires_approval=False,  # 关闭审批，直接进入 handler
                timeout_seconds=t.policy.timeout_seconds,
                max_retries=t.policy.max_retries,
                idempotent=t.policy.idempotent,
            ) if t.name == "transfer" else t.policy,
            handler=t.handler,
            canonical_target=t.canonical_target,
            precheck=_lenient_precheck if t.name == "transfer" else t.precheck,
        )
        for t in tools
    ]

    approvals = ApprovalStore()
    audit = AuditSink()
    from tool_governance_demo import PermissionEngine

    engine = PermissionEngine((), approvals)
    runtime = ToolRuntime(tools, engine, audit)

    ctx = _transfer_context()
    arguments = {
        "from_account": "ACC-A-123456",
        "to_account": "ACC-A-888888",
        "amount": 90000,
    }
    result = await runtime.invoke(ToolCall("call_timeout", "transfer", arguments), ctx)

    # 打印审计日志
    print_audit_logs(audit, "- 超时测试")

    assert result.code == "TIMEOUT_UNKNOWN", (
        f"期望 TIMEOUT_UNKNOWN，实际 {result.code}"
    )
    assert SIDE_EFFECTS["transfer_executions"] <= 1, (
        f"transfer_executions 应 <= 1，实际 {SIDE_EFFECTS['transfer_executions']}"
    )


# ─────────────────────────────────────────────
# 6. 额外测试：验证审计日志中的账号脱敏和 CONFIRM 状态
# ─────────────────────────────────────────────
@pytest.mark.anyio
async def test_audit_log_redaction_and_confirm():
    """验证审计日志中账号已脱敏，且审批流程执行前返回 CONFIRM 状态。"""
    runtime, _, audit = build_runtime()
    ctx = _transfer_context()
    
    arguments = {
        "from_account": "ACC-A-123456",
        "to_account": "ACC-A-654321",
        "amount": 100,
    }
    
    # 执行转账（没有审批ID，应该触发 CONFIRM）
    result = await runtime.invoke(
        ToolCall("call_audit_test", "transfer", arguments),
        ctx
    )
    
    # 打印审计日志
    print_audit_logs(audit, "- 审计日志脱敏和CONFIRM状态测试")
    
    # 验证决策阶段返回 CONFIRM
    decision_record = audit.records[0]
    assert decision_record.phase == "decision", "应该有决策阶段的审计记录"
    assert decision_record.decision == "confirm", (
        f"期望 decision 为 confirm，实际 {decision_record.decision}"
    )
    assert decision_record.code == "APPROVAL_REQUIRED", (
        f"期望 code 为 APPROVAL_REQUIRED，实际 {decision_record.code}"
    )
    
    # 验证审计日志中包含脱敏后的参数
    assert decision_record.redacted_arguments is not None, "审计记录应包含脱敏后的参数"
    redacted = decision_record.redacted_arguments
    
    # 验证账号已脱敏
    assert redacted["from_account"] == "ACC-A-****3456", (
        f"from_account 应脱敏为 ACC-A-****3456，实际 {redacted['from_account']}"
    )
    assert redacted["to_account"] == "ACC-A-****4321", (
        f"to_account 应脱敏为 ACC-A-****4321，实际 {redacted['to_account']}"
    )
    assert redacted["amount"] == 100, "金额不应被脱敏"
    
    # 验证没有执行（因为返回了 CONFIRM）
    assert result.action == "confirm", f"期望 action 为 confirm，实际 {result.action}"
    assert SIDE_EFFECTS["transfer_executions"] == 0, (
        f"CONFIRM 状态下不应执行，实际执行次数 {SIDE_EFFECTS['transfer_executions']}"
    )
