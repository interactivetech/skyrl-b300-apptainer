#!/usr/bin/env python3
"""Apply provisional terminal final-submission handling to SkyRL OpenReward adapter.

Verified against the main-branch env.py block reviewed 2026-11-20.
This is a candidate integration patch, NOT a confirmed working SDK contract:
smoke-test the hidden submit_answer call before paid training.
"""
from pathlib import Path
import sys

path = Path('examples/train_integrations/openreward/env.py')
if not path.exists():
    sys.exit('Run from the SkyRL repository root')
s = path.read_text()
old = '''        # No tool call found — treat as final answer, end episode
        if tool_call is None:
            # No new tool call this turn, so reward = 0 (previous rewards already returned per-step)
            reward = 0.0
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"stop_reason": "no_tool_call"},
            )'''
new = '''        # No tool call: a terminal agent has submitted its final sandbox state.
        if tool_call is None:
            reward = 0.0
            if self.env_name == "GeneralReasoning/TerminalHorizon":
                # TerminalHorizon documents a hidden submit_answer grader.
                # Verify this call against your OpenReward SDK/environment version.
                try:
                    result = _retry_on_server_error(
                        self._session.call_tool,
                        tool_name="submit_answer",
                        input={},
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "TerminalHorizon final grading failed; do not train on zero rewards"
                    ) from exc
                if not result.finished or result.reward is None:
                    raise RuntimeError(
                        f"Unexpected TerminalHorizon grade: finished={result.finished}, reward={result.reward}"
                    )
                reward = float(result.reward)
                self._rewards.append(reward)
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"stop_reason": "no_tool_call"},
            )'''
if new in s:
    print('Already patched')
elif old not in s:
    sys.exit('Upstream env.py changed: inspect and patch manually; file not modified')
else:
    path.with_suffix('.py.pre-terminal-patch').write_text(s)
    path.write_text(s.replace(old, new, 1))
    print('Patched', path)
