#!/usr/bin/env python3
"""Test the provisional OpenReward TerminalHorizon hidden final-grader API.

This is deliberately separate from a full training run because a hidden
submission API may differ across OpenReward SDK/environment versions.
Requires OPENREWARD_API_KEY and network access. This opens one task session.
"""
import os
from openreward import OpenReward


def main() -> None:
    if not os.environ.get("OPENREWARD_API_KEY"):
        raise SystemExit("Set OPENREWARD_API_KEY first")
    client = OpenReward()
    env = client.environments.get(name="GeneralReasoning/TerminalHorizon")

    print("Opening a single TerminalHorizon test session (train index 0).")
    with env.session(split="train", index=0) as session:
        # A no-action submission is expected to grade the initial state (often 0).
        # The key check is that the API returns an actual terminal reward.
        result = session.call_tool(tool_name="submit_answer", input={})
        print("Finished:", result.finished)
        print("Reward:", result.reward)
        if not result.finished or result.reward is None:
            raise RuntimeError("Grader did not return a finished result with reward")

    close = getattr(client, "close", None)
    if callable(close):
        close()
    print("Grader contract test passed. This does NOT guarantee successful RL training.")


if __name__ == "__main__":
    main()
