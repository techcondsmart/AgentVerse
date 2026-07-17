"""Offline integration test for the FULL video_understanding swarm loop.

Runs `TaskSolving.run()` end-to-end — role assignment, horizontal critic
debate, solver, the `video-evidence` executor querying a real evidence index,
and the groundedness evaluator — with a *scripted* deterministic LLM, so it
needs no API keys and no network. This exercises exactly the machinery a real
run uses (prompt templating, output parsers, decision maker, executor seam,
evaluator gating); only the model's text is canned.

    python -m agentverse.video.swarm_selftest

The scripted model recognises which role is prompting it from the prompt text
and replies in that role's expected format, including one full "reject" round:
turn 0 the critics dissent and the evaluator scores 0, turn 1 everything is
grounded and the evaluator accepts — proving the loop iterates and terminates.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import Field

from agentverse.llms.base import BaseChatModel, BaseModelArgs, LLMResult


class ScriptedArgs(BaseModelArgs):
    model: str = Field(default="gpt-4")  # tiktoken-known name for token counting
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.0)


class ScriptedChat(BaseChatModel):
    """Deterministic stand-in for a chat LLM, keyed on prompt content."""

    args: ScriptedArgs = Field(default_factory=ScriptedArgs)
    calls: List[str] = Field(default_factory=list)
    state: Dict[str, int] = Field(default_factory=dict)

    def send_token_limit(self, model: str) -> int:
        return 8192

    def get_spend(self) -> float:
        return 0.0

    # -- role-detection + canned replies -----------------------------------
    def _reply(self, prepend: str, append: str) -> str:
        text = f"{prepend}\n{append}"
        if "recruit" in text and "expert" in text:
            self.calls.append("role_assigner")
            return (
                "1. a visual-scene analyst who reasons over frame captions and OCR.\n"
                "2. an audio/dialogue analyst who reasons over the transcript.\n"
                "3. a temporal-reasoning analyst verifying order and duration.\n"
                "4. a skeptical fact-checker demanding evidence for every claim."
            )
        if "Chain-of-Verification" in text:
            self.calls.append("critic")
            round_ = self.state.get("round", 0)
            if round_ == 0:
                return (
                    "Claim check: the draft says the warning text is red, but the "
                    "evidence at (00:07) shows it on a green background. This claim "
                    "is ungrounded. Corrected: the on-screen text \"SAFETY FIRST\" "
                    "appears at (00:07) on a green background."
                )
            return "All claims are grounded with timestamps. [Agree]"
        if "Provide your grounded answer" in text:
            self.calls.append("solver")
            round_ = self.state.get("round", 0)
            if round_ == 0:
                # deliberately imperfect first draft (colour claim is wrong)
                return (
                    "The video shows a warning in red text (00:07): \"SAFETY FIRST\"."
                )
            return (
                "The on-screen text \"SAFETY FIRST\" appears at (00:07) on a green "
                "background (vision, agreement=0.90). No other safety instruction "
                "is evidenced in the video."
            )
        if "Judge groundedness" in text or "Score:" in append:
            self.calls.append("evaluator")
            round_ = self.state.get("round", 0)
            self.state["round"] = round_ + 1
            if round_ == 0:
                return (
                    "Score: 0\n"
                    "Response: the colour of the text is misreported; cite the "
                    "green-background evidence at 00:07 and re-answer."
                )
            return "Score: 1\nResponse: every claim cites supporting evidence."
        self.calls.append("other")
        return "OK"

    def generate_response(self, prepend_prompt: str = "", history=None,
                          append_prompt: str = "", **kwargs) -> LLMResult:
        content = self._reply(prepend_prompt, append_prompt)
        return LLMResult(content=content, send_tokens=0, recv_tokens=0, total_tokens=0)

    async def agenerate_response(self, prepend_prompt: str = "", history=None,
                                 append_prompt: str = "", **kwargs) -> LLMResult:
        return self.generate_response(prepend_prompt, history, append_prompt)


def _offline_token_counting():
    """Make tiktoken work without network access.

    tiktoken lazily downloads its BPE ranks from openaipublic.blob.core.windows.net
    on first use; in sandboxes where that host is egress-blocked, token counting
    would crash the whole loop. Token counts only budget prompt trimming here, so
    a ~4-chars-per-token approximation is more than good enough for the test.
    Applied only in this self-test — production environments with network use the
    real tokenizer.
    """
    import tiktoken

    class _ApproxEncoding:
        def encode(self, text: str):
            return [0] * max(1, len(text) // 4)

        def decode(self, tokens):
            return ""

    try:  # only patch if the real BPE is unavailable
        tiktoken.encoding_for_model("gpt-4").encode("probe")
    except Exception:
        tiktoken.encoding_for_model = lambda model: _ApproxEncoding()
        tiktoken.get_encoding = lambda name: _ApproxEncoding()


def main() -> int:
    _offline_token_counting()
    from agentverse.tasksolving import TaskSolving
    from agentverse.utils import AGENT_TYPES
    from agentverse.video.captioning import FrameEvidence
    from agentverse.video.asr import TranscriptSegment
    from agentverse.video.retrieval import VideoEvidenceIndex

    # 1) evidence index (what perception would produce for the synthetic video)
    index = VideoEvidenceIndex().build(
        [
            FrameEvidence(timestamp=0.0, scene_id=0, agreement=0.5,
                          caption="red background | Objects: white square | Actions: a square moves"),
            FrameEvidence(timestamp=4.0, scene_id=1, agreement=1.0,
                          caption="blue background | Objects: yellow circle"),
            FrameEvidence(timestamp=7.0, scene_id=2, agreement=0.9,
                          caption='green background | On-screen text: "SAFETY FIRST"'),
        ],
        [TranscriptSegment(2.0, 4.0, "please wear a helmet at all times")],
    )

    # 2) build the swarm from the shipped config
    ts = TaskSolving.from_task("video_understanding", "agentverse/tasks/tasksolving")
    ts.environment.task_description = (
        "What safety message appears in the video, and when?"
    )
    env = ts.environment
    env.rule.executor.index = index
    env.rule.add_execution_result_to_critic = True
    env.rule.add_execution_result_to_solver = True

    # 3) swap every agent's LLM for the shared scripted model
    llm = ScriptedChat()
    for slot, agent_or_list in env.agents.items():
        if isinstance(agent_or_list, list):
            for a in agent_or_list:
                a.llm = llm
        else:
            agent_or_list.llm = llm

    # 4) run the full loop
    plan, result, logs = ts.run()

    # 5) assertions: the loop must have iterated (reject round) then converged
    assert env.success, "swarm never converged to an accepted answer"
    assert env.cnt_turn >= 2, "expected at least one reject round before success"
    assert "SAFETY FIRST" in plan and "(00:07)" in plan and "green" in plan, (
        f"final answer is not the grounded one: {plan!r}"
    )
    assert "SAFETY FIRST" in result, "executor evidence missing from final result"
    roles_called = set(llm.calls)
    assert {"role_assigner", "solver", "critic", "evaluator"} <= roles_called, (
        f"not every role was exercised: {roles_called}"
    )

    print("\nSWARM SELF-TEST PASSED:")
    print(f"  turns: {env.cnt_turn} (1 reject + 1 accept)")
    print(f"  llm calls by role: "
          f"{ {r: llm.calls.count(r) for r in sorted(set(llm.calls))} }")
    print(f"  final grounded answer: {plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
