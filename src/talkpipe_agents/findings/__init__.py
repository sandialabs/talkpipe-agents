"""The findings loop: tools that return findings, judged as they arrive.

An agent answering a question from what its tools turn up needs more
than tool calls: something must decide, after every call, which of the
results are worth keeping, remember what is already known, tell the
agent what it still lacks and how many turns it has left, and end the
run when it has enough. This package is that loop, independent of what
the tools search or compute::

    tool → findings → judge → session → reply to the model

One module per part:

- :mod:`.finding` — :class:`Finding`, one piece of information a tool
  returned: the text the judge and the agent read, a label and a source
  to cite, a key to deduplicate by, and whatever else the tool wants to
  carry along in ``meta`` (a ``SearchResult``, say); and how strings,
  dicts and search results become findings (:func:`as_findings`).
- :mod:`.session` — :class:`FindingsSession`, one question's state: the
  findings seen, kept (graded, best first) and ignored, why, the judge's
  verdict on what is kept, the stop rule, the turn countdown, how the
  run ended (:class:`FindingsOutcome`), and the standard reply the model
  reads.
- :mod:`.judge` — :class:`FindingsJudge`, one batched model call
  deciding keep or ignore, with a relevance grade, for every finding of
  a call, and whether what is kept would answer the question
  (:class:`Verdict`); read leniently (JSON, fenced, in prose, or one
  decision per line). Also the ``judgeFindings`` segment, for pipelines
  without an agent.
- :mod:`.hook` — :class:`JudgeFindings`, the Strands hook that runs the
  judge as a step of the agent loop: after a finding tool's call, before
  the model sees the result. The judging costs the agent no turn, and
  it cannot skip it. At ``max_kept`` a better-graded finding displaces
  the weakest kept one, so the kept list is the best N, not the first N.
  Its :meth:`~JudgeFindings.seed` judges a call made *before* the agent
  starts, so a run can end on the judge's verdict with no agent turn.
- :mod:`.finding_tools` — :func:`finding_tool`, which makes any function
  returning findings (or strings, or dicts, or search results) an agent
  tool that knows nothing about sessions or judges; and the registered
  session tools ``enough_information`` (the stop), ``keep_finding`` and
  ``ignore_finding`` (deciding by hand, for a session without a judge).

``llmAgent[judge=True]`` (or ``findings=True`` to decide by hand)
installs the hook and a session per item; see ``segments``. Everything
below is importable from ``talkpipe_agents.findings`` as before.
"""

from __future__ import annotations

from .finding import (
    DEFAULT_MAX_CHARS,
    Finding,
    FoundFindings,
    as_finding,
    as_findings,
    below_score_floor,
    render_findings,
)
from .finding_tools import (
    FINDINGS_PROTOCOL,
    FINDINGS_PROTOCOL_UNJUDGED,
    enough_information,
    finding_function,
    finding_tool,
    ignore_finding,
    keep_finding,
)
from .hook import FINDINGS_KEY, SESSION_KEY, JudgeFindings, session_of
from .judge import (
    JUDGE_SYSTEM_PROMPT,
    Decision,
    FindingsJudge,
    Judged,
    Judgement,
    Verdict,
    parse_judgements,
    parse_verdict,
)
from .session import (
    DEFAULT_MAX_KEPT,
    DEFAULT_SCORE,
    DEFAULT_STOP_INSTRUCTION,
    ENDINGS,
    JUDGE_ENOUGH_REASON,
    MAX_SCORE,
    MIN_SCORE,
    FindingsOutcome,
    FindingsSession,
    Need,
)

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_KEPT",
    "DEFAULT_SCORE",
    "DEFAULT_STOP_INSTRUCTION",
    "ENDINGS",
    "FINDINGS_KEY",
    "FINDINGS_PROTOCOL",
    "FINDINGS_PROTOCOL_UNJUDGED",
    "JUDGE_ENOUGH_REASON",
    "JUDGE_SYSTEM_PROMPT",
    "MAX_SCORE",
    "MIN_SCORE",
    "SESSION_KEY",
    "Decision",
    "Finding",
    "FindingsJudge",
    "FindingsOutcome",
    "FindingsSession",
    "FoundFindings",
    "JudgeFindings",
    "Judged",
    "Judgement",
    "Need",
    "Verdict",
    "as_finding",
    "as_findings",
    "below_score_floor",
    "enough_information",
    "finding_function",
    "finding_tool",
    "ignore_finding",
    "keep_finding",
    "parse_judgements",
    "parse_verdict",
    "render_findings",
    "session_of",
]
