# Contradiction Checkpoint

A tiny guardrail for agent tool calls: **re-check the last few turns for
contradictions immediately before an irreversible action executes.**

## Why this, specifically

There's already a crowded field of agent guardrails — budget limiters, secret
scanners, prompt-injection filters, static duplication checkers (AgentGuard,
Atlas Guardrails, NeMo Guardrails, and others). None of the ones we could find
do this narrow thing: sit right before an irreversible call (`delete_file`,
`deploy_to_production`, `send_email`, ...) and ask *"does anything said since
the plan was formed contradict this specific action?"*

That's the whole idea. One interception point, one judgment call.

This is built as a direct, honest response to
[AgentAbstain: Do LLM Agents Know When Not to Act?](https://arxiv.org/abs/2607.10059)
(arXiv:2607.10059, July 2026) — a paper that tested 17 frontier LLMs across
263 paired should-act/should-abstain tasks and found the *best* model gets
this kind of judgment call right only 59.5% of the time. **We are not
claiming to have solved that.** This tool ships a cheap, honest, narrower
version: a deterministic keyword/negation matcher that catches the *obvious*
class of contradictions (explicit reversals — "actually, don't...", "wait,
hold off...") for free, with an optional upgrade path to a real LLM judge for
the subtler cases the heuristic will miss. The demo shows both — including a
case it misses on purpose (Scenario 4).

## How it works

```python
from contradiction_checkpoint import CheckpointSession, ContradictionHalt

session = CheckpointSession(plan="Refactor auth. Do not touch tests/.")
session.add_turn("user", "actually keep the old test fixtures as reference, don't delete them")

@session.guard
def delete_file(path):
    os.remove(path)

delete_file("tests/fixtures/old.py")
# -> raises ContradictionHalt: the reversal in the turn above shares
#    keywords with this call, so the delete never runs.
```

`session.guard` wraps any callable. Before it runs, the session's judge is
handed `{plan, last N turns, proposed action}` and returns `ALLOW` or `HALT`.
On `HALT`, the wrapped function is **never called** — `ContradictionHalt` is
raised instead, with the judge's reasoning attached.

For an `async def` function, the check runs when its guarded coroutine starts,
using the history available then. Creating a coroutine object does not authorize
future execution. Only a `CheckResult` containing the actual `Verdict.ALLOW`
enum permits execution; malformed results are logged as `HALT`. Custom judges
must return the declared enum rather than a plain string.

This is an entry check, not continuous authorization across later `await`
points inside a running tool. Generators, async generators and callable objects
that return deferred work require separate interception of their execution.

Two judges ship in the box:

| Judge | Dependencies | API key | What it does |
|---|---|---|---|
| `HeuristicJudge` (default) | none (stdlib only) | none | Deterministic negation/keyword matcher. What the demo runs on. |
| `AnthropicJudge` | `pip install anthropic` | `ANTHROPIC_API_KEY` | Sends the same `{plan, history, action}` to a small Claude model for a real judgment call. |

## 30-second demo

No install, no API key:

```bash
python demo.py
```

You'll see four scripted scenarios print live: an `ALLOW`, two `HALT`s (a
file deletion and a production deploy, each blocked by an explicit reversal
earlier in the conversation), and one deliberately-shown miss where the
heuristic judge lets an action through because the contradiction was phrased
too indirectly for keyword matching to catch — with an explanation of why,
right in the output.

## Running the tests

```bash
python -m unittest discover -s tests -v
```

9 tests, standard library only, no network.

## Using the real LLM judge

```bash
cp .env.example .env        # then fill in ANTHROPIC_API_KEY
pip install -r requirements.txt
```

```python
from contradiction_checkpoint import CheckpointSession, AnthropicJudge

session = CheckpointSession(plan="...", judge=AnthropicJudge())
```

Not required to run or evaluate this repo — the heuristic judge is the
default and what the demo is built around.

## What's in scope for this MVP, and what's cut

**Built:** the core `CheckpointSession`/`guard` decorator, both judges, a
scripted 4-scenario demo, a unit test suite.

**Cut for time** (would be next):
- Framework adapters for LangChain/LangGraph tool-call interception (right
  now you wrap your own functions directly with `@session.guard`, which
  works for any Python callable but isn't a drop-in for a specific agent
  framework's tool-call hook).
- Persistence of session history across process restarts (currently
  in-memory only, which is fine for a single running agent session).
- A real sentence splitter / NLP pipeline for the heuristic judge (currently
  a regex-based sentence split + stopword-filtered keyword overlap — good
  enough to be honest and fast, not a claim of general language
  understanding).

## Project layout

```
contradiction_checkpoint/
  core.py      CheckpointSession, guard decorator, ContradictionHalt
  judges.py    BaseJudge, HeuristicJudge, AnthropicJudge
demo.py        the 30-second demo
tests/         unit tests (stdlib unittest, no dependencies)
```

Requires Python 3.10+.
