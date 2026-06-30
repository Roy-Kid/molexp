"""Fan-out, conditionals, and diamond shapes — all via the DAG.

Matches ``docs/guide/control-flow.md``.

MolExp has no ``IfTask`` / ``ForTask``; control flow is expressed by the
shape of ``depends_on``, by Python inside tasks, and by ``wf.parallel`` for
runtime-sized fan-out. Every task body reads its inputs through named
parameters — the engine binds {upstream outputs} | {run params} | {build-time
config} by name; ``ctx`` is only kept when a body needs ``ctx.workdir``. This
example runs four patterns back to back:

1. Diamond — ``parse`` and ``validate`` run in parallel after ``fetch``.
2. Conditional — ``maybe_clean`` short-circuits based on a config field.
3. Build-time fan-out — two sibling tasks reduce into a third.
4. Runtime fan-out — ``wf.parallel`` maps a body task over an upstream list.

Run directly::

    python examples/workflow/control_flow.py
"""

from __future__ import annotations

import asyncio

from molexp.workflow import WorkflowCompiler, WorkflowRuntime


# ── 1. Diamond fan-out ─────────────────────────────────────────────────────
async def diamond_demo() -> None:
    wf = WorkflowCompiler(name="diamond")

    @wf.task
    async def fetch() -> dict:
        return {"raw": [1, 2, 3]}

    @wf.task(depends_on=["fetch"])
    async def parse(raw: list[int]) -> list[int]:
        # ``fetch``'s single upstream dict binds by name → ``raw``.
        return raw

    @wf.task(depends_on=["fetch"])
    async def validate(raw: list[int]) -> bool:
        return bool(raw)

    @wf.task(depends_on=["parse", "validate"])
    async def merge(parse: list[int], validate: bool) -> dict:
        # Two scalar upstreams bind by their dependency names.
        return {"parsed": parse, "ok": validate}

    result = await WorkflowRuntime().execute(wf.compile())
    print(f"diamond:     {result.outputs['merge']}")


# ── 2. Conditional branch inside a task ────────────────────────────────────
async def conditional_demo(skip: bool) -> None:
    wf = WorkflowCompiler(name="conditional")

    @wf.task
    async def fetch() -> list[int]:
        return [5, -3, 2, -1, 4]

    @wf.task(depends_on=["fetch"])
    async def maybe_clean(items: list[int], skip_cleaning: bool = False) -> list[int]:
        # ``items`` = ``fetch``'s list (single upstream, bound positionally);
        # ``skip_cleaning`` = the build-time ``config`` field, bound by name.
        if skip_cleaning:
            return items
        return [x for x in items if x >= 0]

    result = await WorkflowRuntime().execute(wf.compile(), config={"skip_cleaning": skip})
    tag = "raw    " if skip else "cleaned"
    print(f"conditional {tag}: {result.outputs['maybe_clean']}")


# ── 3. Build-time fan-out ──────────────────────────────────────────────────
async def fanout_demo() -> None:
    wf = WorkflowCompiler(name="fanout")

    @wf.task
    async def load() -> list[int]:
        return [1, 2, 3, 4]

    @wf.task(depends_on=["load"], name="square_evens")
    async def square_evens(items: list[int]) -> int:
        return sum(x * x for x in items if x % 2 == 0)

    @wf.task(depends_on=["load"], name="square_odds")
    async def square_odds(items: list[int]) -> int:
        return sum(x * x for x in items if x % 2 == 1)

    @wf.task(depends_on=["square_evens", "square_odds"])
    async def total(square_evens: int, square_odds: int) -> int:
        return square_evens + square_odds

    result = await WorkflowRuntime().execute(wf.compile())
    print(
        f"fan-out:     squares_evens={result.outputs['square_evens']}, "
        f"squares_odds={result.outputs['square_odds']}, total={result.outputs['total']}"
    )


# ── 4. Runtime fan-out — ``wf.parallel`` over an upstream list ─────────────
async def parallel_demo() -> None:
    wf = WorkflowCompiler(name="parallel", entry="scatter")

    @wf.task
    async def scatter() -> list[int]:
        return [1, 2, 3, 4]

    @wf.task
    async def square(value: int) -> int:
        return value**2  # ``value`` is one fan-out element

    @wf.task
    async def gather(values: list[int]) -> int:
        return sum(values)  # one squared value per element, in order

    wf.parallel(map_over="scatter", body="square", join="gather", max_concurrency=2)

    result = await WorkflowRuntime().execute(wf.compile())
    print(f"parallel:    square={result.outputs['square']}, gather={result.outputs['gather']}")


async def main() -> None:
    await diamond_demo()
    await conditional_demo(skip=False)
    await conditional_demo(skip=True)
    await fanout_demo()
    await parallel_demo()


if __name__ == "__main__":
    asyncio.run(main())
