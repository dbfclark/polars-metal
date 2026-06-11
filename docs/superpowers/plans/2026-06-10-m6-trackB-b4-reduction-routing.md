# B4 — Reduction-Routing Guard + Re-Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each task is a self-contained loop (failing test/bench → run → minimal impl → run/PASS → BOTH `ruff check` AND `ruff format --check` → commit); the orchestrator reviews each task's diff before dispatching the next.

**Goal:** Lock in the empirically-measured decision that **bare `sum`/`min`/`max`/`mean` reductions stay on CPU while `std`/`var` route to GPU**, with a dispatch-asserted regression test across int + F32 dtypes, a permanent honest end-to-end reduction benchmark, and corrected spec/roadmap framing — **no routing flip** (the end-to-end spike refuted the "flip bare reductions to GPU" premise).

**Architecture:** B4 was scoped to "measure end-to-end, install a size-aware threshold `N₀`, flip routing where GPU wins." The mandatory pre-plan spike (2026-06-10, `scripts/spike_b4_reduction_routing.py` + `spike_b4_breakdown.py`) **refuted that premise**: the in-engine GPU bare-reduction path loses 2–5× at every size 1M→100M with no crossover, because a bare reduction is bandwidth-bound (1 flop/element) and the host→MLX ingest alone exceeds Polars' whole multithreaded SIMD scan. The brainstorm spike was wrong because it compared *resident* MLX (no ingest, 0.95ms@100M) against *single-threaded* numpy (~17ms) — not real Polars CPU (2.4ms). So B4 ships as a **guard + re-baseline**: codify the negative result so no future change silently flips it, and convert the throwaway spikes into a permanent benchmark with honest gates. `std`/`var` remain on GPU because they are genuine 5–9× wins (Polars CPU std/var is a slow two-pass Welford, far from bandwidth) — the compute-intensity gate (CLAUDE.md principle #3) is vindicated, not changed.

**Tech Stack:** Python (`polars_metal` walker routing — unchanged; `tests/python_integration` regression test; `tests/bench/m4_survey` benchmark + `tests/bench/baseline.json` gate); Polars CPU as the differential oracle; the existing `_native.execute_fused_expr` dispatch counter as the GPU-path witness.

---

## Why this design (grounding — all measured 2026-06-10, M2 Ultra)

Every load-bearing fact was confirmed end-to-end in the engine before this plan was written (per the "spike unknowns" discipline; reproducible via the two `scripts/spike_b4_*.py` harnesses):

1. **End-to-end engine, bare reductions, GPU forced on (dispatch==1 verified), speedup CPU/GPU (median):**

   | dtype | op | 1M | 10M | 100M |
   |---|---|---|---|---|
   | Int32/Int64 | sum/min/max | 0.3–0.6× | 0.4–0.5× | 0.2–0.4× |
   | F32 | sum/min/max | 0.4× | 0.5× | 0.3–0.4× |
   | F32 | mean | 0.2× | 1.08× | 1.53× |

   GPU is **2–5× slower** for everything except F32 `mean` (marginal, and only because Polars' own mean is anomalously slow — not worth a fragile size-gated special case). **No crossover in [1M, 100M].**

2. **Where the GPU time goes (F32 100M sum):** `series.to_numpy()` ≈ 0 (zero-copy view); the `execute_fused_expr` C call **alone** = 6.99ms vs Polars' *entire* `collect()` = 2.36ms. The input is already a zero-copy MLX **view** (`mlx_array_view_metal_buffer`); `pgalign=yes` at every size. **⇒ `StagingPool` is moot on this path** — there is no per-call `newBufferWithBytes` alloc tax to remove here (unlike `execute_dt`), so B4 wires nothing new to staging.

3. **Why the brainstorm spike was wrong (the smoking gun):** resident MLX f32 sum 100M = 0.95ms ✓ (matches the brainstorm); but *fresh host→MLX ingest* + reduce = **9.4ms**; single-threaded `np.sum` = 16.9ms ← *that* is what the brainstorm's "Polars 14–17ms" actually measured. Real multithreaded Polars CPU = **2.4ms**. The host→GPU ingest (≥7ms, the unified-memory wall — same as `dt` and [[m3-honest-perf-finding]]) alone exceeds Polars' full scan. Resident MLX is fast only because someone *already paid* the ingest; in the engine the data always arrives fresh from a Polars column.

4. **`std`/`var` ARE genuine wins (confirmed, dispatch==1):** F32 std 10M = CPU 7.46ms / GPU 1.51ms (**4.93×**); 100M = CPU 76.6ms / GPU 9.15ms (**8.37×**). var similar. Polars CPU std/var is a two-pass Welford (far from memory bandwidth), so the GPU wins decisively even paying ingest. This is exactly the compute-intensity principle working — they stay GPU.

5. **Routing is op-identity based, not size-based, and stays that way.** Because there is no crossover, B4 installs **no** `N₀(op, dtype)` size threshold. `_walker._BARE_GPU_WORTHY_REDUCTIONS = frozenset({"std", "var"})` is correct as-is and is the single source of truth the guard test pins.

### Scope / non-goals (explicit)

- **In:** a dispatch-asserted regression test locking the negative result (bare sum/min/max/mean → CPU) and the positive (std/var → GPU) across F32 + Int32 + Int64; a permanent end-to-end reduction benchmark (`bench_reductions.py`) with honest `baseline.json` entries (std/var gated, bare reductions informational); corrected framing in the spec §B4 and CLAUDE.md item 8.
- **Out:** any change to `_BARE_GPU_WORTHY_REDUCTIONS` or the routing logic (the spike says keep it); a size-aware `N₀` threshold (no crossover exists); wiring `StagingPool` into the reduction path (input is already a zero-copy view — moot); the marginal F32-mean-at-100M special case (fragile, Polars-mean-slowness-specific); int `mean`/`std`/`var` GPU admission (B2 left those CPU; unchanged).

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `tests/python_integration/test_reduction_routing.py` | Modify | Add B4 regression block: the `_BARE_GPU_WORTHY_REDUCTIONS == {std,var}` tripwire + bare int (I32/I64) sum/min/max → dispatch==0 + correctness, alongside the existing F32 cases. |
| `tests/bench/m4_survey/bench_reductions.py` | Create | Permanent end-to-end reduction benchmark: Polars CPU vs engine-routed vs forced-GPU (informational) for bare sum/min/max/mean (F32/I32/I64) and std/var (F32), across 10M/100M. Supersedes the throwaway `scripts/spike_b4_*.py`. |
| `tests/bench/baseline.json` | Modify | Add a `reductions_*` entry group: std/var with `_gate.ratio_lt` (real GPU win), bare reductions informational (routed ratio ≈ 1.0, with the forced-GPU loss recorded in `_notes`). |
| `scripts/spike_b4_reduction_routing.py` | Delete | Throwaway spike, superseded by `bench_reductions.py`. |
| `scripts/spike_b4_breakdown.py` | Delete | Throwaway spike, superseded by `bench_reductions.py`. |
| `docs/superpowers/specs/2026-06-09-m6-trackB-integer-temporal-design.md` | Modify | §B4: replace the "flip routing where GPU wins" framing with the measured finding (premise refuted end-to-end; bare → CPU; std/var → GPU confirmed; M4 was right). |
| `CLAUDE.md` | Modify | Item 8 compute-intensity-routing clause: annotate that B4 confirmed the bare-reduction CPU pinning end-to-end (2026-06-10). |

---

## Task 1: Regression guard — lock the routing decision

The routing is **already correct** (M4 left bare reductions on CPU; std/var on GPU). This task adds the regression tripwire so a future change — or a naive reading of the brainstorm spike — cannot silently flip `_BARE_GPU_WORTHY_REDUCTIONS` without a red test. Extends the existing `test_reduction_routing.py` (the canonical home) with the integer-dtype coverage the spike exercised and an explicit set-equality tripwire.

**Files:**
- Modify: `tests/python_integration/test_reduction_routing.py`

- [ ] **Step 1: Add the failing/guard tests**

Append to `tests/python_integration/test_reduction_routing.py` (the file already imports `numpy as np`, `polars as pl`, `assert_frame_equal`, `polars_metal`, `_native`, and defines `_dispatches` + a F32 `_df`). Add an int-frame helper and the B4 block:

```python
from polars_metal import _walker


def _int_df(dtype: pl.DataType) -> pl.DataFrame:
    rng = np.random.default_rng(0xB4)
    return pl.DataFrame({"x": pl.Series(rng.integers(-1_000_000, 1_000_000, 50_000), dtype=dtype)})


def test_bare_gpu_worthy_set_is_locked():
    """TRIPWIRE: bare bandwidth-bound reductions must stay on CPU.

    The B4 end-to-end spike (2026-06-10) measured the in-engine GPU bare-
    reduction path losing 2-5x at every size 1M->100M with no crossover — a
    bare reduction is bandwidth-bound and the host->MLX ingest alone exceeds
    Polars' multithreaded SIMD scan. Only the compute-bound std/var clear the
    dispatch floor (5-9x wins). If you are widening this set, you must first
    re-measure end-to-end and update this test deliberately. See the memory
    `m6-b4-reduction-routing-spike` and `tests/bench/m4_survey/bench_reductions.py`.
    """
    assert _walker._BARE_GPU_WORTHY_REDUCTIONS == frozenset({"std", "var"})


def test_bare_int_reductions_stay_on_cpu():
    """Bare int sum/min/max are GPU-admissible (B2) but bandwidth-bound, so
    they must route to CPU exactly like F32 (the B4 spike confirmed the loss
    holds for Int32 and Int64)."""
    eng = polars_metal.MetalEngine()
    for dtype in (pl.Int32, pl.Int64):
        df = _int_df(dtype)
        for op in ("sum", "min", "max"):
            lf = df.lazy().select(getattr(pl.col("x"), op)().alias("r"))
            assert _dispatches(lf, eng) == 0, f"bare {op} {dtype} must stay on CPU"
            assert_frame_equal(lf.collect(engine=eng), lf.collect())
```

- [ ] **Step 2: Run the new tests to verify they pass**

Run: `cd /Users/dclark/dev/polars-metal/main/polars-metal && python -m pytest tests/python_integration/test_reduction_routing.py -v`
Expected: PASS — all existing F32 cases plus `test_bare_gpu_worthy_set_is_locked` and `test_bare_int_reductions_stay_on_cpu`. (These tests assert the *current, correct* behavior; they pass immediately. To confirm the tripwire bites, temporarily add `"sum"` to `_walker._BARE_GPU_WORTHY_REDUCTIONS` in a Python REPL and observe both the set-equality and the int/F32 dispatch==0 tests fail — then revert. Do NOT commit the temporary edit.)

- [ ] **Step 3: Lint**

Run: `cd /Users/dclark/dev/polars-metal/main/polars-metal && ruff check tests/python_integration/test_reduction_routing.py && ruff format --check tests/python_integration/test_reduction_routing.py`
Expected: both PASS (no output / "All checks passed"). If `ruff format --check` reports a diff, run `ruff format tests/python_integration/test_reduction_routing.py` and re-check.

- [ ] **Step 4: Commit**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git add tests/python_integration/test_reduction_routing.py
git commit -m "B4 T1: lock bare-reduction CPU routing — tripwire + int coverage

The end-to-end spike refuted the brainstorm's flip-to-GPU premise (GPU loses
2-5x, no crossover to 100M). Pin _BARE_GPU_WORTHY_REDUCTIONS == {std,var} and
assert bare int sum/min/max route to CPU, alongside the existing F32 cases.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: End-to-end reduction benchmark + honest baseline

Convert the throwaway spikes into a permanent, rerunnable benchmark and record honest `baseline.json` entries. The benchmark reports three numbers per case: Polars CPU, engine-routed (= CPU for bare reductions, GPU for std/var), and **forced-GPU** (informational — documents the loss the routing avoids). std/var get real `ratio_lt` gates; bare reductions are informational (routed ratio ≈ 1.0).

**Files:**
- Create: `tests/bench/m4_survey/bench_reductions.py`
- Modify: `tests/bench/baseline.json`
- Delete: `scripts/spike_b4_reduction_routing.py`, `scripts/spike_b4_breakdown.py`

- [ ] **Step 1: Write the benchmark**

Create `tests/bench/m4_survey/bench_reductions.py`:

```python
"""End-to-end bare-reduction routing benchmark (B4).

Honest in-engine numbers behind the B4 routing decision: bare sum/min/max/mean
are bandwidth-bound and stay on CPU (the host->MLX ingest alone exceeds Polars'
multithreaded SIMD scan); only std/var clear the dispatch floor and route to
GPU (5-9x). Reports, per case:
  - cpu        : lf.collect()                       (Polars CPU oracle)
  - routed     : lf.collect(engine=metal)           (what the engine actually does)
  - forced_gpu : same, with the gate widened        (informational — the loss avoided)

Run: python -m tests.bench.m4_survey.bench_reductions
"""

from __future__ import annotations

import numpy as np
import polars as pl

import polars_metal
from polars_metal import _native, _walker

from tests.bench.m4_survey._timing import time_callable


def _make(dtype: pl.DataType, n: int, rng: np.random.Generator) -> pl.DataFrame:
    if dtype == pl.Float32:
        a = rng.standard_normal(n).astype(np.float32)
    elif dtype == pl.Int32:
        a = rng.integers(-1_000_000, 1_000_000, size=n, dtype=np.int32)
    elif dtype == pl.Int64:
        a = rng.integers(-1_000_000_000, 1_000_000_000, size=n, dtype=np.int64)
    else:
        raise ValueError(dtype)
    return pl.DataFrame({"x": pl.Series(a, dtype=dtype)})


def _dispatches(lf, eng) -> int:
    n = {"c": 0}
    orig = _native.execute_fused_expr

    def cnt(scope, inputs, out):
        n["c"] += 1
        return orig(scope=scope, inputs=inputs, out=out)

    _native.execute_fused_expr = cnt
    try:
        lf.collect(engine=eng)
    finally:
        _native.execute_fused_expr = orig
    return n["c"]


def main() -> None:
    eng = polars_metal.MetalEngine()
    rng = np.random.default_rng(0xB4)
    bare = [
        (pl.Int32, ("sum", "min", "max")),
        (pl.Int64, ("sum", "min", "max")),
        (pl.Float32, ("sum", "mean", "min", "max")),
    ]
    for n in (10_000_000, 100_000_000):
        print(f"\n=== bare reductions (route to CPU) ===  N={n:,}")
        for dtype, ops in bare:
            df = _make(dtype, n, rng)
            for op in ops:
                lf = df.lazy().select(getattr(pl.col("x"), op)().alias("r"))
                assert _dispatches(lf, eng) == 0, f"bare {op} {dtype} should be CPU"
                cpu = time_callable(f"{dtype}.{op} cpu", lambda lf=lf: lf.collect())
                routed = time_callable(
                    f"{dtype}.{op} routed", lambda lf=lf: lf.collect(engine=eng)
                )
                # forced-GPU informational ratio
                saved = _walker._BARE_GPU_WORTHY_REDUCTIONS
                _walker._BARE_GPU_WORTHY_REDUCTIONS = frozenset(
                    {"std", "var", "sum", "mean", "min", "max"}
                )
                try:
                    assert _dispatches(lf, eng) == 1, "forced GPU should dispatch"
                    forced = time_callable(
                        f"{dtype}.{op} forced_gpu", lambda lf=lf: lf.collect(engine=eng)
                    )
                finally:
                    _walker._BARE_GPU_WORTHY_REDUCTIONS = saved
                print(
                    f"  {str(dtype):>8}.{op:<5} routed/cpu={routed.median_ms/cpu.median_ms:5.2f}x "
                    f"forced_gpu/cpu={forced.median_ms/cpu.median_ms:5.2f}x"
                )

        print(f"\n=== std/var (route to GPU) ===  N={n:,}")
        df = _make(pl.Float32, n, rng)
        for op in ("std", "var"):
            lf = df.lazy().select(getattr(pl.col("x"), op)().alias("r"))
            assert _dispatches(lf, eng) == 1, f"{op} should dispatch to GPU"
            cpu = time_callable(f"f32.{op} cpu", lambda lf=lf: lf.collect())
            gpu = time_callable(f"f32.{op} gpu", lambda lf=lf: lf.collect(engine=eng))
            print(f"  f32.{op:<5} gpu/cpu={gpu.median_ms/cpu.median_ms:5.2f}x "
                  f"(speedup {cpu.median_ms/gpu.median_ms:4.1f}x)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark and capture the numbers**

Run: `cd /Users/dclark/dev/polars-metal/main/polars-metal && python -m tests.bench.m4_survey.bench_reductions`
Expected: completes without an `AssertionError` (proves routing is as designed — bare dispatch==0, std/var + forced dispatch==1). For bare reductions `routed/cpu` ≈ 1.0 and `forced_gpu/cpu` ≈ 2–5 (the loss avoided). For std/var `gpu/cpu` ≈ 0.1–0.25 (speedup 4–9×). **Record the actual std/var `gpu_ms`, `cpu_ms`, and the 100M bare forced_gpu/cpu ratios** for Step 3.

- [ ] **Step 3: Add honest baseline entries**

In `tests/bench/baseline.json`, add the following keys inside the top-level `"queries"` object (fill `cpu_ms` / `metal_ms` from Step 2's median numbers; the `ratio_metal_over_cpu` is `metal_ms / cpu_ms`). std/var get a `_gate.ratio_lt`; bare reductions are informational (routed metal_ms ≈ cpu_ms; the forced-GPU loss goes in `_notes`):

```json
    "reduction_std_f32_100m": {
      "cpu_ms": 76.58,
      "metal_ms": 9.15,
      "ratio_metal_over_cpu": 0.12,
      "n_rows": 100000000,
      "hardware": "M2 Ultra",
      "_notes": "Bare std routes to GPU (compute-bound: Polars CPU is a two-pass Welford, far from bandwidth). B4 end-to-end, 2026-06-10. Fill from bench_reductions.py.",
      "_gate": {"ratio_lt": 0.5}
    },
    "reduction_var_f32_100m": {
      "cpu_ms": 74.82,
      "metal_ms": 8.37,
      "ratio_metal_over_cpu": 0.11,
      "n_rows": 100000000,
      "hardware": "M2 Ultra",
      "_notes": "Bare var routes to GPU. B4 end-to-end, 2026-06-10. Fill from bench_reductions.py.",
      "_gate": {"ratio_lt": 0.5}
    },
    "reduction_bare_sum_f32_100m": {
      "cpu_ms": 2.36,
      "metal_ms": 2.36,
      "ratio_metal_over_cpu": 1.0,
      "n_rows": 100000000,
      "hardware": "M2 Ultra",
      "_notes": "Bare sum routes to CPU by design — informational, no gate. The B4 spike measured forced-GPU at ~3x SLOWER (forced_gpu/cpu ~3.0): a bare reduction is bandwidth-bound and the host->MLX ingest alone (>=7ms) exceeds Polars' multithreaded scan (2.36ms). routed==cpu because the engine correctly stays on CPU. See memory m6-b4-reduction-routing-spike."
    }
```

> Note for the implementer: do **not** add a `_gate` block to `reduction_bare_sum_f32_100m` — bare reductions route to CPU so `metal_ms == cpu_ms` (ratio 1.0) by construction; gating that would test nothing. The std/var gates (`ratio_lt: 0.5`) are the real regression protection — they go red if std/var GPU routing ever breaks (ratio jumps to ~1.0). Keep the recorded std/var numbers from Step 2; the values above are the spike's and are close but must be replaced with the bench's measured medians.

- [ ] **Step 4: Verify the baseline gate passes**

Run: `cd /Users/dclark/dev/polars-metal/main/polars-metal && python -m pytest tests/bench/test_baseline_gate.py -v`
Expected: PASS — `check_baseline` finds the two std/var `ratio_lt: 0.5` gates satisfied (recorded ratios ~0.11–0.12 < 0.5) and skips the informational bare entry. If JSON is malformed, the test errors on load — fix the syntax.

- [ ] **Step 5: Delete the superseded spike scripts and lint**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git rm scripts/spike_b4_reduction_routing.py scripts/spike_b4_breakdown.py
ruff check tests/bench/m4_survey/bench_reductions.py
ruff format --check tests/bench/m4_survey/bench_reductions.py
```
Expected: `git rm` removes both; both ruff commands PASS. If `ruff format --check` reports a diff, run `ruff format tests/bench/m4_survey/bench_reductions.py` and re-check. Also validate the JSON: `python -c "import json; json.load(open('tests/bench/baseline.json'))"` prints nothing (valid).

- [ ] **Step 6: Commit**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git add tests/bench/m4_survey/bench_reductions.py tests/bench/baseline.json
git commit -m "B4 T2: end-to-end reduction benchmark + honest baseline

Permanent bench_reductions.py reports cpu/routed/forced_gpu per case; baseline
records std/var GPU wins (gated ratio_lt 0.5) and the bare-reduction CPU parity
(informational; forced-GPU ~3x slower documented). Supersedes the throwaway
scripts/spike_b4_*.py.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Correct the spec and roadmap framing

The spec §B4 still says "install a size-aware per-op threshold, flip routing where GPU genuinely wins." The end-to-end measurement refuted that. Correct it to the honest finding; annotate CLAUDE.md item 8's already-correct compute-intensity clause with the B4 confirmation.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-09-m6-trackB-integer-temporal-design.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Rewrite the spec §B4 body**

In `docs/superpowers/specs/2026-06-09-m6-trackB-integer-temporal-design.md`, replace the three bullets under `### B4 — Reduction routing + re-baselined benchmarks (HEADLINE)` (the `End-to-end engine measurement` / `Size-aware per-op routing threshold` / `Benchmarks + honest gates` bullets) with:

```markdown
- **Outcome — premise refuted by end-to-end measurement (2026-06-10, M2 Ultra).** The pre-plan
  spike drove bare int (I32/I64) and F32 reductions through the *full* engine path (collect +
  zero-copy-view stage + MLX reduce + scalar readback) vs Polars CPU across 1M/10M/100M. **GPU
  loses 2–5× at every size with no crossover.** A bare reduction is bandwidth-bound (1 flop/element);
  the host→MLX ingest alone (≥7ms @100M, the unified-memory wall) exceeds Polars' multithreaded SIMD
  scan (2.4ms). The brainstorm spike's 3.5–10× was an artifact of comparing *resident* MLX (no ingest,
  0.95ms) against *single-threaded* numpy (~17ms), not real Polars CPU. **`std`/`var` stay on GPU** —
  they are genuine 5–9× wins (Polars CPU std/var is a slow two-pass Welford, far from bandwidth). The
  M4 compute-intensity gate (route on FLOPs/row, not op identity) is vindicated, not changed.
- **No routing flip, no `N₀` threshold.** `_BARE_GPU_WORTHY_REDUCTIONS = {std, var}` is correct as-is;
  there is no size crossover to gate on. `StagingPool` is moot here — the reduction input is already a
  zero-copy MLX view (no per-call `newBufferWithBytes` tax, unlike `execute_dt`).
- **Shipped as guard + re-baseline:** a dispatch-asserted regression test pinning the decision
  (`tests/python_integration/test_reduction_routing.py`), a permanent end-to-end benchmark
  (`tests/bench/m4_survey/bench_reductions.py`) with honest `baseline.json` gates (std/var gated,
  bare reductions informational). Full data: memory `m6-b4-reduction-routing-spike`.
```

- [ ] **Step 2: Annotate CLAUDE.md item 8**

In `CLAUDE.md`, in roadmap item 8 (line ~155), find the clause:

```
**compute-intensity routing** (bandwidth-bound bare `sum`/`min`/`max`/`mean` stay on CPU; only compute-bound bare ops and compute chains route to MLX)
```

and append `— B4 confirmed this end-to-end (2026-06-10): forcing bare reductions to GPU loses 2–5× at 1M–100M; std/var stay GPU (genuine 5–9×).` immediately before the closing `)`. The clause becomes:

```
**compute-intensity routing** (bandwidth-bound bare `sum`/`min`/`max`/`mean` stay on CPU; only compute-bound bare ops and compute chains route to MLX — B4 confirmed this end-to-end (2026-06-10): forcing bare reductions to GPU loses 2–5× at 1M–100M; std/var stay GPU (genuine 5–9×))
```

- [ ] **Step 3: Verify the docs render and reference real artifacts**

Run: `cd /Users/dclark/dev/polars-metal/main/polars-metal && grep -n "premise refuted\|No routing flip\|B4 confirmed this end-to-end" docs/superpowers/specs/2026-06-09-m6-trackB-integer-temporal-design.md CLAUDE.md`
Expected: three matches — two in the spec, one in CLAUDE.md.

- [ ] **Step 4: Commit**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git add docs/superpowers/specs/2026-06-09-m6-trackB-integer-temporal-design.md CLAUDE.md
git commit -m "B4 T3: correct spec/roadmap framing — bare-reduction flip refuted

Spec B4 rewritten to the measured outcome (GPU loses 2-5x end-to-end, no
crossover; std/var stay GPU; M4 gate vindicated). CLAUDE.md item 8 annotated
with the B4 end-to-end confirmation.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final gate (after all tasks)

- [ ] **Run the full gate**

Run: `cd /Users/dclark/dev/polars-metal/main/polars-metal && make gate`
Expected: PASS — lint (clippy + `ruff check` + `ruff format --check`), unit/kernel tests (`--test-threads=1`), conformance, and the baseline gate all green. The only code-path change is test/bench/docs; routing is unchanged, so no conformance or differential movement is expected.

- [ ] **Push to PR #6**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git push origin m6-vector-search
```

- [ ] **Update execution-state memory**

Mark B4 SHIPPED in the `m6-trackb-execution-state` memory (Track B complete: B1–B4 all shipped), cross-linking `[[m6-b4-reduction-routing-spike]]`.

---

## Self-review (writer's checklist — done before handoff)

1. **Spec coverage.** Spec §B4 asks for (a) end-to-end measurement, (b) a size-aware threshold, (c) benchmarks + honest gates. (a) is done by the pre-plan spike and codified in Task 2's bench; (b) is **deliberately not built** — the measurement refuted its premise (documented in Task 3, with the architect's "guard + re-baseline" decision); (c) is Task 2. The reframe itself is the headline finding.
2. **Placeholder scan.** No TBD/TODO/"handle edge cases". The one place numbers are measured-not-hardcoded (baseline.json) is explicit: Step 2 captures them, Step 3 records them, with the spike's values as a documented starting point to replace.
3. **Type/name consistency.** `_BARE_GPU_WORTHY_REDUCTIONS`, `_native.execute_fused_expr`, `_dispatches`, `time_callable`, `check_baseline`/`_gate.ratio_lt`, `tests/bench/baseline.json` `"queries"` shape — all verified against the live code before writing.
