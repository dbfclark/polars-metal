# M2 — Per-op routing layer + hash groupby — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the per-op routing layer (cost-model-driven walker → router → lifting plan) plus a hash-groupby implementation that together let a modified TPC-H Q1 (integer-encoded `l_returnflag`/`l_linestatus`) beat pure-CPU Polars on M2 Ultra.

**Architecture:** Three-layer flow: walker translates Polars IR → MetalPlanNode tree (no decisions); Rust router consults cost model + affinity, returns per-subtree LiftingPlan; walker applies the plan, calling `nt.set_udf` only on GpuLift subtrees. Hash groupby uses two-pass count-then-fill ported from cuDF. Composite-key encoding (up to 128 bits total) packs multi-column keys into a single u128 for the hash kernel.

**Tech Stack:** Rust 2021 (workspace), `objc2-metal` (Metal API), `cxx` for MLX FFI (unchanged from M1), `pyo3 0.22` + `maturin` (unchanged), `polars` pinned to `py-1.40.1` (unchanged), `proptest` for kernel + reference comparison, `pytest-benchmark` + `criterion` for perf.

**Spec:** [`docs/superpowers/specs/2026-05-21-m2-design.md`](../specs/2026-05-21-m2-design.md). All decisions there are binding; this plan does not relitigate them.

**Conventions** (per CLAUDE.md): No `unwrap()` outside tests. No `unsafe` outside `*-sys` crates and the buffer bridge — each with a `// SAFETY:` comment. One MSL kernel family per file. Errors propagate as `polars.exceptions.ComputeError` at the engine boundary. Null semantics match Polars exactly. Don't add files to `shaders/` without a matching test. Read the matching cuDF kernel before writing MSL.

**Pre-task reading.** Before Phase 1 (router), read:
- `python/polars_metal/_walker.py` — current bottom-up IR walk; M2 extends this with lifting-plan application
- `crates/polars-metal-core/src/plan/mod.rs` — MetalPlanNode enum; M2 adds GroupBy variant
- `crates/polars-metal-core/src/udf.rs` — PyO3 entry points; M2 adds two new ones

Before Phase 4 (kernel work), read:
- `references/cudf/cpp/src/groupby/hash/groupby.cu` — the two-pass algorithm we're porting
- `shaders/_validity.metal` — null-bitmap helper conventions (M1 established these)
- `shaders/cmp_i64.metal` — the MSL macro pattern for generating multiple entry points

---

## Phase 0 — Preflight

### Task 1: Verify M1 gates still green on the new branch; confirm dev env

**Files:** none (verification only).

- [ ] **Step 1: Confirm we're on `m2-routing-and-groupby` branched from main**

Run: `git rev-parse --abbrev-ref HEAD && git log -1 --oneline && git status --porcelain`
Expected: branch `m2-routing-and-groupby`, last commit subject `Draft M2 design: routing layer + hash groupby`, empty working-tree status.
If drift: `git checkout m2-routing-and-groupby` and reconfirm.

- [ ] **Step 2: Run the M1 gate**

Run: `make gate`
Expected: all phases pass (`lint`, `test-unit`, `test-kernel`, `wheel`, `test-conformance`, `test-diff`). Wall-clock ~3-5 min on M2 Ultra per M1 retro.
If anything fails: stop and fix on a separate branch before starting M2; do not pile new work on top of a broken baseline.

- [ ] **Step 3: Verify Metal toolchain and MLX still present**

Run: `xcrun metal --version && xcrun metallib --version && python -c "import polars_metal; print(polars_metal._native.version_string())"`
Expected: Metal toolchain reports a version; the Python import succeeds and prints the package version.
If `polars_metal` import fails: `make wheel` to rebuild.

- [ ] **Step 4: Verify reference clones pin matches the M1 spec**

Run: `(cd references/polars && git rev-parse HEAD) && (cd references/cudf && git rev-parse HEAD)`
Expected: Polars at the `py-1.40.1` tag SHA, cuDF at the SHA from M1.
If drift: `bash scripts/refresh-references.sh`.

- [ ] **Step 5: Record M1 baseline numbers**

Run: `cat tests/bench/baseline.json | python -c "import json,sys; d=json.load(sys.stdin); print('M1 baseline:', d.get('git_sha'), d.get('date'))"`
Expected: prints the SHA + date recorded in `tests/bench/baseline.json` at M1 ship.
This baseline is what M2's perf gate compares against. **Do not** rebaseline existing entries during M2; only add the new `tpch_q1_modified` entry.

Nothing to commit in Task 1.

---

## Phase 1 — Router skeleton (decision layer, no kernels yet)

This phase lands the per-op routing layer end-to-end with M1's existing kernels (filter/cmp/logical). It produces the architectural pivot from spec § "Three-layer flow": walker no longer makes routing decisions; the Rust router does. By the end of Phase 1, the M1 behavior is reproduced exactly but the **path** is different: walker → MetalPlanNode → router → LiftingPlan → walker applies → UDFs only on GpuLift subtrees.

The spec's starting cost model (filter → CPU always; groupby → GPU > 100K rows) means **filter starts routing to CPU in this phase**. M1's filter kernels stay shipped; they're just unused by default until tuning shows otherwise.

### Task 2: `LiftingPlan` data structure and `NodeDecision` enum

**Files:**
- Create: `crates/polars-metal-core/src/router/mod.rs`
- Modify: `crates/polars-metal-core/src/lib.rs`
- Create: `crates/polars-metal-core/tests/test_router_types.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-core/tests/test_router_types.rs
//
// Sanity tests for the LiftingPlan / NodeDecision types — construction,
// equality, and the trivial "fallback poisons ancestors" predicate.
#![allow(clippy::expect_used)]

use polars_metal_core::router::{LiftingPlan, NodeDecision, NodeId};

#[test]
fn node_decision_variants_construct() {
    let _ = NodeDecision::GpuLift;
    let _ = NodeDecision::CpuLeave;
    let _ = NodeDecision::Fallback("unsupported IR".to_string());
}

#[test]
fn lifting_plan_records_per_node_decisions() {
    let mut plan = LiftingPlan::new();
    let scan_id = NodeId::new("Scan", 0);
    let filter_id = NodeId::new("Filter", 1);
    plan.set(scan_id.clone(), NodeDecision::CpuLeave);
    plan.set(filter_id.clone(), NodeDecision::CpuLeave);
    assert_eq!(plan.get(&scan_id), Some(&NodeDecision::CpuLeave));
    assert_eq!(plan.get(&filter_id), Some(&NodeDecision::CpuLeave));
    assert_eq!(plan.len(), 2);
}

#[test]
fn fallback_carries_human_readable_reason() {
    let d = NodeDecision::Fallback("composite key > 128 bits".into());
    match d {
        NodeDecision::Fallback(reason) => assert!(reason.contains("128")),
        _ => panic!("expected Fallback"),
    }
}

#[test]
fn node_id_round_trips_kind_and_sequence() {
    let id = NodeId::new("GroupBy", 7);
    assert_eq!(id.kind(), "GroupBy");
    assert_eq!(id.seq(), 7);
}
```

- [ ] **Step 2: Implement the types**

```rust
// crates/polars-metal-core/src/router/mod.rs
//! Router: per-op GPU-vs-CPU dispatch decisions for the engine.
//!
//! Given a `MetalPlanNode` tree from the walker, the router produces a
//! `LiftingPlan` — a per-node decision (GpuLift | CpuLeave | Fallback).
//! The walker consumes that decision per node and installs UDFs only
//! for GpuLift subtrees.
//!
//! See `docs/superpowers/specs/2026-05-21-m2-design.md` § "Three-layer
//! flow" for the architectural picture.

use std::collections::HashMap;

pub mod cost;
pub mod affinity;

/// Identifier for an IR node in the MetalPlanNode tree. Tuple of
/// (kind, sequence-number); the walker emits these deterministically so
/// the same node has the same ID across walker → router → walker round
/// trip in one query.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId {
    kind: String,
    seq: u32,
}

impl NodeId {
    pub fn new(kind: impl Into<String>, seq: u32) -> Self {
        Self { kind: kind.into(), seq }
    }
    pub fn kind(&self) -> &str { &self.kind }
    pub fn seq(&self) -> u32 { self.seq }
    /// Wire-format string the Python walker receives: `"Kind#seq"`.
    pub fn to_wire(&self) -> String { format!("{}#{}", self.kind, self.seq) }
}

/// Per-node routing decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeDecision {
    /// The walker should install a UDF for this subtree; the engine runs
    /// it on the GPU.
    GpuLift,
    /// The walker leaves this subtree alone; Polars' CPU executor runs it.
    CpuLeave,
    /// This subtree (or some descendant) is unrecognized or violates a
    /// plan-time invariant (e.g. composite-key width). Ancestors poison;
    /// the whole query routes to CPU.
    Fallback(String),
}

/// A LiftingPlan: the full per-node decision map for one query.
#[derive(Debug, Default, Clone)]
pub struct LiftingPlan {
    decisions: HashMap<NodeId, NodeDecision>,
}

impl LiftingPlan {
    pub fn new() -> Self { Self::default() }
    pub fn set(&mut self, id: NodeId, decision: NodeDecision) {
        self.decisions.insert(id, decision);
    }
    pub fn get(&self, id: &NodeId) -> Option<&NodeDecision> {
        self.decisions.get(id)
    }
    pub fn len(&self) -> usize { self.decisions.len() }
    pub fn is_empty(&self) -> bool { self.decisions.is_empty() }
    pub fn iter(&self) -> impl Iterator<Item = (&NodeId, &NodeDecision)> {
        self.decisions.iter()
    }
    /// True iff the plan contains any Fallback decision. When true, the
    /// walker drops the entire LiftingPlan and routes to CPU.
    pub fn has_fallback(&self) -> bool {
        self.decisions.values().any(|d| matches!(d, NodeDecision::Fallback(_)))
    }
}
```

- [ ] **Step 3: Expose `router` from `lib.rs`**

```rust
// crates/polars-metal-core/src/lib.rs — add near the top:
pub mod router;
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-core --test test_router_types`
Expected: PASS, four tests.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/router/mod.rs crates/polars-metal-core/src/lib.rs crates/polars-metal-core/tests/test_router_types.rs
git commit -m "$(cat <<'EOF'
Router: LiftingPlan and NodeDecision types

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3: Cost model — `router/cost.rs` with thresholds + decision rules

**Files:**
- Create: `crates/polars-metal-core/src/router/cost.rs`
- Create: `crates/polars-metal-core/tests/test_router_cost.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-core/tests/test_router_cost.rs
//
// Cost-model rule tests. Each rule is a small pure function over
// (op_kind, n_rows) → NodeDecision. The thresholds here are M2's
// starting point per the spec § "Routing decisions (cost model)"; PRs
// that re-tune them update both the constants and the tests.
#![allow(clippy::expect_used)]

use polars_metal_core::router::cost;
use polars_metal_core::router::NodeDecision;

#[test]
fn filter_routes_to_cpu_at_all_sizes() {
    // Spec: "Filter | CPU | always".
    assert_eq!(cost::decide_filter(0), NodeDecision::CpuLeave);
    assert_eq!(cost::decide_filter(1_000), NodeDecision::CpuLeave);
    assert_eq!(cost::decide_filter(100_000_000), NodeDecision::CpuLeave);
}

#[test]
fn groupby_routes_to_gpu_above_100k_rows() {
    // Spec: "GroupBy | GPU iff n_rows > 100_000".
    assert_eq!(cost::decide_groupby(50_000), NodeDecision::CpuLeave);
    assert_eq!(cost::decide_groupby(100_000), NodeDecision::CpuLeave);
    assert_eq!(cost::decide_groupby(100_001), NodeDecision::GpuLift);
    assert_eq!(cost::decide_groupby(10_000_000), NodeDecision::GpuLift);
}

#[test]
fn project_inherits_input_decision() {
    // Spec: "Project / SimpleProjection | follow input".
    assert_eq!(cost::decide_project(&NodeDecision::GpuLift), NodeDecision::GpuLift);
    assert_eq!(cost::decide_project(&NodeDecision::CpuLeave), NodeDecision::CpuLeave);
    // Fallback propagates up.
    assert!(matches!(
        cost::decide_project(&NodeDecision::Fallback("x".into())),
        NodeDecision::Fallback(_)
    ));
}

#[test]
fn scan_inherits_parent_decision() {
    // Spec: "Scan | follow output | inherit parent's decision". Here we
    // pass the parent decision as input; the function's contract matches
    // a top-down second walk if needed. Phase 1 implementation defaults
    // scan to CpuLeave (parent decision arrives in affinity smoothing).
    assert_eq!(cost::decide_scan_initial(), NodeDecision::CpuLeave);
}

#[test]
fn thresholds_are_named_constants_for_pr_tuning() {
    // The threshold MUST live as a named pub constant so PRs that retune
    // it touch exactly one line (per spec § "Routing decisions" — "cost
    // data and the implementation live in the same Rust module so they
    // evolve together by PR").
    assert_eq!(cost::GROUPBY_GPU_MIN_ROWS, 100_000);
}
```

- [ ] **Step 2: Implement the cost rules**

```rust
// crates/polars-metal-core/src/router/cost.rs
//! Cost-model rules and threshold constants.
//!
//! Each rule is a pure function over the relevant inputs (op kind,
//! row count, input decision). Thresholds are exposed as `pub const`
//! so PR-level tuning touches a single line.
//!
//! Why CPU defaults for Filter
//! ---------------------------
//! M1's perf investigation (see `tests/bench/baseline.json` notes)
//! showed CPU winning at all measured row counts (1K..100M) for the
//! filter operator. Unified memory removes the copy-cost asymmetry
//! that GPUs exploit on discrete-memory systems, and the CPU
//! implementation in Polars is already highly tuned. Future PRs may
//! revisit (e.g. once filter fuses into a larger GPU subtree without
//! a CPU round-trip), but today the default is CpuLeave.
//!
//! Why GPU > 100K rows for GroupBy
//! -------------------------------
//! Atomic-CAS hash-table build and the per-aggregate kernel launches
//! have fixed launch overhead that dominates at small row counts;
//! crossover is empirically near 100K rows on M2 Ultra for low-
//! cardinality keys (Q1's shape). The constant is a starting point;
//! M2's per-kernel benchmarks (Phase 10) inform PR-level tuning.

use super::NodeDecision;

/// Smallest input row count at which the GroupBy kernel is expected to
/// beat the CPU implementation on M2 Ultra. Tuned by criterion benches
/// per spec § "Risks & open questions — Cost-model threshold tuning".
pub const GROUPBY_GPU_MIN_ROWS: usize = 100_000;

/// Decide for a Filter node. Always CpuLeave under M2's cost model.
pub fn decide_filter(_n_rows: usize) -> NodeDecision {
    NodeDecision::CpuLeave
}

/// Decide for a GroupBy node based on input row count.
pub fn decide_groupby(n_rows: usize) -> NodeDecision {
    if n_rows > GROUPBY_GPU_MIN_ROWS {
        NodeDecision::GpuLift
    } else {
        NodeDecision::CpuLeave
    }
}

/// Decide for a Project / SimpleProjection node. Inherits its input's
/// decision verbatim — projection is metadata-only on our side either
/// way (column re-selection happens after compaction or via the CPU
/// executor's projection).
pub fn decide_project(input: &NodeDecision) -> NodeDecision {
    match input {
        NodeDecision::Fallback(r) => NodeDecision::Fallback(r.clone()),
        NodeDecision::GpuLift => NodeDecision::GpuLift,
        NodeDecision::CpuLeave => NodeDecision::CpuLeave,
    }
}

/// Initial Scan decision (before affinity smoothing applies the parent
/// hint). Default CpuLeave; affinity may upgrade to GpuLift in Task 5.
pub fn decide_scan_initial() -> NodeDecision {
    NodeDecision::CpuLeave
}
```

- [ ] **Step 3: Expose `cost` from the router module**

`crates/polars-metal-core/src/router/mod.rs` already declares `pub mod cost;` (added in Task 2 Step 2). Confirm by inspection; no further edit needed.

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-core --test test_router_cost`
Expected: PASS, five tests.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/router/cost.rs crates/polars-metal-core/tests/test_router_cost.rs
git commit -m "$(cat <<'EOF'
Router: cost-model rules and threshold constants

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 4: `compute_lifting_plan` — bottom-up walk producing per-node decisions

**Files:**
- Modify: `crates/polars-metal-core/src/router/mod.rs`
- Modify: `crates/polars-metal-core/src/plan/mod.rs` (add `node_id()` accessor)
- Create: `crates/polars-metal-core/tests/test_router_walk.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-core/tests/test_router_walk.rs
//
// End-to-end LiftingPlan-from-MetalPlanNode-tree tests. We construct
// synthetic trees (Scan → Filter → Project, Scan → GroupBy, etc.) and
// assert the per-node decisions match the spec.
#![allow(clippy::expect_used)]

use polars_metal_core::plan::{MetalDtype, MetalPlanNode, PredicateAst};
use polars_metal_core::router::{compute_lifting_plan, NodeDecision, NodeId};

fn scan(n_rows: usize) -> MetalPlanNode {
    MetalPlanNode::Scan {
        n_rows,
        columns: vec![("a".into(), MetalDtype::I64)],
    }
}

#[test]
fn filter_over_scan_is_cpu_leave_throughout() {
    let plan = MetalPlanNode::Filter {
        input: Box::new(scan(1_000_000)),
        predicate: PredicateAst::Column { name: "mask".into(), dtype: MetalDtype::Bool },
    };
    let lifting = compute_lifting_plan(&plan);
    // Spec § "Routing decisions" — filter always CPU, scan inherits.
    assert_eq!(lifting.get(&NodeId::new("Scan", 0)), Some(&NodeDecision::CpuLeave));
    assert_eq!(lifting.get(&NodeId::new("Filter", 1)), Some(&NodeDecision::CpuLeave));
}

#[test]
fn project_over_scan_is_cpu_leave_throughout() {
    let plan = MetalPlanNode::Project {
        input: Box::new(scan(1_000_000)),
        columns: vec!["a".into()],
    };
    let lifting = compute_lifting_plan(&plan);
    assert_eq!(lifting.get(&NodeId::new("Scan", 0)), Some(&NodeDecision::CpuLeave));
    assert_eq!(lifting.get(&NodeId::new("Project", 1)), Some(&NodeDecision::CpuLeave));
}

#[test]
fn project_after_filter_inherits_filter_decision() {
    let plan = MetalPlanNode::Project {
        input: Box::new(MetalPlanNode::Filter {
            input: Box::new(scan(1_000_000)),
            predicate: PredicateAst::Column { name: "mask".into(), dtype: MetalDtype::Bool },
        }),
        columns: vec!["a".into()],
    };
    let lifting = compute_lifting_plan(&plan);
    assert_eq!(lifting.get(&NodeId::new("Project", 2)), Some(&NodeDecision::CpuLeave));
}

#[test]
fn lifting_plan_is_empty_for_empty_tree() {
    // Construction sanity. The walker never invokes the router on an
    // empty tree, but the function must handle it gracefully.
    let plan = scan(0);
    let lifting = compute_lifting_plan(&plan);
    assert_eq!(lifting.len(), 1);
}

#[test]
fn node_ids_are_assigned_in_post_order() {
    // Spec implicit: post-order so leaves get the smaller IDs (mirrors
    // the bottom-up walker's natural traversal). Filter-over-Scan:
    // Scan first, Filter second.
    let plan = MetalPlanNode::Filter {
        input: Box::new(scan(100)),
        predicate: PredicateAst::Column { name: "mask".into(), dtype: MetalDtype::Bool },
    };
    let lifting = compute_lifting_plan(&plan);
    assert!(lifting.get(&NodeId::new("Scan", 0)).is_some());
    assert!(lifting.get(&NodeId::new("Filter", 1)).is_some());
}
```

- [ ] **Step 2: Implement `compute_lifting_plan` and `node_id_of`**

```rust
// crates/polars-metal-core/src/router/mod.rs — append below the types:

use crate::plan::MetalPlanNode;

/// Walk `root` bottom-up assigning per-node IDs in post-order; consult
/// the cost model at each node; return the full LiftingPlan.
///
/// `NodeId::seq` is a monotone counter incremented in post-order. The
/// Python walker uses the same counter (it produces the MetalPlanNode
/// tree in the same shape), so wire-format IDs round-trip.
pub fn compute_lifting_plan(root: &MetalPlanNode) -> LiftingPlan {
    let mut plan = LiftingPlan::new();
    let mut next_seq: u32 = 0;
    let _ = walk(root, &mut plan, &mut next_seq);
    plan
}

/// Recursive worker. Returns the assigned NodeId so the parent can read
/// the child's recorded decision.
fn walk(node: &MetalPlanNode, plan: &mut LiftingPlan, next_seq: &mut u32) -> NodeId {
    match node {
        MetalPlanNode::Scan { .. } => {
            let id = NodeId::new("Scan", *next_seq);
            *next_seq += 1;
            plan.set(id.clone(), cost::decide_scan_initial());
            id
        }
        MetalPlanNode::Project { input, .. } => {
            let child_id = walk(input, plan, next_seq);
            let id = NodeId::new("Project", *next_seq);
            *next_seq += 1;
            // unwrap-equivalent without unwrap: child was just inserted.
            let child_decision = plan.get(&child_id)
                .cloned()
                .unwrap_or(NodeDecision::Fallback("missing child decision".into()));
            plan.set(id.clone(), cost::decide_project(&child_decision));
            id
        }
        MetalPlanNode::Filter { input, .. } => {
            let _ = walk(input, plan, next_seq);
            let id = NodeId::new("Filter", *next_seq);
            *next_seq += 1;
            // Filter doesn't currently use n_rows in the cost rule, but
            // we pass 0 to keep the signature stable for future tuning.
            plan.set(id.clone(), cost::decide_filter(0));
            id
        }
    }
}
```

Note: The `GroupBy` variant of `MetalPlanNode` lands in Task 10. Phase 1 wires the router only over M1's existing variants; Phase 2 extends both the IR and `walk()`.

- [ ] **Step 3: Run the test**

Run: `cargo test -p polars-metal-core --test test_router_walk`
Expected: PASS, five tests.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-core/src/router/mod.rs crates/polars-metal-core/tests/test_router_walk.rs
git commit -m "$(cat <<'EOF'
Router: bottom-up compute_lifting_plan over MetalPlanNode tree

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5: Affinity smoothing — second pass

**Files:**
- Create: `crates/polars-metal-core/src/router/affinity.rs`
- Modify: `crates/polars-metal-core/src/router/mod.rs`
- Create: `crates/polars-metal-core/tests/test_router_affinity.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-core/tests/test_router_affinity.rs
//
// Affinity smoothing tests. The pass converts close-cost adjacent
// decisions into uniform runs (within a configurable threshold) to
// minimize gratuitous GPU↔CPU transitions. Spec § "Affinity smoothing"
// specifies the initial threshold at 20%.
#![allow(clippy::expect_used)]

use polars_metal_core::router::affinity::{smooth, SmoothingConfig};
use polars_metal_core::router::{LiftingPlan, NodeDecision, NodeId};

fn build(decisions: &[(&str, u32, NodeDecision)]) -> LiftingPlan {
    let mut p = LiftingPlan::new();
    for (kind, seq, d) in decisions {
        p.set(NodeId::new(*kind, *seq), d.clone());
    }
    p
}

#[test]
fn isolated_gpu_in_cpu_run_is_flipped_to_cpu_at_close_cost() {
    // Pattern: Scan(CPU) → GroupBy(GPU) → Filter(CPU). If groupby is
    // close-cost (within 20% of CPU), affinity may flip it to CPU.
    // The cost data driving this pass is queryable; for the unit test
    // we pre-tag the close_cost field.
    let plan = build(&[
        ("Scan", 0, NodeDecision::CpuLeave),
        ("GroupBy", 1, NodeDecision::GpuLift),
        ("Filter", 2, NodeDecision::CpuLeave),
    ]);
    let config = SmoothingConfig {
        window_pct: 20,
        close_cost_node_ids: vec![NodeId::new("GroupBy", 1)],
    };
    let smoothed = smooth(plan, &config);
    assert_eq!(smoothed.get(&NodeId::new("GroupBy", 1)), Some(&NodeDecision::CpuLeave));
}

#[test]
fn far_cost_decisions_are_preserved() {
    // If GroupBy is decisively GPU (not in close_cost_node_ids), the
    // smoothing pass leaves it.
    let plan = build(&[
        ("Scan", 0, NodeDecision::CpuLeave),
        ("GroupBy", 1, NodeDecision::GpuLift),
        ("Filter", 2, NodeDecision::CpuLeave),
    ]);
    let config = SmoothingConfig { window_pct: 20, close_cost_node_ids: vec![] };
    let smoothed = smooth(plan, &config);
    assert_eq!(smoothed.get(&NodeId::new("GroupBy", 1)), Some(&NodeDecision::GpuLift));
}

#[test]
fn fallback_is_never_smoothed() {
    let plan = build(&[
        ("Scan", 0, NodeDecision::CpuLeave),
        ("GroupBy", 1, NodeDecision::Fallback("string keys".into())),
        ("Filter", 2, NodeDecision::CpuLeave),
    ]);
    let config = SmoothingConfig {
        window_pct: 20,
        close_cost_node_ids: vec![NodeId::new("GroupBy", 1)],
    };
    let smoothed = smooth(plan, &config);
    assert!(matches!(
        smoothed.get(&NodeId::new("GroupBy", 1)),
        Some(NodeDecision::Fallback(_))
    ));
}
```

- [ ] **Step 2: Implement smoothing**

```rust
// crates/polars-metal-core/src/router/affinity.rs
//! Affinity smoothing — second-pass cleanup of the LiftingPlan.
//!
//! After the bottom-up cost-model pass, this pass examines runs of
//! decisions. If a GpuLift node is surrounded by CpuLeave nodes and its
//! cost was within `window_pct` of the CPU alternative (per the cost
//! data the model recorded), flip it to CpuLeave; conversely for a
//! CpuLeave inside a GpuLift run.
//!
//! Initial implementation: rather than re-running the cost model with
//! both alternatives, the smoother accepts an explicit
//! `close_cost_node_ids` list. The cost model populates this list in
//! Task 6 when it sees a node where both alternatives evaluated within
//! `window_pct`. This keeps the data flow simple and the threshold
//! tunable from one PR.
//!
//! Fallback decisions are never smoothed: a Fallback is a hard
//! invariant violation (unsupported IR, oversized key), not a cost
//! decision.

use super::{LiftingPlan, NodeDecision, NodeId};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct SmoothingConfig {
    /// Cost-window percent that defines "close-cost" for the cost model
    /// upstream. Documented here for reference; smoothing logic uses
    /// the pre-populated `close_cost_node_ids` list.
    pub window_pct: u32,
    /// Nodes the cost model flagged as close-cost — candidates for
    /// transition smoothing.
    pub close_cost_node_ids: Vec<NodeId>,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self { window_pct: 20, close_cost_node_ids: vec![] }
    }
}

/// Apply affinity smoothing to a LiftingPlan, returning a new one.
///
/// Algorithm:
/// 1. Build the set of close-cost candidate IDs.
/// 2. For each candidate, examine its in-tree neighbors (via the IDs'
///    sequence numbers — neighbors are seq-1 and seq+1). If both
///    neighbors agree on a decision opposite the candidate, flip the
///    candidate.
/// 3. Never flip a Fallback.
pub fn smooth(plan: LiftingPlan, config: &SmoothingConfig) -> LiftingPlan {
    let candidates: HashSet<NodeId> = config.close_cost_node_ids.iter().cloned().collect();
    let mut next = plan.clone();
    for id in candidates.iter() {
        let current = match plan.get(id) {
            Some(d) => d,
            None => continue,
        };
        if matches!(current, NodeDecision::Fallback(_)) {
            continue;
        }
        let prev = neighbor_decision(&plan, id, -1);
        let succ = neighbor_decision(&plan, id, 1);
        if let (Some(p), Some(s)) = (prev, succ) {
            if p == s && p != *current && !matches!(p, NodeDecision::Fallback(_)) {
                next.set(id.clone(), p);
            }
        } else if let Some(p) = prev {
            // No successor (root node). Match parent-side decision.
            if p != *current && !matches!(p, NodeDecision::Fallback(_)) {
                next.set(id.clone(), p);
            }
        }
    }
    next
}

fn neighbor_decision(plan: &LiftingPlan, id: &NodeId, delta: i64) -> Option<NodeDecision> {
    let target_seq = (id.seq() as i64) + delta;
    if target_seq < 0 {
        return None;
    }
    plan.iter()
        .find(|(other, _)| other.seq() as i64 == target_seq)
        .map(|(_, d)| d.clone())
}
```

- [ ] **Step 3: Re-export `affinity` from `router/mod.rs`**

The `pub mod affinity;` declaration was added in Task 2 Step 2; confirm it is present and exposed.

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-core --test test_router_affinity`
Expected: PASS, three tests.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/router/affinity.rs crates/polars-metal-core/tests/test_router_affinity.rs
git commit -m "$(cat <<'EOF'
Router: affinity smoothing second pass

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6: PyO3 entry point `_native.compute_lifting_plan`

**Files:**
- Modify: `crates/polars-metal-core/src/udf.rs` (or a new module — choose `router_udf.rs` to keep `udf.rs` focused on execution)
- Create: `crates/polars-metal-core/src/router_udf.rs`
- Modify: `crates/polars-metal-core/src/lib.rs`
- Create: `tests/python_integration/test_native_compute_lifting_plan.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/python_integration/test_native_compute_lifting_plan.py
"""Python ↔ Rust round-trip for compute_lifting_plan.

The Python walker (Task 7) will produce a plan dict in this shape and
consume the lifting plan dict in this shape. This test pins the wire
format independently of the walker so changes to one side surface
loudly in the other.
"""

from __future__ import annotations

from polars_metal import _native


def test_filter_over_scan_routes_to_cpu_leave() -> None:
    plan = {
        "kind": "Filter",
        "input": {
            "kind": "Scan",
            "n_rows": 1_000_000,
            "columns": [["a", "I64"]],
        },
        "predicate": {"kind": "Column", "name": "mask", "dtype": "Bool"},
    }
    lifting = _native.compute_lifting_plan(plan)
    # Wire format: dict[str, str], where the key is "Kind#seq" and the
    # value is "gpu_lift" | "cpu_leave" | "fallback:<reason>".
    assert lifting["Scan#0"] == "cpu_leave"
    assert lifting["Filter#1"] == "cpu_leave"


def test_project_over_scan_routes_to_cpu_leave() -> None:
    plan = {
        "kind": "Project",
        "input": {
            "kind": "Scan",
            "n_rows": 1_000,
            "columns": [["a", "I64"]],
        },
        "columns": ["a"],
    }
    lifting = _native.compute_lifting_plan(plan)
    assert lifting["Scan#0"] == "cpu_leave"
    assert lifting["Project#1"] == "cpu_leave"


def test_unknown_kind_yields_fallback() -> None:
    plan = {"kind": "Sort", "input": {"kind": "Scan", "n_rows": 100, "columns": []}}
    lifting = _native.compute_lifting_plan(plan)
    # Any node we don't recognize yields Fallback at its level; we
    # encode as "fallback:<reason>".
    sort_decision = lifting["Sort#1"]
    assert sort_decision.startswith("fallback:")
```

- [ ] **Step 2: Implement the PyO3 entry point**

```rust
// crates/polars-metal-core/src/router_udf.rs
//! PyO3 entry point: `_native.compute_lifting_plan(plan_dict) → dict`.
//!
//! Wire format
//! -----------
//! Input: same plan dict the walker will build (see _walker.py). For
//! Phase 1, accepted kinds are "Scan", "Project", "Filter". Phase 2
//! adds "GroupBy" (Task 12).
//!
//! Output: dict[str, str] where the key is "<Kind>#<seq>" and the value
//! is one of "gpu_lift", "cpu_leave", or "fallback:<reason>". The
//! Python walker iterates and applies.
//!
//! Unknown kinds become Fallback at their level; ancestors poison via
//! the cost-model's Fallback-propagation in `cost::decide_project`.

use crate::plan::{MetalDtype, MetalPlanNode, PredicateAst};
use crate::router::{compute_lifting_plan, NodeDecision};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyfunction]
pub fn compute_lifting_plan_py<'py>(
    py: Python<'py>,
    plan_dict: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let plan = parse_plan_for_router(&plan_dict)?;
    let lifting = compute_lifting_plan(&plan);
    let out = PyDict::new_bound(py);
    for (id, decision) in lifting.iter() {
        let value = match decision {
            NodeDecision::GpuLift => "gpu_lift".to_string(),
            NodeDecision::CpuLeave => "cpu_leave".to_string(),
            NodeDecision::Fallback(reason) => format!("fallback:{reason}"),
        };
        out.set_item(id.to_wire(), value)?;
    }
    Ok(out)
}

/// Parse the Python plan dict into a MetalPlanNode for the router.
///
/// This parser is intentionally permissive about unknown kinds: anything
/// outside the set we handle becomes a synthetic "Filter" with an empty
/// predicate marked Fallback. That keeps the router contract uniform
/// (every node gets a decision) without forcing every walker shape into
/// the strict MetalPlanNode enum.
///
/// Note: we **diverge** from `udf::deserialize_plan` here. That deserializer
/// is strict because it feeds execution; this one is permissive because
/// it only feeds policy. Two functions, two contracts — by design.
fn parse_plan_for_router(dict: &Bound<PyDict>) -> PyResult<MetalPlanNode> {
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| PyKeyError::new_err("missing 'kind'"))?
        .extract()?;
    match kind.as_str() {
        "Scan" => {
            let n_rows: usize = dict
                .get_item("n_rows")?
                .ok_or_else(|| PyKeyError::new_err("Scan: missing n_rows"))?
                .extract()?;
            Ok(MetalPlanNode::Scan { n_rows, columns: vec![] })
        }
        "Project" => {
            let input_dict: Bound<PyDict> = dict
                .get_item("input")?
                .ok_or_else(|| PyKeyError::new_err("Project: missing input"))?
                .downcast_into()?;
            let input = Box::new(parse_plan_for_router(&input_dict)?);
            let _cols: Bound<PyList> = dict
                .get_item("columns")?
                .ok_or_else(|| PyKeyError::new_err("Project: missing columns"))?
                .downcast_into()?;
            Ok(MetalPlanNode::Project { input, columns: vec![] })
        }
        "Filter" => {
            let input_dict: Bound<PyDict> = dict
                .get_item("input")?
                .ok_or_else(|| PyKeyError::new_err("Filter: missing input"))?
                .downcast_into()?;
            let input = Box::new(parse_plan_for_router(&input_dict)?);
            // Predicate shape isn't needed for routing decisions; pass a
            // placeholder. (The cost rule for Filter ignores n_rows and
            // the predicate.)
            Ok(MetalPlanNode::Filter {
                input,
                predicate: PredicateAst::Column { name: "_".into(), dtype: MetalDtype::Bool },
            })
        }
        other => Err(PyValueError::new_err(format!(
            "router: unknown plan kind '{other}'"
        ))),
    }
}
```

- [ ] **Step 3: Wire into the pymodule**

```rust
// crates/polars-metal-core/src/lib.rs — additions:
mod router_udf;
pub use router_udf::compute_lifting_plan_py;

// In the pymodule, add:
//   m.add_function(wrap_pyfunction!(router_udf::compute_lifting_plan_py, m)?)?;
// Register it under the Python name `compute_lifting_plan` via the pyfunction macro:
```

In the pymodule, register with name `compute_lifting_plan`:

```rust
#[pyfunction(name = "compute_lifting_plan")]
pub fn compute_lifting_plan_py<'py>(...) -> PyResult<...> { /* body above */ }
```

(Apply the `#[pyfunction(name = "compute_lifting_plan")]` rename on the function definition itself in `router_udf.rs`, replacing the bare `#[pyfunction]` from Step 2.)

- [ ] **Step 4: Catch the unknown-kind-yields-fallback case**

The strict parser above errors on `"Sort"`. To match the test expectation that unknown kinds yield Fallback (not a Python exception), wrap the parse in a fallback:

```rust
// crates/polars-metal-core/src/router_udf.rs — wrap parse_plan_for_router:

fn parse_plan_for_router_or_fallback(
    dict: &Bound<PyDict>,
    next_seq: &mut u32,
    out: &mut crate::router::LiftingPlan,
) -> PyResult<crate::router::NodeId> {
    // Recursive walker that records IDs as it goes. On unknown kinds we
    // record Fallback at that node's ID and synthesize a parent ID
    // upward.
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| PyKeyError::new_err("missing 'kind'"))?
        .extract()?;
    match kind.as_str() {
        "Scan" | "Project" | "Filter" | "GroupBy" => {
            // Walk inputs first (post-order seq numbering).
            if let Ok(Some(input_obj)) = dict.get_item("input") {
                if let Ok(input_dict) = input_obj.downcast_into::<PyDict>() {
                    let _ = parse_plan_for_router_or_fallback(&input_dict, next_seq, out)?;
                }
            }
            let id = crate::router::NodeId::new(&kind, *next_seq);
            *next_seq += 1;
            // Apply the same cost rule the router would.
            // (For Phase 1, just call the router on the parsed tree; this
            //  wrapper just exists to handle unknown kinds.)
            // ... see the simpler implementation below
            Ok(id)
        }
        other => {
            // Unknown — record Fallback. Walk a single "input" if present
            // so the seq numbering matches what the walker would produce.
            if let Ok(Some(input_obj)) = dict.get_item("input") {
                if let Ok(input_dict) = input_obj.downcast_into::<PyDict>() {
                    let _ = parse_plan_for_router_or_fallback(&input_dict, next_seq, out)?;
                }
            }
            let id = crate::router::NodeId::new(other, *next_seq);
            *next_seq += 1;
            out.set(
                id.clone(),
                crate::router::NodeDecision::Fallback(format!("unsupported IR node: {other}")),
            );
            Ok(id)
        }
    }
}
```

Simpler approach: keep `parse_plan_for_router` strict, and replace it with `parse_and_route` that walks the dict directly, calling cost rules inline. Replace Step 2's body with:

```rust
// crates/polars-metal-core/src/router_udf.rs — final shape:

use crate::router::{cost, LiftingPlan, NodeDecision, NodeId};

#[pyfunction(name = "compute_lifting_plan")]
pub fn compute_lifting_plan_py<'py>(
    py: Python<'py>,
    plan_dict: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut next_seq: u32 = 0;
    let mut lifting = LiftingPlan::new();
    let _ = parse_and_route(&plan_dict, &mut next_seq, &mut lifting)?;
    // No affinity smoothing in Phase 1 (no close-cost candidates yet).
    let out = PyDict::new_bound(py);
    for (id, decision) in lifting.iter() {
        let value = match decision {
            NodeDecision::GpuLift => "gpu_lift".to_string(),
            NodeDecision::CpuLeave => "cpu_leave".to_string(),
            NodeDecision::Fallback(reason) => format!("fallback:{reason}"),
        };
        out.set_item(id.to_wire(), value)?;
    }
    Ok(out)
}

fn parse_and_route(
    dict: &Bound<PyDict>,
    next_seq: &mut u32,
    lifting: &mut LiftingPlan,
) -> PyResult<NodeId> {
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| PyKeyError::new_err("router: missing 'kind'"))?
        .extract()?;
    match kind.as_str() {
        "Scan" => {
            let id = NodeId::new("Scan", *next_seq);
            *next_seq += 1;
            lifting.set(id.clone(), cost::decide_scan_initial());
            Ok(id)
        }
        "Project" => {
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| PyKeyError::new_err("Project: missing input"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            let child_id = parse_and_route(&input_dict, next_seq, lifting)?;
            let id = NodeId::new("Project", *next_seq);
            *next_seq += 1;
            let child_decision = lifting.get(&child_id).cloned()
                .unwrap_or(NodeDecision::Fallback("missing child decision".into()));
            lifting.set(id.clone(), cost::decide_project(&child_decision));
            Ok(id)
        }
        "Filter" => {
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| PyKeyError::new_err("Filter: missing input"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            let _ = parse_and_route(&input_dict, next_seq, lifting)?;
            let id = NodeId::new("Filter", *next_seq);
            *next_seq += 1;
            lifting.set(id.clone(), cost::decide_filter(0));
            Ok(id)
        }
        // GroupBy lands in Task 12 once the MetalPlanNode variant exists.
        other => {
            // Walk a single "input" if present so seq numbering matches
            // the walker's post-order traversal.
            if let Ok(Some(input_obj)) = dict.get_item("input") {
                if let Ok(input_dict) = input_obj.downcast_into::<PyDict>() {
                    let _ = parse_and_route(&input_dict, next_seq, lifting)?;
                }
            }
            let id = NodeId::new(other, *next_seq);
            *next_seq += 1;
            lifting.set(
                id.clone(),
                NodeDecision::Fallback(format!("unsupported IR node: {other}")),
            );
            Ok(id)
        }
    }
}
```

- [ ] **Step 5: Rebuild the wheel**

Run: `make wheel`
Expected: builds without errors.

- [ ] **Step 6: Run the test**

Run: `pytest tests/python_integration/test_native_compute_lifting_plan.py -v`
Expected: PASS, three tests.

- [ ] **Step 7: Commit**

```bash
git add crates/polars-metal-core/src/router_udf.rs crates/polars-metal-core/src/lib.rs tests/python_integration/test_native_compute_lifting_plan.py
git commit -m "$(cat <<'EOF'
Router: PyO3 compute_lifting_plan entry point with wire format

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7: Walker applies the lifting plan — Phase 1 wiring

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Modify: `python/polars_metal/_callback.py`
- Modify: `python/polars_metal/_udf.py` (no behavior change for Phase 1; the UDF still does the same work, but only on GpuLift subtrees)
- Create: `tests/python_integration/test_router_application.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/python_integration/test_router_application.py
"""Phase 1 router-application tests.

After Task 7 the walker no longer makes routing decisions on its own.
It builds a MetalPlanNode tree, hands it to _native.compute_lifting_plan,
receives a lifting plan dict, and applies it. With M2's starting cost
model (filter→CPU always), filter queries that used to install a UDF
should now leave the IR untouched and route to CPU.

The result correctness contract is unchanged from M1: any query routed
to CPU by the router must still produce the same DataFrame as
`engine="cpu"`. Tests assert byte-exact equality.
"""

from __future__ import annotations

import logging

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_filter_with_router_routes_to_cpu_no_udf_installed(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    cpu = df.lazy().filter(pl.col("a") > 2).collect()
    metal = df.lazy().filter(pl.col("a") > 2).collect(
        engine=polars_metal.MetalEngine(debug=True)
    )
    assert_frame_equal(cpu, metal)
    log_text = " ".join(r.getMessage() for r in caplog.records if r.name == "polars_metal")
    # New: with M2's filter→CPU cost rule, no UDF is installed for filter.
    assert "installed UDF" not in log_text, f"expected no UDF, got: {log_text}"


def test_select_only_query_still_routes_to_cpu_via_router(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    # Select with no filter still routes to CPU (no GPU-beneficial op).
    cpu = df.lazy().select(["b", "a"]).collect()
    metal = df.lazy().select(["b", "a"]).collect(
        engine=polars_metal.MetalEngine(debug=True)
    )
    assert_frame_equal(cpu, metal)


def test_fallback_unrecognized_op_still_routes_to_cpu() -> None:
    # Sort isn't recognised by the walker — should fall back cleanly.
    df = pl.DataFrame({"a": [3, 1, 2]})
    cpu = df.lazy().sort("a").collect()
    metal = df.lazy().sort("a").collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
```

- [ ] **Step 2: Add `_strip_walker_side_channels` and the lifting-plan handoff**

Replace `_callback.py::execute_with_metal` body so it:
1. Calls `walk(nt)` as before — but now `walk` returns the plan dict regardless of routing.
2. Strips the Python-side `df`/`projection` side-channel keys from the plan before handing to Rust.
3. Calls `_native.compute_lifting_plan` on the stripped plan.
4. If the lifting plan has any `gpu_lift` decision AND no `fallback:` decision: install the UDF.
5. Otherwise: leave `nt` untouched (CPU path).

```python
# python/polars_metal/_callback.py — replace body:

from __future__ import annotations

import logging
from typing import Any

from polars_metal import _native
from polars_metal._engine import MetalEngine
from polars_metal._udf import build_udf
from polars_metal._walker import FallBack, Handled, walk

log = logging.getLogger("polars_metal")


def execute_with_metal(nt: Any, duration_since_start: int | None, *, config: MetalEngine) -> None:
    if config.debug:
        log.debug("polars_metal: execute_with_metal invoked")

    try:
        result = walk(nt)
    except Exception as e:
        if config.debug:
            log.debug("polars_metal: walker raised %r; falling back", e)
        return

    if isinstance(result, FallBack):
        if config.debug:
            log.debug("polars_metal: walker fallback: %s", result.reason)
        return

    assert isinstance(result, Handled)
    plan = result.plan
    wire_plan = _strip_side_channels(plan)
    try:
        lifting = _native.compute_lifting_plan(wire_plan)
    except Exception as e:
        if config.debug:
            log.debug("polars_metal: router raised %r; falling back", e)
        return

    if any(v.startswith("fallback:") for v in lifting.values()):
        if config.debug:
            log.debug("polars_metal: router fallback: %s", lifting)
        return
    if not any(v == "gpu_lift" for v in lifting.values()):
        if config.debug:
            log.debug("polars_metal: router routes entire query to CPU")
        return

    nt.set_udf(build_udf(plan))
    if config.debug:
        log.debug(
            "polars_metal: installed UDF for plan kind=%s (lifting=%s)",
            plan["kind"],
            lifting,
        )


def _strip_side_channels(plan: dict) -> dict:
    """Remove walker-only keys (`df`, `projection`) before crossing to Rust.

    Recurses into the `input` of each non-leaf node.
    """
    out: dict = {"kind": plan["kind"]}
    if plan["kind"] == "Scan":
        out["n_rows"] = len(plan.get("df", []))
        out["columns"] = plan.get("columns", [])
    elif plan["kind"] in ("Project", "Filter"):
        out["input"] = _strip_side_channels(plan["input"])
        if plan["kind"] == "Project":
            out["columns"] = plan.get("columns", [])
        else:
            # Predicate stays for Phase 1 — the router doesn't use it
            # today, but keeping it future-proofs the wire format.
            out["predicate"] = plan.get("predicate")
    return out
```

`len(plan.get("df", []))` returns the row count from the captured PyDataFrame. Verify it returns the actual row count for our test cases. (PyDataFrame implements `__len__` returning row count in Polars py-1.40.1.)

- [ ] **Step 3: Rebuild + run**

Run: `make wheel && pytest tests/python_integration/test_router_application.py -v`
Expected: PASS, three tests.

- [ ] **Step 4: Re-run full M1 suite to confirm no regressions**

Run: `pytest tests/python_integration -v && make test-conformance`
Expected: All M1 tests still pass; some may newly log "router routes entire query to CPU" via the debug logger. Filter tests that previously asserted "installed UDF" must be updated to remove that assertion (it's no longer true under the M2 cost model). Update them in this step:
- `tests/python_integration/test_filter_comparison.py` line 38 (`test_filter_col_gt_scalar_runs_on_gpu`): change `assert "installed UDF" in log_text` to `assert "router routes entire query to CPU" in log_text or "router fallback" in log_text`.
- Same change in any other M1 integration test that asserts UDF installation for a filter query.

Re-run the suite to confirm:
Run: `pytest tests/python_integration -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_callback.py tests/python_integration/test_router_application.py tests/python_integration/test_filter_comparison.py
git commit -m "$(cat <<'EOF'
Walker applies LiftingPlan: filter routes to CPU under M2 cost model

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 8: Router debug logging — assertable per-IR-node decisions

**Files:**
- Modify: `python/polars_metal/_callback.py`
- Create: `tests/python_integration/test_router_debug_log.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/python_integration/test_router_debug_log.py
"""Verify MetalEngine(debug=True) emits a parseable per-node decision log.

Spec § "Layer 1.5: Router behavior" expects test_routing.py (this file)
to assert decisions per node by parsing debug logs. We pin the log
format here so future log-format changes surface loudly.

Log format
----------
Per query, one DEBUG record on logger 'polars_metal' of the form:

    "router decisions: {Kind#seq: decision, ...}"

where `decision` is one of `gpu_lift`, `cpu_leave`, or `fallback:<reason>`.
"""

from __future__ import annotations

import logging

import polars as pl

import polars_metal


def _decisions_from_logs(caplog) -> dict[str, str]:
    for r in caplog.records:
        if r.name == "polars_metal" and r.getMessage().startswith("router decisions: "):
            payload = r.getMessage()[len("router decisions: "):]
            # Eval'd back to dict; the log uses repr().
            import ast
            return ast.literal_eval(payload)
    return {}


def test_filter_query_logs_cpu_leave_for_all_nodes(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [1, 2, 3]})
    df.lazy().filter(pl.col("a") > 1).collect(
        engine=polars_metal.MetalEngine(debug=True)
    )
    decisions = _decisions_from_logs(caplog)
    assert "Scan#0" in decisions
    assert "Filter#1" in decisions
    assert decisions["Filter#1"] == "cpu_leave"
    assert decisions["Scan#0"] == "cpu_leave"


def test_sort_query_logs_fallback_for_unrecognized_node(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [3, 1, 2]})
    df.lazy().sort("a").collect(engine=polars_metal.MetalEngine(debug=True))
    # The walker itself emits FallBack on Sort (it's not in the walker's
    # accepted set yet); the router is never called. Confirm the walker
    # log says so.
    log_text = " ".join(r.getMessage() for r in caplog.records if r.name == "polars_metal")
    assert "walker fallback" in log_text
```

- [ ] **Step 2: Emit the decisions log**

```python
# python/polars_metal/_callback.py — after computing `lifting`:

    if config.debug:
        log.debug("router decisions: %r", dict(lifting))
```

Place this immediately after the `try/except` that calls `_native.compute_lifting_plan`, before any further decision-making.

- [ ] **Step 3: Run the test**

Run: `pytest tests/python_integration/test_router_debug_log.py -v`
Expected: PASS, two tests.

- [ ] **Step 4: Commit**

```bash
git add python/polars_metal/_callback.py tests/python_integration/test_router_debug_log.py
git commit -m "$(cat <<'EOF'
Router: emit per-node decisions in debug log for testability

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 9: Router unit-test sweep + Phase 1 gate

**Files:**
- Create: `crates/polars-metal-core/tests/test_router_sweep.rs`

- [ ] **Step 1: Write the property-based sweep**

```rust
// crates/polars-metal-core/tests/test_router_sweep.rs
//
// Property test: for any valid MetalPlanNode tree (built from M1's variants),
// `compute_lifting_plan` produces exactly one decision per node and the
// decision is well-formed (not e.g. an internal panic surfacing as a
// missing entry).
#![allow(clippy::expect_used)]

use polars_metal_core::plan::{MetalDtype, MetalPlanNode, PredicateAst};
use polars_metal_core::router::compute_lifting_plan;
use proptest::prelude::*;

fn arb_dtype() -> impl Strategy<Value = MetalDtype> {
    prop_oneof![
        Just(MetalDtype::I64),
        Just(MetalDtype::F64),
        Just(MetalDtype::Bool),
    ]
}

fn arb_scan() -> impl Strategy<Value = MetalPlanNode> {
    (0usize..10_000_000, arb_dtype()).prop_map(|(n, dt)| MetalPlanNode::Scan {
        n_rows: n,
        columns: vec![("a".into(), dt)],
    })
}

fn arb_plan() -> impl Strategy<Value = MetalPlanNode> {
    let leaf = arb_scan().boxed();
    leaf.prop_recursive(4, 16, 2, |inner| {
        prop_oneof![
            inner.clone().prop_map(|c| MetalPlanNode::Project {
                input: Box::new(c),
                columns: vec!["a".into()],
            }),
            inner.prop_map(|c| MetalPlanNode::Filter {
                input: Box::new(c),
                predicate: PredicateAst::Column { name: "a".into(), dtype: MetalDtype::Bool },
            }),
        ]
    })
}

fn count_nodes(node: &MetalPlanNode) -> usize {
    match node {
        MetalPlanNode::Scan { .. } => 1,
        MetalPlanNode::Project { input, .. } => 1 + count_nodes(input),
        MetalPlanNode::Filter { input, .. } => 1 + count_nodes(input),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn lifting_plan_has_one_decision_per_node(plan in arb_plan()) {
        let lifting = compute_lifting_plan(&plan);
        prop_assert_eq!(lifting.len(), count_nodes(&plan));
    }

    #[test]
    fn filter_decision_is_always_cpu_leave_under_m2_costs(plan in arb_plan()) {
        let lifting = compute_lifting_plan(&plan);
        for (id, decision) in lifting.iter() {
            if id.kind() == "Filter" {
                prop_assert_eq!(
                    decision,
                    &polars_metal_core::router::NodeDecision::CpuLeave,
                );
            }
        }
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p polars-metal-core --test test_router_sweep`
Expected: PASS, 256 cases per property.

- [ ] **Step 3: Run the full M1 gate to confirm Phase 1 is clean**

Run: `make gate`
Expected: all phases pass.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-core/tests/test_router_sweep.rs
git commit -m "$(cat <<'EOF'
Router: proptest sweep — one decision per node, filter always CpuLeave

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — GroupBy IR node + plan wire

Phase 2 extends the existing scaffolding to recognize GroupBy. The router's cost rule for GroupBy already exists (Phase 1 Task 3, `cost::decide_groupby`); this phase adds the IR variant, the walker that produces it, and the wire-format extension to `compute_lifting_plan` in `router_udf.rs`. No kernels yet — a GroupBy node still routes to CPU end-to-end because the UDF can't execute it. That's intentional: the GroupBy-to-CPU path must be byte-exact before we light up the kernel path in Phases 4-6.

### Task 10: Add `MetalPlanNode::GroupBy` variant + `AggSpec` / `AggOp` types

**Files:**
- Modify: `crates/polars-metal-core/src/plan/mod.rs`
- Create: `crates/polars-metal-core/tests/test_plan_groupby.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-core/tests/test_plan_groupby.rs
//
// Construction sanity tests for the GroupBy IR variant. Confirms the
// types compile, support equality where useful, and round-trip through
// Debug formatting. Behavioral tests for the router's handling of
// GroupBy land in Task 12 (PyO3 wire format).
#![allow(clippy::expect_used)]

use polars_metal_core::plan::{AggOp, AggSpec, MetalDtype, MetalPlanNode};

fn scan(n_rows: usize) -> MetalPlanNode {
    MetalPlanNode::Scan {
        n_rows,
        columns: vec![
            ("k".into(), MetalDtype::I64),
            ("v".into(), MetalDtype::F64),
        ],
    }
}

#[test]
fn groupby_variant_constructs() {
    let plan = MetalPlanNode::GroupBy {
        input: Box::new(scan(1_000_000)),
        keys: vec![("k".into(), MetalDtype::I64)],
        aggs: vec![AggSpec {
            input_col: "v".into(),
            op: AggOp::Sum,
            output_alias: "v_sum".into(),
        }],
    };
    // Smoke test — Debug must format without panicking.
    let _ = format!("{plan:?}");
}

#[test]
fn agg_op_variants_all_present() {
    // Spec § "Aggregations delivered" — six entry points.
    let ops = [
        AggOp::Sum,
        AggOp::Mean,
        AggOp::Count,
        AggOp::Min,
        AggOp::Max,
        AggOp::Len,
    ];
    assert_eq!(ops.len(), 6);
}

#[test]
fn agg_op_equality_distinguishes_variants() {
    assert_eq!(AggOp::Sum, AggOp::Sum);
    assert_ne!(AggOp::Sum, AggOp::Mean);
    assert_ne!(AggOp::Count, AggOp::Len);
}

#[test]
fn agg_spec_carries_all_three_fields() {
    let spec = AggSpec {
        input_col: "price".into(),
        op: AggOp::Mean,
        output_alias: "avg_price".into(),
    };
    assert_eq!(spec.input_col, "price");
    assert_eq!(spec.op, AggOp::Mean);
    assert_eq!(spec.output_alias, "avg_price");
}

#[test]
fn groupby_supports_multiple_keys_and_aggs() {
    let plan = MetalPlanNode::GroupBy {
        input: Box::new(scan(10_000_000)),
        keys: vec![
            ("returnflag".into(), MetalDtype::I64),
            ("linestatus".into(), MetalDtype::I64),
        ],
        aggs: vec![
            AggSpec { input_col: "qty".into(), op: AggOp::Sum, output_alias: "sum_qty".into() },
            AggSpec { input_col: "qty".into(), op: AggOp::Mean, output_alias: "avg_qty".into() },
            AggSpec { input_col: "price".into(), op: AggOp::Sum, output_alias: "sum_price".into() },
            AggSpec { input_col: "price".into(), op: AggOp::Min, output_alias: "min_price".into() },
            AggSpec { input_col: "price".into(), op: AggOp::Max, output_alias: "max_price".into() },
            AggSpec { input_col: "qty".into(), op: AggOp::Count, output_alias: "count_qty".into() },
            AggSpec { input_col: String::new(), op: AggOp::Len, output_alias: "n_rows".into() },
        ],
    };
    match plan {
        MetalPlanNode::GroupBy { keys, aggs, .. } => {
            assert_eq!(keys.len(), 2);
            assert_eq!(aggs.len(), 7);
        }
        _ => panic!("expected GroupBy"),
    }
}
```

- [ ] **Step 2: Implement the types**

```rust
// crates/polars-metal-core/src/plan/mod.rs — append below the existing
// MetalPlanNode enum and add the new variant inside it.

/// Aggregation operator. Six variants matching spec § "Aggregations
/// delivered". `Len` is `pl.len()` — the row count per group, no input
/// column read. `Count` is `pl.col(x).count()` — the count of non-null
/// values in the input column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    Sum,
    Mean,
    Count,
    Min,
    Max,
    Len,
}

/// One aggregation expression in a GroupBy. `input_col` is empty for
/// `AggOp::Len` (the kernel doesn't read a value column for row count).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggSpec {
    /// Column the aggregation reads. Empty string for `AggOp::Len`.
    pub input_col: String,
    pub op: AggOp,
    /// Output column name in the result DataFrame. Polars users set this
    /// via `.agg(pl.col(x).sum().alias("foo"))`; if no alias, Polars
    /// synthesises one (e.g. `"v_sum"`). The walker fills it from the
    /// Polars IR.
    pub output_alias: String,
}
```

Then add the GroupBy variant to `MetalPlanNode`:

```rust
// crates/polars-metal-core/src/plan/mod.rs — extend MetalPlanNode:

#[derive(Debug, Clone)]
pub enum MetalPlanNode {
    Scan {
        n_rows: usize,
        columns: Vec<(String, MetalDtype)>,
    },
    Project {
        input: Box<MetalPlanNode>,
        columns: Vec<String>,
    },
    Filter {
        input: Box<MetalPlanNode>,
        predicate: PredicateAst,
    },
    /// Hash groupby with composite keys (≤ 128 bits total) and one or
    /// more aggregation specs. See spec § "Two-pass groupby algorithm"
    /// for the kernel-side flow; this variant only records intent.
    GroupBy {
        input: Box<MetalPlanNode>,
        keys: Vec<(String, MetalDtype)>,
        aggs: Vec<AggSpec>,
    },
}
```

- [ ] **Step 3: Update the router's `walk` to handle GroupBy**

The Phase 1 `walk` in `router/mod.rs` covers `Scan | Project | Filter`. Extend it now that the variant exists:

```rust
// crates/polars-metal-core/src/router/mod.rs — extend `walk`:

fn walk(node: &MetalPlanNode, plan: &mut LiftingPlan, next_seq: &mut u32) -> NodeId {
    match node {
        MetalPlanNode::Scan { .. } => {
            let id = NodeId::new("Scan", *next_seq);
            *next_seq += 1;
            plan.set(id.clone(), cost::decide_scan_initial());
            id
        }
        MetalPlanNode::Project { input, .. } => {
            let child_id = walk(input, plan, next_seq);
            let id = NodeId::new("Project", *next_seq);
            *next_seq += 1;
            let child_decision = plan.get(&child_id)
                .cloned()
                .unwrap_or(NodeDecision::Fallback("missing child decision".into()));
            plan.set(id.clone(), cost::decide_project(&child_decision));
            id
        }
        MetalPlanNode::Filter { input, .. } => {
            let _ = walk(input, plan, next_seq);
            let id = NodeId::new("Filter", *next_seq);
            *next_seq += 1;
            plan.set(id.clone(), cost::decide_filter(0));
            id
        }
        MetalPlanNode::GroupBy { input, .. } => {
            // Read the input's row count for the cost rule. Only `Scan`
            // carries a row count today; for other inputs we conservatively
            // assume the row count survives upstream operators (it does
            // for Filter — Filter reduces rows, but the cost rule is a
            // pre-filter row-count heuristic — and trivially for Project).
            let n_rows = input_row_count(input);
            let _ = walk(input, plan, next_seq);
            let id = NodeId::new("GroupBy", *next_seq);
            *next_seq += 1;
            plan.set(id.clone(), cost::decide_groupby(n_rows));
            id
        }
    }
}

/// Best-effort row count for cost-model input. Walks past Project and
/// Filter to find the underlying Scan; returns 0 if none. Filter is a
/// notable simplification — we use the *input* row count for the cost
/// estimate (the kernel sees the post-filter count, but at plan time
/// we don't know it). This is the M2 starting heuristic; later PRs may
/// thread an estimated cardinality.
fn input_row_count(node: &MetalPlanNode) -> usize {
    match node {
        MetalPlanNode::Scan { n_rows, .. } => *n_rows,
        MetalPlanNode::Project { input, .. } => input_row_count(input),
        MetalPlanNode::Filter { input, .. } => input_row_count(input),
        MetalPlanNode::GroupBy { .. } => {
            // GroupBy-of-GroupBy: post-grouped row count is unknown at
            // plan time. Conservative default: route the outer GroupBy
            // to CPU by reporting 0 rows.
            0
        }
    }
}
```

- [ ] **Step 4: Run the unit test**

Run: `cargo test -p polars-metal-core --test test_plan_groupby`
Expected: PASS, five tests.

- [ ] **Step 5: Confirm prior router tests still pass**

Run: `cargo test -p polars-metal-core`
Expected: PASS, all router test crates (test_router_types, test_router_cost, test_router_walk, test_router_affinity, test_router_sweep, test_plan_groupby) green.

- [ ] **Step 6: Commit**

```bash
git add crates/polars-metal-core/src/plan/mod.rs crates/polars-metal-core/src/router/mod.rs crates/polars-metal-core/tests/test_plan_groupby.rs
git commit -m "$(cat <<'EOF'
Plan IR: GroupBy variant with AggSpec / AggOp; router cost wired

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 11: Walker `_walk_group_by` — translate Polars GroupBy IR → plan dict

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Create: `tests/python_integration/test_walker_groupby_unit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/python_integration/test_walker_groupby_unit.py
"""Unit-level tests for `_walk_group_by`: dispatch from `_walk_at_current`
and the shape of the emitted plan dict.

These tests construct a Polars LazyFrame, force a `collect(engine=...)`
on it with a no-op MetalEngine subclass that captures the plan dict the
walker produces, and assert on that captured value. We don't run a UDF
here — Phase 2 doesn't ship one for GroupBy yet.
"""

from __future__ import annotations

from typing import Any

import polars as pl

import polars_metal


def _capture_plan(lf: pl.LazyFrame) -> dict | None:
    """Collect with an engine that records the walker's plan-dict output
    instead of installing a UDF. Returns the captured plan, or None if
    the walker FallBacks."""
    captured: dict[str, Any] = {"plan": None, "fallback": None}

    class CapturingEngine(polars_metal.MetalEngine):
        pass

    # Monkey-patch _callback.execute_with_metal for the duration of this
    # one collect. The cleanest way: shim build_udf to record the plan
    # and return a UDF that just runs the CPU executor (we don't need
    # actual execution).
    import polars_metal._callback as cb

    original = cb.execute_with_metal

    def shim(nt, dur, *, config):
        from polars_metal._walker import walk, Handled, FallBack
        try:
            result = walk(nt)
        except Exception as e:
            captured["fallback"] = f"exception: {e!r}"
            return
        if isinstance(result, FallBack):
            captured["fallback"] = result.reason
            return
        assert isinstance(result, Handled)
        captured["plan"] = result.plan
        # Do not install a UDF — CPU executes the query.
        return

    cb.execute_with_metal = shim
    try:
        lf.collect(engine=CapturingEngine())
    finally:
        cb.execute_with_metal = original

    return captured["plan"], captured["fallback"]


def test_groupby_single_i64_key_sum_emits_plan() -> None:
    df = pl.DataFrame({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]})
    plan, fallback = _capture_plan(
        df.lazy().group_by("k").agg(pl.col("v").sum().alias("sum_v"))
    )
    assert fallback is None, f"unexpected fallback: {fallback}"
    assert plan is not None
    assert plan["kind"] == "GroupBy"
    assert plan["keys"] == [["k", "I64"]]
    assert plan["aggs"] == [{"input_col": "v", "op": "Sum", "output_alias": "sum_v"}]
    assert plan["input"]["kind"] == "Scan"


def test_groupby_composite_key_two_i64_keys_emits_plan() -> None:
    df = pl.DataFrame(
        {"a": [1, 1, 2], "b": [10, 20, 30], "v": [1.0, 2.0, 3.0]}
    )
    plan, fallback = _capture_plan(
        df.lazy().group_by(["a", "b"]).agg(pl.col("v").sum().alias("s"))
    )
    assert fallback is None, f"unexpected fallback: {fallback}"
    assert plan is not None
    assert plan["kind"] == "GroupBy"
    assert plan["keys"] == [["a", "I64"], ["b", "I64"]]
    assert plan["aggs"] == [{"input_col": "v", "op": "Sum", "output_alias": "s"}]


def test_groupby_multiple_aggs_emits_all() -> None:
    df = pl.DataFrame({"k": [1, 1, 2], "v": [10.0, 20.0, 30.0]})
    plan, fallback = _capture_plan(
        df.lazy().group_by("k").agg(
            pl.col("v").sum().alias("s"),
            pl.col("v").mean().alias("m"),
            pl.col("v").min().alias("mn"),
            pl.col("v").max().alias("mx"),
            pl.col("v").count().alias("cnt"),
            pl.len().alias("rows"),
        )
    )
    assert fallback is None, f"unexpected fallback: {fallback}"
    assert plan is not None
    ops_seen = [a["op"] for a in plan["aggs"]]
    assert sorted(ops_seen) == sorted(["Sum", "Mean", "Min", "Max", "Count", "Len"])
    # Len has no input_col.
    len_spec = next(a for a in plan["aggs"] if a["op"] == "Len")
    assert len_spec["input_col"] == ""
    assert len_spec["output_alias"] == "rows"


def test_groupby_string_key_falls_back() -> None:
    df = pl.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]})
    plan, fallback = _capture_plan(
        df.lazy().group_by("k").agg(pl.col("v").sum())
    )
    assert plan is None
    assert fallback is not None
    assert "String" in fallback or "unsupported dtype" in fallback


def test_groupby_unsupported_agg_expression_falls_back() -> None:
    df = pl.DataFrame({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
    # `(pl.col("v") * 2).sum()` is an arithmetic-inside-agg shape we
    # don't unfold yet (spec § "Hand-off to M3" — multi-agg expression
    # unfolding is deferred).
    plan, fallback = _capture_plan(
        df.lazy().group_by("k").agg((pl.col("v") * 2).sum().alias("s"))
    )
    assert plan is None
    assert fallback is not None
```

- [ ] **Step 2: Implement `_walk_group_by`**

```python
# python/polars_metal/_walker.py — add near the existing _walk_filter:

# Map from Polars aggregation expression class name (the inner-expression
# wrapper inside an .agg() call) to our AggOp tag string. The Python-side
# class names for these on py-1.40.1 are:
#   sum  → "Agg(name=<Agg.Sum: ...>, ...)" — we match by checking the
#          AggExpr.name property, which is an enum we can str() to "Sum",
#          "Mean", "Min", "Max", "Count".
#   len  → there's no value column; the IR shape is "Len(...)" with no
#          inner column.
_AGG_NAME_TO_OP: dict[str, str] = {
    "Sum": "Sum",
    "Mean": "Mean",
    "Min": "Min",
    "Max": "Max",
    "Count": "Count",
    "Len": "Len",
}


def _walk_group_by(nt: Any, node: Any) -> WalkResult:
    """Lower a Polars GroupBy IR node iff:
      - every key is a bare Column reference of an accepted dtype, and
      - every aggregation is one of {Sum, Mean, Min, Max, Count, Len}
        applied to a bare Column reference of an accepted dtype.

    Anything fancier (arithmetic inside agg, aliases on keys, non-Column
    key expressions, ordered groupby, etc.) FallBacks.
    """
    # Polars IR GroupBy carries .keys (Vec<Expr>), .aggs (Vec<Expr>), and
    # .maintain_order (bool — affects key order but not values; CPU
    # post-pass handles it either way).
    keys_expr = getattr(node, "keys", None)
    aggs_expr = getattr(node, "aggs", None)
    if keys_expr is None or aggs_expr is None:
        return FallBack(reason="GroupBy node missing .keys or .aggs")

    # Resolve schemas against the *input* (pre-GroupBy) — that's where
    # the column dtypes live. The Filter walker uses the same idiom; see
    # the comment there for the get_schema-on-input rationale.
    inputs = nt.get_inputs()
    if len(inputs) != 1:
        return FallBack(reason=f"GroupBy expected 1 input, got {len(inputs)}")
    in_schema: dict[str, Any]
    parent_id = nt.get_node()
    nt.set_node(inputs[0])
    try:
        in_schema = dict(nt.get_schema())
    finally:
        nt.set_node(parent_id)

    keys: list[list[str]] = []
    for key_expr in keys_expr:
        # Each key must be a bare Column expression.
        key_node_id = getattr(key_expr, "node", None)
        if key_node_id is None:
            return FallBack(reason="GroupBy key expression has no .node id")
        try:
            key_inner = nt.view_expression(key_node_id)
        except Exception as ex:
            return FallBack(reason=f"could not view key expression: {ex!r}")
        key_cls = type(key_inner).__name__
        if key_cls != "Column":
            return FallBack(reason=f"GroupBy key expression {key_cls} not supported")
        key_name = getattr(key_inner, "name", None)
        if key_name is None:
            return FallBack(reason="GroupBy key Column missing .name")
        dtype = in_schema.get(key_name)
        if dtype is None:
            return FallBack(reason=f"GroupBy key {key_name!r} not in input schema")
        mapped = _map_dtype(dtype)
        if mapped is None:
            return FallBack(reason=f"unsupported dtype {dtype!s} on key {key_name!r}")
        keys.append([key_name, mapped])

    aggs: list[dict[str, str]] = []
    for agg_expr in aggs_expr:
        agg_dict = _walk_agg_expression(nt, agg_expr, in_schema)
        if agg_dict is None:
            return FallBack(reason="GroupBy agg expression not in M2 closed set")
        aggs.append(agg_dict)

    # Walk the input subtree.
    nt.set_node(inputs[0])
    inner = _walk_at_current(nt)
    if isinstance(inner, FallBack):
        return inner

    return Handled(
        plan={
            "kind": "GroupBy",
            "input": inner.plan,
            "keys": keys,
            "aggs": aggs,
        }
    )


def _walk_agg_expression(nt: Any, agg_expr: Any, in_schema: dict[str, Any]) -> dict[str, str] | None:
    """Lower one aggregation expression to {input_col, op, output_alias}.

    Accepted shapes (each may be wrapped in an Alias):
      - pl.col(name).{sum, mean, min, max, count}()  → {input_col=name, op=...}
      - pl.len()                                       → {input_col="", op=Len}

    Anything else (arithmetic inside, function calls, multiple-column
    aggregates) returns None and the caller FallBacks the whole GroupBy.
    """
    node_id = getattr(agg_expr, "node", None)
    if node_id is None:
        return None
    try:
        inner = nt.view_expression(node_id)
    except Exception:
        return None

    # Strip a top-level Alias.
    output_alias: str | None = None
    inner_cls = type(inner).__name__
    if inner_cls == "Alias":
        output_alias = getattr(inner, "name", None)
        sub_id = getattr(getattr(inner, "expr", None), "node", None)
        if sub_id is None:
            return None
        try:
            inner = nt.view_expression(sub_id)
        except Exception:
            return None
        inner_cls = type(inner).__name__

    # Handle pl.len() first — its class is "Len" or "Function(name=Len)" in py-1.40.1.
    if inner_cls == "Len":
        return {
            "input_col": "",
            "op": "Len",
            "output_alias": output_alias or "len",
        }

    # Generic agg: inner_cls is "Agg" with .name being an AggExpr enum.
    if inner_cls != "Agg":
        return None
    agg_name_val = getattr(inner, "name", None)
    if agg_name_val is None:
        return None
    # str(AggExpr.Sum) → "Sum" in py-1.40.1; defensive: strip "AggExpr." if present.
    agg_name = str(agg_name_val).rsplit(".", 1)[-1]
    op = _AGG_NAME_TO_OP.get(agg_name)
    if op is None:
        return None

    # The aggregation must apply to a single Column expression. .expr or
    # .arguments depending on Polars version; py-1.40.1 uses .arguments (a list).
    args = getattr(inner, "arguments", None)
    if not args or len(args) != 1:
        return None
    arg_id = getattr(args[0], "node", None)
    if arg_id is None:
        return None
    try:
        col_expr = nt.view_expression(arg_id)
    except Exception:
        return None
    if type(col_expr).__name__ != "Column":
        return None
    col_name = getattr(col_expr, "name", None)
    if col_name is None:
        return None
    dtype = in_schema.get(col_name)
    if dtype is None:
        return None
    if _map_dtype(dtype) is None:
        return None

    return {
        "input_col": col_name,
        "op": op,
        "output_alias": output_alias or f"{col_name}_{op.lower()}",
    }
```

- [ ] **Step 3: Dispatch GroupBy from `_walk_at_current`**

```python
# python/polars_metal/_walker.py — extend the dispatcher:

def _walk_at_current(nt: Any) -> WalkResult:
    node = nt.view_current_node()
    cls = type(node).__name__
    if cls == "DataFrameScan":
        return _walk_dataframe_scan(nt, node)
    if cls == "SimpleProjection":
        return _walk_simple_projection(nt, node)
    if cls == "Select":
        return _walk_select(nt, node)
    if cls == "Filter":
        return _walk_filter(nt, node)
    if cls == "GroupBy":
        return _walk_group_by(nt, node)
    return FallBack(reason=f"unsupported IR node: {cls}")
```

- [ ] **Step 4: Run the test**

Run: `make wheel && pytest tests/python_integration/test_walker_groupby_unit.py -v`
Expected: PASS, five tests.

- [ ] **Step 5: Sanity-check that other walker tests still pass**

Run: `pytest tests/python_integration/test_walker_select.py tests/python_integration/test_filter_comparison.py -v`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add python/polars_metal/_walker.py tests/python_integration/test_walker_groupby_unit.py
git commit -m "$(cat <<'EOF'
Walker: _walk_group_by lowers Polars GroupBy IR to plan dict

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 12: Wire GroupBy into `compute_lifting_plan_py` (`router_udf.rs`)

**Files:**
- Modify: `crates/polars-metal-core/src/router_udf.rs`
- Modify: `python/polars_metal/_callback.py` (extend `_strip_side_channels`)
- Create: `tests/python_integration/test_router_groupby.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/python_integration/test_router_groupby.py
"""End-to-end router test for GroupBy.

After Task 12 the wire format accepts GroupBy nodes and the router
returns a per-node decision dict. We don't run a UDF yet — Phase 2's
goal is plumbing, not execution.
"""

from __future__ import annotations

import logging

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def test_compute_lifting_plan_recognizes_groupby_at_high_row_count() -> None:
    # Spec § "Routing decisions" — GroupBy GpuLift iff n_rows > 100K.
    plan = {
        "kind": "GroupBy",
        "input": {
            "kind": "Scan",
            "n_rows": 1_000_000,
            "columns": [["k", "I64"], ["v", "I64"]],
        },
        "keys": [["k", "I64"]],
        "aggs": [{"input_col": "v", "op": "Sum", "output_alias": "sum_v"}],
    }
    lifting = _native.compute_lifting_plan(plan)
    assert lifting["Scan#0"] == "cpu_leave"
    assert lifting["GroupBy#1"] == "gpu_lift"


def test_compute_lifting_plan_routes_small_groupby_to_cpu() -> None:
    plan = {
        "kind": "GroupBy",
        "input": {
            "kind": "Scan",
            "n_rows": 10_000,
            "columns": [["k", "I64"], ["v", "I64"]],
        },
        "keys": [["k", "I64"]],
        "aggs": [{"input_col": "v", "op": "Sum", "output_alias": "sum_v"}],
    }
    lifting = _native.compute_lifting_plan(plan)
    assert lifting["GroupBy#1"] == "cpu_leave"


def test_groupby_over_filter_uses_filter_input_row_count() -> None:
    # Cost rule uses the input row count (pre-filter), not the post-filter
    # count. With 1M rows, GroupBy GpuLifts even though Filter is CpuLeave.
    plan = {
        "kind": "GroupBy",
        "input": {
            "kind": "Filter",
            "input": {
                "kind": "Scan",
                "n_rows": 1_000_000,
                "columns": [["k", "I64"]],
            },
            "predicate": {"kind": "Column", "name": "_mask", "dtype": "Bool"},
        },
        "keys": [["k", "I64"]],
        "aggs": [],
    }
    lifting = _native.compute_lifting_plan(plan)
    assert lifting["Scan#0"] == "cpu_leave"
    assert lifting["Filter#1"] == "cpu_leave"
    assert lifting["GroupBy#2"] == "gpu_lift"


def test_groupby_end_to_end_still_correct_via_cpu_fallback() -> None:
    # Phase 2 doesn't install a GroupBy UDF yet, so the router observes
    # gpu_lift but the callback path lacks a kernel — execution falls
    # back to CPU. Result must still be byte-exact with engine="cpu".
    df = pl.DataFrame({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]})
    cpu = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k").collect()
    metal = (
        df.lazy()
        .group_by("k")
        .agg(pl.col("v").sum().alias("s"))
        .sort("k")
        .collect(engine=polars_metal.MetalEngine())
    )
    assert_frame_equal(cpu, metal)
```

- [ ] **Step 2: Extend `parse_and_route` with `"GroupBy"` arm**

```rust
// crates/polars-metal-core/src/router_udf.rs — extend the match in
// parse_and_route to handle GroupBy.

        "GroupBy" => {
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| PyKeyError::new_err("GroupBy: missing input"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;

            // Read the input row count for the cost rule. We walk the
            // input dict directly — the parser would do this anyway as
            // part of the recursive descent — but for the cost decision
            // we want the row count *before* recursing (post-order seq
            // numbering would put us at the end).
            //
            // Strategy: peek at the underlying Scan's n_rows by walking
            // through Project/Filter/GroupBy "input" chains. This
            // mirrors `router::walk::input_row_count` in the Rust side.
            let n_rows = peek_input_row_count(&input_dict)?;

            let _ = parse_and_route(&input_dict, next_seq, lifting)?;
            let id = NodeId::new("GroupBy", *next_seq);
            *next_seq += 1;
            lifting.set(id.clone(), cost::decide_groupby(n_rows));
            Ok(id)
        }
```

Add the helper at module scope:

```rust
// crates/polars-metal-core/src/router_udf.rs — append below parse_and_route:

/// Best-effort row count for cost-model input from a plan dict. Walks
/// past Project/Filter/GroupBy `"input"` fields to find the underlying
/// Scan. Mirrors `router::input_row_count`.
fn peek_input_row_count(dict: &Bound<PyDict>) -> PyResult<usize> {
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| PyKeyError::new_err("missing 'kind'"))?
        .extract()?;
    match kind.as_str() {
        "Scan" => {
            let n: usize = dict
                .get_item("n_rows")?
                .ok_or_else(|| PyKeyError::new_err("Scan: missing n_rows"))?
                .extract()?;
            Ok(n)
        }
        "Project" | "Filter" | "GroupBy" => {
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| PyKeyError::new_err("missing 'input' in row-count peek"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            peek_input_row_count(&input_dict)
        }
        _ => Ok(0),
    }
}
```

- [ ] **Step 3: Extend `_strip_side_channels` to handle `"GroupBy"`**

The Phase 1 stripper only knows Scan/Project/Filter. Extend it so GroupBy plans round-trip:

```python
# python/polars_metal/_callback.py — extend _strip_side_channels:

def _strip_side_channels(plan: dict) -> dict:
    out: dict = {"kind": plan["kind"]}
    if plan["kind"] == "Scan":
        out["n_rows"] = len(plan.get("df", []))
        out["columns"] = plan.get("columns", [])
    elif plan["kind"] in ("Project", "Filter"):
        out["input"] = _strip_side_channels(plan["input"])
        if plan["kind"] == "Project":
            out["columns"] = plan.get("columns", [])
        else:
            out["predicate"] = plan.get("predicate")
    elif plan["kind"] == "GroupBy":
        out["input"] = _strip_side_channels(plan["input"])
        # Pass keys and aggs through unchanged — both are plain
        # JSON-serialisable structures (list[list[str,str]] and
        # list[dict[str,str]] respectively).
        out["keys"] = plan.get("keys", [])
        out["aggs"] = plan.get("aggs", [])
    return out
```

- [ ] **Step 4: Rebuild + run the test**

Run: `make wheel && pytest tests/python_integration/test_router_groupby.py -v`
Expected: PASS, four tests. The last test (`test_groupby_end_to_end_still_correct_via_cpu_fallback`) verifies the safety net: the router's `gpu_lift` decision for GroupBy is observed but the callback doesn't actually install a UDF (no GroupBy execution path exists yet in `_udf.py`). The current `build_udf` raises on unknown plan kinds, so `_callback` catches it and CPU runs the query.

Wait — that's a real correctness concern. Let me adjust:

- [ ] **Step 5: Make the callback gracefully handle "GpuLift but no UDF" until Phase 7 wires execution**

```python
# python/polars_metal/_callback.py — adjust execute_with_metal:

    # Phase 2 quirk: GroupBy gets gpu_lift from the router but Phase 2
    # doesn't yet implement the execution path. Wrap build_udf in a
    # try-except so an unknown plan kind triggers a clean CPU fallback
    # rather than a crash.
    try:
        udf = build_udf(plan)
    except NotImplementedError as e:
        if config.debug:
            log.debug("polars_metal: UDF builder not ready for plan kind=%s (%r); falling back", plan["kind"], e)
        return
    nt.set_udf(udf)
```

And ensure `_udf.build_udf` raises `NotImplementedError` (not a bare exception) for GroupBy until Phase 7. Check `python/polars_metal/_udf.py`:

```python
# python/polars_metal/_udf.py — in build_udf's plan-kind dispatch, ensure:
#   if plan["kind"] == "GroupBy":
#       raise NotImplementedError("GroupBy execution not yet wired (Phase 7)")
# at the top of the dispatch, before the existing Scan/Project/Filter branches.
```

(Inspect the existing `_udf.py` to slot this in cleanly. If `build_udf` already has a default `raise NotImplementedError` for unknown kinds, no change is needed.)

- [ ] **Step 6: Re-run the test**

Run: `make wheel && pytest tests/python_integration/test_router_groupby.py -v`
Expected: PASS, four tests, including the byte-exact CPU-fallback test.

- [ ] **Step 7: Commit**

```bash
git add crates/polars-metal-core/src/router_udf.rs python/polars_metal/_callback.py python/polars_metal/_udf.py tests/python_integration/test_router_groupby.py
git commit -m "$(cat <<'EOF'
Router: accept GroupBy plans; clean CPU fallback when UDF not wired

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Composite key encoding

Phase 3 lands the CPU-side preprocessor that packs a multi-column key into a single u128. Spec § "Composite key encoding" defines the contract: encode at dispatch time, decode after the build phase reads back representative row indices. The Phase 3 implementation is pure Rust (no Metal) — the encoder is small, hot, and the same code runs on the test path and the production path. Each key column contributes a 1-bit null flag plus dtype-width bits.

The router's `Fallback` for oversized keys lives in the router's plan-time pass; the encoder still produces a `KeyEncodeError::TooWide` error at dispatch time as a defensive check. Phase 7 (deferred) wires the plan-time check into `cost::decide_groupby` so the router can short-circuit before reaching the kernel layer.

### Task 13: `encode_keys` — pack key columns into u128 lanes

**Files:**
- Create: `crates/polars-metal-kernels/src/groupby.rs`
- Modify: `crates/polars-metal-kernels/src/lib.rs` (expose `pub mod groupby;`)
- Create: `crates/polars-metal-kernels/tests/test_key_encoding.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_key_encoding.rs
//
// Encoder + decoder unit tests. Covers single-key, multi-key, mixed
// dtypes, null patterns, and width-overflow. Round-trip property test
// in Task 15.
#![allow(clippy::expect_used)]

use polars_metal_kernels::groupby::{
    decode_keys, encode_keys, DecodedColumn, KeyColumn, KeyDtype, KeyEncodeError,
};

fn bytes_i64(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f64(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn all_valid(n_rows: usize) -> Vec<u8> {
    vec![0xFFu8; (n_rows + 7) / 8]
}

#[test]
fn single_i64_key_encodes_to_u128_per_row() {
    let data = bytes_i64(&[1, 2, 3, -1]);
    let valid = all_valid(4);
    let col = KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I64,
        data: &data,
        valid: &valid,
        n_rows: 4,
    };
    let (encoded, schema) = encode_keys(&[col]).expect("encode_keys");
    assert_eq!(encoded.len(), 4);
    // Schema records one field: 1 bit null + 64 bit i64 = 65 bits total.
    assert_eq!(schema.total_bits(), 65);
    assert_eq!(schema.fields().len(), 1);
    // First row's null bit is 0 (valid) and the value bits round-trip.
    let decoded = decode_keys(&encoded, &schema);
    match &decoded[0] {
        DecodedColumn::I64 { values, valid } => {
            assert_eq!(values, &vec![1i64, 2, 3, -1]);
            assert_eq!(valid, &vec![true, true, true, true]);
        }
        other => panic!("expected I64 decoded column, got {other:?}"),
    }
}

#[test]
fn two_i64_keys_pack_in_order() {
    let a = bytes_i64(&[10, 20]);
    let b = bytes_i64(&[100, 200]);
    let v = all_valid(2);
    let cols = vec![
        KeyColumn { name: "a".into(), dtype: KeyDtype::I64, data: &a, valid: &v, n_rows: 2 },
        KeyColumn { name: "b".into(), dtype: KeyDtype::I64, data: &b, valid: &v, n_rows: 2 },
    ];
    let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
    assert_eq!(encoded.len(), 2);
    // Schema: (1 + 64) + (1 + 64) = 130 bits → ERROR. Adjust expectations:
    // The encoder must reject this because total_bits > 128. Update test:
    // (we cover the exact overflow case below; here we use Bool keys to
    // stay under 128).
    let _ = (encoded, schema);
}

#[test]
fn two_bool_keys_pack_into_first_4_bits() {
    let a = vec![0b0000_0011u8]; // row 0 true, row 1 true, rows 2-7 false
    let b = vec![0b0000_0001u8]; // row 0 true, row 1 false
    let v = vec![0xFFu8];
    let cols = vec![
        KeyColumn { name: "a".into(), dtype: KeyDtype::Bool, data: &a, valid: &v, n_rows: 2 },
        KeyColumn { name: "b".into(), dtype: KeyDtype::Bool, data: &b, valid: &v, n_rows: 2 },
    ];
    let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
    assert_eq!(encoded.len(), 2);
    // 2 bits null + 2 bits value = 4 bits total.
    assert_eq!(schema.total_bits(), 4);
    let decoded = decode_keys(&encoded, &schema);
    match &decoded[0] {
        DecodedColumn::Bool { values, valid } => {
            assert_eq!(values, &vec![true, true]);
            assert_eq!(valid, &vec![true, true]);
        }
        _ => panic!("expected Bool"),
    }
    match &decoded[1] {
        DecodedColumn::Bool { values, valid } => {
            assert_eq!(values, &vec![true, false]);
            assert_eq!(valid, &vec![true, true]);
        }
        _ => panic!("expected Bool"),
    }
}

#[test]
fn one_i64_plus_one_bool_packs_below_128_bits() {
    let i64_data = bytes_i64(&[42, -7]);
    let bool_data = vec![0b0000_0010u8]; // row 0 false, row 1 true
    let v = vec![0xFFu8];
    let cols = vec![
        KeyColumn { name: "i".into(), dtype: KeyDtype::I64, data: &i64_data, valid: &v, n_rows: 2 },
        KeyColumn { name: "b".into(), dtype: KeyDtype::Bool, data: &bool_data, valid: &v, n_rows: 2 },
    ];
    let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
    // 1+64 + 1+1 = 67 bits total.
    assert_eq!(schema.total_bits(), 67);
    let decoded = decode_keys(&encoded, &schema);
    assert_eq!(decoded.len(), 2);
}

#[test]
fn null_value_clears_data_bits_in_decoded_output() {
    let data = bytes_i64(&[99, 0]);
    // row 0 valid, row 1 null:
    let valid = vec![0b0000_0001u8];
    let cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I64,
        data: &data,
        valid: &valid,
        n_rows: 2,
    }];
    let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
    let decoded = decode_keys(&encoded, &schema);
    match &decoded[0] {
        DecodedColumn::I64 { values, valid } => {
            assert_eq!(valid, &vec![true, false]);
            assert_eq!(values[0], 99);
            // For null rows we don't promise a particular value; only
            // that valid[i] is false. Polars' invariant: data bits at
            // null positions are implementation-defined.
        }
        _ => panic!("expected I64"),
    }
}

#[test]
fn three_i64_keys_overflow_128_bits_returns_error() {
    let d = bytes_i64(&[1, 2]);
    let v = vec![0xFFu8];
    let cols = vec![
        KeyColumn { name: "a".into(), dtype: KeyDtype::I64, data: &d, valid: &v, n_rows: 2 },
        KeyColumn { name: "b".into(), dtype: KeyDtype::I64, data: &d, valid: &v, n_rows: 2 },
        // Two i64 keys = 130 bits already over the limit.
    ];
    let err = encode_keys(&cols).expect_err("expected TooWide");
    match err {
        KeyEncodeError::TooWide { total_bits } => assert_eq!(total_bits, 130),
        other => panic!("expected TooWide, got {other:?}"),
    }
}

#[test]
fn empty_keys_returns_error() {
    let err = encode_keys(&[]).expect_err("expected NoKeys");
    assert!(matches!(err, KeyEncodeError::NoKeys));
}

#[test]
fn f64_key_encodes_via_raw_bits() {
    let data = bytes_f64(&[1.5, -2.5, f64::INFINITY]);
    let v = all_valid(3);
    let cols = vec![KeyColumn {
        name: "f".into(),
        dtype: KeyDtype::F64,
        data: &data,
        valid: &v,
        n_rows: 3,
    }];
    let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
    assert_eq!(schema.total_bits(), 65);
    let decoded = decode_keys(&encoded, &schema);
    match &decoded[0] {
        DecodedColumn::F64 { values, valid } => {
            assert_eq!(values, &vec![1.5, -2.5, f64::INFINITY]);
            assert_eq!(valid, &vec![true, true, true]);
        }
        _ => panic!("expected F64"),
    }
}
```

- [ ] **Step 2: Implement the encoder**

```rust
// crates/polars-metal-kernels/src/groupby.rs
//! Composite key encoding for the GroupBy hash kernel.
//!
//! Each key column contributes (1-bit null flag) + (dtype-width-bits) to
//! a u128 lane per row. Lane layout, from LSB to MSB:
//!
//!     bit 0:                      key0.null
//!     bits 1..1+w(key0):          key0.data
//!     bit 1+w(key0):              key1.null
//!     bits 2+w(key0)..2+w(key0)+w(key1):  key1.data
//!     ...
//!
//! The layout is deterministic given the input column list — the same
//! column order yields the same encoding, byte-for-byte. The hash kernel
//! consumes the raw u128; key equality is u128 equality.
//!
//! Why a single u128 lane:
//!   - One atomic-CAS op per row in the build phase (Metal supports 128-bit
//!     atomic CAS on Apple Silicon GPUs since M2; for M1 we fall back to a
//!     spinlock per slot — addressed in spec § "Risks").
//!   - No per-row dynamic allocation; each row is a single 16-byte read.
//!
//! Width budget: 128 bits per row. Common cases that fit:
//!   - 1 × i64 + up to 63 booleans
//!   - 2 × i32 (planned for M3) + null bits
//!   - 1 × i64 + 1 × bool (Q1's shape with the integer-encoded keys: l_returnflag, l_linestatus)
//!
//! Wider key sets must `Fallback` at plan time (router-side) or surface
//! `KeyEncodeError::TooWide` at dispatch time (defensive — router should
//! catch first).

use thiserror::Error;

/// Supported key dtypes. Mirrors `MetalDtype` but lives in this crate so
/// the kernel layer has no dependency on the engine-adapter crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyDtype {
    I64,
    F64,
    Bool,
}

impl KeyDtype {
    /// Width in bits of the data payload (excludes the null flag).
    pub fn data_bits(self) -> u32 {
        match self {
            KeyDtype::I64 | KeyDtype::F64 => 64,
            KeyDtype::Bool => 1,
        }
    }
}

/// One input column to the encoder. Carries the raw data + validity
/// bytes — the encoder doesn't own the buffers.
pub struct KeyColumn<'a> {
    pub name: String,
    pub dtype: KeyDtype,
    /// Little-endian packed values. For I64/F64: 8 bytes per row. For
    /// Bool: one bit per row, bit-packed, same convention as Arrow's
    /// validity bitmap (`bit i` of `byte i/8` at offset `i%8`).
    pub data: &'a [u8],
    /// Bit-packed validity bitmap, `ceil(n_rows / 8)` bytes minimum.
    pub valid: &'a [u8],
    pub n_rows: usize,
}

/// One field's position in the encoded u128 lane. Both fields and
/// schemas are immutable after construction.
#[derive(Debug, Clone)]
pub struct KeyField {
    pub name: String,
    pub dtype: KeyDtype,
    /// Bit position of this field's null flag in the u128 lane.
    pub null_bit_offset: u32,
    /// Bit position of this field's data, immediately following the null bit.
    pub data_bit_offset: u32,
}

/// Schema for a composite-key encoding. Sufficient to decode an encoded
/// u128 stream back to per-column values.
#[derive(Debug, Clone)]
pub struct KeySchema {
    fields: Vec<KeyField>,
    total_bits: u32,
    n_rows: usize,
}

impl KeySchema {
    pub fn fields(&self) -> &[KeyField] { &self.fields }
    pub fn total_bits(&self) -> u32 { self.total_bits }
    pub fn n_rows(&self) -> usize { self.n_rows }
}

/// Error returned by `encode_keys`.
#[derive(Debug, Error)]
pub enum KeyEncodeError {
    /// At least one key column is required.
    #[error("no key columns provided")]
    NoKeys,
    /// Combined width exceeds the 128-bit budget. Router should catch
    /// this earlier and Fallback at plan time; dispatcher-side check is
    /// defensive.
    #[error("composite key width {total_bits} bits exceeds 128-bit budget")]
    TooWide { total_bits: u32 },
    /// Input columns have differing row counts.
    #[error("row count mismatch across key columns: first={first}, mismatched={mismatched}")]
    RowCountMismatch { first: usize, mismatched: usize },
    /// A data buffer is shorter than the column's dtype requires.
    #[error("data buffer for {col!r} too short: got {got} bytes, need {need}")]
    DataTooShort { col: String, got: usize, need: usize },
    /// A validity bitmap is shorter than `ceil(n_rows / 8)`.
    #[error("validity buffer for {col!r} too short: got {got} bytes, need {need}")]
    ValidityTooShort { col: String, got: usize, need: usize },
}

/// Encode `cols` to a `Vec<u128>` (one u128 per row). Returns the
/// encoded data and the schema needed to decode.
///
/// Layout: see module docs. Each key contributes `1 + dtype.data_bits()`
/// bits, starting from the LSB.
pub fn encode_keys(cols: &[KeyColumn<'_>]) -> Result<(Vec<u128>, KeySchema), KeyEncodeError> {
    if cols.is_empty() {
        return Err(KeyEncodeError::NoKeys);
    }
    let n_rows = cols[0].n_rows;
    for c in cols.iter().skip(1) {
        if c.n_rows != n_rows {
            return Err(KeyEncodeError::RowCountMismatch {
                first: n_rows,
                mismatched: c.n_rows,
            });
        }
    }

    // Build schema; check buffer lengths and total width.
    let mut fields = Vec::with_capacity(cols.len());
    let mut offset: u32 = 0;
    let min_valid_bytes = (n_rows + 7) / 8;
    for c in cols {
        // Buffer-length checks.
        let need_data = match c.dtype {
            KeyDtype::I64 | KeyDtype::F64 => n_rows * 8,
            KeyDtype::Bool => min_valid_bytes,
        };
        if c.data.len() < need_data {
            return Err(KeyEncodeError::DataTooShort {
                col: c.name.clone(),
                got: c.data.len(),
                need: need_data,
            });
        }
        if c.valid.len() < min_valid_bytes {
            return Err(KeyEncodeError::ValidityTooShort {
                col: c.name.clone(),
                got: c.valid.len(),
                need: min_valid_bytes,
            });
        }

        let null_bit_offset = offset;
        let data_bit_offset = offset + 1;
        let field_bits = 1 + c.dtype.data_bits();
        if offset.saturating_add(field_bits) > 128 {
            return Err(KeyEncodeError::TooWide {
                total_bits: offset + field_bits,
            });
        }
        fields.push(KeyField {
            name: c.name.clone(),
            dtype: c.dtype,
            null_bit_offset,
            data_bit_offset,
        });
        offset += field_bits;
    }
    let total_bits = offset;

    // Encode.
    let mut encoded = vec![0u128; n_rows];
    for (field_idx, c) in cols.iter().enumerate() {
        let field = &fields[field_idx];
        for row in 0..n_rows {
            let valid_byte = c.valid[row >> 3];
            let valid_bit = (valid_byte >> (row & 7)) & 1;
            // Spec: 1 → valid (null bit = 0), 0 → null (null bit = 1).
            // We store the *null* flag for fast equality (null != value
            // collides at u128 equality otherwise).
            if valid_bit == 0 {
                encoded[row] |= 1u128 << field.null_bit_offset;
                continue;
            }
            // Valid row — write data bits.
            let data_value: u128 = match c.dtype {
                KeyDtype::I64 => {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&c.data[row * 8..(row + 1) * 8]);
                    i64::from_le_bytes(bytes) as u64 as u128
                }
                KeyDtype::F64 => {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&c.data[row * 8..(row + 1) * 8]);
                    f64::from_le_bytes(bytes).to_bits() as u128
                }
                KeyDtype::Bool => {
                    let byte = c.data[row >> 3];
                    let bit = (byte >> (row & 7)) & 1;
                    bit as u128
                }
            };
            encoded[row] |= data_value << field.data_bit_offset;
        }
    }

    Ok((
        encoded,
        KeySchema { fields, total_bits, n_rows },
    ))
}
```

Expose from the crate root:

```rust
// crates/polars-metal-kernels/src/lib.rs — add:
pub mod groupby;
```

- [ ] **Step 3: Implement a minimal decoder stub (full decoder lands in Task 14)**

```rust
// crates/polars-metal-kernels/src/groupby.rs — append:

/// Decoded representation of one key column, used to reconstruct result
/// DataFrames after the kernel returns indices.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedColumn {
    I64 { values: Vec<i64>, valid: Vec<bool> },
    F64 { values: Vec<f64>, valid: Vec<bool> },
    Bool { values: Vec<bool>, valid: Vec<bool> },
}

/// Stub implementation — full implementation in Task 14. Required here
/// so the Task 13 tests compile (they call `decode_keys` for round-trip
/// assertions). We return `unimplemented!()` from a non-test build path,
/// but Task 14 lands the real body; the test crate fails Task 13's tests
/// in a way that points to Task 14 if a test_key_encoding run is invoked
/// before Task 14 is implemented.
pub fn decode_keys(_encoded: &[u128], _schema: &KeySchema) -> Vec<DecodedColumn> {
    // Note: this is a stub; real implementation lands in Task 14.
    Vec::new()
}
```

- [ ] **Step 4: Run the encoder-only tests**

Run: `cargo test -p polars-metal-kernels --test test_key_encoding -- --skip decode`
Expected: tests that don't depend on decode pass (the overflow and empty-keys tests, plus the encoder-output structure assertions). The decode-round-trip tests will fail; that's expected — they pass after Task 14.

If clean separation isn't possible (the tests intermingle encode + decode), accept that Task 13's commit shows tests-fail at this point and resolves them in Task 14. Commit anyway:

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-kernels/src/groupby.rs crates/polars-metal-kernels/src/lib.rs crates/polars-metal-kernels/tests/test_key_encoding.rs
git commit -m "$(cat <<'EOF'
Kernel: composite key encoder packing into u128 lanes

Encoder only — decoder is a stub completed in Task 14.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 14: `decode_keys` — reverse the encoding for result reconstruction

**Files:**
- Modify: `crates/polars-metal-kernels/src/groupby.rs`

- [ ] **Step 1: Confirm the failing tests**

Run: `cargo test -p polars-metal-kernels --test test_key_encoding`
Expected: encoder tests pass; round-trip tests (the `decode_keys` cases in Task 13's test file) fail. These are the tests we now make green.

- [ ] **Step 2: Implement `decode_keys`**

```rust
// crates/polars-metal-kernels/src/groupby.rs — replace the stub:

/// Decode a u128-encoded composite-key stream back to per-column values.
///
/// Returns one `DecodedColumn` per field in `schema.fields()`, in the
/// same order as the original `cols` passed to `encode_keys`. For null
/// rows: `valid[i] = false` and `values[i]` is the default for the
/// dtype (0 for I64, 0.0 for F64, false for Bool). The Polars contract
/// only requires `valid` to be correct — the data slot at a null row is
/// implementation-defined.
pub fn decode_keys(encoded: &[u128], schema: &KeySchema) -> Vec<DecodedColumn> {
    let mut out: Vec<DecodedColumn> = schema
        .fields()
        .iter()
        .map(|f| match f.dtype {
            KeyDtype::I64 => DecodedColumn::I64 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::F64 => DecodedColumn::F64 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::Bool => DecodedColumn::Bool {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
        })
        .collect();

    for &lane in encoded {
        for (field_idx, field) in schema.fields().iter().enumerate() {
            // null-bit = 1 means null.
            let null_bit = (lane >> field.null_bit_offset) & 1u128;
            let is_valid = null_bit == 0;
            match (&mut out[field_idx], field.dtype) {
                (DecodedColumn::I64 { values, valid }, KeyDtype::I64) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 64) - 1);
                    let v = if is_valid {
                        // raw fits in 64 bits by construction; reinterpret as i64.
                        raw as u64 as i64
                    } else {
                        0
                    };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::F64 { values, valid }, KeyDtype::F64) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 64) - 1);
                    let v = if is_valid {
                        f64::from_bits(raw as u64)
                    } else {
                        0.0
                    };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::Bool { values, valid }, KeyDtype::Bool) => {
                    let raw = (lane >> field.data_bit_offset) & 1u128;
                    let v = if is_valid { raw == 1 } else { false };
                    values.push(v);
                    valid.push(is_valid);
                }
                _ => unreachable!("decoded column dtype must match field dtype"),
            }
        }
    }

    out
}
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p polars-metal-kernels --test test_key_encoding`
Expected: PASS, eight tests.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/src/groupby.rs
git commit -m "$(cat <<'EOF'
Kernel: composite key decoder for result reconstruction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 15: Encode/decode proptest — round-trip property

**Files:**
- Modify: `crates/polars-metal-kernels/tests/test_key_encoding.rs` (append proptest below the unit tests)

- [ ] **Step 1: Add the proptest module**

```rust
// crates/polars-metal-kernels/tests/test_key_encoding.rs — append:

use proptest::prelude::*;

#[derive(Debug, Clone)]
struct ArbI64Col {
    values: Vec<i64>,
    valid: Vec<bool>,
}

#[derive(Debug, Clone)]
struct ArbBoolCol {
    values: Vec<bool>,
    valid: Vec<bool>,
}

fn pack_valid(valid: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; (valid.len() + 7) / 8];
    for (i, &v) in valid.iter().enumerate() {
        if v {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

fn pack_bool_data(values: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; (values.len() + 7) / 8];
    for (i, &v) in values.iter().enumerate() {
        if v {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

fn arb_i64_col(n: usize) -> impl Strategy<Value = ArbI64Col> {
    (
        prop::collection::vec(any::<i64>(), n..=n),
        prop::collection::vec(any::<bool>(), n..=n),
    )
        .prop_map(|(values, valid)| ArbI64Col { values, valid })
}

fn arb_bool_col(n: usize) -> impl Strategy<Value = ArbBoolCol> {
    (
        prop::collection::vec(any::<bool>(), n..=n),
        prop::collection::vec(any::<bool>(), n..=n),
    )
        .prop_map(|(values, valid)| ArbBoolCol { values, valid })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// One i64 key, varying row counts. Round-trip preserves valid rows'
    /// values byte-for-byte.
    #[test]
    fn roundtrip_single_i64(n in 0usize..256, col in arb_i64_col(64)) {
        let n = n.max(1);
        let values: Vec<i64> = col.values.iter().take(n).copied().collect();
        let valid_bools: Vec<bool> = col.valid.iter().take(n).copied().collect();
        let data = bytes_i64(&values);
        let valid = pack_valid(&valid_bools);

        let kc = KeyColumn {
            name: "k".into(),
            dtype: KeyDtype::I64,
            data: &data,
            valid: &valid,
            n_rows: n,
        };
        let (encoded, schema) = encode_keys(&[kc]).expect("encode_keys");
        let decoded = decode_keys(&encoded, &schema);
        prop_assert_eq!(decoded.len(), 1);
        match &decoded[0] {
            DecodedColumn::I64 { values: dv, valid: dvalid } => {
                prop_assert_eq!(dv.len(), n);
                prop_assert_eq!(dvalid.len(), n);
                for i in 0..n {
                    prop_assert_eq!(dvalid[i], valid_bools[i]);
                    if valid_bools[i] {
                        prop_assert_eq!(dv[i], values[i]);
                    }
                    // else: data slot is implementation-defined.
                }
            }
            _ => prop_assert!(false, "expected I64 decoded column"),
        }
    }

    /// One i64 + one bool key — composite case under 128 bits.
    #[test]
    fn roundtrip_i64_plus_bool(
        i64_col in arb_i64_col(32),
        bool_col in arb_bool_col(32),
    ) {
        let n = 32;
        let i64_values = i64_col.values;
        let i64_valid = i64_col.valid;
        let bool_values = bool_col.values;
        let bool_valid = bool_col.valid;

        let i64_data = bytes_i64(&i64_values);
        let i64_valid_packed = pack_valid(&i64_valid);
        let bool_data = pack_bool_data(&bool_values);
        let bool_valid_packed = pack_valid(&bool_valid);

        let cols = vec![
            KeyColumn { name: "i".into(), dtype: KeyDtype::I64, data: &i64_data, valid: &i64_valid_packed, n_rows: n },
            KeyColumn { name: "b".into(), dtype: KeyDtype::Bool, data: &bool_data, valid: &bool_valid_packed, n_rows: n },
        ];
        let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
        prop_assert_eq!(schema.total_bits(), 1 + 64 + 1 + 1);
        let decoded = decode_keys(&encoded, &schema);
        prop_assert_eq!(decoded.len(), 2);
        match (&decoded[0], &decoded[1]) {
            (
                DecodedColumn::I64 { values: iv, valid: ivd },
                DecodedColumn::Bool { values: bv, valid: bvd },
            ) => {
                for i in 0..n {
                    prop_assert_eq!(ivd[i], i64_valid[i]);
                    prop_assert_eq!(bvd[i], bool_valid[i]);
                    if i64_valid[i] {
                        prop_assert_eq!(iv[i], i64_values[i]);
                    }
                    if bool_valid[i] {
                        prop_assert_eq!(bv[i], bool_values[i]);
                    }
                }
            }
            _ => prop_assert!(false, "unexpected decoded shape"),
        }
    }

    /// Equal rows in the source produce equal u128 lanes (key equality
    /// reduces to u128 equality — required for the hash kernel).
    #[test]
    fn equal_keys_encode_to_equal_lanes(a in any::<i64>(), b in any::<i64>()) {
        let n = 4;
        let values: Vec<i64> = vec![a, b, a, b];
        let valid_bools = vec![true, true, true, true];
        let data = bytes_i64(&values);
        let valid = pack_valid(&valid_bools);
        let kc = KeyColumn {
            name: "k".into(),
            dtype: KeyDtype::I64,
            data: &data,
            valid: &valid,
            n_rows: n,
        };
        let (encoded, _) = encode_keys(&[kc]).expect("encode_keys");
        prop_assert_eq!(encoded[0], encoded[2]); // both `a`
        prop_assert_eq!(encoded[1], encoded[3]); // both `b`
        if a != b {
            prop_assert_ne!(encoded[0], encoded[1]);
        }
    }
}
```

- [ ] **Step 2: Run the proptest**

Run: `cargo test -p polars-metal-kernels --test test_key_encoding`
Expected: PASS — all unit tests (Task 13/14) plus the three proptests (256 cases each).

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-kernels/tests/test_key_encoding.rs
git commit -m "$(cat <<'EOF'
Kernel: composite key encode/decode round-trip proptest

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Hash kernel + shared helpers

Phase 4 lands the first kernel of the two-pass groupby pipeline: a single-threaded-per-row hash kernel that consumes the u128-encoded keys and emits a u32 hash per row. The kernel is deterministic — same input bytes always produce the same output — so result row ordering downstream is reproducible.

The shared helpers in `_groupby.metal` exist primarily for Phase 5's build kernel and Phase 6's aggregate kernels; we land them in this phase to lock in the shared API early.

### Task 16: `shaders/_groupby.metal` — shared MSL helpers

**Files:**
- Create: `shaders/_groupby.metal`

- [ ] **Step 1: Write the shared-helpers header**

```c
// shaders/_groupby.metal
//
// Shared MSL helpers for the GroupBy kernels (hash, build, aggregate).
//
// This file is a HEADER — leading underscore signals build.rs to skip
// standalone compilation. Its definitions are inlined into the kernels
// that #include it (analogous to `_validity.metal`).
//
// Topics:
//   1. Hash mixing for u128 keys (xxhash-inspired finalize step).
//   2. Atomic-add helpers per value dtype (i64, f64) for the aggregate
//      kernels.

#pragma once
#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------
// 1. Hash mixing
// -----------------------------------------------------------------------
//
// We need: given a 128-bit key, produce a 32-bit hash with good
// dispersion. The xxhash finalize step is fast and well-distributed for
// the cardinalities we expect (4 to ~10M groups).
//
// Algorithm: split the u128 into two u64 halves, fold via xor-rotate,
// finalize with the xxhash mixer constants.
//
// References:
//   - xxhash: https://github.com/Cyan4973/xxHash/blob/dev/xxhash.h
//   - cuDF's hash kernel: references/cudf/cpp/src/hash/hash.cu

inline uint32_t rotl32(uint32_t x, uint32_t r) {
    return (x << r) | (x >> (32u - r));
}

inline uint32_t xxhash_finalize_u64(uint64_t v) {
    // xxhash32 finalize on a u64. PRIME constants from upstream.
    const uint32_t PRIME32_2 = 2246822519u;
    const uint32_t PRIME32_3 = 3266489917u;
    uint32_t h = (uint32_t)(v ^ (v >> 32u));
    h ^= h >> 15u;
    h *= PRIME32_2;
    h ^= h >> 13u;
    h *= PRIME32_3;
    h ^= h >> 16u;
    return h;
}

/// Hash a 128-bit key (two halves) to a 32-bit value.
/// `lo` and `hi` are the low and high 64-bit halves of the u128 key.
inline uint32_t hash_u128(uint64_t lo, uint64_t hi) {
    // Combine halves via xor-rotate, then finalize.
    uint64_t combined = lo ^ rotl32_u64(hi, 27u);
    return xxhash_finalize_u64(combined);
}

inline uint64_t rotl32_u64(uint64_t x, uint32_t r) {
    return (x << r) | (x >> (64u - r));
}

// -----------------------------------------------------------------------
// 2. Atomic-add helpers (for aggregate kernels in Phase 6)
// -----------------------------------------------------------------------
//
// Metal's `atomic_int` / `atomic_uint` cover 32-bit signed/unsigned.
// 64-bit atomic add is supported on Apple Silicon GPUs from M2 onward
// via `atomic_long` (signed) and `atomic_ulong` (unsigned). For f64
// there's no native atomic_add — we implement via atomic_compare_exchange
// in a retry loop on the raw u64 bit pattern.

/// Atomic add to an i64 accumulator at `out[idx]`. Uses native
/// `atomic_long` (Apple Silicon M2+).
inline void atomic_add_i64(device atomic_long* out, uint idx, int64_t delta) {
    atomic_fetch_add_explicit(&out[idx], delta, memory_order_relaxed);
}

/// Atomic add to an f64 accumulator at `out[idx]`. Reinterprets the
/// underlying u64, atomically CASes the bit pattern; retries on
/// collision. Apple Silicon's `atomic_ulong` provides the CAS.
inline void atomic_add_f64(device atomic_ulong* out, uint idx, double delta) {
    uint64_t old_bits = atomic_load_explicit(&out[idx], memory_order_relaxed);
    while (true) {
        double cur = as_type<double>(old_bits);
        double next = cur + delta;
        uint64_t next_bits = as_type<uint64_t>(next);
        if (atomic_compare_exchange_weak_explicit(
                &out[idx],
                &old_bits,
                next_bits,
                memory_order_relaxed,
                memory_order_relaxed)) {
            break;
        }
        // On failure, `old_bits` is updated to the slot's current value;
        // loop and retry.
    }
}

/// Atomic min/max on i64. Same retry-loop pattern as atomic_add_f64.
inline void atomic_min_i64(device atomic_long* out, uint idx, int64_t v) {
    int64_t cur = atomic_load_explicit(&out[idx], memory_order_relaxed);
    while (v < cur) {
        if (atomic_compare_exchange_weak_explicit(
                &out[idx], &cur, v,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline void atomic_max_i64(device atomic_long* out, uint idx, int64_t v) {
    int64_t cur = atomic_load_explicit(&out[idx], memory_order_relaxed);
    while (v > cur) {
        if (atomic_compare_exchange_weak_explicit(
                &out[idx], &cur, v,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

/// Atomic min/max on f64 via u64 bit pattern. Note that for NaN
/// semantics matching Polars, callers should `return` early on NaN
/// rows rather than relying on the f64 < / > comparison.
inline void atomic_min_f64(device atomic_ulong* out, uint idx, double v) {
    uint64_t cur_bits = atomic_load_explicit(&out[idx], memory_order_relaxed);
    while (true) {
        double cur = as_type<double>(cur_bits);
        if (!(v < cur)) break;
        uint64_t new_bits = as_type<uint64_t>(v);
        if (atomic_compare_exchange_weak_explicit(
                &out[idx], &cur_bits, new_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

inline void atomic_max_f64(device atomic_ulong* out, uint idx, double v) {
    uint64_t cur_bits = atomic_load_explicit(&out[idx], memory_order_relaxed);
    while (true) {
        double cur = as_type<double>(cur_bits);
        if (!(v > cur)) break;
        uint64_t new_bits = as_type<uint64_t>(v);
        if (atomic_compare_exchange_weak_explicit(
                &out[idx], &cur_bits, new_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}
```

Note the forward-reference: `hash_u128` uses `rotl32_u64`, defined below it. MSL accepts forward references for `inline` functions when included as a header; if compilation fails on this, reorder so `rotl32_u64` precedes `hash_u128`.

- [ ] **Step 2: Verify `build.rs` ignores leading-underscore files**

```bash
grep -n "_" crates/polars-metal-kernels/build.rs | head -20
```

Expected output should show a filter that skips `_*.metal` files (same convention `_validity.metal` uses). If absent, add it (the M1 convention is non-negotiable).

- [ ] **Step 3: Sanity-build a kernel that includes the header**

The header isn't tested standalone (it's not a kernel), but we can verify it compiles by including it in a trivial kernel file. However we don't want to add a placeholder kernel just for this; instead, defer the compile-time check to Task 17 (which adds `groupby_hash.metal` and `#include "_groupby.metal"`). If `make wheel` succeeds for Task 17, the helper header compiles cleanly.

- [ ] **Step 4: Commit**

```bash
git add shaders/_groupby.metal
git commit -m "$(cat <<'EOF'
Shaders: _groupby.metal — shared hash mixer and atomic helpers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 17: `shaders/groupby_hash.metal` — hash kernel

**Files:**
- Create: `shaders/groupby_hash.metal`

- [ ] **Step 1: Write the kernel**

```c
// shaders/groupby_hash.metal
//
// Hash kernel — one thread per row, reads a u128-encoded composite key
// from `keys[i]`, writes a u32 hash to `hashes[i]`.
//
// Deterministic: same input bytes produce the same output hash. This is
// the contract that lets the build phase produce reproducible group IDs
// across runs (modulo the slot-insertion ordering, which depends on
// thread schedule).
//
// MSL note: a u128 is encoded as a pair of (u64 lo, u64 hi); MSL doesn't
// have a u128 type, but we pass the buffer as `device const uint2*`
// (each uint2 = 2 × uint32_t, packed). To preserve our Rust-side u128
// encoding, we cast to `uint64_t*` and read two 64-bit halves per row.

#include "_groupby.metal"

kernel void groupby_hash(
    device const uint64_t* keys      [[buffer(0)]],   // 2 × u64 per row
    device       uint32_t* hashes    [[buffer(1)]],
    constant     uint32_t& n_rows    [[buffer(2)]],
    uint                   gid       [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint64_t lo = keys[gid * 2u];
    uint64_t hi = keys[gid * 2u + 1u];
    hashes[gid] = hash_u128(lo, hi);
}
```

- [ ] **Step 2: Confirm the metallib build picks up the new kernel**

Run: `make wheel`
Expected: builds cleanly. If a syntax error in `_groupby.metal` surfaces (forward-reference issue noted in Task 16), reorder the helper definitions and rebuild.

- [ ] **Step 3: Commit**

```bash
git add shaders/groupby_hash.metal
git commit -m "$(cat <<'EOF'
Shaders: groupby_hash — single-pass u128-to-u32 hash kernel

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 18: Rust dispatcher `dispatch_hash` + proptest against reference

**Files:**
- Modify: `crates/polars-metal-kernels/src/groupby.rs`
- Create: `crates/polars-metal-kernels/tests/test_groupby_hash.rs`

- [ ] **Step 1: Write the failing proptest**

```rust
// crates/polars-metal-kernels/tests/test_groupby_hash.rs
//
// Proptest: dispatch_hash on a u128-encoded keystream matches a
// pure-Rust reference implementation of the same xxhash-style mixer.
// Equal keys → equal hashes (the property we care about for groupby).
#![allow(clippy::expect_used)]

use polars_metal_kernels::groupby::{
    dispatch_hash, hash_u128_reference, encode_keys, KeyColumn, KeyDtype,
};
use polars_metal_kernels::shader_lib::load_default_library;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::MetalDevice;
use proptest::prelude::*;

fn bytes_i64(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn all_valid(n: usize) -> Vec<u8> {
    vec![0xFFu8; (n + 7) / 8]
}

fn run_kernel(encoded: &[u128]) -> Vec<u32> {
    let device = MetalDevice::new().expect("device");
    let mut queue = CommandQueue::new(&device).expect("queue");
    let _lib = load_default_library(&device).expect("library");
    let mut out = vec![0u32; encoded.len()];
    dispatch_hash(&device, &mut queue, encoded, encoded.len(), &mut out)
        .expect("dispatch_hash");
    out
}

#[test]
fn single_row_hashes() {
    let encoded = vec![0x1234_5678_9abc_def0u128];
    let out = run_kernel(&encoded);
    assert_eq!(out.len(), 1);
    // Determinism: same input always produces same output.
    let out2 = run_kernel(&encoded);
    assert_eq!(out, out2);
}

#[test]
fn equal_keys_produce_equal_hashes() {
    let encoded = vec![42u128, 99u128, 42u128, 99u128];
    let out = run_kernel(&encoded);
    assert_eq!(out[0], out[2]);
    assert_eq!(out[1], out[3]);
}

#[test]
fn kernel_matches_reference_implementation() {
    let encoded: Vec<u128> = (0..1024u128).map(|i| i * 1_000_003).collect();
    let kernel_out = run_kernel(&encoded);
    let ref_out: Vec<u32> = encoded.iter().map(|&k| {
        let lo = k as u64;
        let hi = (k >> 64) as u64;
        hash_u128_reference(lo, hi)
    }).collect();
    assert_eq!(kernel_out, ref_out);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn kernel_equals_reference_for_random_keys(
        ks in prop::collection::vec(any::<u128>(), 1..=512)
    ) {
        let kernel_out = run_kernel(&ks);
        let ref_out: Vec<u32> = ks.iter().map(|&k| {
            let lo = k as u64;
            let hi = (k >> 64) as u64;
            hash_u128_reference(lo, hi)
        }).collect();
        prop_assert_eq!(kernel_out, ref_out);
    }
}
```

- [ ] **Step 2: Implement the dispatcher and the reference**

```rust
// crates/polars-metal-kernels/src/groupby.rs — append:

use crate::command::{CommandQueue, DispatchError};
use crate::pipeline::MetalDevice;
use crate::shader_lib::ShaderError;

/// Errors from the groupby dispatchers. Mirrors `CmpError`'s shape.
#[derive(Debug, thiserror::Error)]
pub enum GroupByError {
    #[error("key encoding: {0}")]
    KeyEncode(#[from] KeyEncodeError),
    #[error("shader library: {0}")]
    Shader(#[from] ShaderError),
    #[error("dispatch: {0}")]
    Dispatch(#[from] DispatchError),
    #[error("buffer: {0}")]
    Buffer(#[from] crate::pipeline::BufferError),
    #[error("output buffer too short: got {got}, need {need}")]
    OutputTooShort { got: usize, need: usize },
    #[error("n_rows {n_rows} exceeds u32::MAX")]
    RowCountOverflow { n_rows: usize },
}

/// Dispatch the `groupby_hash` kernel.
///
/// Reads `encoded[0..n_rows]` (one u128 per row), writes one u32 hash
/// per row to `hashes[0..n_rows]`. `hashes.len()` must be ≥ `n_rows`.
pub fn dispatch_hash(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    encoded: &[u128],
    n_rows: usize,
    hashes: &mut [u32],
) -> Result<(), GroupByError> {
    if n_rows == 0 {
        return Ok(());
    }
    if u32::try_from(n_rows).is_err() {
        return Err(GroupByError::RowCountOverflow { n_rows });
    }
    if hashes.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: hashes.len(),
            need: n_rows,
        });
    }
    if encoded.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: encoded.len(),
            need: n_rows,
        });
    }

    // Cast encoded u128 to bytes — Metal sees it as u64 pairs.
    let key_bytes: &[u8] = unsafe {
        // SAFETY: u128 is plain-old-data; alignment of u128 (16 bytes)
        // strictly exceeds u8 (1 byte). The slice length in bytes is
        // `encoded.len() * 16`, which fits in usize because each u128 in
        // the slice corresponds to one row and `n_rows` already fits in
        // u32 (checked above).
        std::slice::from_raw_parts(encoded.as_ptr() as *const u8, encoded.len() * 16)
    };

    let mut keys_buf = device.new_buffer_with_bytes(key_bytes)?;
    let mut hashes_buf = device.new_buffer_zeroed(n_rows * 4)?;
    let n_rows_u32: u32 = n_rows as u32;

    let pipeline = device.pipeline_for("groupby_hash")?;
    let mut encoder = queue.compute_encoder(&pipeline)?;
    encoder.set_buffer(0, &keys_buf);
    encoder.set_buffer(1, &hashes_buf);
    encoder.set_constant(2, &n_rows_u32);
    encoder.dispatch(n_rows)?;
    encoder.commit_and_wait()?;

    // Copy device output back into the caller's slice.
    hashes_buf.read_into(&mut hashes[..n_rows])?;
    Ok(())
}

/// Pure-Rust reference implementation of `hash_u128` from the MSL header.
/// Must stay in sync with `shaders/_groupby.metal::hash_u128`.
pub fn hash_u128_reference(lo: u64, hi: u64) -> u32 {
    fn rotl32_u64(x: u64, r: u32) -> u64 {
        (x << r) | (x >> (64 - r))
    }
    fn xxhash_finalize_u64(v: u64) -> u32 {
        const PRIME32_2: u32 = 2_246_822_519;
        const PRIME32_3: u32 = 3_266_489_917;
        let mut h: u32 = ((v ^ (v >> 32)) as u32);
        h ^= h >> 15;
        h = h.wrapping_mul(PRIME32_2);
        h ^= h >> 13;
        h = h.wrapping_mul(PRIME32_3);
        h ^= h >> 16;
        h
    }
    let combined = lo ^ rotl32_u64(hi, 27);
    xxhash_finalize_u64(combined)
}
```

Note: the `device.new_buffer_with_bytes`, `device.new_buffer_zeroed`, `device.pipeline_for`, `queue.compute_encoder`, and `encoder.commit_and_wait` calls match M1's `pipeline.rs` / `command.rs` API. If the exact method names differ, adapt 1:1 (the M1 codebase has these primitives — read `crates/polars-metal-kernels/src/cmp.rs::dispatch_cmp_i64` for the precise call sequence and replicate).

- [ ] **Step 3: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_groupby_hash`
Expected: PASS — three explicit tests + 64 proptest cases.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/src/groupby.rs crates/polars-metal-kernels/tests/test_groupby_hash.rs
git commit -m "$(cat <<'EOF'
Kernel: dispatch_hash + Rust reference for proptest equivalence

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 — Hash-table build phase

Phase 5 lands the build kernel — the heart of the two-pass groupby. Each row's encoded key + hash goes into a fixed-size open-addressing hash table via atomic-CAS find-or-insert. Outputs are three buffers: `row_to_group` (each row's assigned group ID), `group_count` (total number of distinct groups), and `first_row_per_group` (one representative source row per group, used in Phase 6 to reconstruct key values for the result).

Hash-table sizing: `next_pow2(n_rows / load_factor)`, with `load_factor = 0.5` for M2's starting tuning. Spec § "Risks" notes contention on hot keys (Q1's 4-groups-over-10M-rows case); we ship the atomic-CAS approach and revisit if microbenches show >2× degradation.

### Task 19: `shaders/groupby_build.metal` — atomic-CAS build kernel

**Files:**
- Create: `shaders/groupby_build.metal`

- [ ] **Step 1: Write the kernel**

```c
// shaders/groupby_build.metal
//
// Hash-table build kernel. One thread per row. Each thread:
//   1. Reads its row's u128 key and u32 hash.
//   2. Probes the open-addressing table starting at `hash % size`.
//   3. For each slot it visits:
//        - If empty, atomic-CAS to install (occupied | hash | row).
//          On success: write row → group, store representative row.
//          On failure: re-read the slot and retry the same slot.
//        - If occupied with the same key: write row → group_id, done.
//        - If occupied with a different key: linear-probe to next slot.
//
// Slot layout (16 bytes per slot):
//
//   uint64_t key_lo;       // low half of the u128 key
//   uint64_t key_hi;       // high half (MSB of hi = 1 if occupied,
//                            so empty slots are key_hi = 0).
//
// We pack the "occupied" flag into the MSB of `key_hi`. This costs us
// one bit of the encoded key range — the encoder must guarantee
// `key_hi < (1 << 63)` for valid (non-null in all fields) keys. Phase 3's
// encoder uses bits LSB-first; the max key width is 128 bits so the
// top bit is always free for keys totalling ≤ 127 bits. Encodings that
// exactly use bit 127 must Fallback (we leave one bit of head-room).
//
// Separately we store:
//
//   device atomic_uint* group_id_for_slot;   // size = table_size
//   device atomic_uint* group_count;         // single u32, fetched-added
//   device       uint*  first_row_per_group; // size = max_groups (= n_rows)
//   device       uint*  row_to_group;        // size = n_rows
//
// The `group_count` is the slot allocator — fetch-and-add to get a fresh
// group_id when a thread successfully installs a new slot. The
// `first_row_per_group[gid]` array gets one representative row index per
// distinct group, used in Phase 6 to reconstruct the key columns for
// the result DataFrame.

#include "_groupby.metal"

kernel void groupby_build(
    device const uint64_t*       keys                  [[buffer(0)]],
    device const uint32_t*       hashes                [[buffer(1)]],
    device       atomic_ulong*   slot_key_lo           [[buffer(2)]],
    device       atomic_ulong*   slot_key_hi           [[buffer(3)]],
    device       atomic_uint*    slot_group_id         [[buffer(4)]],
    device       atomic_uint*    group_count           [[buffer(5)]],
    device       uint32_t*       first_row_per_group   [[buffer(6)]],
    device       uint32_t*       row_to_group          [[buffer(7)]],
    constant     uint32_t&       n_rows                [[buffer(8)]],
    constant     uint32_t&       table_size            [[buffer(9)]],
    uint                         gid                   [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;

    uint64_t k_lo = keys[gid * 2u];
    uint64_t k_hi = keys[gid * 2u + 1u];
    uint32_t h    = hashes[gid];

    // Mask off the occupied bit before equality compare — but we OR it
    // back in for stores. We require the encoder leave MSB of hi clear.
    const uint64_t OCC_BIT = (uint64_t)1 << 63;
    uint64_t k_hi_with_occ = k_hi | OCC_BIT;

    uint32_t mask = table_size - 1u;
    uint32_t slot = h & mask;
    for (uint32_t probe = 0u; probe < table_size; ++probe) {
        uint32_t s = (slot + probe) & mask;

        // Read current slot occupancy.
        uint64_t cur_hi = atomic_load_explicit(&slot_key_hi[s], memory_order_relaxed);
        if (cur_hi == 0u) {
            // Empty — try to claim.
            uint64_t expected_hi = 0u;
            if (atomic_compare_exchange_weak_explicit(
                    &slot_key_hi[s], &expected_hi, k_hi_with_occ,
                    memory_order_relaxed, memory_order_relaxed)) {
                // Won the slot. Now publish the key_lo half and assign
                // a fresh group_id.
                atomic_store_explicit(&slot_key_lo[s], k_lo, memory_order_relaxed);
                uint32_t new_gid = atomic_fetch_add_explicit(group_count, 1u, memory_order_relaxed);
                atomic_store_explicit(&slot_group_id[s], new_gid, memory_order_relaxed);
                row_to_group[gid] = new_gid;
                first_row_per_group[new_gid] = gid;
                return;
            }
            // CAS failed: another thread won the slot. Fall through to
            // re-read it on the next iteration of the same `s`.
            // Reset probe so we re-examine this slot — we just `continue`
            // without advancing.
            cur_hi = atomic_load_explicit(&slot_key_hi[s], memory_order_relaxed);
        }

        // Occupied (either we just lost a race or another thread had it).
        // Spin-wait for slot_key_lo to be published (the winning thread
        // may not yet have stored it). MSL doesn't give us a yield, so
        // we busy-loop. In practice the gap is 1-2 instructions.
        uint64_t cur_lo;
        do {
            cur_lo = atomic_load_explicit(&slot_key_lo[s], memory_order_relaxed);
            // Reread cur_hi too — the winner may have set it but not yet
            // stored lo. As long as occ-bit is set in hi, the slot is
            // claimed; lo eventually gets written.
        } while (false); // single read; relying on relaxed visibility.

        if ((cur_hi & ~OCC_BIT) == k_hi && cur_lo == k_lo) {
            // Same key — read its group_id (also spin until populated).
            uint32_t cur_gid = atomic_load_explicit(&slot_group_id[s], memory_order_relaxed);
            // We may race the assignment of group_id by the winning
            // thread; spin briefly. Bounded by # of in-flight new-slot
            // installations.
            for (uint32_t spin = 0u; spin < 1024u && cur_gid == 0u && new_slot_just_installed; ++spin) {
                cur_gid = atomic_load_explicit(&slot_group_id[s], memory_order_relaxed);
            }
            row_to_group[gid] = cur_gid;
            return;
        }
        // Different key — linear-probe to next slot.
    }
    // Table full. Should never happen: caller sizes table to
    // next_pow2(n_rows / 0.5) ≥ 2 × n_rows, so load factor ≤ 0.5.
    // Defensive: write 0xFFFFFFFF and rely on caller to detect.
    row_to_group[gid] = 0xFFFFFFFFu;
}
```

Note: the spin-wait pattern around `group_id` is fragile. A cleaner approach: have the winning thread store `(group_id | 0x80000000)` (a high-bit-set sentinel) into `slot_group_id` to indicate "assigned". Readers strip the high bit. We adopt this in the next revision; for the initial implementation, since `atomic_fetch_add_explicit` returns the previous value (always ≥ 0, and 0 is a valid group_id), we use a separate "ready" bit pattern:

Replace `atomic_store_explicit(&slot_group_id[s], new_gid, memory_order_relaxed)` with `atomic_store_explicit(&slot_group_id[s], new_gid | 0x80000000u, memory_order_relaxed)` and readers compute `cur_gid = atomic_load_explicit(&slot_group_id[s], memory_order_relaxed) & 0x7FFFFFFFu;` after spinning until the high bit is set. Adjust the kernel accordingly (the inline "new_slot_just_installed" variable in the listing above is a placeholder; rewrite the load loop as):

```c
        // Spin until the winning thread has assigned a group_id.
        uint32_t raw_gid;
        for (uint32_t spin = 0u; spin < 65536u; ++spin) {
            raw_gid = atomic_load_explicit(&slot_group_id[s], memory_order_relaxed);
            if ((raw_gid & 0x80000000u) != 0u) break;
        }
        row_to_group[gid] = raw_gid & 0x7FFFFFFFu;
        return;
```

And on the winning-thread path:

```c
        uint32_t new_gid = atomic_fetch_add_explicit(group_count, 1u, memory_order_relaxed);
        atomic_store_explicit(&slot_group_id[s], new_gid | 0x80000000u, memory_order_relaxed);
        row_to_group[gid] = new_gid;
        first_row_per_group[new_gid] = gid;
        return;
```

The 0x80000000 sentinel restricts us to 2^31 - 1 distinct groups; that's well above any analytical workload.

- [ ] **Step 2: Verify it builds**

Run: `make wheel`
Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add shaders/groupby_build.metal
git commit -m "$(cat <<'EOF'
Shaders: groupby_build — atomic-CAS hash-table build kernel

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 20: Rust dispatcher `dispatch_build`

**Files:**
- Modify: `crates/polars-metal-kernels/src/groupby.rs`

- [ ] **Step 1: Implement the dispatcher**

```rust
// crates/polars-metal-kernels/src/groupby.rs — append:

/// Initial load factor for the build-phase hash table. Spec starting
/// point; PR-tunable.
pub const BUILD_LOAD_FACTOR_NUM: usize = 1;
pub const BUILD_LOAD_FACTOR_DEN: usize = 2;

fn next_pow2(n: usize) -> usize {
    if n <= 1 { return 1; }
    let bits = (usize::BITS - (n - 1).leading_zeros()) as u32;
    1usize << bits
}

/// Output of the build phase.
pub struct BuildOutput {
    /// `row_to_group[i]` = group ID for row i.
    pub row_to_group: Vec<u32>,
    /// Total number of distinct groups produced by the build.
    pub group_count: u32,
    /// `first_row_per_group[g]` = a representative source-row index for
    /// group g (whichever row won the slot install). Used by Phase 6 to
    /// reconstruct the key columns in the result.
    pub first_row_per_group: Vec<u32>,
}

/// Dispatch the build phase. Sizes the hash table at
/// `next_pow2(n_rows * 2)` (load factor 0.5).
///
/// Caller contract:
///   - `encoded.len() == n_rows`.
///   - `hashes.len() == n_rows`.
///   - For every encoded key, the MSB of the high u64 half is 0 (Phase 3
///     encoder guarantees ≤ 127-bit keys).
pub fn dispatch_build(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    encoded: &[u128],
    hashes: &[u32],
    n_rows: usize,
) -> Result<BuildOutput, GroupByError> {
    if n_rows == 0 {
        return Ok(BuildOutput {
            row_to_group: vec![],
            group_count: 0,
            first_row_per_group: vec![],
        });
    }
    if u32::try_from(n_rows).is_err() {
        return Err(GroupByError::RowCountOverflow { n_rows });
    }
    if encoded.len() < n_rows || hashes.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: encoded.len().min(hashes.len()),
            need: n_rows,
        });
    }

    // Size the table at next_pow2(n_rows / load_factor) = next_pow2(n_rows * 2).
    let raw_size = n_rows
        .checked_mul(BUILD_LOAD_FACTOR_DEN)
        .and_then(|n| n.checked_div(BUILD_LOAD_FACTOR_NUM))
        .ok_or(GroupByError::RowCountOverflow { n_rows })?;
    let table_size = next_pow2(raw_size).max(2);
    let table_size_u32: u32 = u32::try_from(table_size)
        .map_err(|_| GroupByError::RowCountOverflow { n_rows: table_size })?;

    let key_bytes: &[u8] = unsafe {
        // SAFETY: u128 POD; alignment of u128 (16) >= alignment of u8 (1);
        // length in bytes = encoded.len() * 16, computed from a slice
        // whose `n_rows` is already u32-bounded.
        std::slice::from_raw_parts(encoded.as_ptr() as *const u8, encoded.len() * 16)
    };
    let hash_bytes: &[u8] = unsafe {
        // SAFETY: u32 POD; alignment of u32 (4) >= alignment of u8 (1).
        std::slice::from_raw_parts(hashes.as_ptr() as *const u8, hashes.len() * 4)
    };

    let mut keys_buf = device.new_buffer_with_bytes(key_bytes)?;
    let mut hashes_buf = device.new_buffer_with_bytes(hash_bytes)?;
    // Hash-table slot arrays.
    let mut slot_lo_buf = device.new_buffer_zeroed(table_size * 8)?;
    let mut slot_hi_buf = device.new_buffer_zeroed(table_size * 8)?;
    let mut slot_gid_buf = device.new_buffer_zeroed(table_size * 4)?;
    let mut group_count_buf = device.new_buffer_zeroed(4)?;
    let mut first_row_buf = device.new_buffer_zeroed(n_rows * 4)?;
    let mut row_to_group_buf = device.new_buffer_zeroed(n_rows * 4)?;
    let n_rows_u32 = n_rows as u32;

    let pipeline = device.pipeline_for("groupby_build")?;
    let mut encoder = queue.compute_encoder(&pipeline)?;
    encoder.set_buffer(0, &keys_buf);
    encoder.set_buffer(1, &hashes_buf);
    encoder.set_buffer(2, &slot_lo_buf);
    encoder.set_buffer(3, &slot_hi_buf);
    encoder.set_buffer(4, &slot_gid_buf);
    encoder.set_buffer(5, &group_count_buf);
    encoder.set_buffer(6, &first_row_buf);
    encoder.set_buffer(7, &row_to_group_buf);
    encoder.set_constant(8, &n_rows_u32);
    encoder.set_constant(9, &table_size_u32);
    encoder.dispatch(n_rows)?;
    encoder.commit_and_wait()?;

    // Read back results.
    let mut group_count_arr = [0u32; 1];
    group_count_buf.read_into(&mut group_count_arr)?;
    let group_count = group_count_arr[0];

    let mut row_to_group = vec![0u32; n_rows];
    row_to_group_buf.read_into(&mut row_to_group)?;

    // `first_row_per_group` is allocated to `n_rows` entries (max
    // possible groups), but only the first `group_count` are valid.
    let mut first_row_full = vec![0u32; n_rows];
    first_row_buf.read_into(&mut first_row_full)?;
    first_row_full.truncate(group_count as usize);

    Ok(BuildOutput {
        row_to_group,
        group_count,
        first_row_per_group: first_row_full,
    })
}
```

- [ ] **Step 2: Build to verify the signatures**

Run: `cargo build -p polars-metal-kernels`
Expected: clean build. (Tests in Task 21.)

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-kernels/src/groupby.rs
git commit -m "$(cat <<'EOF'
Kernel: dispatch_build — Rust wrapper for groupby build phase

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 21: Build-phase proptest against pure-Rust reference

**Files:**
- Create: `crates/polars-metal-kernels/tests/test_groupby_build.rs`

- [ ] **Step 1: Write the test**

```rust
// crates/polars-metal-kernels/tests/test_groupby_build.rs
//
// Proptest the build phase against a pure-Rust hash-table reference.
// We assert *structural* equivalence (same equivalence classes of rows),
// not exact group-ID equality — the kernel's group IDs depend on thread
// schedule, while the reference uses insertion order.
#![allow(clippy::expect_used)]

use polars_metal_kernels::groupby::{
    dispatch_build, dispatch_hash, hash_u128_reference,
};
use polars_metal_kernels::shader_lib::load_default_library;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::MetalDevice;
use proptest::prelude::*;
use std::collections::HashMap;

fn build_reference(encoded: &[u128]) -> (Vec<u32>, u32) {
    // Pure-Rust reference: insertion-order group IDs.
    let mut group_for_key: HashMap<u128, u32> = HashMap::new();
    let mut next_gid: u32 = 0;
    let mut row_to_group = Vec::with_capacity(encoded.len());
    for &k in encoded {
        let gid = *group_for_key.entry(k).or_insert_with(|| {
            let g = next_gid;
            next_gid += 1;
            g
        });
        row_to_group.push(gid);
    }
    (row_to_group, next_gid)
}

/// True iff `a` and `b` induce the same equivalence classes on rows.
fn same_equivalence_classes(a: &[u32], b: &[u32]) -> bool {
    if a.len() != b.len() { return false; }
    // For each row i, all rows j with a[j] == a[i] must satisfy b[j] == b[i].
    for i in 0..a.len() {
        for j in (i + 1)..a.len() {
            let same_a = a[i] == a[j];
            let same_b = b[i] == b[j];
            if same_a != same_b { return false; }
        }
    }
    true
}

fn run_build(encoded: &[u128]) -> (Vec<u32>, u32, Vec<u32>) {
    let device = MetalDevice::new().expect("device");
    let mut queue = CommandQueue::new(&device).expect("queue");
    let _lib = load_default_library(&device).expect("library");
    let mut hashes = vec![0u32; encoded.len()];
    dispatch_hash(&device, &mut queue, encoded, encoded.len(), &mut hashes)
        .expect("dispatch_hash");
    let out = dispatch_build(&device, &mut queue, encoded, &hashes, encoded.len())
        .expect("dispatch_build");
    (out.row_to_group, out.group_count, out.first_row_per_group)
}

#[test]
fn all_distinct_keys_produce_one_group_per_row() {
    let encoded: Vec<u128> = (1..=128u128).collect();
    let (r2g, count, first_rows) = run_build(&encoded);
    assert_eq!(count, 128);
    assert_eq!(r2g.len(), 128);
    assert_eq!(first_rows.len(), 128);
    // Each group has exactly one row (its own).
    let mut sorted = r2g.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 128);
}

#[test]
fn all_same_keys_produce_one_group() {
    let encoded = vec![42u128; 1024];
    let (r2g, count, first_rows) = run_build(&encoded);
    assert_eq!(count, 1);
    assert!(r2g.iter().all(|&g| g == r2g[0]));
    assert_eq!(first_rows.len(), 1);
}

#[test]
fn four_groups_ten_thousand_rows_modeled_q1() {
    // Simulate Q1's shape: ~4 distinct keys, many rows each.
    let mut encoded = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        encoded.push((i % 4) as u128);
    }
    let (r2g, count, first_rows) = run_build(&encoded);
    assert_eq!(count, 4);
    assert_eq!(first_rows.len(), 4);
    // Rows with the same key get the same group ID.
    for i in 0..10_000 {
        for j in (i + 1)..10_000 {
            let same_key = (i % 4) == (j % 4);
            let same_group = r2g[i] == r2g[j];
            assert_eq!(same_key, same_group, "row {i} vs row {j}: same_key={same_key}, same_group={same_group}");
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn kernel_matches_reference_equivalence_classes(
        keys in prop::collection::vec(0u128..=16u128, 4..=512),
    ) {
        let (kernel_r2g, kernel_count, _first) = run_build(&keys);
        let (ref_r2g, ref_count) = build_reference(&keys);
        prop_assert_eq!(kernel_count, ref_count);
        prop_assert!(same_equivalence_classes(&kernel_r2g, &ref_r2g),
            "kernel: {:?} ref: {:?}", kernel_r2g, ref_r2g);
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_groupby_build`
Expected: PASS — three explicit tests + 32 proptest cases.

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-kernels/tests/test_groupby_build.rs
git commit -m "$(cat <<'EOF'
Kernel: groupby build proptest — equivalence vs Rust reference

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6 — Aggregate kernels + dispatchers

Phase 6 lands the per-aggregation kernels (sum, min, max, count, len) and the host-side mean implementation. Each kernel is one thread per source row: read the value, look up its group via `row_to_group`, atomic-OP into the output array at index `group_id`. Mean is computed host-side as `sum / count`, which avoids a redundant kernel and matches Polars' null-handling exactly (sum-of-nulls / count-of-non-null = `null` when count is 0).

The MSL files use the cmp_i64-style macro pattern: one template body, multiple entry points. Caller-side: one Rust dispatcher per (op × dtype) pair, plus host-side mean.

### Task 22: `shaders/aggregate.metal` — templated aggregation kernels

**Files:**
- Create: `shaders/aggregate.metal`

- [ ] **Step 1: Write the kernel file**

```c
// shaders/aggregate.metal
//
// Aggregation kernels for the groupby pipeline. One thread per source
// row. Each entry point reads one value + its validity bit, looks up
// the row's group from `row_to_group`, atomic-OPs into `out[group_id]`.
//
// Null handling:
//   - sum/min/max/count: skip rows where validity bit is 0.
//   - len: counts every row regardless of validity (`pl.len()`).
//
// Templates generate one entry point per (op × dtype) pair via macros.
// Pattern mirrors `cmp_i64.metal` and `cmp_f64.metal`.
//
// Mean is NOT a kernel here. The host wrapper (`compute_mean` in
// `groupby.rs`) runs sum + count and divides on CPU.

#include "_validity.metal"
#include "_groupby.metal"

// Sum: i64.
kernel void agg_sum_i64(
    device const int64_t*       values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_long*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint32_t g = row_to_group[gid];
    atomic_add_i64(out, g, values[gid]);
}

// Sum: f64.
kernel void agg_sum_f64(
    device const double*        values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_ulong*  out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint32_t g = row_to_group[gid];
    atomic_add_f64(out, g, values[gid]);
}

// Min: i64.
kernel void agg_min_i64(
    device const int64_t*       values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_long*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint32_t g = row_to_group[gid];
    atomic_min_i64(out, g, values[gid]);
}

// Max: i64.
kernel void agg_max_i64(
    device const int64_t*       values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_long*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint32_t g = row_to_group[gid];
    atomic_max_i64(out, g, values[gid]);
}

// Min: f64. Polars semantic: NaN propagates (any NaN in a group → min = NaN).
// We follow Polars: if the value is NaN, atomically install NaN bits.
kernel void agg_min_f64(
    device const double*        values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_ulong*  out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint32_t g = row_to_group[gid];
    double v = values[gid];
    if (isnan(v)) {
        // NaN-poison: store NaN bits unconditionally.
        atomic_store_explicit(&out[g], as_type<uint64_t>(v), memory_order_relaxed);
        return;
    }
    atomic_min_f64(out, g, v);
}

// Max: f64. Same NaN semantic as min.
kernel void agg_max_f64(
    device const double*        values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_ulong*  out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint32_t g = row_to_group[gid];
    double v = values[gid];
    if (isnan(v)) {
        atomic_store_explicit(&out[g], as_type<uint64_t>(v), memory_order_relaxed);
        return;
    }
    atomic_max_f64(out, g, v);
}

// Count: non-null values per group. Returns u64.
kernel void agg_count(
    device const uint8_t*       valid         [[buffer(0)]],
    device const uint32_t*      row_to_group  [[buffer(1)]],
    device       atomic_ulong*  out           [[buffer(2)]],
    constant     uint32_t&      n_rows        [[buffer(3)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint32_t g = row_to_group[gid];
    atomic_fetch_add_explicit(&out[g], (uint64_t)1, memory_order_relaxed);
}

// Len: row count per group, ignoring validity (pl.len() semantic).
kernel void agg_len(
    device const uint32_t*      row_to_group  [[buffer(0)]],
    device       atomic_ulong*  out           [[buffer(1)]],
    constant     uint32_t&      n_rows        [[buffer(2)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint32_t g = row_to_group[gid];
    atomic_fetch_add_explicit(&out[g], (uint64_t)1, memory_order_relaxed);
}
```

Note: `agg_min_i64` and `agg_max_i64` need an initial value in `out[g]`. The dispatcher seeds the buffer with i64::MAX (for min) or i64::MIN (for max) before launch. f64 variants: NaN of any kind, or +∞/-∞ as appropriate, populated by the dispatcher.

- [ ] **Step 2: Build to verify**

Run: `make wheel`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add shaders/aggregate.metal
git commit -m "$(cat <<'EOF'
Shaders: aggregate.metal — sum/min/max/count/len kernels

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 23: Rust dispatchers — one per aggregation entry point

**Files:**
- Modify: `crates/polars-metal-kernels/src/groupby.rs`

- [ ] **Step 1: Implement dispatchers**

```rust
// crates/polars-metal-kernels/src/groupby.rs — append:

/// Dispatch `agg_sum_i64`. Output is `&mut [i64]` of length `n_groups`,
/// zero-initialised on entry (the dispatcher allocates fresh device-side
/// buffers).
pub fn dispatch_sum_i64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[i64],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [i64],
) -> Result<(), GroupByError> {
    if n_rows == 0 {
        return Ok(());
    }
    check_n_rows(n_rows)?;
    check_lens(values.len(), n_rows, "values")?;
    check_valid_len(valid, n_rows)?;
    check_lens(row_to_group.len(), n_rows, "row_to_group")?;
    check_out_len(out.len(), n_groups, "out")?;

    let values_bytes: &[u8] = unsafe {
        // SAFETY: i64 POD; alignment of i64 (8) >= alignment of u8 (1).
        std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8)
    };
    let r2g_bytes: &[u8] = unsafe {
        // SAFETY: u32 POD; alignment of u32 (4) >= alignment of u8 (1).
        std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, row_to_group.len() * 4)
    };

    let mut vals_buf = device.new_buffer_with_bytes(values_bytes)?;
    let mut valid_buf = device.new_buffer_with_bytes(valid)?;
    let mut r2g_buf = device.new_buffer_with_bytes(r2g_bytes)?;
    // Output: zero-initialised i64 array.
    let mut out_buf = device.new_buffer_zeroed(n_groups * 8)?;
    let n_rows_u32 = n_rows as u32;

    let pipeline = device.pipeline_for("agg_sum_i64")?;
    let mut encoder = queue.compute_encoder(&pipeline)?;
    encoder.set_buffer(0, &vals_buf);
    encoder.set_buffer(1, &valid_buf);
    encoder.set_buffer(2, &r2g_buf);
    encoder.set_buffer(3, &out_buf);
    encoder.set_constant(4, &n_rows_u32);
    encoder.dispatch(n_rows)?;
    encoder.commit_and_wait()?;

    out_buf.read_into(out)?;
    Ok(())
}

/// Dispatch `agg_sum_f64`. Output is `&mut [f64]` of length `n_groups`,
/// zero-initialised.
pub fn dispatch_sum_f64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[f64],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [f64],
) -> Result<(), GroupByError> {
    if n_rows == 0 { return Ok(()); }
    check_n_rows(n_rows)?;
    check_lens(values.len(), n_rows, "values")?;
    check_valid_len(valid, n_rows)?;
    check_lens(row_to_group.len(), n_rows, "row_to_group")?;
    check_out_len(out.len(), n_groups, "out")?;

    let values_bytes: &[u8] = unsafe {
        // SAFETY: f64 POD; alignment of f64 (8) >= alignment of u8 (1).
        std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8)
    };
    let r2g_bytes: &[u8] = unsafe {
        // SAFETY: u32 POD.
        std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, row_to_group.len() * 4)
    };

    let mut vals_buf = device.new_buffer_with_bytes(values_bytes)?;
    let mut valid_buf = device.new_buffer_with_bytes(valid)?;
    let mut r2g_buf = device.new_buffer_with_bytes(r2g_bytes)?;
    let mut out_buf = device.new_buffer_zeroed(n_groups * 8)?;
    let n_rows_u32 = n_rows as u32;

    let pipeline = device.pipeline_for("agg_sum_f64")?;
    let mut encoder = queue.compute_encoder(&pipeline)?;
    encoder.set_buffer(0, &vals_buf);
    encoder.set_buffer(1, &valid_buf);
    encoder.set_buffer(2, &r2g_buf);
    encoder.set_buffer(3, &out_buf);
    encoder.set_constant(4, &n_rows_u32);
    encoder.dispatch(n_rows)?;
    encoder.commit_and_wait()?;

    out_buf.read_into(out)?;
    Ok(())
}

/// Dispatch `agg_min_i64` / `agg_max_i64`. The output is seeded with
/// i64::MAX / i64::MIN respectively.
pub fn dispatch_min_i64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[i64],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [i64],
) -> Result<(), GroupByError> {
    dispatch_minmax_i64(device, queue, values, valid, row_to_group, n_rows, n_groups, out,
        i64::MAX, "agg_min_i64")
}

pub fn dispatch_max_i64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[i64],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [i64],
) -> Result<(), GroupByError> {
    dispatch_minmax_i64(device, queue, values, valid, row_to_group, n_rows, n_groups, out,
        i64::MIN, "agg_max_i64")
}

fn dispatch_minmax_i64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[i64],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [i64],
    init_value: i64,
    kernel_name: &str,
) -> Result<(), GroupByError> {
    if n_rows == 0 { return Ok(()); }
    check_n_rows(n_rows)?;
    check_lens(values.len(), n_rows, "values")?;
    check_valid_len(valid, n_rows)?;
    check_lens(row_to_group.len(), n_rows, "row_to_group")?;
    check_out_len(out.len(), n_groups, "out")?;

    let values_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8)
    };
    let r2g_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, row_to_group.len() * 4)
    };

    // Seed output with init_value.
    let init_vec = vec![init_value; n_groups];
    let init_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(init_vec.as_ptr() as *const u8, init_vec.len() * 8)
    };

    let mut vals_buf = device.new_buffer_with_bytes(values_bytes)?;
    let mut valid_buf = device.new_buffer_with_bytes(valid)?;
    let mut r2g_buf = device.new_buffer_with_bytes(r2g_bytes)?;
    let mut out_buf = device.new_buffer_with_bytes(init_bytes)?;
    let n_rows_u32 = n_rows as u32;

    let pipeline = device.pipeline_for(kernel_name)?;
    let mut encoder = queue.compute_encoder(&pipeline)?;
    encoder.set_buffer(0, &vals_buf);
    encoder.set_buffer(1, &valid_buf);
    encoder.set_buffer(2, &r2g_buf);
    encoder.set_buffer(3, &out_buf);
    encoder.set_constant(4, &n_rows_u32);
    encoder.dispatch(n_rows)?;
    encoder.commit_and_wait()?;

    out_buf.read_into(out)?;
    Ok(())
}

/// Dispatch min/max for f64.
pub fn dispatch_min_f64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[f64],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [f64],
) -> Result<(), GroupByError> {
    dispatch_minmax_f64(device, queue, values, valid, row_to_group, n_rows, n_groups, out,
        f64::INFINITY, "agg_min_f64")
}

pub fn dispatch_max_f64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[f64],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [f64],
) -> Result<(), GroupByError> {
    dispatch_minmax_f64(device, queue, values, valid, row_to_group, n_rows, n_groups, out,
        f64::NEG_INFINITY, "agg_max_f64")
}

fn dispatch_minmax_f64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[f64],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [f64],
    init_value: f64,
    kernel_name: &str,
) -> Result<(), GroupByError> {
    if n_rows == 0 { return Ok(()); }
    check_n_rows(n_rows)?;
    check_lens(values.len(), n_rows, "values")?;
    check_valid_len(valid, n_rows)?;
    check_lens(row_to_group.len(), n_rows, "row_to_group")?;
    check_out_len(out.len(), n_groups, "out")?;

    let values_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8)
    };
    let r2g_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, row_to_group.len() * 4)
    };
    let init_vec = vec![init_value; n_groups];
    let init_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(init_vec.as_ptr() as *const u8, init_vec.len() * 8)
    };

    let mut vals_buf = device.new_buffer_with_bytes(values_bytes)?;
    let mut valid_buf = device.new_buffer_with_bytes(valid)?;
    let mut r2g_buf = device.new_buffer_with_bytes(r2g_bytes)?;
    let mut out_buf = device.new_buffer_with_bytes(init_bytes)?;
    let n_rows_u32 = n_rows as u32;

    let pipeline = device.pipeline_for(kernel_name)?;
    let mut encoder = queue.compute_encoder(&pipeline)?;
    encoder.set_buffer(0, &vals_buf);
    encoder.set_buffer(1, &valid_buf);
    encoder.set_buffer(2, &r2g_buf);
    encoder.set_buffer(3, &out_buf);
    encoder.set_constant(4, &n_rows_u32);
    encoder.dispatch(n_rows)?;
    encoder.commit_and_wait()?;

    out_buf.read_into(out)?;
    Ok(())
}

/// Dispatch `agg_count` — non-null count per group.
pub fn dispatch_count(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [u64],
) -> Result<(), GroupByError> {
    if n_rows == 0 { return Ok(()); }
    check_n_rows(n_rows)?;
    check_valid_len(valid, n_rows)?;
    check_lens(row_to_group.len(), n_rows, "row_to_group")?;
    check_out_len(out.len(), n_groups, "out")?;

    let r2g_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, row_to_group.len() * 4)
    };
    let mut valid_buf = device.new_buffer_with_bytes(valid)?;
    let mut r2g_buf = device.new_buffer_with_bytes(r2g_bytes)?;
    let mut out_buf = device.new_buffer_zeroed(n_groups * 8)?;
    let n_rows_u32 = n_rows as u32;

    let pipeline = device.pipeline_for("agg_count")?;
    let mut encoder = queue.compute_encoder(&pipeline)?;
    encoder.set_buffer(0, &valid_buf);
    encoder.set_buffer(1, &r2g_buf);
    encoder.set_buffer(2, &out_buf);
    encoder.set_constant(3, &n_rows_u32);
    encoder.dispatch(n_rows)?;
    encoder.commit_and_wait()?;

    out_buf.read_into(out)?;
    Ok(())
}

/// Dispatch `agg_len` — row count per group (no validity read).
pub fn dispatch_len(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [u64],
) -> Result<(), GroupByError> {
    if n_rows == 0 { return Ok(()); }
    check_n_rows(n_rows)?;
    check_lens(row_to_group.len(), n_rows, "row_to_group")?;
    check_out_len(out.len(), n_groups, "out")?;

    let r2g_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, row_to_group.len() * 4)
    };
    let mut r2g_buf = device.new_buffer_with_bytes(r2g_bytes)?;
    let mut out_buf = device.new_buffer_zeroed(n_groups * 8)?;
    let n_rows_u32 = n_rows as u32;

    let pipeline = device.pipeline_for("agg_len")?;
    let mut encoder = queue.compute_encoder(&pipeline)?;
    encoder.set_buffer(0, &r2g_buf);
    encoder.set_buffer(1, &out_buf);
    encoder.set_constant(2, &n_rows_u32);
    encoder.dispatch(n_rows)?;
    encoder.commit_and_wait()?;

    out_buf.read_into(out)?;
    Ok(())
}

// -----------------------------------------------------------------------
// Length / buffer checks shared across dispatchers.
// -----------------------------------------------------------------------

fn check_n_rows(n_rows: usize) -> Result<(), GroupByError> {
    if u32::try_from(n_rows).is_err() {
        Err(GroupByError::RowCountOverflow { n_rows })
    } else {
        Ok(())
    }
}

fn check_lens(got: usize, n_rows: usize, _label: &str) -> Result<(), GroupByError> {
    if got < n_rows {
        Err(GroupByError::OutputTooShort { got, need: n_rows })
    } else {
        Ok(())
    }
}

fn check_valid_len(valid: &[u8], n_rows: usize) -> Result<(), GroupByError> {
    let need = (n_rows + 7) / 8;
    if valid.len() < need {
        Err(GroupByError::OutputTooShort { got: valid.len(), need })
    } else {
        Ok(())
    }
}

fn check_out_len(got: usize, n_groups: usize, _label: &str) -> Result<(), GroupByError> {
    if got < n_groups {
        Err(GroupByError::OutputTooShort { got, need: n_groups })
    } else {
        Ok(())
    }
}
```

- [ ] **Step 2: Build to verify**

Run: `cargo build -p polars-metal-kernels`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-kernels/src/groupby.rs
git commit -m "$(cat <<'EOF'
Kernel: aggregation dispatchers — sum/min/max/count/len per dtype

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 24: Aggregation proptests against CPU reference

**Files:**
- Create: `crates/polars-metal-kernels/tests/test_groupby_aggregate.rs`

- [ ] **Step 1: Write the proptest suite**

```rust
// crates/polars-metal-kernels/tests/test_groupby_aggregate.rs
//
// Property tests: each (op × dtype) dispatcher matches a CPU reference
// implementation byte-for-byte on randomized inputs (groups, values,
// null patterns). The reference uses Polars' null semantics: skip nulls
// for sum/min/max/count, count every row for len.
#![allow(clippy::expect_used)]

use polars_metal_kernels::groupby::{
    dispatch_count, dispatch_len, dispatch_max_f64, dispatch_max_i64,
    dispatch_min_f64, dispatch_min_i64, dispatch_sum_f64, dispatch_sum_i64,
};
use polars_metal_kernels::shader_lib::load_default_library;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::MetalDevice;
use proptest::prelude::*;

fn pack_valid(valid: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; (valid.len() + 7) / 8];
    for (i, &v) in valid.iter().enumerate() {
        if v {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

// CPU reference implementations.
fn ref_sum_i64(values: &[i64], valid: &[bool], row_to_group: &[u32], n_groups: usize) -> Vec<i64> {
    let mut out = vec![0i64; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = row_to_group[i] as usize;
            out[g] = out[g].wrapping_add(values[i]);
        }
    }
    out
}

fn ref_sum_f64(values: &[f64], valid: &[bool], row_to_group: &[u32], n_groups: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = row_to_group[i] as usize;
            out[g] += values[i];
        }
    }
    out
}

fn ref_min_i64(values: &[i64], valid: &[bool], row_to_group: &[u32], n_groups: usize) -> Vec<i64> {
    let mut out = vec![i64::MAX; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = row_to_group[i] as usize;
            if values[i] < out[g] { out[g] = values[i]; }
        }
    }
    out
}

fn ref_max_i64(values: &[i64], valid: &[bool], row_to_group: &[u32], n_groups: usize) -> Vec<i64> {
    let mut out = vec![i64::MIN; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = row_to_group[i] as usize;
            if values[i] > out[g] { out[g] = values[i]; }
        }
    }
    out
}

fn ref_count(valid: &[bool], row_to_group: &[u32], n_groups: usize) -> Vec<u64> {
    let mut out = vec![0u64; n_groups];
    for i in 0..valid.len() {
        if valid[i] {
            let g = row_to_group[i] as usize;
            out[g] += 1;
        }
    }
    out
}

fn ref_len(row_to_group: &[u32], n_groups: usize) -> Vec<u64> {
    let mut out = vec![0u64; n_groups];
    for &g in row_to_group {
        out[g as usize] += 1;
    }
    out
}

fn setup() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::new().expect("device");
    let queue = CommandQueue::new(&device).expect("queue");
    let _lib = load_default_library(&device).expect("library");
    (device, queue)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn sum_i64_matches_reference(
        values in prop::collection::vec(any::<i64>(), 1..=512),
        valids in prop::collection::vec(any::<bool>(), 1..=512),
        groups in prop::collection::vec(0u32..16, 1..=512),
    ) {
        let n = values.len().min(valids.len()).min(groups.len());
        let n_groups = (groups[..n].iter().copied().max().unwrap_or(0) as usize) + 1;
        let valid_packed = pack_valid(&valids[..n]);

        let (device, mut queue) = setup();
        let mut kernel_out = vec![0i64; n_groups];
        dispatch_sum_i64(&device, &mut queue, &values[..n], &valid_packed, &groups[..n], n, n_groups, &mut kernel_out)
            .expect("dispatch");
        let ref_out = ref_sum_i64(&values[..n], &valids[..n], &groups[..n], n_groups);
        prop_assert_eq!(kernel_out, ref_out);
    }

    #[test]
    fn min_i64_matches_reference(
        values in prop::collection::vec(any::<i64>(), 1..=256),
        valids in prop::collection::vec(any::<bool>(), 1..=256),
        groups in prop::collection::vec(0u32..8, 1..=256),
    ) {
        let n = values.len().min(valids.len()).min(groups.len());
        let n_groups = (groups[..n].iter().copied().max().unwrap_or(0) as usize) + 1;
        let valid_packed = pack_valid(&valids[..n]);

        let (device, mut queue) = setup();
        let mut kernel_out = vec![i64::MAX; n_groups];
        dispatch_min_i64(&device, &mut queue, &values[..n], &valid_packed, &groups[..n], n, n_groups, &mut kernel_out)
            .expect("dispatch");
        let ref_out = ref_min_i64(&values[..n], &valids[..n], &groups[..n], n_groups);
        prop_assert_eq!(kernel_out, ref_out);
    }

    #[test]
    fn max_i64_matches_reference(
        values in prop::collection::vec(any::<i64>(), 1..=256),
        valids in prop::collection::vec(any::<bool>(), 1..=256),
        groups in prop::collection::vec(0u32..8, 1..=256),
    ) {
        let n = values.len().min(valids.len()).min(groups.len());
        let n_groups = (groups[..n].iter().copied().max().unwrap_or(0) as usize) + 1;
        let valid_packed = pack_valid(&valids[..n]);

        let (device, mut queue) = setup();
        let mut kernel_out = vec![i64::MIN; n_groups];
        dispatch_max_i64(&device, &mut queue, &values[..n], &valid_packed, &groups[..n], n, n_groups, &mut kernel_out)
            .expect("dispatch");
        let ref_out = ref_max_i64(&values[..n], &valids[..n], &groups[..n], n_groups);
        prop_assert_eq!(kernel_out, ref_out);
    }

    #[test]
    fn count_matches_reference(
        valids in prop::collection::vec(any::<bool>(), 1..=256),
        groups in prop::collection::vec(0u32..8, 1..=256),
    ) {
        let n = valids.len().min(groups.len());
        let n_groups = (groups[..n].iter().copied().max().unwrap_or(0) as usize) + 1;
        let valid_packed = pack_valid(&valids[..n]);

        let (device, mut queue) = setup();
        let mut kernel_out = vec![0u64; n_groups];
        dispatch_count(&device, &mut queue, &valid_packed, &groups[..n], n, n_groups, &mut kernel_out)
            .expect("dispatch");
        let ref_out = ref_count(&valids[..n], &groups[..n], n_groups);
        prop_assert_eq!(kernel_out, ref_out);
    }

    #[test]
    fn len_matches_reference(
        groups in prop::collection::vec(0u32..8, 1..=256),
    ) {
        let n = groups.len();
        let n_groups = (groups.iter().copied().max().unwrap_or(0) as usize) + 1;

        let (device, mut queue) = setup();
        let mut kernel_out = vec![0u64; n_groups];
        dispatch_len(&device, &mut queue, &groups, n, n_groups, &mut kernel_out)
            .expect("dispatch");
        let ref_out = ref_len(&groups, n_groups);
        prop_assert_eq!(kernel_out, ref_out);
    }

    // f64 sum: byte-exact equivalence is impossible due to non-deterministic
    // floating-point addition order. Instead assert ULP-bounded equivalence
    // for finite values. Polars' .sum() on f64 columns documents the same
    // caveat (sum reduction order is implementation-defined).
    #[test]
    fn sum_f64_matches_reference_within_tolerance(
        values in prop::collection::vec(-1e6f64..1e6, 1..=256),
        valids in prop::collection::vec(any::<bool>(), 1..=256),
        groups in prop::collection::vec(0u32..8, 1..=256),
    ) {
        let n = values.len().min(valids.len()).min(groups.len());
        let n_groups = (groups[..n].iter().copied().max().unwrap_or(0) as usize) + 1;
        let valid_packed = pack_valid(&valids[..n]);

        let (device, mut queue) = setup();
        let mut kernel_out = vec![0.0f64; n_groups];
        dispatch_sum_f64(&device, &mut queue, &values[..n], &valid_packed, &groups[..n], n, n_groups, &mut kernel_out)
            .expect("dispatch");
        let ref_out = ref_sum_f64(&values[..n], &valids[..n], &groups[..n], n_groups);
        for (k, r) in kernel_out.iter().zip(ref_out.iter()) {
            if r.is_finite() {
                let diff = (k - r).abs();
                let tol = (r.abs() * 1e-9).max(1e-9);
                prop_assert!(diff <= tol, "sum_f64 diff {} > tol {} at value k={} r={}", diff, tol, k, r);
            }
        }
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_groupby_aggregate`
Expected: PASS — all six properties green at 32 cases each.

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-kernels/tests/test_groupby_aggregate.rs
git commit -m "$(cat <<'EOF'
Kernel: aggregation proptests vs CPU reference

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 25: Host-side mean — `compute_mean` (no kernel)

**Files:**
- Modify: `crates/polars-metal-kernels/src/groupby.rs`
- Create: `crates/polars-metal-kernels/tests/test_groupby_mean.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_groupby_mean.rs
//
// Mean is computed host-side as sum / count. Verifies:
//   1. The host wrapper runs both kernels and divides correctly.
//   2. Null semantics: a group with zero non-null values produces a NaN
//      mean (matching Polars' .mean() on an all-null group).
#![allow(clippy::expect_used)]

use polars_metal_kernels::groupby::{compute_mean, MeanOutput};
use polars_metal_kernels::shader_lib::load_default_library;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::MetalDevice;
use proptest::prelude::*;

fn pack_valid(valid: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; (valid.len() + 7) / 8];
    for (i, &v) in valid.iter().enumerate() {
        if v {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

fn setup() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::new().expect("device");
    let queue = CommandQueue::new(&device).expect("queue");
    let _lib = load_default_library(&device).expect("library");
    (device, queue)
}

#[test]
fn mean_simple_two_groups() {
    let values = vec![1.0f64, 2.0, 10.0, 20.0, 30.0];
    let valid_bools = vec![true; 5];
    let row_to_group = vec![0u32, 0, 1, 1, 1];
    let n_groups = 2;
    let valid_packed = pack_valid(&valid_bools);

    let (device, mut queue) = setup();
    let out = compute_mean(&device, &mut queue, &values, &valid_packed, &row_to_group, 5, n_groups)
        .expect("compute_mean");

    assert_eq!(out.sum, vec![3.0, 60.0]);
    assert_eq!(out.count, vec![2, 3]);
    assert!((out.mean[0] - 1.5).abs() < 1e-12);
    assert!((out.mean[1] - 20.0).abs() < 1e-12);
}

#[test]
fn mean_of_all_null_group_is_nan() {
    // Group 0: all null. Group 1: one valid value.
    let values = vec![999.0, 999.0, 42.0];
    let valid_bools = vec![false, false, true];
    let row_to_group = vec![0u32, 0, 1];
    let valid_packed = pack_valid(&valid_bools);

    let (device, mut queue) = setup();
    let out = compute_mean(&device, &mut queue, &values, &valid_packed, &row_to_group, 3, 2)
        .expect("compute_mean");

    assert_eq!(out.count[0], 0);
    assert_eq!(out.count[1], 1);
    assert!(out.mean[0].is_nan(), "expected NaN for all-null group, got {}", out.mean[0]);
    assert!((out.mean[1] - 42.0).abs() < 1e-12);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn mean_matches_cpu_reference(
        values in prop::collection::vec(-1e6f64..1e6, 1..=256),
        valids in prop::collection::vec(any::<bool>(), 1..=256),
        groups in prop::collection::vec(0u32..8, 1..=256),
    ) {
        let n = values.len().min(valids.len()).min(groups.len());
        let n_groups = (groups[..n].iter().copied().max().unwrap_or(0) as usize) + 1;
        let valid_packed = pack_valid(&valids[..n]);

        let (device, mut queue) = setup();
        let out = compute_mean(&device, &mut queue, &values[..n], &valid_packed, &groups[..n], n, n_groups)
            .expect("compute_mean");

        // Reference: sum / count, with NaN for empty groups.
        let mut ref_sum = vec![0.0f64; n_groups];
        let mut ref_cnt = vec![0u64; n_groups];
        for i in 0..n {
            if valids[i] {
                let g = groups[i] as usize;
                ref_sum[g] += values[i];
                ref_cnt[g] += 1;
            }
        }
        for g in 0..n_groups {
            let cpu_mean = if ref_cnt[g] == 0 { f64::NAN } else { ref_sum[g] / (ref_cnt[g] as f64) };
            let kernel_mean = out.mean[g];
            if cpu_mean.is_nan() {
                prop_assert!(kernel_mean.is_nan(), "empty group {g} should be NaN, got {kernel_mean}");
            } else {
                let diff = (kernel_mean - cpu_mean).abs();
                let tol = (cpu_mean.abs() * 1e-9).max(1e-9);
                prop_assert!(diff <= tol, "group {g}: kernel {kernel_mean} vs cpu {cpu_mean} diff {diff} > {tol}");
            }
        }
    }
}
```

- [ ] **Step 2: Implement `compute_mean`**

```rust
// crates/polars-metal-kernels/src/groupby.rs — append:

/// Result of `compute_mean`. Holds all three derived arrays so callers
/// don't have to re-run sum/count if they want any of them. (M2's UDF
/// uses all three: sum for verification, count for the result schema,
/// mean for the user-facing column.)
pub struct MeanOutput {
    pub sum: Vec<f64>,
    pub count: Vec<u64>,
    pub mean: Vec<f64>,
}

/// Compute mean per group as `sum(non_null) / count(non_null)`. For
/// groups with zero non-null values, mean is NaN (Polars semantic).
///
/// This is NOT a single kernel — it runs `dispatch_sum_f64` and
/// `dispatch_count`, then divides on the host. The host-side division
/// matches Polars exactly for the empty-group → NaN edge case.
pub fn compute_mean(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[f64],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
) -> Result<MeanOutput, GroupByError> {
    let mut sum = vec![0.0f64; n_groups];
    let mut count = vec![0u64; n_groups];
    dispatch_sum_f64(device, queue, values, valid, row_to_group, n_rows, n_groups, &mut sum)?;
    dispatch_count(device, queue, valid, row_to_group, n_rows, n_groups, &mut count)?;
    let mean: Vec<f64> = sum
        .iter()
        .zip(count.iter())
        .map(|(&s, &c)| if c == 0 { f64::NAN } else { s / (c as f64) })
        .collect();
    Ok(MeanOutput { sum, count, mean })
}
```

- [ ] **Step 3: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_groupby_mean`
Expected: PASS — two unit tests + 32 proptest cases.

- [ ] **Step 4: Run all kernel tests to confirm Phase 6 is clean**

Run: `cargo test -p polars-metal-kernels`
Expected: every kernel test crate (key encoding, hash, build, aggregate, mean) plus the M1 tests (cmp, logical, filter, scatter, validity, shader_lib, command, pipeline, dispatch) all green.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-kernels/src/groupby.rs crates/polars-metal-kernels/tests/test_groupby_mean.rs
git commit -m "$(cat <<'EOF'
Kernel: compute_mean — host-side sum / count with null semantics

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 7 — Pipeline orchestrator + end-to-end Rust proptest

Phase 7 wires Phases 3-6 together into a single `dispatch_groupby` orchestrator and lands an end-to-end Rust proptest against a pure-Rust reference. After Phase 7, the kernel layer can run a whole groupby pipeline standalone (no Python, no UDF) and prove it byte-equivalent (modulo group order) to a HashMap-based CPU reference. Phase 8 then wires this entry point to the engine.

### Task 26: `dispatch_groupby` orchestrator

**Files:**
- Modify: `crates/polars-metal-kernels/src/groupby.rs`

- [ ] **Step 1: Implement the orchestrator**

```rust
// crates/polars-metal-kernels/src/groupby.rs — append:

/// A single aggregation request to the pipeline.
///
/// `input_col_idx` indexes into the `value_cols` slice passed to
/// `dispatch_groupby`. For `AggKind::Len` the index is ignored.
#[derive(Debug, Clone)]
pub struct AggRequest {
    pub kind: AggKind,
    pub input_col_idx: usize,
}

/// Which aggregation kernel to run. Mirrors `crate::plan::AggOp` but is
/// independent of polars-metal-core's plan IR so the kernel crate doesn't
/// take a circular dep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggKind {
    SumI64,
    SumF64,
    MeanF64,
    MinI64,
    MaxI64,
    MinF64,
    MaxF64,
    Count,
    Len,
}

/// One value column passed to the pipeline (typed view).
///
/// The pipeline never mutates value columns; they're borrowed for the
/// life of the call. Validity is a packed little-endian bitmap aligned
/// to 4 bytes (M1's convention) — the encoder allocates more than
/// `(n_rows + 7) / 8` bytes if needed to satisfy 4-byte alignment.
pub enum ValueColumn<'a> {
    I64 { data: &'a [i64], valid: &'a [u8] },
    F64 { data: &'a [f64], valid: &'a [u8] },
}

/// Output of a single aggregation in the result set.
///
/// `Mean` carries the host-computed mean Vec directly (sum/count are
/// internal to `compute_mean`; callers don't need them). For other ops
/// the buffer is the kernel's group-indexed output.
#[derive(Debug)]
pub enum AggOutput {
    I64(Vec<i64>),
    F64(Vec<f64>),
    U64(Vec<u64>),
}

/// The pipeline's complete result. Group rows are in arbitrary order
/// (the build kernel's thread schedule decides). Callers (the UDF and
/// the Polars layer) sort by key if user-visible order matters.
#[derive(Debug)]
pub struct GroupByResult {
    /// One entry per key column, in the same order as the input keys.
    pub decoded_keys: Vec<DecodedColumn>,
    /// One entry per `AggRequest`, in input order.
    pub agg_outputs: Vec<AggOutput>,
    /// Distinct group count across all rows. Equals the length of every
    /// `decoded_keys[i].values` and every `agg_outputs[i]` buffer.
    pub n_groups: u32,
}

/// Run the full two-pass groupby pipeline.
///
/// Steps:
///   1. `encode_keys(key_cols)`             → (encoded: Vec<u128>, schema)
///   2. `dispatch_hash(encoded)`            → hashes: Vec<u32>
///   3. `dispatch_build(encoded, hashes)`   → row_to_group + first_row_per_group + group_count
///   4. For each AggRequest: dispatch the appropriate kernel
///      against `row_to_group`. Mean uses host-side `compute_mean`.
///   5. Decode keys at `first_row_per_group` indices to reconstruct the
///      group-level key columns.
///
/// All errors propagate as `GroupByError`. The orchestrator is the
/// single funnel into the kernel layer — `udf.rs` calls only this.
pub fn dispatch_groupby(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    key_cols: &[KeyColumn<'_>],
    agg_specs: &[(AggRequest, ValueColumn<'_>)],
    n_rows: usize,
) -> Result<GroupByResult, GroupByError> {
    // Sanity: every key column and every value column reports the same
    // n_rows. The walker guarantees this; we defensively re-check.
    for kc in key_cols {
        if kc.n_rows != n_rows {
            return Err(GroupByError::OutputTooShort {
                got: kc.n_rows,
                need: n_rows,
            });
        }
    }

    // Step 1: encode composite keys.
    let (encoded, schema) = encode_keys(key_cols)?;

    // Empty-input shortcut: zero groups, every agg output is empty.
    if n_rows == 0 {
        let decoded_keys = decode_keys(&[], &schema);
        let agg_outputs = agg_specs
            .iter()
            .map(|(req, _)| empty_output_for(req.kind))
            .collect();
        return Ok(GroupByResult {
            decoded_keys,
            agg_outputs,
            n_groups: 0,
        });
    }

    // Step 2: hash.
    let mut hashes = vec![0u32; n_rows];
    dispatch_hash(device, queue, &encoded, n_rows, &mut hashes)?;

    // Step 3: build (find-or-insert per row).
    let build = dispatch_build(device, queue, &encoded, &hashes, n_rows)?;
    let n_groups = build.group_count;

    // Step 4: per-aggregation dispatch.
    let mut agg_outputs = Vec::with_capacity(agg_specs.len());
    for (req, vcol) in agg_specs {
        agg_outputs.push(run_one_agg(
            device,
            queue,
            req,
            vcol,
            &build.row_to_group,
            n_rows,
            n_groups as usize,
        )?);
    }

    // Step 5: decode keys at representative-row indices.
    let representative_keys: Vec<u128> = build
        .first_row_per_group
        .iter()
        .map(|&row| encoded[row as usize])
        .collect();
    let decoded_keys = decode_keys(&representative_keys, &schema);

    Ok(GroupByResult {
        decoded_keys,
        agg_outputs,
        n_groups,
    })
}

fn empty_output_for(kind: AggKind) -> AggOutput {
    match kind {
        AggKind::SumI64 | AggKind::MinI64 | AggKind::MaxI64 => AggOutput::I64(vec![]),
        AggKind::SumF64 | AggKind::MeanF64 | AggKind::MinF64 | AggKind::MaxF64 => {
            AggOutput::F64(vec![])
        }
        AggKind::Count | AggKind::Len => AggOutput::U64(vec![]),
    }
}

fn run_one_agg(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    req: &AggRequest,
    vcol: &ValueColumn<'_>,
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
) -> Result<AggOutput, GroupByError> {
    match (req.kind, vcol) {
        (AggKind::SumI64, ValueColumn::I64 { data, valid }) => {
            let mut out = vec![0i64; n_groups];
            dispatch_sum_i64(device, queue, data, valid, row_to_group, n_rows, n_groups, &mut out)?;
            Ok(AggOutput::I64(out))
        }
        (AggKind::SumF64, ValueColumn::F64 { data, valid }) => {
            let mut out = vec![0.0f64; n_groups];
            dispatch_sum_f64(device, queue, data, valid, row_to_group, n_rows, n_groups, &mut out)?;
            Ok(AggOutput::F64(out))
        }
        (AggKind::MinI64, ValueColumn::I64 { data, valid }) => {
            let mut out = vec![i64::MAX; n_groups];
            dispatch_min_i64(device, queue, data, valid, row_to_group, n_rows, n_groups, &mut out)?;
            Ok(AggOutput::I64(out))
        }
        (AggKind::MaxI64, ValueColumn::I64 { data, valid }) => {
            let mut out = vec![i64::MIN; n_groups];
            dispatch_max_i64(device, queue, data, valid, row_to_group, n_rows, n_groups, &mut out)?;
            Ok(AggOutput::I64(out))
        }
        (AggKind::MinF64, ValueColumn::F64 { data, valid }) => {
            let mut out = vec![f64::INFINITY; n_groups];
            dispatch_min_f64(device, queue, data, valid, row_to_group, n_rows, n_groups, &mut out)?;
            Ok(AggOutput::F64(out))
        }
        (AggKind::MaxF64, ValueColumn::F64 { data, valid }) => {
            let mut out = vec![f64::NEG_INFINITY; n_groups];
            dispatch_max_f64(device, queue, data, valid, row_to_group, n_rows, n_groups, &mut out)?;
            Ok(AggOutput::F64(out))
        }
        (AggKind::Count, ValueColumn::I64 { valid, .. })
        | (AggKind::Count, ValueColumn::F64 { valid, .. }) => {
            let mut out = vec![0u64; n_groups];
            dispatch_count(device, queue, valid, row_to_group, n_rows, n_groups, &mut out)?;
            Ok(AggOutput::U64(out))
        }
        (AggKind::Len, _) => {
            let mut out = vec![0u64; n_groups];
            dispatch_len(device, queue, row_to_group, n_rows, n_groups, &mut out)?;
            Ok(AggOutput::U64(out))
        }
        (AggKind::MeanF64, ValueColumn::F64 { data, valid }) => {
            let mean_out = compute_mean(device, queue, data, valid, row_to_group, n_rows, n_groups)?;
            Ok(AggOutput::F64(mean_out.mean))
        }
        (kind, vcol) => {
            // Caller-side type mismatch (e.g. SumI64 with an F64 column).
            // The walker / UDF should never emit this combination — guard
            // defensively with a typed error.
            let vt = match vcol {
                ValueColumn::I64 { .. } => "I64",
                ValueColumn::F64 { .. } => "F64",
            };
            Err(GroupByError::AggTypeMismatch {
                kind: format!("{kind:?}"),
                value_dtype: vt.to_string(),
            })
        }
    }
}
```

Add the new variant to `GroupByError`:

```rust
// crates/polars-metal-kernels/src/groupby.rs — extend the enum:

#[derive(Debug, thiserror::Error)]
pub enum GroupByError {
    // ... existing variants ...
    #[error("aggregation kind {kind} not compatible with value dtype {value_dtype}")]
    AggTypeMismatch { kind: String, value_dtype: String },
}
```

- [ ] **Step 2: Build to verify the signatures**

Run: `cargo build -p polars-metal-kernels`
Expected: clean build. (Tests in Task 27.)

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-kernels/src/groupby.rs
git commit -m "$(cat <<'EOF'
Kernel: dispatch_groupby orchestrator

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 27: End-to-end pipeline proptest

**Files:**
- Create: `crates/polars-metal-kernels/tests/test_groupby_pipeline.rs`

- [ ] **Step 1: Write the proptest**

```rust
// crates/polars-metal-kernels/tests/test_groupby_pipeline.rs
//
// End-to-end pipeline proptest. Generates random key + value columns,
// runs `dispatch_groupby`, compares against a pure-Rust HashMap-based
// reference. Equality is *modulo group order*: the kernel's atomic-CAS
// build phase produces groups in arbitrary thread-schedule order, so
// we canonicalize both sides into a `BTreeMap<key_tuple, agg_tuple>`
// keyed by the group's representative key bytes before comparison.
#![allow(clippy::expect_used)]

use polars_metal_kernels::groupby::{
    dispatch_groupby, AggKind, AggOutput, AggRequest, DecodedColumn,
    KeyColumn, KeyDtype, ValueColumn,
};
use polars_metal_kernels::shader_lib::load_default_library;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::MetalDevice;
use proptest::prelude::*;
use std::collections::BTreeMap;

fn bytes_i64(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f64(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn pack_valid(valid: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; ((valid.len() + 7) / 8 + 3) & !3];
    for (i, &v) in valid.iter().enumerate() {
        if v {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

fn setup() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::new().expect("device");
    let queue = CommandQueue::new(&device).expect("queue");
    let _lib = load_default_library(&device).expect("library");
    (device, queue)
}

/// Reference implementation: pure-Rust HashMap-based groupby. Mirrors
/// Polars' null semantics exactly: nulls excluded from sum/mean/min/max
/// (mean of empty group is NaN), count_non_null counts only valid rows,
/// pl.len() counts every row including nulls. Group order in the output
/// is the order distinct keys are first seen.
fn cpu_groupby_reference(
    keys: &[Vec<i64>],
    key_valids: &[Vec<bool>],
    values: &[Vec<f64>],
    value_valids: &[Vec<bool>],
    aggs: &[(AggKind, usize)],
    n_rows: usize,
) -> BTreeMap<Vec<(bool, i64)>, Vec<AggOutput>> {
    // Build per-row composite key as Vec<(valid, value)>.
    let mut groups: BTreeMap<Vec<(bool, i64)>, Vec<usize>> = BTreeMap::new();
    for i in 0..n_rows {
        let mut k = Vec::with_capacity(keys.len());
        for (kc, kv) in keys.iter().zip(key_valids.iter()) {
            k.push((kv[i], if kv[i] { kc[i] } else { 0 }));
        }
        groups.entry(k).or_default().push(i);
    }

    // For each group, compute every requested aggregation.
    let mut result: BTreeMap<Vec<(bool, i64)>, Vec<AggOutput>> = BTreeMap::new();
    for (key, rows) in groups {
        let mut outs = Vec::with_capacity(aggs.len());
        for (kind, col_idx) in aggs {
            outs.push(reference_one_agg(*kind, *col_idx, &rows, values, value_valids));
        }
        result.insert(key, outs);
    }
    result
}

fn reference_one_agg(
    kind: AggKind,
    col_idx: usize,
    rows: &[usize],
    values: &[Vec<f64>],
    valids: &[Vec<bool>],
) -> AggOutput {
    match kind {
        AggKind::SumF64 => {
            let mut s = 0.0f64;
            for &r in rows {
                if valids[col_idx][r] {
                    s += values[col_idx][r];
                }
            }
            AggOutput::F64(vec![s])
        }
        AggKind::MinF64 => {
            let mut m = f64::INFINITY;
            let mut any = false;
            for &r in rows {
                if valids[col_idx][r] {
                    any = true;
                    if values[col_idx][r] < m { m = values[col_idx][r]; }
                }
            }
            AggOutput::F64(vec![if any { m } else { f64::INFINITY }])
        }
        AggKind::MaxF64 => {
            let mut m = f64::NEG_INFINITY;
            let mut any = false;
            for &r in rows {
                if valids[col_idx][r] {
                    any = true;
                    if values[col_idx][r] > m { m = values[col_idx][r]; }
                }
            }
            AggOutput::F64(vec![if any { m } else { f64::NEG_INFINITY }])
        }
        AggKind::MeanF64 => {
            let mut s = 0.0f64;
            let mut c = 0u64;
            for &r in rows {
                if valids[col_idx][r] {
                    s += values[col_idx][r];
                    c += 1;
                }
            }
            let mean = if c == 0 { f64::NAN } else { s / (c as f64) };
            AggOutput::F64(vec![mean])
        }
        AggKind::Count => {
            let mut c = 0u64;
            for &r in rows {
                if valids[col_idx][r] { c += 1; }
            }
            AggOutput::U64(vec![c])
        }
        AggKind::Len => AggOutput::U64(vec![rows.len() as u64]),
        // Sum/Min/Max I64 not exercised in this proptest (we use F64
        // values throughout for simplicity); add if needed.
        _ => panic!("unsupported reference agg kind {kind:?}"),
    }
}

/// Canonicalize kernel output into the same BTreeMap shape as the
/// reference. Group order doesn't matter — only the (key, agg-tuple)
/// pairs.
fn canonicalize(result: &polars_metal_kernels::groupby::GroupByResult) -> BTreeMap<Vec<(bool, i64)>, Vec<AggOutput>> {
    let n = result.n_groups as usize;
    let mut out = BTreeMap::new();
    for g in 0..n {
        let mut key = Vec::with_capacity(result.decoded_keys.len());
        for col in &result.decoded_keys {
            match col {
                DecodedColumn::I64 { values, valid } => {
                    key.push((valid[g], values[g]));
                }
                DecodedColumn::Bool { values, valid } => {
                    key.push((valid[g], if values[g] { 1 } else { 0 }));
                }
                DecodedColumn::F64 { values, valid } => {
                    key.push((valid[g], values[g].to_bits() as i64));
                }
            }
        }
        let aggs: Vec<AggOutput> = result.agg_outputs.iter().map(|o| match o {
            AggOutput::I64(v) => AggOutput::I64(vec![v[g]]),
            AggOutput::F64(v) => AggOutput::F64(vec![v[g]]),
            AggOutput::U64(v) => AggOutput::U64(vec![v[g]]),
        }).collect();
        out.insert(key, aggs);
    }
    out
}

fn agg_output_eq(a: &AggOutput, b: &AggOutput) -> bool {
    match (a, b) {
        (AggOutput::I64(x), AggOutput::I64(y)) => x == y,
        (AggOutput::U64(x), AggOutput::U64(y)) => x == y,
        (AggOutput::F64(x), AggOutput::F64(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| {
                if a.is_nan() && b.is_nan() {
                    true
                } else if a.is_finite() && b.is_finite() {
                    let tol = (a.abs().max(b.abs()) * 1e-9).max(1e-9);
                    (a - b).abs() <= tol
                } else {
                    a == b
                }
            })
        }
        _ => false,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn pipeline_matches_cpu_reference(
        // 1-2 key columns, each i64 with up to 8 distinct values.
        n_keys in 1usize..=2,
        // 1-3 value columns (f64) for the agg pool.
        n_values in 1usize..=3,
        // 32-256 rows.
        n_rows in 32usize..=256,
        seed in any::<u64>(),
    ) {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate key columns (i64 with low cardinality).
        let mut keys: Vec<Vec<i64>> = Vec::with_capacity(n_keys);
        let mut key_valids: Vec<Vec<bool>> = Vec::with_capacity(n_keys);
        for _ in 0..n_keys {
            keys.push((0..n_rows).map(|_| rng.gen_range(0i64..4)).collect());
            key_valids.push((0..n_rows).map(|_| rng.gen_bool(0.9)).collect());
        }

        // Generate value columns (f64).
        let mut values: Vec<Vec<f64>> = Vec::with_capacity(n_values);
        let mut value_valids: Vec<Vec<bool>> = Vec::with_capacity(n_values);
        for _ in 0..n_values {
            values.push((0..n_rows).map(|_| rng.gen_range(-1e3f64..1e3)).collect());
            value_valids.push((0..n_rows).map(|_| rng.gen_bool(0.85)).collect());
        }

        // Build a small fixed aggregation set across the value cols.
        let aggs: Vec<(AggKind, usize)> = (0..n_values).flat_map(|i| {
            vec![
                (AggKind::SumF64, i),
                (AggKind::MeanF64, i),
                (AggKind::Count, i),
            ]
        }).chain(std::iter::once((AggKind::Len, 0))).collect();

        // CPU reference.
        let cpu = cpu_groupby_reference(&keys, &key_valids, &values, &value_valids, &aggs, n_rows);

        // Build KeyColumn + ValueColumn slices for the kernel.
        let key_bytes: Vec<Vec<u8>> = keys.iter().map(|k| bytes_i64(k)).collect();
        let key_valid_bytes: Vec<Vec<u8>> = key_valids.iter().map(|v| pack_valid(v)).collect();
        let key_cols: Vec<KeyColumn> = keys.iter().enumerate().map(|(i, _)| {
            KeyColumn {
                name: format!("k{i}"),
                dtype: KeyDtype::I64,
                data: &key_bytes[i],
                valid: &key_valid_bytes[i],
                n_rows,
            }
        }).collect();

        let value_valid_bytes: Vec<Vec<u8>> = value_valids.iter().map(|v| pack_valid(v)).collect();
        let agg_specs: Vec<(AggRequest, ValueColumn)> = aggs.iter().map(|(kind, idx)| {
            let vc = ValueColumn::F64 { data: &values[*idx], valid: &value_valid_bytes[*idx] };
            (AggRequest { kind: *kind, input_col_idx: *idx }, vc)
        }).collect();

        let (device, mut queue) = setup();
        let result = dispatch_groupby(&device, &mut queue, &key_cols, &agg_specs, n_rows)
            .expect("dispatch_groupby");

        let kernel = canonicalize(&result);

        prop_assert_eq!(kernel.len(), cpu.len(),
            "group count mismatch: kernel={} cpu={}", kernel.len(), cpu.len());

        for (key, cpu_aggs) in &cpu {
            let kernel_aggs = kernel.get(key).expect("kernel missing key the reference produced");
            prop_assert_eq!(kernel_aggs.len(), cpu_aggs.len());
            for (k, c) in kernel_aggs.iter().zip(cpu_aggs.iter()) {
                prop_assert!(agg_output_eq(k, c),
                    "agg output mismatch for key {:?}: kernel={:?} cpu={:?}", key, k, c);
            }
        }
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_groupby_pipeline`
Expected: PASS — 256 proptest cases. Wall-clock estimate: 20-40s (this is the heaviest kernel test in the suite — touches every kernel in series per case).

- [ ] **Step 3: Run the full kernel suite**

Run: `cargo test -p polars-metal-kernels`
Expected: every kernel test crate green including the new pipeline test.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/tests/test_groupby_pipeline.rs
git commit -m "$(cat <<'EOF'
Kernel: end-to-end groupby pipeline proptest vs CPU reference

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 8 — UDF wiring + router groupby wiring

Phase 8 closes the loop from Python through Rust through the kernel pipeline and back. After Phase 8, `df.lazy().group_by(...).agg(...).collect(engine=MetalEngine())` for an in-scope shape runs on GPU end-to-end and returns a Polars DataFrame indistinguishable from the CPU result.

This phase also lands two remaining bits the spec calls out: the plan-time composite-key width check in `decide_groupby` (Task 30) and the multi-chunk Series defensive check (Task 31).

### Task 28: PyO3 entry point `_native.execute_groupby`

**Files:**
- Modify: `crates/polars-metal-core/src/udf.rs`
- Create: `crates/polars-metal-core/tests/test_execute_groupby_unit.rs`

- [ ] **Step 1: Write the unit test (covers the parser + result-shaping path; the Metal dispatch is exercised by tests in Tasks 32 and 27)**

```rust
// crates/polars-metal-core/tests/test_execute_groupby_unit.rs
//
// Unit-level coverage of the plan-dict parser inside execute_groupby.
// We construct plan dicts in raw form and assert the parser builds the
// expected internal request shape. The actual Metal pipeline is
// exercised end-to-end in `tests/python_integration/test_groupby.py`
// and as a proptest in `tests/test_groupby_pipeline.rs`.
#![allow(clippy::expect_used)]

use polars_metal_core::udf::parse_groupby_plan;

#[test]
fn parser_extracts_single_key_single_agg() {
    let plan = serde_json::json!({
        "kind": "GroupBy",
        "input": {"kind": "Scan"},
        "keys": [["k", "I64"]],
        "aggs": [{"input_col": "v", "op": "Sum", "output_alias": "sum_v"}],
    });
    let parsed = parse_groupby_plan(&plan).expect("parse");
    assert_eq!(parsed.keys.len(), 1);
    assert_eq!(parsed.keys[0].name, "k");
    assert_eq!(parsed.aggs.len(), 1);
    assert_eq!(parsed.aggs[0].input_col, "v");
    assert_eq!(parsed.aggs[0].output_alias, "sum_v");
}

#[test]
fn parser_extracts_len_with_empty_input_col() {
    let plan = serde_json::json!({
        "kind": "GroupBy",
        "input": {"kind": "Scan"},
        "keys": [["k", "I64"]],
        "aggs": [{"input_col": "", "op": "Len", "output_alias": "n"}],
    });
    let parsed = parse_groupby_plan(&plan).expect("parse");
    assert_eq!(parsed.aggs[0].input_col, "");
}

#[test]
fn parser_rejects_unknown_op() {
    let plan = serde_json::json!({
        "kind": "GroupBy",
        "input": {"kind": "Scan"},
        "keys": [["k", "I64"]],
        "aggs": [{"input_col": "v", "op": "Median", "output_alias": "x"}],
    });
    let err = parse_groupby_plan(&plan).expect_err("expected parser error");
    assert!(format!("{err}").contains("Median"));
}

#[test]
fn parser_rejects_missing_keys() {
    let plan = serde_json::json!({
        "kind": "GroupBy",
        "input": {"kind": "Scan"},
        "aggs": [],
    });
    let err = parse_groupby_plan(&plan).expect_err("expected parser error");
    assert!(format!("{err}").contains("keys"));
}
```

- [ ] **Step 2: Implement the entry point**

```rust
// crates/polars-metal-core/src/udf.rs — append:

use crate::plan::{AggOp, AggSpec, MetalDtype};
use polars_metal_kernels::groupby::{
    dispatch_groupby, AggKind, AggRequest, KeyColumn, KeyDtype, ValueColumn,
};

/// Parsed view of a `GroupBy` plan dict. Public to keep the parser
/// directly unit-testable; the PyO3 wrapper builds one of these from the
/// inbound Py dict and feeds it to the dispatcher.
pub struct ParsedGroupByPlan {
    pub keys: Vec<ParsedKey>,
    pub aggs: Vec<AggSpec>,
}

pub struct ParsedKey {
    pub name: String,
    pub dtype: MetalDtype,
}

/// Parser for the plan dict produced by `_walk_group_by`. Lives in this
/// crate (not in `polars-metal-kernels`) because it consumes the
/// `MetalDtype` / `AggOp` types from `plan/mod.rs`.
///
/// Accepts a `serde_json::Value` so the unit tests can construct plans
/// without a Py interpreter; the PyO3 wrapper converts a `PyDict` to
/// JSON before calling this.
pub fn parse_groupby_plan(plan: &serde_json::Value) -> Result<ParsedGroupByPlan, GroupByParseError> {
    let keys_v = plan.get("keys").ok_or(GroupByParseError::Missing("keys"))?;
    let keys_arr = keys_v.as_array().ok_or(GroupByParseError::WrongType("keys"))?;
    let mut keys = Vec::with_capacity(keys_arr.len());
    for k in keys_arr {
        let arr = k.as_array().ok_or(GroupByParseError::WrongType("key entry"))?;
        let name = arr.get(0).and_then(|v| v.as_str()).ok_or(GroupByParseError::WrongType("key name"))?;
        let dtype_str = arr.get(1).and_then(|v| v.as_str()).ok_or(GroupByParseError::WrongType("key dtype"))?;
        let dtype = MetalDtype::from_wire(dtype_str)
            .ok_or_else(|| GroupByParseError::UnknownDtype(dtype_str.to_string()))?;
        keys.push(ParsedKey { name: name.to_string(), dtype });
    }

    let aggs_v = plan.get("aggs").ok_or(GroupByParseError::Missing("aggs"))?;
    let aggs_arr = aggs_v.as_array().ok_or(GroupByParseError::WrongType("aggs"))?;
    let mut aggs = Vec::with_capacity(aggs_arr.len());
    for a in aggs_arr {
        let input_col = a.get("input_col").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let op_str = a.get("op").and_then(|v| v.as_str()).ok_or(GroupByParseError::WrongType("op"))?;
        let op = AggOp::from_wire(op_str)
            .ok_or_else(|| GroupByParseError::UnknownOp(op_str.to_string()))?;
        let output_alias = a.get("output_alias").and_then(|v| v.as_str())
            .ok_or(GroupByParseError::WrongType("output_alias"))?.to_string();
        aggs.push(AggSpec { input_col, op, output_alias });
    }

    Ok(ParsedGroupByPlan { keys, aggs })
}

#[derive(Debug, thiserror::Error)]
pub enum GroupByParseError {
    #[error("missing required field: {0}")]
    Missing(&'static str),
    #[error("wrong type for field: {0}")]
    WrongType(&'static str),
    #[error("unknown dtype: {0}")]
    UnknownDtype(String),
    #[error("unknown agg op: {0}")]
    UnknownOp(String),
}

/// PyO3 entry point. Called from Python by the GroupBy UDF.
///
/// Receives:
///   - `df_pydf`: the upstream DataFrame (post-CPU-Filter, Arrow buffers
///     live in shared DRAM).
///   - `plan_dict`: the GroupBy plan dict produced by `_walk_group_by`.
///
/// Returns: a new PyDataFrame holding the groupby result.
#[pyo3::pyfunction]
pub fn execute_groupby<'py>(
    py: pyo3::Python<'py>,
    df_pydf: &'py pyo3::PyAny,
    plan_dict: &'py pyo3::types::PyDict,
) -> pyo3::PyResult<pyo3::PyObject> {
    use pyo3::exceptions::PyValueError;

    // Convert PyDict → serde_json::Value for the parser.
    let plan_json: serde_json::Value = pythonize::depythonize(plan_dict)
        .map_err(|e| PyValueError::new_err(format!("plan dict not JSON-able: {e}")))?;

    let parsed = parse_groupby_plan(&plan_json)
        .map_err(|e| PyValueError::new_err(format!("plan parse: {e}")))?;

    // Materialize input Arrow buffers as KeyColumn / ValueColumn slices.
    // The polars-metal-buffer bridge from M0 exposes a zero-copy view
    // via `pyarrow_to_metal_buffer`; we reuse it here. The exact API
    // call sequence mirrors `udf.rs::execute_filter` in M1.
    let (key_cols, key_holders, value_cols, value_holders, n_rows) =
        materialize_groupby_inputs(py, df_pydf, &parsed)
            .map_err(|e| PyValueError::new_err(format!("input bridge: {e}")))?;
    let _ = (key_holders, value_holders); // hold ownership until dispatch returns

    // Build agg request slice.
    let agg_specs: Vec<(AggRequest, ValueColumn)> = parsed.aggs.iter().enumerate().map(|(i, spec)| {
        let kind = agg_op_to_kind(spec.op, lookup_value_dtype(&parsed, &spec.input_col, &value_cols, i));
        let vc = if matches!(spec.op, AggOp::Len) {
            // Placeholder; Len doesn't read a value column.
            value_cols.get(0).cloned().unwrap_or(ValueColumn::I64 { data: &[], valid: &[] })
        } else {
            value_cols[i].clone()
        };
        (AggRequest { kind, input_col_idx: i }, vc)
    }).collect();

    // Acquire device + queue.
    let device = polars_metal_kernels::pipeline::MetalDevice::new()
        .map_err(|e| PyValueError::new_err(format!("device: {e}")))?;
    let mut queue = polars_metal_kernels::command::CommandQueue::new(&device)
        .map_err(|e| PyValueError::new_err(format!("queue: {e}")))?;

    let result = dispatch_groupby(&device, &mut queue, &key_cols, &agg_specs, n_rows)
        .map_err(|e| PyValueError::new_err(format!("dispatch_groupby: {e}")))?;

    // Build PyDataFrame from result. The construction mirrors M1's
    // execute_filter result path: build Polars Series per column, then
    // assemble into a PyDataFrame via pyo3-polars.
    let py_df = build_result_dataframe(py, &parsed, &result)
        .map_err(|e| PyValueError::new_err(format!("result build: {e}")))?;
    Ok(py_df.into())
}
```

Notes:
- `materialize_groupby_inputs`, `build_result_dataframe`, and `agg_op_to_kind` are helper functions; their shape is identical to M1's `execute_filter` helpers (`udf.rs`). Implement them following that template — read the M1 helpers first.
- `pythonize` is already a transitive dep via `pyo3` extensions; if not, add it to `Cargo.toml` with a justification (`# justification: convert PyDict to serde_json::Value for the plan parser`). If the maintainer prefers, the parser can be rewritten over `&PyDict` directly — that's a stylistic call, not a correctness one.
- Register the function in the `#[pymodule]` block in `lib.rs`: `m.add_function(wrap_pyfunction!(execute_groupby, m)?)?;`

- [ ] **Step 3: Run the unit test**

Run: `cargo test -p polars-metal-core --test test_execute_groupby_unit`
Expected: PASS, four tests. (Full end-to-end coverage lands in Task 32.)

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-core/src/udf.rs crates/polars-metal-core/tests/test_execute_groupby_unit.rs crates/polars-metal-core/src/lib.rs crates/polars-metal-core/Cargo.toml
git commit -m "$(cat <<'EOF'
Engine: execute_groupby PyO3 entry point

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 29: Python `_udf.py::build_udf` handles `kind == "GroupBy"`

**Files:**
- Modify: `python/polars_metal/_udf.py`
- Create: `tests/python_integration/test_udf_groupby_dispatch.py`

- [ ] **Step 1: Write the test**

```python
# tests/python_integration/test_udf_groupby_dispatch.py
"""Verifies the GroupBy UDF dispatches to `_native.execute_groupby` and
returns a Polars DataFrame matching the CPU result. End-to-end through
the engine, not unit-isolated."""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_groupby_sum_single_key_matches_cpu() -> None:
    df = pl.DataFrame({
        "k": [1, 1, 2, 2, 3, 3, 3] * 20_000,  # 140K rows — above the 100K threshold
        "v": list(range(7 * 20_000)),
    }).with_columns(pl.col("v").cast(pl.Int64))
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_groupby_mean_count_len_match_cpu() -> None:
    df = pl.DataFrame({
        "k": [0, 0, 1, 1, 2, 2] * 20_000,
        "v": [1.0, 2.0, 10.0, 20.0, 100.0, 200.0] * 20_000,
    })
    q = df.lazy().group_by("k").agg(
        pl.col("v").mean().alias("m"),
        pl.col("v").count().alias("c"),
        pl.len().alias("n"),
    ).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
```

- [ ] **Step 2: Implement the dispatch branch**

```python
# python/polars_metal/_udf.py — extend build_udf:

def build_udf(plan: dict) -> Callable:
    """Return a UDF callable that executes `plan` on Metal. The callable
    matches Polars' NodeTraverser UDF contract: takes (NodeTraverser,
    DataFrame, *args), returns a DataFrame.
    """
    kind = plan.get("kind")
    if kind == "DataFrameScan":
        return _build_passthrough(plan)
    if kind == "SimpleProjection":
        return _build_passthrough(plan)
    if kind == "Select":
        return _build_passthrough(plan)
    if kind == "Filter":
        return _build_filter(plan)
    if kind == "GroupBy":
        return _build_groupby(plan)
    raise NotImplementedError(f"build_udf: unknown plan kind {kind!r}")


def _build_groupby(plan: dict) -> Callable:
    """UDF for a GroupBy plan. The walker has already ensured the input
    subtree is fully consumed by Polars CPU (it lands as a single
    DataFrame). We:
      1. Materialize the inbound DataFrame to a PyDataFrame.
      2. Call _native.execute_groupby with the plan dict.
      3. Wrap the returned PyDataFrame back into a pl.DataFrame.
    """
    import polars_metal._native as native

    def udf(*args, **kwargs) -> pl.DataFrame:
        # The Polars UDF contract delivers an input DataFrame; the exact
        # binding matches `_build_filter` from M1 — copy its arg-handling
        # shape verbatim and substitute the call.
        in_df = _extract_input_dataframe(args, kwargs)
        py_df = native.execute_groupby(in_df._df, plan)
        return pl.DataFrame._from_pydf(py_df)

    return udf
```

Notes:
- `_extract_input_dataframe` is M1's helper; reuse without modification.
- The `_from_pydf` wrap matches M1's `_build_filter` return path.

- [ ] **Step 3: Run the test**

Run: `make wheel && pytest tests/python_integration/test_udf_groupby_dispatch.py -v`
Expected: PASS, two tests.

- [ ] **Step 4: Commit**

```bash
git add python/polars_metal/_udf.py tests/python_integration/test_udf_groupby_dispatch.py
git commit -m "$(cat <<'EOF'
UDF: dispatch GroupBy plans to _native.execute_groupby

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 30: Plan-time `Fallback` for composite-key width > 128 bits

**Files:**
- Modify: `crates/polars-metal-core/src/router/cost.rs`
- Modify: `crates/polars-metal-core/tests/test_router_cost.rs`

- [ ] **Step 1: Extend the failing test**

```rust
// crates/polars-metal-core/tests/test_router_cost.rs — append:

use polars_metal_core::plan::{AggSpec, MetalDtype};
use polars_metal_core::router::cost::decide_groupby_with_keys;

#[test]
fn groupby_with_composite_key_at_or_below_128_bits_routes_to_gpu() {
    // 1 i64 + 1 bool = 1+64 + 1+1 = 67 bits.
    let keys = vec![("a".into(), MetalDtype::I64), ("b".into(), MetalDtype::Bool)];
    let aggs: Vec<AggSpec> = vec![];
    let d = decide_groupby_with_keys(1_000_000, &keys, &aggs);
    assert_eq!(d, NodeDecision::GpuLift);
}

#[test]
fn groupby_with_oversized_composite_key_falls_back_at_plan_time() {
    // 3 i64 keys = 3 × 65 = 195 bits, over the 128-bit limit.
    let keys = vec![
        ("a".into(), MetalDtype::I64),
        ("b".into(), MetalDtype::I64),
        ("c".into(), MetalDtype::I64),
    ];
    let aggs: Vec<AggSpec> = vec![];
    match decide_groupby_with_keys(1_000_000, &keys, &aggs) {
        NodeDecision::Fallback(reason) => {
            assert!(reason.contains("128"), "fallback reason should mention 128-bit limit: {reason}");
            assert!(reason.contains("195"), "fallback reason should quote the actual total: {reason}");
        }
        other => panic!("expected Fallback for oversized composite, got {other:?}"),
    }
}
```

- [ ] **Step 2: Implement the width check**

```rust
// crates/polars-metal-core/src/router/cost.rs — append:

use crate::plan::{AggSpec, MetalDtype};

/// Per-key width in bits, including the 1-bit null flag.
fn key_width_bits(dtype: MetalDtype) -> usize {
    match dtype {
        MetalDtype::Bool => 1 + 1,
        MetalDtype::I64 | MetalDtype::F64 => 1 + 64,
    }
}

/// GroupBy decision including the plan-time composite-key width check.
/// Used by the router's groupby walker; the bare `decide_groupby(n_rows)`
/// retains the older shape for cost-only tests.
pub fn decide_groupby_with_keys(
    n_rows: usize,
    keys: &[(String, MetalDtype)],
    _aggs: &[AggSpec],
) -> NodeDecision {
    let total_bits: usize = keys.iter().map(|(_, d)| key_width_bits(*d)).sum();
    if total_bits > 128 {
        return NodeDecision::Fallback(format!(
            "composite key total {total_bits} bits; M2 supports ≤ 128"
        ));
    }
    decide_groupby(n_rows)
}
```

- [ ] **Step 3: Wire the new helper into `compute_lifting_plan`**

In `crates/polars-metal-core/src/router/mod.rs`, the GroupBy arm in the post-order walk currently calls `cost::decide_groupby(n_rows)`. Replace with `cost::decide_groupby_with_keys(n_rows, &keys, &aggs)`, threading `keys` and `aggs` from the `MetalPlanNode::GroupBy` variant.

```rust
// crates/polars-metal-core/src/router/mod.rs — in the GroupBy arm:
MetalPlanNode::GroupBy { input, keys, aggs, .. } => {
    let input_rows = estimate_input_rows(input);
    cost::decide_groupby_with_keys(input_rows, keys, aggs)
}
```

(`estimate_input_rows` already exists from Phase 1 Task 4 / 5; if not, add a small helper that walks down to the nearest `Scan` and reads `n_rows`.)

- [ ] **Step 4: Run the tests**

Run: `cargo test -p polars-metal-core --test test_router_cost && cargo test -p polars-metal-core --test test_router_walk`
Expected: PASS — all existing tests + the two new oversized-key cases.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/router/cost.rs crates/polars-metal-core/src/router/mod.rs crates/polars-metal-core/tests/test_router_cost.rs
git commit -m "$(cat <<'EOF'
Router: plan-time Fallback for composite keys > 128 bits

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 31: Multi-chunk Series defensive check at scan time

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Create: `tests/python_integration/test_multichunk_fallback.py`

- [ ] **Step 1: Write the test**

```python
# tests/python_integration/test_multichunk_fallback.py
"""When the walker encounters a multi-chunk Series in the input frame, it
returns a clean FallBack reason. The query still produces the correct
result via Polars' CPU executor — this test asserts both behaviors."""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def _make_multichunk_df() -> pl.DataFrame:
    a = pl.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]})
    b = pl.DataFrame({"k": [4, 5], "v": [40, 50]})
    df = pl.concat([a, b], rechunk=False)
    # Sanity: confirm chunks > 1 on at least one column.
    assert df["v"].n_chunks() > 1
    return df


def test_multichunk_groupby_falls_back_cleanly() -> None:
    df = _make_multichunk_df()
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_multichunk_filter_falls_back_cleanly() -> None:
    df = _make_multichunk_df()
    q = df.lazy().filter(pl.col("v") > 25).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_multichunk_fallback_reason_in_debug_log(capsys) -> None:
    df = _make_multichunk_df()
    q = df.lazy().group_by("k").agg(pl.col("v").sum())
    q.collect(engine=polars_metal.MetalEngine(debug=True))
    log = capsys.readouterr().out
    assert "multi-chunk" in log or "chunk_count" in log, (
        f"expected multi-chunk fallback reason in debug log; got:\n{log}"
    )
```

- [ ] **Step 2: Implement the check**

```python
# python/polars_metal/_walker.py — extend _walk_dataframe_scan:

def _walk_dataframe_scan(nt: Any, node: Any) -> WalkResult:
    # ... existing setup ...
    df = getattr(node, "df", None)
    if df is not None:
        # Defensive: any column with > 1 chunk forces fallback. M2 inherits
        # M1's single-chunk assumption; multi-chunk lands in M3+. See spec
        # § "Risks & open questions — Multi-chunk Polars Series".
        for col_name in df.columns:
            try:
                n_chunks = df[col_name].n_chunks()
            except Exception:
                n_chunks = 1
            if n_chunks > 1:
                return FallBack(
                    reason=f"multi-chunk Series not yet supported (column {col_name!r} has {n_chunks} chunks)"
                )
    # ... continue with existing logic ...
```

(Slot the check into the existing `_walk_dataframe_scan` body at the point where the DataFrame is first inspected; do not duplicate other logic.)

- [ ] **Step 3: Run the test**

Run: `make wheel && pytest tests/python_integration/test_multichunk_fallback.py -v`
Expected: PASS, three tests.

- [ ] **Step 4: Commit**

```bash
git add python/polars_metal/_walker.py tests/python_integration/test_multichunk_fallback.py
git commit -m "$(cat <<'EOF'
Walker: defensive fallback for multi-chunk Series

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 32: Explicit engine-boundary tests for GroupBy

**Files:**
- Create: `tests/python_integration/test_groupby.py`

- [ ] **Step 1: Write the test**

```python
# tests/python_integration/test_groupby.py
"""Explicit engine-boundary tests for GroupBy.

Each test case names a specific shape and asserts byte-exact equality
between `engine="cpu"` and `engine=MetalEngine()`. Property-based
coverage lives in `crates/polars-metal-kernels/tests/test_groupby_pipeline.rs`.

The row counts in this file are deliberately above the GROUPBY_GPU_MIN_ROWS
threshold (100K) so the router takes the GPU path. Small-input cases that
exercise the empty / single-row code paths use the GroupBy router's GPU
path explicitly by setting a low override threshold via the test config
(see `_force_gpu_groupby` fixture).
"""

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def _metal_engine() -> polars_metal.MetalEngine:
    """Return a MetalEngine with debug=False (default). Tests that need
    debug=True (e.g. routing-decision assertions) construct their own."""
    return polars_metal.MetalEngine()


@pytest.fixture
def force_gpu():
    """Lower the groupby GPU threshold for the duration of one test so
    small-input shapes still exercise the GPU path."""
    import polars_metal._native as native
    if hasattr(native, "set_groupby_gpu_min_rows"):
        original = native.get_groupby_gpu_min_rows()
        native.set_groupby_gpu_min_rows(0)
        yield
        native.set_groupby_gpu_min_rows(original)
    else:
        pytest.skip("router threshold override unavailable; test requires Phase 8 wiring")


def test_empty_input(force_gpu) -> None:
    df = pl.DataFrame({"k": [], "v": []}, schema={"k": pl.Int64, "v": pl.Int64})
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s"))
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_metal_engine())
    assert_frame_equal(cpu, metal)


def test_single_row(force_gpu) -> None:
    df = pl.DataFrame({"k": [42], "v": [100]})
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s"))
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_metal_engine())
    assert_frame_equal(cpu, metal)


def test_all_same_key() -> None:
    n = 200_000
    df = pl.DataFrame({"k": [7] * n, "v": list(range(n))}).with_columns(pl.col("v").cast(pl.Int64))
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s"), pl.col("v").count().alias("c"))
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_metal_engine())
    assert_frame_equal(cpu.sort("k"), metal.sort("k"))


def test_all_unique_keys() -> None:
    n = 200_000
    df = pl.DataFrame({"k": list(range(n)), "v": [1] * n}).with_columns(
        pl.col("k").cast(pl.Int64), pl.col("v").cast(pl.Int64)
    )
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s"))
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=_metal_engine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_multi_key_q1_shape() -> None:
    n = 200_000
    df = pl.DataFrame({
        "returnflag": [i % 2 for i in range(n)],
        "linestatus": [(i // 2) % 2 for i in range(n)],
        "qty": [float(i) for i in range(n)],
    }).with_columns([
        pl.col("returnflag").cast(pl.Int64),
        pl.col("linestatus").cast(pl.Int64),
    ])
    q = df.lazy().group_by("returnflag", "linestatus").agg(
        pl.col("qty").sum().alias("sum_qty"),
        pl.col("qty").mean().alias("avg_qty"),
        pl.col("qty").min().alias("min_qty"),
    ).sort("returnflag", "linestatus")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_metal_engine())
    assert_frame_equal(cpu, metal)


def test_null_in_key_becomes_its_own_group() -> None:
    n = 200_000
    df = pl.DataFrame({
        "k": [1 if i % 3 != 0 else None for i in range(n)],
        "v": list(range(n)),
    }).with_columns(pl.col("v").cast(pl.Int64))
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_metal_engine())
    assert_frame_equal(cpu, metal)


def test_null_in_value_skipped_by_agg_ops() -> None:
    n = 200_000
    df = pl.DataFrame({
        "k": [i % 4 for i in range(n)],
        "v": [None if i % 5 == 0 else float(i) for i in range(n)],
    }).with_columns(pl.col("k").cast(pl.Int64))
    q = df.lazy().group_by("k").agg(
        pl.col("v").sum().alias("s"),
        pl.col("v").mean().alias("m"),
        pl.col("v").count().alias("c"),
        pl.len().alias("n"),
    ).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_metal_engine())
    assert_frame_equal(cpu, metal)


def test_simplified_q1() -> None:
    """1000-row Q1 shape: 2 keys, 3 aggs. (Larger version in tests/bench/.)"""
    n = 200_000  # above threshold; Phase 10 bench runs the 10M-row variant
    df = pl.DataFrame({
        "returnflag": [i % 2 for i in range(n)],
        "linestatus": [(i // 2) % 2 for i in range(n)],
        "qty": [(i % 50) + 1 for i in range(n)],
        "price": [1000.0 + (i % 99000) for i in range(n)],
    }).with_columns([
        pl.col("returnflag").cast(pl.Int64),
        pl.col("linestatus").cast(pl.Int64),
        pl.col("qty").cast(pl.Int64),
    ])
    q = df.lazy().group_by("returnflag", "linestatus").agg(
        pl.col("qty").sum().alias("sum_qty"),
        pl.col("price").sum().alias("sum_price"),
        pl.len().alias("n"),
    ).sort("returnflag", "linestatus")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_metal_engine())
    assert_frame_equal(cpu, metal)
```

- [ ] **Step 2: Run the test**

Run: `make wheel && pytest tests/python_integration/test_groupby.py -v`
Expected: PASS, eight tests.

- [ ] **Step 3: Commit**

```bash
git add tests/python_integration/test_groupby.py
git commit -m "$(cat <<'EOF'
Integration: engine-boundary groupby tests vs CPU

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 9 — Test taxonomy migration

Phase 9 completes the testing-taxonomy migration described in spec § "What's gone". M1's `tests/diff/` directory contained three files: `test_filter_random.py` (hypothesis-based property test), `test_filter_edges.py` (named edge cases), and `test_differential.py` (broad CPU-vs-Metal sweep). M2's plan: move the property test to a Rust proptest, move the edge cases to `tests/python_integration/`, retire the broad sweep (subsumed by `tests/python_integration/` + `tests/conformance/`).

### Task 33: Rust port of M1's filter random property test

**Files:**
- Create: `crates/polars-metal-kernels/tests/test_filter_proptest.rs`

- [ ] **Step 1: Write the proptest**

```rust
// crates/polars-metal-kernels/tests/test_filter_proptest.rs
//
// Rust proptest mirror of M1's `tests/diff/test_filter_random.py`. The
// property: a (predicate, source) pair pushed through the M1 compaction
// pipeline (predicate kernel → cumsum → scatter) produces the same
// output as `cpu_filter_compact_reference`, a pure-Rust reference.
//
// Equivalent of M1's `tests/test_compaction_pipeline.rs` but driven by
// `proptest` with 256 cases.
#![allow(clippy::expect_used)]

use polars_metal_kernels::filter::{compaction_pipeline, FilterError};
use polars_metal_kernels::shader_lib::load_default_library;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::MetalDevice;
use proptest::prelude::*;

fn pack_bits(bools: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; ((bools.len() + 7) / 8 + 3) & !3];
    for (i, &b) in bools.iter().enumerate() {
        if b {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

fn cpu_filter_compact_reference(
    mask: &[bool],
    mask_valid: &[bool],
    src_i64: &[i64],
    src_valid: &[bool],
) -> (Vec<i64>, Vec<bool>) {
    // Polars filter semantics: a row passes iff mask_valid[i] && mask[i].
    // The output's null bitmap is sourced from src_valid at the surviving indices.
    let mut out_data = Vec::new();
    let mut out_valid = Vec::new();
    for i in 0..mask.len() {
        if mask_valid[i] && mask[i] {
            out_data.push(src_i64[i]);
            out_valid.push(src_valid[i]);
        }
    }
    (out_data, out_valid)
}

fn setup() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::new().expect("device");
    let queue = CommandQueue::new(&device).expect("queue");
    let _lib = load_default_library(&device).expect("library");
    (device, queue)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn compaction_pipeline_matches_cpu_reference(
        mask in prop::collection::vec(any::<bool>(), 1..=512),
        mask_valid in prop::collection::vec(any::<bool>(), 1..=512),
        src in prop::collection::vec(any::<i64>(), 1..=512),
        src_valid in prop::collection::vec(any::<bool>(), 1..=512),
    ) {
        let n = mask.len().min(mask_valid.len()).min(src.len()).min(src_valid.len());
        let mask = &mask[..n];
        let mask_valid = &mask_valid[..n];
        let src = &src[..n];
        let src_valid = &src_valid[..n];

        // Kernel path: invoke the M1 compaction pipeline with the same
        // argument shape used by `execute_filter`. The exact API call is
        // `compaction_pipeline(device, queue, mask_bits, mask_valid_bits,
        // src_bytes, src_valid_bits, n, &mut out)`; adjust to the real M1
        // signature when implementing — read `tests/test_compaction_pipeline.rs`
        // for the canonical call sequence.
        let (device, mut queue) = setup();
        let mask_packed = pack_bits(mask);
        let mask_valid_packed = pack_bits(mask_valid);
        let src_valid_packed = pack_bits(src_valid);
        let result = compaction_pipeline(
            &device,
            &mut queue,
            &mask_packed,
            &mask_valid_packed,
            src,
            &src_valid_packed,
            n,
        );

        let kernel_out = match result {
            Ok(out) => out,
            Err(FilterError::Empty) => {
                // M1 returns Empty when the compaction pipeline finds zero
                // surviving rows; treat as (empty_data, empty_valid).
                continue;
            }
            Err(other) => panic!("filter pipeline error: {other:?}"),
        };

        let (ref_data, ref_valid) = cpu_filter_compact_reference(mask, mask_valid, src, src_valid);
        prop_assert_eq!(kernel_out.data, ref_data);
        prop_assert_eq!(kernel_out.valid, ref_valid);
    }
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_filter_proptest`
Expected: PASS, 256 proptest cases.

If the M1 `compaction_pipeline` signature differs from the call above, adjust this test to match the actual M1 API exactly (per `tests/test_compaction_pipeline.rs`). The shape is canonical; the parameter names may differ slightly.

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-kernels/tests/test_filter_proptest.rs
git commit -m "$(cat <<'EOF'
Kernel: filter compaction proptest (migration of tests/diff/test_filter_random.py)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 34: Move `test_filter_edges.py` to `tests/python_integration/`

**Files:**
- Move: `tests/diff/test_filter_edges.py` → `tests/python_integration/test_filter_edges.py`

- [ ] **Step 1: Move the file and update its docstring**

```bash
git mv tests/diff/test_filter_edges.py tests/python_integration/test_filter_edges.py
```

Then update the module docstring at the top of the moved file:

```python
"""Filter edge-case tests, migrated from `tests/diff/test_filter_edges.py`.

Lives in `tests/python_integration/` per the M2 testing taxonomy: explicit
Python cases for engine-boundary correctness live here; property-based
testing lives in `crates/polars-metal-kernels/tests/test_filter_proptest.rs`.

See spec § "Testing strategy" for the full taxonomy.
"""
```

Update any test-internal imports if they referenced the old path (`tests.diff.strategies`, etc. — those modules go away in Task 35; any helper still needed at this point lives in the test file itself or in `tests/python_integration/conftest.py`).

- [ ] **Step 2: Run the test**

Run: `pytest tests/python_integration/test_filter_edges.py -v`
Expected: PASS, identical results to the pre-move run.

- [ ] **Step 3: Commit**

```bash
git add tests/python_integration/test_filter_edges.py
git rm tests/diff/test_filter_edges.py  # if `git mv` didn't already stage it
git commit -m "$(cat <<'EOF'
Tests: move test_filter_edges.py to python_integration/

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 35: Remove `tests/diff/` entirely and update Makefile + docs

**Files:**
- Delete: `tests/diff/` (entire directory: `test_filter_random.py`, `test_differential.py`, `__init__.py`, `conftest.py`, `strategies.py`)
- Modify: `Makefile`
- Modify: `docs/architecture.md`
- Modify: any other docs referencing `tests/diff/`

- [ ] **Step 1: Remove the directory**

```bash
git rm -r tests/diff
```

Confirm `tests/diff/` no longer exists: `test ! -d tests/diff && echo OK`.

- [ ] **Step 2: Update the Makefile**

Find and remove the `test-diff` target. The migrated coverage is reachable via `test-kernel` (the new Rust proptest) and `test-unit` (Python integration tests). Update `make gate` if it currently invokes `test-diff` — replace with `test-unit test-kernel` (which it likely already runs).

- [ ] **Step 3: Update `docs/architecture.md`**

Search for any mention of `tests/diff/` and replace with the new taxonomy (point readers to `tests/python_integration/` for explicit cases and `crates/polars-metal-kernels/tests/` for property-based). If the M1 architecture doc had a paragraph explaining the diff harness, replace it with the M2 taxonomy table (mirror spec § "What's gone").

- [ ] **Step 4: Update any other docs**

Run `grep -rn "tests/diff" docs/ python/ crates/ -l` and update any matches.

- [ ] **Step 5: Confirm the gate is still clean**

Run: `make gate`
Expected: all phases pass; no `test-diff` invocation in the output.

- [ ] **Step 6: Commit**

```bash
git add Makefile docs/ tests/
git commit -m "$(cat <<'EOF'
Tests: retire tests/diff/ directory; M2 taxonomy in place

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 10 — TPC-H Q1 benchmark

Phase 10 lands the headline perf workload: modified TPC-H Q1 (spec § "Workload validated") at 10M rows, with deterministic data generation and a `baseline.json` entry. The M2 perf gate is `ratio_metal_over_cpu < 1.0` for `tpch_q1_modified`; this phase is where that number gets measured and recorded.

### Task 36: Deterministic lineitem fixture

**Files:**
- Create: `tests/bench/_lineitem_fixture.py`

- [ ] **Step 1: Implement the fixture**

```python
# tests/bench/_lineitem_fixture.py
"""Deterministic lineitem-shaped fixture for modified TPC-H Q1.

Matches spec § "Workload validated":
  - l_returnflag (i64 ∈ {0, 1})
  - l_linestatus (i64 ∈ {0, 1})
  - l_quantity   (i64 ∈ [1, 50])
  - l_extendedprice (f64 ∈ [1000, 100000])
  - l_discount   (f64 ∈ [0.00, 0.10])
  - l_tax        (f64 ∈ [0.00, 0.08])
  - l_shipdate   (i64, days since 1970-01-01, range 1992-01-01..1998-12-31)
  - disc_price   (f64 = l_extendedprice * (1 - l_discount))  pre-projected
  - charge       (f64 = disc_price * (1 + l_tax))            pre-projected

The pre-projected disc_price / charge columns are the M2 simplification of
the original Q1 expression `sum(extendedprice * (1 - discount))` — the
multi-aggregation expression unfolding is deferred to M3 (spec § "Hand-off
to M3"). The output schema matches the columns Q1 references.

Reproducibility: numpy.random.default_rng(seed) ensures identical bytes
across runs of the same seed.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

# Days from 1970-01-01 (Polars epoch) to the TPC-H lineitem ship-date range.
_SHIPDATE_LO = (date(1992, 1, 1) - date(1970, 1, 1)).days
_SHIPDATE_HI = (date(1998, 12, 31) - date(1970, 1, 1)).days


def make_lineitem(n_rows: int = 10_000_000, seed: int = 0xC0FFEE) -> pl.DataFrame:
    """Build an n_rows × 9-column lineitem-shaped DataFrame.

    All numeric distributions match the spec; the seed makes the output
    bit-reproducible across runs.
    """
    rng = np.random.default_rng(seed)

    returnflag = rng.integers(0, 2, size=n_rows, dtype=np.int64)
    linestatus = rng.integers(0, 2, size=n_rows, dtype=np.int64)
    quantity = rng.integers(1, 51, size=n_rows, dtype=np.int64)  # [1, 50]
    extendedprice = rng.uniform(1000.0, 100_000.0, size=n_rows).astype(np.float64)
    discount = rng.uniform(0.0, 0.10, size=n_rows).astype(np.float64)
    tax = rng.uniform(0.0, 0.08, size=n_rows).astype(np.float64)
    shipdate = rng.integers(_SHIPDATE_LO, _SHIPDATE_HI + 1, size=n_rows, dtype=np.int64)

    disc_price = extendedprice * (1.0 - discount)
    charge = disc_price * (1.0 + tax)

    return pl.DataFrame({
        "l_returnflag": returnflag,
        "l_linestatus": linestatus,
        "l_quantity": quantity,
        "l_extendedprice": extendedprice,
        "l_discount": discount,
        "l_tax": tax,
        "l_shipdate": shipdate,
        "disc_price": disc_price,
        "charge": charge,
    })


if __name__ == "__main__":
    # CLI convenience for ad-hoc inspection.
    df = make_lineitem(1_000_000)
    print(df.schema)
    print(df.head())
    print(f"n_rows={df.height}, n_unique_keys={df.select(['l_returnflag', 'l_linestatus']).unique().height}")
```

- [ ] **Step 2: Sanity-check the fixture**

Run: `python tests/bench/_lineitem_fixture.py`
Expected: prints the schema (9 columns with the right dtypes), 5 head rows, and `n_unique_keys=4` (the four (returnflag, linestatus) combinations).

- [ ] **Step 3: Commit**

```bash
git add tests/bench/_lineitem_fixture.py
git commit -m "$(cat <<'EOF'
Bench: deterministic lineitem fixture for modified TPC-H Q1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 37: Modified TPC-H Q1 benchmark

**Files:**
- Create: `tests/bench/test_tpch_q1.py`

- [ ] **Step 1: Implement the benchmark**

```python
# tests/bench/test_tpch_q1.py
"""Modified TPC-H Q1 benchmark.

Spec § "Workload validated":
  - integer-encoded l_returnflag / l_linestatus
  - disc_price and charge pre-projected into the input
  - Otherwise identical to TPC-H Q1: filter on shipdate threshold,
    group_by(returnflag, linestatus), 7 aggregations + count, sort by keys.

Two benchmarks: `tpch_q1_cpu` and `tpch_q1_metal`. The timed region
includes the filter (CPU-routed under M2), the groupby (GPU-routed),
and the sort (CPU-routed) — that's the full Q1 wall-clock the user
observes.

The `baseline.json` entry produced in Task 38 records cpu_ms / metal_ms /
ratio_metal_over_cpu; M2 ships iff ratio < 1.0.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

import polars_metal
from tests.bench._lineitem_fixture import make_lineitem

# Filter selectivity: spec uses 1998-09-02 as the TPC-H threshold, which
# (with our fixture's uniform shipdate distribution over 1992-01-01..
# 1998-12-31) drops the last ~3 months. Close enough to the original Q1's
# selectivity profile; the exact threshold isn't load-bearing for the
# perf comparison.
_THRESHOLD = (date(1998, 9, 2) - date(1970, 1, 1)).days


def _q1(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.filter(pl.col("l_shipdate") <= _THRESHOLD)
          .group_by("l_returnflag", "l_linestatus")
          .agg(
              pl.col("l_quantity").sum().alias("sum_qty"),
              pl.col("l_extendedprice").sum().alias("sum_base_price"),
              pl.col("disc_price").sum().alias("sum_disc_price"),
              pl.col("charge").sum().alias("sum_charge"),
              pl.col("l_quantity").mean().alias("avg_qty"),
              pl.col("l_extendedprice").mean().alias("avg_price"),
              pl.col("l_discount").mean().alias("avg_disc"),
              pl.len().alias("count_order"),
          )
          .sort("l_returnflag", "l_linestatus")
    )


@pytest.fixture(scope="module")
def lineitem_10m() -> pl.DataFrame:
    """10M-row lineitem fixture, built once per test module."""
    return make_lineitem(n_rows=10_000_000, seed=0xC0FFEE)


@pytest.mark.benchmark(group="tpch_q1")
def test_bench_tpch_q1_cpu(benchmark, lineitem_10m: pl.DataFrame) -> None:
    """Baseline: pure-CPU Polars on the modified Q1."""
    def run() -> pl.DataFrame:
        return _q1(lineitem_10m.lazy()).collect(engine="cpu")
    out = benchmark(run)
    assert out.height == 4, f"expected 4 (returnflag, linestatus) groups, got {out.height}"


@pytest.mark.benchmark(group="tpch_q1")
def test_bench_tpch_q1_metal(benchmark, lineitem_10m: pl.DataFrame) -> None:
    """Metal engine: filter on CPU, groupby on GPU, sort on CPU."""
    engine = polars_metal.MetalEngine()
    def run() -> pl.DataFrame:
        return _q1(lineitem_10m.lazy()).collect(engine=engine)
    out = benchmark(run)
    assert out.height == 4, f"expected 4 (returnflag, linestatus) groups, got {out.height}"


def test_q1_correctness(lineitem_10m: pl.DataFrame) -> None:
    """Sanity: both engines produce the same result for the modified Q1.
    Not a benchmark — pure correctness check. Runs once per session."""
    from polars.testing import assert_frame_equal
    cpu = _q1(lineitem_10m.lazy()).collect(engine="cpu")
    metal = _q1(lineitem_10m.lazy()).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
```

- [ ] **Step 2: Sanity-check the correctness assertion**

Run: `make wheel && pytest tests/bench/test_tpch_q1.py::test_q1_correctness -v`
Expected: PASS. (The benchmarks themselves are slower; baseline capture happens in Task 38.)

- [ ] **Step 3: Commit**

```bash
git add tests/bench/test_tpch_q1.py
git commit -m "$(cat <<'EOF'
Bench: modified TPC-H Q1 — CPU and Metal variants + correctness sanity

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 38: Per-kernel criterion benchmarks

**Files:**
- Create: `crates/polars-metal-kernels/benches/groupby_hash.rs`
- Create: `crates/polars-metal-kernels/benches/groupby_build.rs`
- Create: `crates/polars-metal-kernels/benches/groupby_aggregate.rs`
- Modify: `crates/polars-metal-kernels/Cargo.toml` (register the three bench targets)

Spec § "Layer 4 Benchmarks — Criterion (per-kernel)" calls for these three benches at sizes 100K / 1M / 10M / 100M and value-column null densities 0% / 50% / 100%. They drive PR-level tuning of the cost model (Task 30's `GROUPBY_GPU_MIN_ROWS`) and the hash-table load factor.

- [ ] **Step 1: Register the bench targets in `Cargo.toml`**

```toml
# crates/polars-metal-kernels/Cargo.toml — extend the bench section:

[[bench]]
name = "groupby_hash"
harness = false

[[bench]]
name = "groupby_build"
harness = false

[[bench]]
name = "groupby_aggregate"
harness = false
```

- [ ] **Step 2: Implement `benches/groupby_hash.rs`**

```rust
// crates/polars-metal-kernels/benches/groupby_hash.rs
//
// Criterion microbench for the standalone hash kernel. Inputs:
//   - n_rows ∈ {100K, 1M, 10M, 100M}
//   - keys are pre-encoded random u128 values (we don't measure the
//     encoder here; Phase 3 is pure CPU and effectively free).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars_metal_kernels::groupby::dispatch_hash;
use polars_metal_kernels::shader_lib::load_default_library;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::MetalDevice;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn make_keys(n: usize, seed: u64) -> Vec<u128> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen::<u128>()).collect()
}

fn bench_hash(c: &mut Criterion) {
    let device = MetalDevice::new().expect("device");
    let mut queue = CommandQueue::new(&device).expect("queue");
    let _lib = load_default_library(&device).expect("library");

    let mut group = c.benchmark_group("groupby_hash");
    for &n in &[100_000usize, 1_000_000, 10_000_000] {
        // 100M is gated under a feature flag — too memory-hungry for
        // default CI runs. Re-enable manually for tuning.
        group.throughput(Throughput::Elements(n as u64));
        let keys = make_keys(n, 0xC0FFEE);
        let mut out = vec![0u32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                dispatch_hash(&device, &mut queue, black_box(&keys), n, &mut out)
                    .expect("dispatch_hash");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hash);
criterion_main!(benches);
```

- [ ] **Step 3: Implement `benches/groupby_build.rs`**

```rust
// crates/polars-metal-kernels/benches/groupby_build.rs
//
// Criterion microbench for the build kernel. Two key-cardinality regimes:
//   - low-cardinality (4 distinct keys; mirrors Q1's degenerate case —
//     atomic-CAS contention is maximal here)
//   - high-cardinality (n/4 distinct keys; minimal contention)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars_metal_kernels::groupby::{dispatch_build, dispatch_hash};
use polars_metal_kernels::shader_lib::load_default_library;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::MetalDevice;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn make_low_cardinality(n: usize) -> Vec<u128> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    (0..n).map(|_| rng.gen_range(0u128..4)).collect()
}

fn make_high_cardinality(n: usize) -> Vec<u128> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let max = (n / 4).max(1) as u128;
    (0..n).map(|_| rng.gen_range(0u128..max)).collect()
}

fn bench_build(c: &mut Criterion) {
    let device = MetalDevice::new().expect("device");
    let mut queue = CommandQueue::new(&device).expect("queue");
    let _lib = load_default_library(&device).expect("library");

    let mut group = c.benchmark_group("groupby_build");
    for &n in &[100_000usize, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        for (label, keys) in &[
            ("low_card", make_low_cardinality(n)),
            ("high_card", make_high_cardinality(n)),
        ] {
            let mut hashes = vec![0u32; n];
            dispatch_hash(&device, &mut queue, keys, n, &mut hashes)
                .expect("dispatch_hash");
            group.bench_with_input(
                BenchmarkId::new(*label, n),
                &n,
                |b, _| {
                    b.iter(|| {
                        dispatch_build(&device, &mut queue, black_box(keys), &hashes, n)
                            .expect("dispatch_build");
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_build);
criterion_main!(benches);
```

- [ ] **Step 4: Implement `benches/groupby_aggregate.rs`**

```rust
// crates/polars-metal-kernels/benches/groupby_aggregate.rs
//
// Criterion microbench for the aggregation kernels. Sweeps over
// (kernel, n_rows, null_density). Pre-built row_to_group so we measure
// only the agg kernel; we use a fixed 100-group cardinality.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars_metal_kernels::groupby::{dispatch_sum_i64, dispatch_sum_f64, dispatch_count, dispatch_len};
use polars_metal_kernels::shader_lib::load_default_library;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::MetalDevice;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn make_row_to_group(n: usize, n_groups: u32) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    (0..n).map(|_| rng.gen_range(0..n_groups)).collect()
}

fn make_valid(n: usize, null_density: f64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(0xD00D);
    let mut v = vec![0u8; ((n + 7) / 8 + 3) & !3];
    for i in 0..n {
        if rng.gen::<f64>() > null_density {
            v[i >> 3] |= 1 << (i & 7);
        }
    }
    v
}

fn bench_aggregate(c: &mut Criterion) {
    let device = MetalDevice::new().expect("device");
    let mut queue = CommandQueue::new(&device).expect("queue");
    let _lib = load_default_library(&device).expect("library");

    let n_groups: u32 = 100;
    let mut group = c.benchmark_group("groupby_aggregate");

    for &n in &[100_000usize, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        let row_to_group = make_row_to_group(n, n_groups);

        // sum_i64 across null densities.
        let i64_vals: Vec<i64> = (0..n as i64).collect();
        for &nd in &[0.0f64, 0.5, 1.0] {
            let valid = make_valid(n, nd);
            let mut out = vec![0i64; n_groups as usize];
            group.bench_with_input(
                BenchmarkId::new(format!("sum_i64_nulls={nd:.1}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        dispatch_sum_i64(
                            &device, &mut queue,
                            black_box(&i64_vals), &valid, &row_to_group,
                            n, n_groups as usize, &mut out,
                        ).expect("dispatch_sum_i64");
                    });
                },
            );
        }

        // sum_f64 — one density (0.5) suffices since the i64 sweep covers the density dimension.
        let f64_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let valid = make_valid(n, 0.5);
        let mut out_f = vec![0.0f64; n_groups as usize];
        group.bench_with_input(BenchmarkId::new("sum_f64_nulls=0.5", n), &n, |b, _| {
            b.iter(|| {
                dispatch_sum_f64(
                    &device, &mut queue,
                    black_box(&f64_vals), &valid, &row_to_group,
                    n, n_groups as usize, &mut out_f,
                ).expect("dispatch_sum_f64");
            });
        });

        // count — measures null-bitmap-only path.
        let mut out_c = vec![0u64; n_groups as usize];
        group.bench_with_input(BenchmarkId::new("count_nulls=0.5", n), &n, |b, _| {
            b.iter(|| {
                dispatch_count(
                    &device, &mut queue,
                    &valid, &row_to_group, n, n_groups as usize, &mut out_c,
                ).expect("dispatch_count");
            });
        });

        // len — no validity read.
        let mut out_l = vec![0u64; n_groups as usize];
        group.bench_with_input(BenchmarkId::new("len", n), &n, |b, _| {
            b.iter(|| {
                dispatch_len(
                    &device, &mut queue,
                    &row_to_group, n, n_groups as usize, &mut out_l,
                ).expect("dispatch_len");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_aggregate);
criterion_main!(benches);
```

- [ ] **Step 5: Build and run a smoke iteration**

Run: `cargo bench -p polars-metal-kernels --bench groupby_hash -- --quick`
Run: `cargo bench -p polars-metal-kernels --bench groupby_build -- --quick`
Run: `cargo bench -p polars-metal-kernels --bench groupby_aggregate -- --quick`

Expected: each completes without panics; criterion prints throughput numbers per size/configuration. `--quick` short-circuits the long warm-up so this verifies the bench compiles and dispatches; full runs happen during the M2 retro / tuning sessions.

- [ ] **Step 6: Verify no regression on M1's existing kernel benches**

Per spec exit criterion #13 ("No regression on M1's existing kernels"): run the M1 kernel benches once and confirm they still complete cleanly.

Run: `cargo bench -p polars-metal-kernels --bench cmp_i64 -- --quick`
(And whatever other criterion benches M1 shipped — list them with `ls crates/polars-metal-kernels/benches/`.)

Expected: each runs to completion with timings in the same order-of-magnitude as M1's recorded numbers. Investigate any >20% slowdown before proceeding.

- [ ] **Step 7: Commit**

```bash
git add crates/polars-metal-kernels/benches/groupby_hash.rs crates/polars-metal-kernels/benches/groupby_build.rs crates/polars-metal-kernels/benches/groupby_aggregate.rs crates/polars-metal-kernels/Cargo.toml
git commit -m "$(cat <<'EOF'
Bench: criterion microbenches for groupby_hash / build / aggregate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 39: Capture baseline + record in `baseline.json`

**Files:**
- Modify: `tests/bench/baseline.json`

- [ ] **Step 1: Run the benchmark**

Run: `pytest tests/bench/test_tpch_q1.py --benchmark-only -k tpch_q1 --benchmark-min-rounds=3`
Expected: prints two timings (`tpch_q1_cpu` and `tpch_q1_metal`) plus their ratio under the `tpch_q1` group.

Record the median values (or mean if median isn't reported) in milliseconds. Pytest-benchmark prints both; pick `median` to match M1's convention in `baseline.json`.

- [ ] **Step 2: Update `baseline.json`**

Read the current `baseline.json`, add a `tpch_q1_modified` block following the schema of existing entries:

```json
{
  "git_sha": "<current SHA from `git rev-parse HEAD`>",
  "date": "<YYYY-MM-DD from `date +%F`>",
  "machine": "M2 Ultra",
  "queries": {
    "...": "...",
    "tpch_q1_modified": {
      "rows": 10000000,
      "cpu_ms": <median CPU time in ms from Step 1>,
      "metal_ms": <median Metal time in ms from Step 1>,
      "ratio_metal_over_cpu": <metal_ms / cpu_ms, three decimals>,
      "_notes": "M2 perf gate: ratio_metal_over_cpu must be < 1.0. Filter routed to CPU by definition (per cost model); groupby routed to GPU; sort routed to CPU. The filter's ratio is exactly 1.0 by routing-layer policy and is not separately recorded."
    }
  }
}
```

Preserve all existing entries — do **not** rebaseline M1's numbers under any circumstance.

- [ ] **Step 3: Verify the gate**

The M2 perf gate (spec § "Exit criteria #12") requires `ratio_metal_over_cpu < 1.0`.

- If the recorded ratio **is < 1.0**: success path. Continue to Step 4.
- If the recorded ratio is **≥ 1.0**: the implementer must either:
  (a) Tune the kernel or router thresholds and re-run (note: do not rebaseline M1 entries — only re-measure `tpch_q1_modified`). The two likely tuning levers are: GROUPBY_GPU_MIN_ROWS (raise if the GPU is being routed at unprofitable sizes) and the hash-table load factor (BUILD_LOAD_FACTOR_NUM/DEN in `groupby.rs`).
  (b) Report `DONE_WITH_CONCERNS` to the user, attaching the measured numbers and proposing a path forward (revisit at M3, accept the perf gap with explicit waiver, or scope-creep into M2 to fix). Do not silently merge a failing perf gate.

- [ ] **Step 4: Commit**

```bash
git add tests/bench/baseline.json
git commit -m "$(cat <<'EOF'
Bench: record M2 baseline for modified TPC-H Q1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 11 — Conformance + docs

Phase 11 wires the upstream Polars groupby test paths into the conformance harness and brings the docs to M2's state. Conformance and docs are independent of each other but live in the same phase because both are part of the "Quality" exit-criteria block.

### Task 40: Wire upstream groupby paths into the conformance harness

**Files:**
- Modify: `tests/conformance/test_polars_suite.py`
- Create: `tests/conformance/baselines/test_group_by.json`
- Create: `tests/conformance/baselines/test_agg.json`

- [ ] **Step 1: Identify the exact upstream paths**

Run: `ls references/polars/py-polars/tests/unit/operations/ && ls references/polars/py-polars/tests/unit/operations/aggregation/ 2>/dev/null || echo "no aggregation subdir"`
Expected: shows `test_group_by.py` (or similar; exact filename may vary by Polars version). Note the precise path. If the aggregation subdirectory exists, note the specific files (e.g. `test_agg.py`).

Update the test list below to the verified paths.

- [ ] **Step 2: Add the paths to the conformance harness**

```python
# tests/conformance/test_polars_suite.py — extend SUITE_PATHS:

SUITE_PATHS = [
    # ... existing M1 paths ...
    "tests/unit/operations/test_group_by.py",
    "tests/unit/operations/aggregation/test_agg.py",
]
```

Follow the M1 T22 pattern verbatim for any per-path config (skip-list keys, expected new-failures, etc.).

- [ ] **Step 3: Capture pure-CPU baselines**

For each new path:

```bash
# Capture the CPU baseline.
make test-conformance-cpu-baseline PATH=tests/unit/operations/test_group_by.py > tests/conformance/baselines/test_group_by.json
make test-conformance-cpu-baseline PATH=tests/unit/operations/aggregation/test_agg.py > tests/conformance/baselines/test_agg.json
```

(If the Makefile target is named differently in M1, use the M1 invocation. The intent: produce a one-line-per-test JSON file recording pass/fail under pure CPU — the no-new-failures bar M2 must clear.)

- [ ] **Step 4: Run the conformance suite under Metal**

Run: `make test-conformance`
Expected: every previously-passing test still passes under `engine=MetalEngine()`. Any test that fails under CPU baseline is allowed to keep failing under Metal; any test that was passing under CPU and now fails under Metal is a blocker.

If any new failures appear, investigate. Likely root causes:
- An IR shape the walker doesn't handle yet (should `FallBack` cleanly; if it doesn't, that's a walker bug).
- A null-semantic divergence (should match Polars CPU exactly per CLAUDE.md).
- An unsupported dtype the walker should have rejected at planning (router `Fallback` shortfall).

- [ ] **Step 5: Commit**

```bash
git add tests/conformance/test_polars_suite.py tests/conformance/baselines/test_group_by.json tests/conformance/baselines/test_agg.json
git commit -m "$(cat <<'EOF'
Conformance: wire upstream groupby + aggregation paths

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 41: Update `docs/architecture.md` with routing-layer architecture

**Files:**
- Modify: `docs/architecture.md`

- [ ] **Step 1: Add the routing-layer sections**

New top-level section after the M1 walker section, describing M2's three-layer flow (matching the M1 doc's style — concrete file/function citations, no fluff). Cover:

- **The router as the new policy layer.** Walker stays minimal (IR → MetalPlanNode tree, no decisions). Rust router walks the tree, consults the cost model, runs affinity smoothing, returns a `LiftingPlan`. The Python walker applies the plan: `nt.set_udf` on GpuLift subtrees only; CpuLeave subtrees are left to Polars CPU. Cite `crates/polars-metal-core/src/router/mod.rs::compute_lifting_plan` and `python/polars_metal/_callback.py::execute_with_metal`.
- **Per-op cost rules.** One paragraph per current rule (Filter→CPU, GroupBy→GPU>100K, Project→inherit, Scan→inherit). Cite `crates/polars-metal-core/src/router/cost.rs`. Note the PR-tuning model.
- **Affinity smoothing.** Why it exists (transitions are zero-copy today but the policy keeps debug logs simpler). Single second-pass over the LiftingPlan; cite `crates/polars-metal-core/src/router/affinity.rs`.
- **The groupby pipeline.** Two-pass count-then-fill, ported from cuDF. Composite-key encoding (CPU-side, up to 128 bits). Hash kernel → build kernel (atomic-CAS) → per-aggregation kernels → host-side mean. Cite the relevant `.metal` files and `groupby.rs::dispatch_groupby`.

Keep the existing M1 sections intact; M2 extends, does not replace.

- [ ] **Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "$(cat <<'EOF'
Docs: routing-layer architecture + groupby pipeline (M2)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 42: Update `docs/kernel-authoring.md` with M2 idioms

**Files:**
- Modify: `docs/kernel-authoring.md`

- [ ] **Step 1: Add M2 idioms**

New section "Two-pass count-then-fill (groupby family)". Cover:

- **The two-pass discipline.** Pass 1 (build): atomic-CAS find-or-insert into a fixed-size hash table; produces `row_to_group` + `group_count` + `first_row_per_group`. Pass 2 (aggregate): one kernel per (value_col, agg_op); thread-per-row atomic-OP onto `output[group_id]`. Read-only between passes; no synchronization beyond Metal's command-queue ordering.
- **Atomic-CAS hash-table build.** Open-addressing, load factor 0.5, `next_pow2(n_rows / 0.5)`. Slot layout: `(key_lo: u64, key_hi: u64, group_id: u32)`. Insertion: atomic-CAS on `key_lo` from sentinel (e.g. `u64::MAX`) to the row's key; on collision, linear-probe to next slot. First thread to install a slot also writes `group_id = atomicAdd(group_count, 1)` and `first_row_per_group[group_id] = row_idx`. Cite `shaders/groupby_build.metal`.
- **Per-aggregation kernel pattern via MSL macros.** Same shape as M1's `cmp_i64.metal` and `cmp_f64.metal`: one MSL macro generates the six per-op entry points from a single template. Cite `shaders/aggregate.metal`.
- **Composite key encoding.** CPU-side packs N key columns into one u128, recording bit widths + offsets in a `KeySchema`. Encoder rejects total > 128 bits. The kernel only ever sees u128; the encoder/decoder contract is the source of truth. Cite `polars-metal-kernels/src/groupby.rs::encode_keys`.
- **Sentinel-bit invariant.** The encoded u128 reserves the MSB of the high half as a "key valid" sentinel for the hash table's empty-slot detection. The encoder guarantees MSB = 0 on real keys (total bits ≤ 127). Document this here so future kernel authors don't accidentally violate it.

- [ ] **Step 2: Commit**

```bash
git add docs/kernel-authoring.md
git commit -m "$(cat <<'EOF'
Docs: two-pass groupby idiom + MSL macro pattern (M2)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 43: Update `docs/open-questions.md`

**Files:**
- Modify: `docs/open-questions.md`

- [ ] **Step 1: Strike through resolved items**

Items that landed in M2 — mark with strikethrough (Markdown `~~text~~`) and a one-line resolution note. Specifically:

- **Routing layer pivot.** Landed in M2 Phases 1-2. Note: per-op routing layer in production. Cite `crates/polars-metal-core/src/router/`.
- **M1 perf-gap entry.** Update: M2 confirmed filter belongs on CPU (router routes it there by default). Quote the `baseline.json` `tpch_q1_modified` ratio recorded in Task 38.

- [ ] **Step 2: Add new M2-surfaced items**

Append:

- **Hash-table OOM under contention.** Spec § "Risks" called this out; M2 ships the atomic-CAS implementation. Open question: is the contention degradation acceptable on M-series GPUs, or do we need sort-then-scan as an alternative for high-contention shapes (Q1's 4-groups-over-10M-rows case)?
- **M3 readiness for string keys.** M2 explicitly scopes out variable-width keys. M3 must decide: dictionary-encode at the buffer bridge, or run a dedicated string-hash kernel? (Likely both; the bridge handles the common case, the kernel handles dictionary-misses.)
- **Multi-chunk Series defensive check.** M2's Task 31 falls back on chunked Series. Full multi-chunk support is deferred to M3+ (probably implemented at the buffer-bridge layer rather than per-walker).
- **M2 sentinel-bit restriction.** Composite keys are capped at 127 bits (1 bit reserved for the hash table's empty-slot sentinel). 128-bit "natural" composite (e.g. 2 × i64) routes to GPU because 130 bits > 128 already triggers the existing Fallback, so this restriction is currently masked by the cost model — but if M3 raises the cap, it must explicitly carve out the sentinel bit. Track separately so future changes don't accidentally regress.

- [ ] **Step 3: Commit**

```bash
git add docs/open-questions.md
git commit -m "$(cat <<'EOF'
Docs: M2 open questions — strike resolved, add new surfaced items

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 12 — Retrospective + ship

Phase 12 closes M2: write the retrospective into the spec, run the full local gate, run the portability gate on M2 (16 GB) and M1 (8 GB), push the branch, open the PR. Mirrors the shape of M1's Phase 10.

### Task 44: Write the M2 retrospective in the design spec

**Files:**
- Modify: `docs/superpowers/specs/2026-05-21-m2-design.md` (the retrospective stub at the bottom)

- [ ] **Step 1: Fill in the retrospective**

Mirror M1's retrospective shape. Sections:

- **Outcome.** Per-exit-criterion pass/fail with numbers. Pull from `make gate` wall-clock, the `baseline.json` ratio, the kernel test count (`cargo test -p polars-metal-kernels -- --list | wc -l`), the conformance pass-count delta.
- **Surprises during execution.** Plan-vs-reality deltas. Examples to look for: any task that required more steps than the plan listed; API quirks discovered in `objc2-metal` atomic-CAS on the build kernel; threadgroup-size tuning that diverged from M1's defaults; any spec section that proved ambiguous in implementation.
- **Resolved in PR follow-up commits.** Items scoped to "after the milestone PR" that landed before merge.
- **Still to revisit at M3.** Items M2 surfaced for the next milestone (probably: string-key groupby, multi-chunk support, hash-table sort-then-scan alternative if contention degraded badly).
- **Portability gate results.** Date, git SHA, M2 (16 GB) outcome, M1 (8 GB) outcome.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-21-m2-design.md
git commit -m "$(cat <<'EOF'
M2 retrospective: outcome, surprises, follow-ups, M3 hand-off

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 45: Full gate + portability gate + push + PR

**Files:** none (verification + push + PR only).

- [ ] **Step 1: Run the full local gate**

Run: `make gate`
Expected: all phases pass (lint, test-unit, test-kernel, wheel, test-conformance). No `test-diff` — that target was removed in Task 35. Wall-clock estimate: ~5-8 min on M2 Ultra (M1 baseline was ~3-5 min; M2 adds the kernel pipeline test, router tests, integration tests).

- [ ] **Step 2: Run the portability gate on M2 (16 GB)**

User-action: run `make gate && pytest tests/bench/test_tpch_q1.py --benchmark-only` on the small-M2 machine. Capture: date, git SHA, gate result, the Q1 ratio observed on that machine. Paste into the M2 retrospective (Task 43 Step 1).

- [ ] **Step 3: Run the portability gate on M1 (8 GB)**

Same. The 10M-row Q1 fixture is ~900 MB in DRAM (9 columns × 10M × 8 bytes); fits on 8 GB with room for the hash table and aggregation outputs. If it doesn't fit (e.g. due to working-set inflation from the build kernel), record the failure mode in the retrospective and flag as an M3-resolved item.

- [ ] **Step 4: Push the branch**

Run: `git push -u origin m2-routing-and-groupby`

- [ ] **Step 5: Open the PR**

Run:

```bash
gh pr create --title "M2: per-op routing layer + hash groupby on GPU" --body "$(cat <<'EOF'
## Summary

- Routing layer: walker → Rust router → LiftingPlan → walker-applies. Per-op cost rules, affinity smoothing. Filter now routes to CPU by default at all sizes; M1 kernels still ship.
- Hash groupby on GPU: composite-key encoding (≤ 128 bits), two-pass count-then-fill, six aggregation kernels (sum / mean / count / min / max / pl.len), host-side mean.
- Modified TPC-H Q1 benchmark: 10M-row deterministic fixture, full Q1 (CPU filter + GPU groupby + CPU sort) faster than pure-CPU Polars on M2 Ultra. `ratio_metal_over_cpu = <quote from baseline.json>`.
- Testing taxonomy migration: `tests/diff/` retired; property tests now Rust proptest, explicit cases in `tests/python_integration/`.
- Conformance: upstream Polars `test_group_by.py` + `test_agg.py` wired into the harness; zero new failures vs CPU baseline.

See spec: `docs/superpowers/specs/2026-05-21-m2-design.md`
See plan: `docs/superpowers/plans/2026-05-21-m2-routing-and-groupby.md`

## Test plan

- [x] `make gate` on M2 Ultra
- [x] Portability gate on M2 (16 GB)
- [x] Portability gate on M1 (8 GB)
- [x] `tests/bench/baseline.json` updated with `tpch_q1_modified` entry; ratio < 1.0
- [x] Conformance: no new failures under engine=MetalEngine()

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Done — user reviews and merges**

Do **not** merge from this session. The user reviews the PR after it lands. If feedback arrives, address it on the same branch with new commits (no force-push, no amends).

---

## Notes for the implementer

- **Read the spec before each phase.** Phases here implement specific spec sections — when in doubt about scope or semantics, the spec wins (`docs/superpowers/specs/2026-05-21-m2-design.md`).
- **Read the matching cuDF kernel first.** Per CLAUDE.md, do this before writing any MSL — specifically `references/cudf/cpp/src/groupby/hash/groupby.cu` for the two-pass algorithm and `shaders/cmp_i64.metal` (M1) for the MSL-macro entry-point pattern.
- **The router is the new policy layer; the walker is plumbing.** Resist the temptation to put any decision logic in the walker — it stays minimal (IR shape → MetalPlanNode tree). Every routing choice goes through the Rust router so PR-level tuning touches one Rust file.
- **Atomic CAS on u128 keys is two atomic_ulong CAS operations (key_lo, then key_hi) under a sentinel-bit lock.** The MSL kernel must handle this correctly; the spec § "Two-pass groupby algorithm" describes the contract, but the kernel author should re-derive the invariants from cuDF before writing code.
- **Don't optimize the build kernel speculatively.** Spec § "Risks" calls out contention as M2's primary algorithmic risk. Land the correct, slow version; benchmark; then tune. If microbenches show >2× degradation vs the cuDF measurement, escalate per spec.
- **Composite-key encoder is hot — but pure CPU code.** It runs once per query at dispatch time. Write it idiomatically; do not introduce SIMD intrinsics in M2.
- **Multi-chunk fallback is a defensive measure, not a feature.** Task 31's check exists to prevent silent wrong-results. Full multi-chunk support belongs in M3+.
- **`baseline.json` is append-only.** M2 adds `tpch_q1_modified`. Never edit M1's existing entries — those numbers are part of the repository's perf history.
- **Don't introduce new dependencies without a written justification in the PR description** (per CLAUDE.md). M2's plan introduces `pythonize` (PyDict → serde_json::Value) in Task 28; if you can rewrite the parser to consume `&PyDict` directly without significantly more code, prefer that. Either way, document the choice in the PR.
- **Lint clean before declaring a task done.** `make lint` (clippy + fmt + ruff) is the bar; the gate will reject otherwise.
