// crates/polars-metal-core/tests/test_execute_groupby_unit.rs
//
// Unit tests for the `parse_groupby_plan` function and related helpers that
// live in `udf.rs`. These tests exercise the PyDict → ParsedGroupByPlan
// parser in isolation (no Metal device required). End-to-end tests
// (`execute_groupby` with real column buffers) land in T32.
#![allow(clippy::expect_used, clippy::panic)]

use polars_metal_native::plan::{AggOp, MetalDtype};
use polars_metal_native::{parse_groupby_plan, GroupByParseError, ParsedGroupByPlan};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

// ---------------------------------------------------------------------------
// Helper: build a 2-element PyList for a key entry [name, dtype_tag]
// ---------------------------------------------------------------------------
fn key_entry<'py>(py: Python<'py>, name: &str, dtype: &str) -> Bound<'py, PyList> {
    let items: Vec<Bound<'py, PyAny>> = vec![
        PyString::new_bound(py, name).into_any(),
        PyString::new_bound(py, dtype).into_any(),
    ];
    PyList::new_bound(py, items)
}

// ---------------------------------------------------------------------------
// Helper: build a minimal valid plan dict with one key and one agg
// ---------------------------------------------------------------------------
fn make_plan_dict<'py>(
    py: Python<'py>,
    key_name: &str,
    key_dtype: &str,
    input_col: &str,
    op: &str,
    output_alias: &str,
) -> Bound<'py, PyDict> {
    let plan = PyDict::new_bound(py);

    let keys = PyList::empty_bound(py);
    keys.append(key_entry(py, key_name, key_dtype))
        .expect("append key entry");
    plan.set_item("keys", keys).expect("set keys");

    let aggs = PyList::empty_bound(py);
    let a_entry = PyDict::new_bound(py);
    a_entry
        .set_item("input_col", input_col)
        .expect("set input_col");
    a_entry.set_item("op", op).expect("set op");
    a_entry
        .set_item("output_alias", output_alias)
        .expect("set output_alias");
    aggs.append(a_entry).expect("append agg entry");
    plan.set_item("aggs", aggs).expect("set aggs");

    plan
}

// ---------------------------------------------------------------------------
// Parser correctness tests
// ---------------------------------------------------------------------------

#[test]
fn parser_extracts_single_key_single_agg() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let plan = make_plan_dict(py, "k", "I64", "v", "Sum", "sum_v");
        let parsed: ParsedGroupByPlan = parse_groupby_plan(&plan).expect("parse must succeed");

        assert_eq!(parsed.keys.len(), 1);
        assert_eq!(parsed.keys[0].name, "k");
        assert_eq!(parsed.keys[0].dtype, MetalDtype::I64);

        assert_eq!(parsed.aggs.len(), 1);
        assert_eq!(parsed.aggs[0].input_col, "v");
        assert_eq!(parsed.aggs[0].op, AggOp::Sum);
        assert_eq!(parsed.aggs[0].output_alias, "sum_v");
    });
}

#[test]
fn parser_extracts_f64_key_and_mean_agg() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let plan = make_plan_dict(py, "price", "F64", "price", "Mean", "avg_price");
        let parsed = parse_groupby_plan(&plan).expect("parse must succeed");
        assert_eq!(parsed.keys[0].dtype, MetalDtype::F64);
        assert_eq!(parsed.aggs[0].op, AggOp::Mean);
    });
}

#[test]
fn parser_extracts_len_with_empty_input_col() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let plan = PyDict::new_bound(py);

        let keys = PyList::empty_bound(py);
        keys.append(key_entry(py, "g", "I64")).expect("append key");
        plan.set_item("keys", keys).expect("set keys");

        let aggs = PyList::empty_bound(py);
        let a_entry = PyDict::new_bound(py);
        // input_col is intentionally absent — Len has no input column.
        a_entry.set_item("op", "Len").expect("set op");
        a_entry
            .set_item("output_alias", "n")
            .expect("set output_alias");
        aggs.append(a_entry).expect("append agg");
        plan.set_item("aggs", aggs).expect("set aggs");

        let parsed =
            parse_groupby_plan(&plan).expect("parse must succeed for Len with no input_col");
        assert_eq!(parsed.aggs[0].op, AggOp::Len);
        assert_eq!(parsed.aggs[0].input_col, "");
        assert_eq!(parsed.aggs[0].output_alias, "n");
    });
}

#[test]
fn parser_accepts_all_six_agg_ops() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let wire_ops = ["Sum", "Mean", "Count", "Min", "Max", "Len"];
        let expected_ops = [
            AggOp::Sum,
            AggOp::Mean,
            AggOp::Count,
            AggOp::Min,
            AggOp::Max,
            AggOp::Len,
        ];
        for (&wire, &expected) in wire_ops.iter().zip(expected_ops.iter()) {
            let plan = PyDict::new_bound(py);

            let keys = PyList::empty_bound(py);
            keys.append(key_entry(py, "k", "I64")).expect("append key");
            plan.set_item("keys", keys).expect("set keys");

            let aggs = PyList::empty_bound(py);
            let a_entry = PyDict::new_bound(py);
            a_entry.set_item("input_col", "x").expect("set input_col");
            a_entry.set_item("op", wire).expect("set op");
            a_entry
                .set_item("output_alias", "out")
                .expect("set output_alias");
            aggs.append(a_entry).expect("append agg");
            plan.set_item("aggs", aggs).expect("set aggs");

            let parsed = parse_groupby_plan(&plan)
                .unwrap_or_else(|e| panic!("parse failed for op {wire}: {e}"));
            assert_eq!(
                parsed.aggs[0].op, expected,
                "op wire string {wire} must parse to {expected:?}"
            );
        }
    });
}

#[test]
fn parser_accepts_bool_key_dtype() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let plan = make_plan_dict(py, "flag", "Bool", "v", "Count", "cnt");
        let parsed = parse_groupby_plan(&plan).expect("parse must succeed");
        assert_eq!(parsed.keys[0].dtype, MetalDtype::Bool);
    });
}

#[test]
fn parser_accepts_multiple_keys_and_aggs() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let plan = PyDict::new_bound(py);

        let keys = PyList::empty_bound(py);
        keys.append(key_entry(py, "returnflag", "I64"))
            .expect("append key");
        keys.append(key_entry(py, "linestatus", "I64"))
            .expect("append key");
        plan.set_item("keys", keys).expect("set keys");

        let aggs = PyList::empty_bound(py);
        for (input_col, op, alias) in [("qty", "Sum", "sum_qty"), ("price", "Mean", "avg_price")] {
            let a = PyDict::new_bound(py);
            a.set_item("input_col", input_col).expect("set input_col");
            a.set_item("op", op).expect("set op");
            a.set_item("output_alias", alias).expect("set output_alias");
            aggs.append(a).expect("append agg");
        }
        plan.set_item("aggs", aggs).expect("set aggs");

        let parsed = parse_groupby_plan(&plan).expect("parse must succeed");
        assert_eq!(parsed.keys.len(), 2);
        assert_eq!(parsed.aggs.len(), 2);
        assert_eq!(parsed.keys[0].name, "returnflag");
        assert_eq!(parsed.keys[1].name, "linestatus");
        assert_eq!(parsed.aggs[0].op, AggOp::Sum);
        assert_eq!(parsed.aggs[1].op, AggOp::Mean);
    });
}

// ---------------------------------------------------------------------------
// Parser error tests
// ---------------------------------------------------------------------------

#[test]
fn parser_rejects_missing_keys_field() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let plan = PyDict::new_bound(py);
        // "keys" field entirely absent.
        let aggs = PyList::empty_bound(py);
        plan.set_item("aggs", aggs).expect("set aggs");

        let err = parse_groupby_plan(&plan).expect_err("must fail on missing keys");
        assert!(
            matches!(err, GroupByParseError::Missing("keys")),
            "expected Missing(\"keys\"), got {err:?}"
        );
    });
}

#[test]
fn parser_rejects_missing_aggs_field() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let plan = PyDict::new_bound(py);
        let keys = PyList::empty_bound(py);
        plan.set_item("keys", keys).expect("set keys");
        // "aggs" intentionally absent.

        let err = parse_groupby_plan(&plan).expect_err("must fail on missing aggs");
        assert!(
            matches!(err, GroupByParseError::Missing("aggs")),
            "expected Missing(\"aggs\"), got {err:?}"
        );
    });
}

#[test]
fn parser_rejects_unknown_dtype() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let plan = make_plan_dict(py, "k", "NotADtype", "v", "Sum", "sum_v");
        let err = parse_groupby_plan(&plan).expect_err("must fail on bad dtype");
        assert!(
            matches!(err, GroupByParseError::UnknownDtype(_)),
            "expected UnknownDtype, got {err:?}"
        );
    });
}

#[test]
fn parser_rejects_unknown_op() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let plan = make_plan_dict(py, "k", "I64", "v", "StdDev", "stddev_v");
        let err = parse_groupby_plan(&plan).expect_err("must fail on unknown op");
        assert!(
            matches!(err, GroupByParseError::UnknownOp(_)),
            "expected UnknownOp, got {err:?}"
        );
    });
}

#[test]
fn parser_rejects_missing_output_alias() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let plan = PyDict::new_bound(py);

        let keys = PyList::empty_bound(py);
        keys.append(key_entry(py, "k", "I64")).expect("append key");
        plan.set_item("keys", keys).expect("set keys");

        let aggs = PyList::empty_bound(py);
        let a_entry = PyDict::new_bound(py);
        a_entry.set_item("input_col", "v").expect("set input_col");
        a_entry.set_item("op", "Sum").expect("set op");
        // output_alias intentionally absent.
        aggs.append(a_entry).expect("append agg");
        plan.set_item("aggs", aggs).expect("set aggs");

        let err = parse_groupby_plan(&plan).expect_err("must fail on missing output_alias");
        assert!(
            matches!(err, GroupByParseError::WrongType("output_alias")),
            "expected WrongType(\"output_alias\"), got {err:?}"
        );
    });
}
