use crate::plan::{MetalDtype, MetalPlanNode, PredicateAst};
use polars_metal_kernels::cmp::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Deserialize a Python `dict` into a [`MetalPlanNode`].
///
/// Plan dict schema (mirrors the Python walker output):
/// - `{"kind": "Scan", "n_rows": int, "columns": [(name, dtype_tag), ...]}`
/// - `{"kind": "Project", "input": <plan>, "columns": list[str]}`
/// - `{"kind": "Filter", "input": <plan>, "predicate": <pred>}`
///
/// dtype_tag is one of `"I64"`, `"F64"`, `"Bool"`.
pub(crate) fn deserialize_plan(dict: &Bound<PyDict>) -> PyResult<MetalPlanNode> {
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("plan dict missing 'kind'"))?
        .extract()?;

    match kind.as_str() {
        "Scan" => {
            // n_rows is informational; Rust does not need it for the no-op
            // pass-through, but we parse it for shape validation / future use.
            let n_rows: usize = dict
                .get_item("n_rows")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Scan missing 'n_rows'"))?
                .extract()?;
            let cols_obj = dict
                .get_item("columns")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Scan missing 'columns'"))?;
            let cols_list: Bound<PyList> = cols_obj.downcast_into()?;
            let mut columns = Vec::with_capacity(cols_list.len());
            for entry in cols_list.iter() {
                // Each column entry is a 2-element (name, dtype_tag) pair. The
                // walker normalizes to tuples, but tests construct lists; we
                // accept both by going through `Vec<String>` and matching on
                // arity. (Pyo3's `extract::<(String, String)>` would reject
                // lists, so we use a Vec and validate length.)
                let pair: Vec<String> = entry.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "Scan.columns entries must be (name, dtype) 2-element sequences",
                    )
                })?;
                let [name, dtype_s] = <[String; 2]>::try_from(pair).map_err(|got| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Scan.columns entry expected 2 elements, got {}",
                        got.len()
                    ))
                })?;
                let dtype = parse_dtype(&dtype_s)?;
                columns.push((name, dtype));
            }
            Ok(MetalPlanNode::Scan { n_rows, columns })
        }
        "Project" => {
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Project missing 'input'"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            let input = Box::new(deserialize_plan(&input_dict)?);
            let cols: Vec<String> = dict
                .get_item("columns")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Project missing 'columns'"))?
                .extract()?;
            Ok(MetalPlanNode::Project {
                input,
                columns: cols,
            })
        }
        "Filter" => {
            // Deserialize for completeness so `execute_node` can raise a clean
            // NotImplementedError on dispatch (rather than panicking during
            // deserialization). The predicate AST is not consumed today.
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Filter missing 'input'"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            let input = Box::new(deserialize_plan(&input_dict)?);
            let pred_obj = dict.get_item("predicate")?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("Filter missing 'predicate'")
            })?;
            let pred_dict: Bound<PyDict> = pred_obj.downcast_into()?;
            let predicate = deserialize_predicate(&pred_dict)?;
            Ok(MetalPlanNode::Filter { input, predicate })
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "polars_metal: unknown plan kind {other:?}"
        ))),
    }
}

pub(crate) fn deserialize_predicate(dict: &Bound<PyDict>) -> PyResult<PredicateAst> {
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("predicate dict missing 'kind'"))?
        .extract()?;
    match kind.as_str() {
        "Column" => {
            let name: String = dict
                .get_item("name")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Column missing 'name'"))?
                .extract()?;
            let dtype_s: String = dict
                .get_item("dtype")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Column missing 'dtype'"))?
                .extract()?;
            Ok(PredicateAst::Column {
                name,
                dtype: parse_dtype(&dtype_s)?,
            })
        }
        "LiteralI64" => {
            let v: i64 = dict
                .get_item("value")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("LiteralI64 missing 'value'"))?
                .extract()?;
            Ok(PredicateAst::LiteralI64(v))
        }
        "LiteralF64" => {
            let v: f64 = dict
                .get_item("value")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("LiteralF64 missing 'value'"))?
                .extract()?;
            Ok(PredicateAst::LiteralF64(v))
        }
        "LiteralBool" => {
            let v: bool = dict
                .get_item("value")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err("LiteralBool missing 'value'")
                })?
                .extract()?;
            Ok(PredicateAst::LiteralBool(v))
        }
        "Compare" => {
            let op_s: String = dict
                .get_item("op")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Compare missing 'op'"))?
                .extract()?;
            let op = parse_compare_op(&op_s)?;
            let lhs_obj = dict
                .get_item("lhs")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Compare missing 'lhs'"))?;
            let lhs_dict: Bound<PyDict> = lhs_obj.downcast_into()?;
            let lhs = Box::new(deserialize_predicate(&lhs_dict)?);
            let rhs_obj = dict
                .get_item("rhs")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Compare missing 'rhs'"))?;
            let rhs_dict: Bound<PyDict> = rhs_obj.downcast_into()?;
            let rhs = Box::new(deserialize_predicate(&rhs_dict)?);
            Ok(PredicateAst::Compare {
                op: match op {
                    CompareOp::Eq => crate::plan::CompareOp::Eq,
                    CompareOp::Ne => crate::plan::CompareOp::Ne,
                    CompareOp::Lt => crate::plan::CompareOp::Lt,
                    CompareOp::Le => crate::plan::CompareOp::Le,
                    CompareOp::Gt => crate::plan::CompareOp::Gt,
                    CompareOp::Ge => crate::plan::CompareOp::Ge,
                },
                lhs,
                rhs,
            })
        }
        "And" | "Or" => {
            // Filter dispatch flows entirely through the Python UDF in
            // M1; the Rust `execute_plan` Filter branch raises
            // NotImplementedError before ever reaching the AST. We
            // still accept the AND/OR shape here so the parse step
            // doesn't fail if a Filter plan ever round-trips through
            // `execute_plan` (e.g. a future test or a debug-print
            // path).
            let lhs_obj = dict
                .get_item("lhs")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("And/Or missing 'lhs'"))?;
            let lhs_dict: Bound<PyDict> = lhs_obj.downcast_into()?;
            let lhs = Box::new(deserialize_predicate(&lhs_dict)?);
            let rhs_obj = dict
                .get_item("rhs")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("And/Or missing 'rhs'"))?;
            let rhs_dict: Bound<PyDict> = rhs_obj.downcast_into()?;
            let rhs = Box::new(deserialize_predicate(&rhs_dict)?);
            if kind == "And" {
                Ok(PredicateAst::And(lhs, rhs))
            } else {
                Ok(PredicateAst::Or(lhs, rhs))
            }
        }
        other => Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "polars_metal: predicate kind {other:?} lands in a later phase"
        ))),
    }
}

pub(crate) fn parse_dtype(s: &str) -> PyResult<MetalDtype> {
    MetalDtype::from_wire(s).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "polars_metal: unknown MetalDtype tag {s:?}"
        ))
    })
}

/// Parse the wire-format op tag (matching `CompareOp::Eq/Ne/Lt/Le/Gt/Ge`)
/// into the kernel-side `CompareOp`. Used both at predicate-AST
/// deserialization time and by the `cmp_*` pyfunctions below.
pub(crate) fn parse_compare_op(s: &str) -> PyResult<CompareOp> {
    match s {
        "Eq" => Ok(CompareOp::Eq),
        "Ne" => Ok(CompareOp::Ne),
        "Lt" => Ok(CompareOp::Lt),
        "Le" => Ok(CompareOp::Le),
        "Gt" => Ok(CompareOp::Gt),
        "Ge" => Ok(CompareOp::Ge),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "polars_metal: unknown CompareOp tag {other:?}"
        ))),
    }
}
