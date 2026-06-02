//! PyO3 wrappers for FusionScope construction from Python.
//!
//! The Python-side analyzer in `_fusion_analyzer.py` builds scopes via this
//! API. The Rust router (Phase 5) then consumes the constructed scope.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::density::{density_routes_gpu, RouteDecision};
use super::scope::{FusionScope, InputDtype, NodeIdx};
use super::supported_ops::{op_spec, OpId};

#[pyclass(name = "PyFusionScope")]
pub struct PyFusionScope {
    pub(crate) inner: FusionScope,
}

#[pymethods]
impl PyFusionScope {
    #[new]
    fn new() -> Self {
        Self {
            inner: FusionScope::new(),
        }
    }

    fn add_input(&mut self, name: &str, dtype: &str) -> PyResult<u32> {
        let d = match dtype {
            "F32" => InputDtype::F32,
            "F64" => InputDtype::F64,
            "Bool" => InputDtype::Bool,
            "I32" => InputDtype::I32,
            other if other.starts_with("ArrayF32(") => {
                let n: usize = other
                    .trim_start_matches("ArrayF32(")
                    .trim_end_matches(')')
                    .parse()
                    .map_err(|_| PyValueError::new_err(format!("bad ArrayF32 dim: {other}")))?;
                InputDtype::ArrayF32(n)
            }
            "ListF32" => InputDtype::ListF32,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown InputDtype: {other}"
                )))
            }
        };
        Ok(self.inner.add_input(name, d).0)
    }

    #[pyo3(signature = (op_str, args, param = None))]
    fn push_op(&mut self, op_str: &str, args: Vec<u32>, param: Option<i64>) -> PyResult<u32> {
        let op = op_id_from_str(op_str)
            .ok_or_else(|| PyValueError::new_err(format!("unknown OpId: {op_str}")))?;
        let spec = op_spec(op);
        if args.len() as u32 != spec.n_args {
            return Err(PyValueError::new_err(format!(
                "{op_str} expects {} args, got {}",
                spec.n_args,
                args.len()
            )));
        }
        let args_idx: Vec<NodeIdx> = args.into_iter().map(NodeIdx).collect();
        Ok(self.inner.push_op_param(op, args_idx, param).0)
    }

    fn mark_output(&mut self, idx: u32) {
        self.inner.mark_output(NodeIdx(idx));
    }

    fn n_inputs(&self) -> usize {
        self.inner.inputs.len()
    }
    fn n_ops(&self) -> usize {
        self.inner.ops.len()
    }

    fn est_flops(&self, n_rows: usize) -> u64 {
        self.inner.est_flops_for(n_rows)
    }

    /// String form of the routing decision: `"Gpu"` or `"Cpu(<reason>)"`.
    fn route_decision(&self, n_rows: usize) -> String {
        match density_routes_gpu(&self.inner, n_rows) {
            RouteDecision::Gpu => "Gpu".to_string(),
            RouteDecision::Cpu(reason) => format!("Cpu({reason:?})"),
        }
    }
}

fn op_id_from_str(s: &str) -> Option<OpId> {
    use OpId::*;
    Some(match s {
        "Add" => Add,
        "Sub" => Sub,
        "Mul" => Mul,
        "Div" => Div,
        "Mod" => Mod,
        "Pow" => Pow,
        "Neg" => Neg,
        "Abs" => Abs,
        "Square" => Square,
        "Eq" => Eq,
        "Ne" => Ne,
        "Lt" => Lt,
        "Le" => Le,
        "Gt" => Gt,
        "Ge" => Ge,
        "LogicalAnd" => LogicalAnd,
        "LogicalOr" => LogicalOr,
        "LogicalNot" => LogicalNot,
        "Where" => Where,
        "Sin" => Sin,
        "Cos" => Cos,
        "Tan" => Tan,
        "Sinh" => Sinh,
        "Cosh" => Cosh,
        "Tanh" => Tanh,
        "Asin" => Asin,
        "Acos" => Acos,
        "Atan" => Atan,
        "Atan2" => Atan2,
        "Log" => Log,
        "Log2" => Log2,
        "Log10" => Log10,
        "Log1p" => Log1p,
        "Exp" => Exp,
        "Exp2" => Exp2,
        "Sqrt" => Sqrt,
        "Cbrt" => Cbrt,
        "Floor" => Floor,
        "Ceil" => Ceil,
        "Round" => Round,
        "CastF32" => CastF32,
        "CastF64" => CastF64,
        "CastI32" => CastI32,
        "CastBool" => CastBool,
        "Sum" => Sum,
        "Mean" => Mean,
        "Min" => Min,
        "Max" => Max,
        "Std" => Std,
        "Var" => Var,
        "ArgMin" => ArgMin,
        "ArgMax" => ArgMax,
        "Sort" => Sort,
        "ArgPartition" => ArgPartition,
        "CumSum" => CumSum,
        "CumProd" => CumProd,
        "CumMax" => CumMax,
        "CumMin" => CumMin,
        "MatMul" => MatMul,
        "Fft" => Fft,
        "Ifft" => Ifft,
        "Shift" => Shift,
        _ => return None,
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFusionScope>()?;
    Ok(())
}
