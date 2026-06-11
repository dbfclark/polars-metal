//! FusionScope: a DAG of expression nodes (inputs + ops) plus output refs.
//! The analyzer (Phase 3) builds these by walking Polars expression IR;
//! the subgraph builder (Phase 4) consumes them and emits MLX calls.

use super::supported_ops::{op_spec, OpId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputDtype {
    F32,
    F64,
    Bool,
    I32,
    I8,
    I16,
    I64,
    U8,
    U16,
    U32,
    U64,
    ArrayF32(usize),
    ListF32,
}

impl InputDtype {
    /// Element width in bytes for the scalar numeric dtypes. `ArrayF32(d)`
    /// returns `d * 4`; `ListF32` is variable-length and returns `4` (the
    /// per-element width — callers that need a row stride must not use this).
    pub fn element_size(self) -> usize {
        match self {
            InputDtype::Bool | InputDtype::I8 | InputDtype::U8 => 1,
            InputDtype::I16 | InputDtype::U16 => 2,
            InputDtype::F32 | InputDtype::I32 | InputDtype::U32 | InputDtype::ListF32 => 4,
            InputDtype::F64 | InputDtype::I64 | InputDtype::U64 => 8,
            InputDtype::ArrayF32(d) => d * 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InputRef {
    pub column_name: String,
    pub dtype: InputDtype,
}

/// Index into FusionScope::ops or FusionScope::inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeIdx(pub u32);

#[derive(Clone, Debug)]
pub struct OpNode {
    pub op: OpId,
    pub args: Vec<NodeIdx>,
    /// Optional scalar parameter (e.g. shift amount for `OpId::Shift`).
    pub param: Option<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct FusionScope {
    pub inputs: Vec<InputRef>,
    pub ops: Vec<OpNode>,
    pub outputs: Vec<NodeIdx>,
}

impl FusionScope {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_input(&mut self, name: &str, dtype: InputDtype) -> NodeIdx {
        let idx = NodeIdx(self.inputs.len() as u32);
        self.inputs.push(InputRef {
            column_name: name.to_string(),
            dtype,
        });
        idx
    }

    pub fn push_op(&mut self, op: OpId, args: Vec<NodeIdx>) -> NodeIdx {
        let idx = NodeIdx(self.inputs.len() as u32 + self.ops.len() as u32);
        self.ops.push(OpNode {
            op,
            args,
            param: None,
        });
        idx
    }

    /// Like [`push_op`] but carries an optional scalar parameter (e.g. the
    /// shift amount for `OpId::Shift`).
    pub fn push_op_param(&mut self, op: OpId, args: Vec<NodeIdx>, param: Option<i64>) -> NodeIdx {
        let idx = NodeIdx(self.inputs.len() as u32 + self.ops.len() as u32);
        self.ops.push(OpNode { op, args, param });
        idx
    }

    pub fn mark_output(&mut self, idx: NodeIdx) {
        self.outputs.push(idx);
    }

    pub fn est_flops_for(&self, n_rows: usize) -> u64 {
        let mut total: u64 = 0;
        for node in &self.ops {
            let spec = op_spec(node.op);
            if !spec.dynamic_flops {
                total += spec.flops_per_row as u64 * n_rows as u64;
            }
        }
        total
    }

    pub fn has_terminator(&self) -> bool {
        use OpId::*;
        self.ops.iter().any(|n| {
            matches!(
                n.op,
                Sum | Mean
                    | Min
                    | Max
                    | Std
                    | Var
                    | ArgMin
                    | ArgMax
                    | Sort
                    | ArgPartition
                    | MatMul
                    | Fft
                    | Ifft
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::supported_ops::OpId;
    use super::{FusionScope, InputDtype};

    #[test]
    fn shift_op_carries_window_param() {
        let mut s = FusionScope::new();
        let a = s.add_input("x", InputDtype::F32);
        // push_op_param with param=Some(3); a is NodeIdx(0), first op is at
        // ops-index 0 (NodeIdx = n_inputs + ops.len() before push = 1).
        let sh = s.push_op_param(OpId::Shift, vec![a], Some(3));
        // sh.0 == 1 (1 input + 0 prior ops); op is at self.ops[sh.0 - n_inputs]
        let n_inputs = s.inputs.len() as u32;
        let op_idx = (sh.0 - n_inputs) as usize;
        assert_eq!(s.ops[op_idx].param, Some(3));
    }
}

#[cfg(test)]
mod scope_dtype_tests {
    use super::InputDtype;

    #[test]
    fn int_input_dtype_element_sizes() {
        assert_eq!(InputDtype::I8.element_size(), 1);
        assert_eq!(InputDtype::I16.element_size(), 2);
        assert_eq!(InputDtype::I32.element_size(), 4);
        assert_eq!(InputDtype::I64.element_size(), 8);
        assert_eq!(InputDtype::U8.element_size(), 1);
        assert_eq!(InputDtype::U16.element_size(), 2);
        assert_eq!(InputDtype::U32.element_size(), 4);
        assert_eq!(InputDtype::U64.element_size(), 8);
        assert_eq!(InputDtype::F32.element_size(), 4);
    }
}
