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
    ArrayF32(usize),
    ListF32,
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
        self.ops.push(OpNode { op, args });
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
