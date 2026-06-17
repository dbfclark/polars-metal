//! M10 Task 2.3: OpId::Take subgraph eval with a mixed-length contract.
//!
//! The whole point of resident gather is that the gathered-from `source`
//! column is SHORT (a dim table of length `dim_n`) while the `index` column
//! (and therefore the output) is LONG (length `N`). `from_fusion_scope_buffers`
//! views each input at its own length, so a 4-element source and a 5-element
//! index coexist naturally; the Take output is index-length, and every
//! downstream op operates on that index-length array.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_native::fusion::scope::{FusionScope, InputDtype};
use polars_metal_native::fusion::subgraph::MlxSubgraph;
use polars_metal_native::fusion::supported_ops::OpId;

/// Scope: in0 = source(F32, len 4), in1 = index(I32, len 5);
/// op0 = Take(in0, in1) -> len 5; op1 = Sqrt(op0). Mixed-length: source 4,
/// output 5.
#[test]
fn take_then_sqrt_mixed_length() {
    let device = MetalDevice::system_default().expect("metal");

    let mut s = FusionScope::new();
    let src = s.add_input("src", InputDtype::F32); // len 4
    let idx = s.add_input("idx", InputDtype::I32); // len 5
    let t = s.push_op_param(OpId::Take, vec![src, idx], None);
    let r = s.push_op_param(OpId::Sqrt, vec![t], None);
    s.mark_output(r);

    // source = [1,4,9,16]; idx = [3,0,2,1,0] -> take=[16,1,9,4,1] -> sqrt=[4,1,3,2,1]
    let src_buf = Arc::new(
        MetalBuffer::from_f32_slice(&device, &[1.0f32, 4.0, 9.0, 16.0]).expect("stage src"),
    );
    let idx_buf =
        Arc::new(MetalBuffer::from_i32_slice(&device, &[3i32, 0, 2, 1, 0]).expect("stage idx"));

    let sg = MlxSubgraph::from_fusion_scope_buffers(&s, &[src_buf, idx_buf]).expect("build");
    let outs = sg.eval_to_metal_buffers(&device).expect("eval");
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].to_f32_vec(), vec![4.0f32, 1.0, 3.0, 2.0, 1.0]);
}
