//! M7 C1 Task 1: pin every numeric dtype through the buffer-path subgraph
//! (`eval_to_metal_buffers` readback) so B2's `per_dtype!` macro fold cannot
//! silently break a dtype arm. Mirrors `test_subgraph_int.rs` (I32) for all
//! 9 buffer-path-supported numeric dtypes (F32 + the 8 integer dtypes; F64/Bool
//! return UnsupportedInputDtype and have no readback to pin).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_native::fusion::scope::{FusionScope, InputDtype};
use polars_metal_native::fusion::subgraph::MlxSubgraph;
use polars_metal_native::fusion::supported_ops::OpId;
use proptest::prelude::*;

/// Identity-subgraph round-trip: stage a typed buffer, run a no-op subgraph,
/// assert the readback equals the input. Generated per dtype by the macro below.
macro_rules! identity_round_trip {
    ($name:ident, $rs:ty, $from:ident, $to:ident, $input_dt:expr, $vals:expr) => {
        #[test]
        fn $name() {
            let device = MetalDevice::system_default().expect("metal");
            let vals: Vec<$rs> = $vals;
            let buf = Arc::new(MetalBuffer::$from(&device, &vals).expect("stage"));

            let mut scope = FusionScope::new();
            let a = scope.add_input("a", $input_dt);
            scope.mark_output(a);

            let sg = MlxSubgraph::from_fusion_scope_buffers(&scope, &[buf]).expect("build");
            let outs = sg.eval_to_metal_buffers(&device).expect("eval");
            assert_eq!(outs.len(), 1);
            assert_eq!(outs[0].$to(), vals);
        }
    };
}

identity_round_trip!(
    f32_identity_round_trips,
    f32,
    from_f32_slice,
    to_f32_vec,
    InputDtype::F32,
    vec![f32::MIN, -1.5, 0.0, 1.0, 42.0, f32::MAX]
);
identity_round_trip!(
    i8_identity_round_trips,
    i8,
    from_i8_slice,
    to_i8_vec,
    InputDtype::I8,
    vec![-128, -1, 0, 1, 127]
);
identity_round_trip!(
    i16_identity_round_trips,
    i16,
    from_i16_slice,
    to_i16_vec,
    InputDtype::I16,
    vec![-32768, -1, 0, 1, 32767]
);
identity_round_trip!(
    i32_identity_round_trips,
    i32,
    from_i32_slice,
    to_i32_vec,
    InputDtype::I32,
    vec![-7, 0, 1, 100, 2_000_000_000]
);
identity_round_trip!(
    i64_identity_round_trips,
    i64,
    from_i64_slice,
    to_i64_vec,
    InputDtype::I64,
    vec![i64::MIN, -1, 0, 1, 9_000_000_000_000, i64::MAX]
);
identity_round_trip!(
    u8_identity_round_trips,
    u8,
    from_u8_slice,
    to_u8_vec,
    InputDtype::U8,
    vec![0, 1, 127, 255]
);
identity_round_trip!(
    u16_identity_round_trips,
    u16,
    from_u16_slice,
    to_u16_vec,
    InputDtype::U16,
    vec![0, 1, 65535]
);
identity_round_trip!(
    u32_identity_round_trips,
    u32,
    from_u32_slice,
    to_u32_vec,
    InputDtype::U32,
    vec![0, 1, 4_000_000_000]
);
identity_round_trip!(
    u64_identity_round_trips,
    u64,
    from_u64_slice,
    to_u64_vec,
    InputDtype::U64,
    vec![0, 1, u64::MAX]
);

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    /// Per-dtype `Add` through the buffer path uses integer (not float-
    /// reinterpreted) semantics — the assertion fails loudly if a fold arm
    /// reads the bit-pattern as the wrong dtype. Ranges narrowed to avoid wrap.
    #[test]
    fn i32_add_buffer_path_matches_scalar(
        a in prop::collection::vec(-1_000_000i32..1_000_000, 1..32),
        b in prop::collection::vec(-1_000_000i32..1_000_000, 1..32),
    ) {
        let device = MetalDevice::system_default().expect("metal");
        let len = a.len().min(b.len());
        let (a, b) = (a[..len].to_vec(), b[..len].to_vec());
        let expect: Vec<i32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

        let a_buf = Arc::new(MetalBuffer::from_i32_slice(&device, &a).expect("stage a"));
        let b_buf = Arc::new(MetalBuffer::from_i32_slice(&device, &b).expect("stage b"));

        let mut scope = FusionScope::new();
        let ai = scope.add_input("a", InputDtype::I32);
        let bi = scope.add_input("b", InputDtype::I32);
        let m = scope.push_op(OpId::Add, vec![ai, bi]);
        scope.mark_output(m);

        let sg = MlxSubgraph::from_fusion_scope_buffers(&scope, &[a_buf, b_buf]).expect("build");
        let outs = sg.eval_to_metal_buffers(&device).expect("eval");
        prop_assert_eq!(outs[0].to_i32_vec(), expect);
    }
}
