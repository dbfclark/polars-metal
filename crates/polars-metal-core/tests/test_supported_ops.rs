use polars_metal_native::fusion::supported_ops::{all_op_ids, op_spec, OpId, OpSpec};

#[test]
fn registry_has_expected_op_count() {
    let n = all_op_ids().count();
    assert!(n >= 35, "expected >= 35 supported ops, got {n}");
}

#[test]
fn add_op_is_supported() {
    let spec: OpSpec = op_spec(OpId::Add);
    assert_eq!(spec.n_args, 2);
    assert_eq!(spec.flops_per_row, 1);
}

#[test]
fn sin_op_is_supported() {
    let spec: OpSpec = op_spec(OpId::Sin);
    assert_eq!(spec.n_args, 1);
    assert_eq!(spec.flops_per_row, 10);
}

#[test]
fn matmul_flops_are_dynamic() {
    let spec = op_spec(OpId::MatMul);
    assert_eq!(spec.n_args, 2);
    assert_eq!(spec.flops_per_row, 0);
    assert!(spec.dynamic_flops);
}
