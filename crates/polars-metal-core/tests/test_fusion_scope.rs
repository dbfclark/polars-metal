use polars_metal_native::fusion::scope::{FusionScope, InputDtype};
use polars_metal_native::fusion::supported_ops::OpId;

#[test]
fn scope_records_ops_inputs_outputs() {
    let mut scope = FusionScope::new();
    let input_a = scope.add_input("a", InputDtype::F32);
    let input_b = scope.add_input("b", InputDtype::F32);
    let add = scope.push_op(OpId::Add, vec![input_a, input_b]);
    scope.mark_output(add);

    assert_eq!(scope.inputs.len(), 2);
    assert_eq!(scope.ops.len(), 1);
    assert_eq!(scope.outputs.len(), 1);
}

#[test]
fn scope_est_flops_sums_per_op() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let sin_a = scope.push_op(OpId::Sin, vec![a]); // 10 FLOPs/row
    let cos_a = scope.push_op(OpId::Cos, vec![a]); // 10
    let prod = scope.push_op(OpId::Mul, vec![sin_a, cos_a]); // 1
    scope.mark_output(prod);

    let total = scope.est_flops_for(10_000_000);
    assert_eq!(total, 21 * 10_000_000);
}

#[test]
fn scope_clone_preserves_structure() {
    let mut s = FusionScope::new();
    let a = s.add_input("a", InputDtype::F32);
    s.push_op(OpId::Sqrt, vec![a]);
    let cloned = s.clone();
    assert_eq!(s.ops.len(), cloned.ops.len());
}
