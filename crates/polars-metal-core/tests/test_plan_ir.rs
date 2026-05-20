#![allow(clippy::panic)]

use polars_metal_native::plan::{CompareOp, MetalDtype, MetalPlanNode, PredicateAst};

#[test]
fn constructs_and_inspects_scan_node() {
    let scan = MetalPlanNode::Scan {
        n_rows: 100,
        columns: vec![("a".into(), MetalDtype::I64), ("b".into(), MetalDtype::F64)],
    };
    match scan {
        MetalPlanNode::Scan { n_rows, columns } => {
            assert_eq!(n_rows, 100);
            assert_eq!(columns.len(), 2);
            assert_eq!(columns[0].0, "a");
            assert!(matches!(columns[0].1, MetalDtype::I64));
            assert_eq!(columns[1].0, "b");
            assert!(matches!(columns[1].1, MetalDtype::F64));
        }
        _ => panic!("expected Scan variant"),
    }
}

#[test]
fn constructs_filter_with_compound_predicate() {
    let pred = PredicateAst::And(
        Box::new(PredicateAst::Compare {
            op: CompareOp::Gt,
            lhs: Box::new(PredicateAst::Column {
                name: "a".into(),
                dtype: MetalDtype::I64,
            }),
            rhs: Box::new(PredicateAst::LiteralI64(0)),
        }),
        Box::new(PredicateAst::Compare {
            op: CompareOp::Lt,
            lhs: Box::new(PredicateAst::Column {
                name: "b".into(),
                dtype: MetalDtype::I64,
            }),
            rhs: Box::new(PredicateAst::Column {
                name: "c".into(),
                dtype: MetalDtype::I64,
            }),
        }),
    );
    let _filter = MetalPlanNode::Filter {
        input: Box::new(MetalPlanNode::Scan {
            n_rows: 100,
            columns: vec![],
        }),
        predicate: pred,
    };
}

#[test]
fn project_nests_inside_filter() {
    let scan = MetalPlanNode::Scan {
        n_rows: 10,
        columns: vec![("a".into(), MetalDtype::I64), ("b".into(), MetalDtype::I64)],
    };
    let filter = MetalPlanNode::Filter {
        input: Box::new(scan),
        predicate: PredicateAst::Column {
            name: "a".into(),
            dtype: MetalDtype::Bool,
        },
    };
    let _project = MetalPlanNode::Project {
        input: Box::new(filter),
        columns: vec!["a".into()],
    };
}

#[test]
fn dtype_variants_distinct() {
    assert_ne!(
        std::mem::discriminant(&MetalDtype::I64),
        std::mem::discriminant(&MetalDtype::F64),
    );
    assert_ne!(
        std::mem::discriminant(&MetalDtype::I64),
        std::mem::discriminant(&MetalDtype::Bool),
    );
}

#[test]
fn all_six_compare_ops_distinguishable() {
    let ops = [
        CompareOp::Eq,
        CompareOp::Ne,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ];
    // All distinct discriminants.
    for i in 0..ops.len() {
        for j in (i + 1)..ops.len() {
            assert_ne!(
                std::mem::discriminant(&ops[i]),
                std::mem::discriminant(&ops[j])
            );
        }
    }
}
