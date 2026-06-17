//! Resident exp-decay rerank: reranked = sim * exp(-weight[hit]).
#![allow(clippy::unwrap_used, clippy::expect_used)]
use polars_metal_native::vector_search::vector_search_topk_rerank;

const OP_COSINE: u32 = 0;

#[test]
fn cosine_topk_with_exp_decay_rerank() {
    // 1 query, 3 corpus rows, D=2, k=2.
    // query = [1,0]; corpus rows: r0=[1,0] (sim 1), r1=[0.6,0.8] (sim .6), r2=[0,1] (sim 0)
    // weights: w=[2.0, 0.0, 0.0] -> reranked r0 = 1*exp(-2)=.135, r1=.6*exp(0)=.6, r2=0
    // top-2 by reranked: r1 (.6), r0 (.135).
    let q = vec![1.0f32, 0.0];
    let c = vec![1.0f32, 0.0, 0.6, 0.8, 0.0, 1.0];
    let w = vec![2.0f32, 0.0, 0.0];
    let (idx, val) =
        vector_search_topk_rerank(&q, 1, &c, 3, 2, 2, OP_COSINE, Some(&w)).expect("rerank search");
    // returns flattened (Q*k) idx + reranked val. Recompute expected per returned index.
    assert_eq!(idx.len(), 2);
    assert_eq!(val.len(), 2);
    // For each returned (i, score): score ≈ cos_sim(query, c_i) * exp(-w[i]).
    let cos = |i: usize| {
        let r = [c[i * 2], c[i * 2 + 1]];
        let dot = q[0] * r[0] + q[1] * r[1];
        let nq = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let nr = (r[0] * r[0] + r[1] * r[1]).sqrt();
        dot / (nq * nr)
    };
    for (ix, sc) in idx.iter().zip(val.iter()) {
        let i = *ix as usize;
        let expect = cos(i) * (-w[i]).exp();
        assert!(
            (sc - expect).abs() < 1e-3,
            "idx {i}: got {sc}, expect {expect}"
        );
    }
    // The two returned indices must be the true top-2 by reranked score: {1, 0}.
    let mut got: Vec<i32> = idx.clone();
    got.sort();
    assert_eq!(got, vec![0, 1]);
}

#[test]
fn rerank_none_matches_plain_topk() {
    // weight=None must behave exactly like vector_search_topk (no rerank).
    use polars_metal_native::vector_search::{vector_search_topk, vector_search_topk_rerank};
    let q = vec![1.0f32, 0.0, 0.0, 1.0];
    let c = vec![1.0f32, 0.0, 0.0, 1.0, 0.7, 0.7];
    let (i0, v0) = vector_search_topk(&q, 2, &c, 3, 2, 2, OP_COSINE).expect("plain search");
    let (i1, v1) =
        vector_search_topk_rerank(&q, 2, &c, 3, 2, 2, OP_COSINE, None).expect("rerank none");
    assert_eq!(i0, i1);
    for (a, b) in v0.iter().zip(v1.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}
