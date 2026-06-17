import numpy as np, polars as pl, pytest
from polars_metal import MetalEngine
from polars.testing import assert_frame_equal


def _run(fact, dim, chain, how="left"):
    lf = fact.lazy().join(dim.lazy(), on="id", how=how).with_columns(chain)
    assert_frame_equal(
        lf.collect(),
        lf.collect(engine=MetalEngine(force_fusion=True)),
        check_dtypes=True,
        rel_tol=1e-3,
        abs_tol=1e-3,
    )


def _dim(dim_n, rng, cols=("price", "rating")):
    d = {"id": rng.permutation(dim_n).astype(np.int64)}
    for c in cols:
        d[c] = rng.uniform(0.5, 3.0, dim_n).astype(np.float32)
    return pl.DataFrame(d)


@pytest.mark.parametrize("how", ["left", "inner"])
def test_two_cols_dense(how):
    rng = np.random.default_rng(20)
    dim_n, n = 2000, 200_000
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, dim_n, n).astype(np.int64),
            "sc": rng.uniform(0, 1, n).astype(np.float32),
        }
    )
    chain = (pl.col("sc") * pl.col("price").exp() * pl.col("rating").log()).alias("rr")
    _run(fact, _dim(dim_n, rng), chain, how)


def test_three_cols_chain_reads_subset():
    rng = np.random.default_rng(21)
    dim_n, n = 1000, 100_000
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, dim_n, n).astype(np.int64),
            "sc": rng.uniform(0, 1, n).astype(np.float32),
        }
    )
    dim = _dim(dim_n, rng, cols=("price", "rating", "weight"))
    chain = (pl.col("sc") * pl.col("price").log() + pl.col("rating")).alias(
        "rr"
    )  # weight passthrough
    _run(fact, dim, chain, "left")


def test_left_join_missing_keys_nulls():
    fact = pl.DataFrame({"id": np.int64([0, 1, 2, 3]), "sc": np.float32([1, 2, 3, 4])})
    dim = pl.DataFrame(
        {"id": np.int64([0, 2]), "price": np.float32([0.5, 0.7]), "rating": np.float32([2.0, 3.0])}
    )
    _run(fact, dim, (pl.col("sc") * pl.col("price") + pl.col("rating")).alias("rr"), "left")


def test_nondense_sparse_falls_back_correct():
    rng = np.random.default_rng(23)
    keys = rng.choice(50_000, 1000, replace=False).astype(np.int64)
    fact = pl.DataFrame(
        {
            "id": rng.choice(keys, 20_000).astype(np.int64),
            "sc": rng.uniform(0, 1, 20_000).astype(np.float32),
        }
    )
    dim = pl.DataFrame(
        {
            "id": keys,
            "price": rng.uniform(0.5, 3, len(keys)).astype(np.float32),
            "rating": rng.uniform(1, 5, len(keys)).astype(np.float32),
        }
    )
    _run(
        fact,
        dim,
        (pl.col("sc") * pl.col("price").exp() * pl.col("rating").log()).alias("rr"),
        "left",
    )


def test_non_f32_dim_col_falls_back():
    rng = np.random.default_rng(24)
    dim_n, n = 500, 50_000
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, dim_n, n).astype(np.int64),
            "sc": rng.uniform(0, 1, n).astype(np.float32),
        }
    )
    dim = pl.DataFrame(
        {
            "id": rng.permutation(dim_n).astype(np.int64),
            "price": rng.uniform(0.5, 3, dim_n).astype(np.float32),
            "code": rng.integers(0, 9, dim_n).astype(np.int64),
        }
    )  # non-F32 passthrough
    _run(fact, dim, (pl.col("sc") * pl.col("price").exp()).alias("rr"), "left")


def test_chain_references_join_key_falls_back():
    # Chain reads the join key itself -> resident path must DECLINE (Part A guard),
    # CPU-lookup is byte-exact. (Regression guard: previously crashed under force_fusion.)
    rng = np.random.default_rng(25)
    dim_n, n = 500, 50_000
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, dim_n, n).astype(np.int64),
            "sc": rng.uniform(0, 1, n).astype(np.float32),
        }
    )
    dim = pl.DataFrame(
        {
            "id": rng.permutation(dim_n).astype(np.int64),
            "price": rng.uniform(0.5, 3, dim_n).astype(np.float32),
        }
    )
    chain = (pl.col("id").cast(pl.Float32) * pl.col("price").log() + pl.col("sc")).alias("rr")
    _run(fact, dim, chain, "left")
