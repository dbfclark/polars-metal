# Makefile — single entry point per gate. See docs/superpowers/specs/2026-05-19-m0-skeleton-design.md.
.PHONY: build wheel test-unit test-kernel test-conformance test-diff bench lint gate refresh-refs help

help:
	@grep '^[a-zA-Z][a-zA-Z-]*:' $(MAKEFILE_LIST) | grep -v '^help:' | awk -F: '{print $$1}' | sort

build:
	cargo build --workspace --release

wheel:
	VIRTUAL_ENV=$$(python3 -c "import sys; print(sys.prefix)") maturin develop --release

test-unit:
	cargo test --workspace -- --test-threads=1

test-kernel:
	# kernel correctness suite (proptest + differential, --test-threads=1 for Metal queue safety)
	cargo test -p polars-metal-kernels -- --test-threads=1

test-conformance:
	pytest tests/conformance -k "not skip_metal"

test-diff:
	# Rust differential/property nets (kernel + dispatch layer)
	cargo test -p polars-metal-core --test proptest_subgraph --test test_subgraph_int --test test_subgraph_dtype_readback -- --test-threads=1
	cargo test -p polars-metal-kernels --test test_cmp_i64 --test test_cmp_f64 --test test_fused_vs_per_agg -- --test-threads=1
	# Python plan-level differential slice (engine=metal vs CPU)
	pytest tests/diff -q

bench:
	cargo bench --workspace
	pytest tests/bench --benchmark-only

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --check
	ruff check .
	ruff format --check .

gate: lint test-unit test-kernel wheel test-conformance test-diff
	@echo "M0 gate passed."

refresh-refs:
	bash scripts/refresh-references.sh
