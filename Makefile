# Makefile — single entry point per gate. See docs/superpowers/specs/2026-05-19-m0-skeleton-design.md.
.PHONY: build wheel test-unit test-kernel test-conformance test-diff bench lint gate refresh-refs help

help:
	@grep '^[a-zA-Z][a-zA-Z-]*:' $(MAKEFILE_LIST) | grep -v '^help:' | awk -F: '{print $$1}' | sort

build:
	cargo build --workspace --release

wheel:
	VIRTUAL_ENV=$$(python3 -c "import sys; print(sys.prefix)") maturin develop --release

test-unit:
	cargo test --workspace

test-kernel:
	@echo "test-kernel target expands as crates land"
	cargo test -p polars-metal-kernels

test-conformance:
	@echo "test-conformance target lands in Task 33"
	@false

test-diff:
	@echo "test-diff target lands in Task 30"
	@false

bench:
	@echo "bench target lands in Task 32"

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --check
	ruff check .
	ruff format --check .

gate: lint test-unit test-kernel test-conformance test-diff
	@echo "M0 gate passed."

refresh-refs:
	bash scripts/refresh-references.sh
