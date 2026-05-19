#!/usr/bin/env bash
# Refresh the local reference clones under references/.
#
# These clones exist so Claude (and human readers) can grep and Read cuDF and
# Polars source directly when authoring kernels and IR adapters. They are not
# build dependencies — nothing in this repo links against them. The references/
# directory is gitignored.
#
# Pins below should match the Polars rev pinned in Cargo.toml. cuDF can track
# a known-good main commit; bump deliberately at milestone boundaries.

set -euo pipefail

CUDF_REPO="https://github.com/rapidsai/cudf.git"
POLARS_REPO="https://github.com/pola-rs/polars.git"

# Pinned commits. Update at milestone boundaries; cuDF when porting a new
# kernel family, Polars whenever Cargo.toml's polars rev is bumped.
CUDF_REF="f57751a743ccf6a1b693375eb089a5cf08723bef"  # main as of 2026-05-19
POLARS_REF="9d8a77e9569779550405fd6ce7fecefcf58f5ca4"  # main as of 2026-05-19

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REFS_DIR="$REPO_ROOT/references"
mkdir -p "$REFS_DIR"

refresh() {
  local name="$1" repo="$2" ref="$3"
  local dir="$REFS_DIR/$name"

  if [ ! -d "$dir/.git" ]; then
    echo "==> cloning $name at $ref"
    git clone --depth=1 "$repo" "$dir"
    (cd "$dir" && git fetch --depth=1 origin "$ref" && git checkout "$ref")
  else
    echo "==> updating $name to $ref"
    (cd "$dir" && git fetch --depth=1 origin "$ref" && git checkout "$ref")
  fi
}

refresh cudf   "$CUDF_REPO"   "$CUDF_REF"
refresh polars "$POLARS_REPO" "$POLARS_REF"

echo
echo "References refreshed:"
(cd "$REFS_DIR/cudf"   && echo "  cudf   $(git rev-parse HEAD)")
(cd "$REFS_DIR/polars" && echo "  polars $(git rev-parse HEAD)")
