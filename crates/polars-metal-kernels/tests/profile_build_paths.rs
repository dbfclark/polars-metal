//! Profiling: break down where time is spent in A1 GPU build, A2 GPU build,
//! and M2's CPU HashMap build. Phase 5 retrospective input — informs Phase 6
//! routing strategy.
//!
//! Run with:
//!   cargo test -p polars-metal-kernels --test profile_build_paths \
//!     --release -- --ignored --nocapture --test-threads=1

#![allow(clippy::expect_used, clippy::print_stdout, clippy::print_stderr)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::shader_lib::shared_library;
use std::mem::size_of_val;
use std::time::Instant;

const WARMUP_ITERS: u32 = 3;
const MEASURE_ITERS: u32 = 5;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn run_times<F: FnMut() -> Vec<f64>>(mut f: F) -> Vec<f64> {
    // Warmup.
    for _ in 0..WARMUP_ITERS {
        let _ = f();
    }
    // Collect per-iteration breakdowns; return median per phase.
    let mut all: Vec<Vec<f64>> = Vec::with_capacity(MEASURE_ITERS as usize);
    for _ in 0..MEASURE_ITERS {
        all.push(f());
    }
    let n_phases = all[0].len();
    (0..n_phases)
        .map(|p| median(all.iter().map(|run| run[p]).collect()))
        .collect()
}

// ============================================================
//  A1 instrumented
// ============================================================

fn profile_a1(device: &MetalDevice, keys: &[u128], n_partitions: u32) -> Vec<f64> {
    use polars_metal_kernels::groupby_build_partitioned::reference::partition_id;

    let mut times = vec![0.0f64; 8];
    let t_start = Instant::now();

    // 1. Key split.
    let t0 = Instant::now();
    let keys_lo: Vec<u64> = keys.iter().map(|k| *k as u64).collect();
    let keys_hi: Vec<u64> = keys.iter().map(|k| (*k >> 64) as u64).collect();
    let n_rows: u32 = keys.len() as u32;
    let log2_tgsm = 10u32;
    times[0] = t0.elapsed().as_secs_f64() * 1000.0; // key_split

    // 2. Buffer allocation (scatter phase).
    let t0 = Instant::now();
    let u64_bytes = |s: &[u64]| unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, size_of_val(s))
    };
    let u32_bytes = |s: &[u32]| unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, size_of_val(s))
    };
    let lib = shared_library(device).expect("lib");
    let pso_count = lib.pipeline("partition_count").expect("pso_count");
    let pso_scatter = lib.pipeline("partition_scatter").expect("pso_scatter");
    let pso_build = lib.pipeline("partition_build").expect("pso_build");
    let buf_keys_lo = device.new_buffer_from_bytes(u64_bytes(&keys_lo)).expect("blo");
    let buf_keys_hi = device.new_buffer_from_bytes(u64_bytes(&keys_hi)).expect("bhi");
    let buf_counts = device.new_buffer_zeroed(n_partitions as usize * 4).expect("bc");
    let buf_n_rows = device.new_buffer_from_bytes(&n_rows.to_le_bytes()).expect("bn");
    let buf_n_part = device.new_buffer_from_bytes(&n_partitions.to_le_bytes()).expect("bp");
    let buf_log2 = device.new_buffer_from_bytes(&log2_tgsm.to_le_bytes()).expect("bl");
    times[1] = t0.elapsed().as_secs_f64() * 1000.0; // alloc_scatter

    // 3. partition_count dispatch + wait.
    let t0 = Instant::now();
    let mut queue = CommandQueue::new(device).expect("queue");
    queue.dispatch_1d(
        &pso_count,
        &[&buf_keys_lo, &buf_keys_hi, &buf_counts, &buf_n_rows, &buf_n_part, &buf_log2],
        n_rows as usize,
    ).expect("count");
    queue.wait_until_complete().expect("wait");
    times[2] = t0.elapsed().as_secs_f64() * 1000.0; // dispatch_count

    // 4. CPU exclusive scan of counts.
    let t0 = Instant::now();
    let counts_bytes = buf_counts.as_slice();
    let mut partition_offsets = vec![0u32; n_partitions as usize + 1];
    for i in 0..n_partitions as usize {
        let b = &counts_bytes[i * 4..(i + 1) * 4];
        partition_offsets[i + 1] = partition_offsets[i] +
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    let buf_offsets = device.new_buffer_from_bytes(u32_bytes(&partition_offsets)).expect("bo");
    let buf_cursors = device.new_buffer_zeroed(n_partitions as usize * 4).expect("bcu");
    let buf_row_idx = device.new_buffer_zeroed(n_rows as usize * 4).expect("bri");
    times[3] = t0.elapsed().as_secs_f64() * 1000.0; // cpu_scan_1

    // 5. partition_scatter dispatch + wait.
    let t0 = Instant::now();
    queue.dispatch_1d(
        &pso_scatter,
        &[&buf_keys_lo, &buf_keys_hi, &buf_offsets, &buf_cursors, &buf_row_idx,
          &buf_n_rows, &buf_n_part, &buf_log2],
        n_rows as usize,
    ).expect("scatter");
    queue.wait_until_complete().expect("wait");
    times[4] = t0.elapsed().as_secs_f64() * 1000.0; // dispatch_scatter

    // 6. Build buffers + dispatch.
    let t0 = Instant::now();
    let buf_r2lg = device.new_buffer_zeroed(n_rows as usize * 4).expect("br2");
    let buf_ng_per_part = device.new_buffer_zeroed(n_partitions as usize * 4).expect("bng");
    let buf_overflow = device.new_buffer_zeroed(4).expect("bov");
    let tg_width = 256usize;
    queue.dispatch_1d_with_tg(
        &pso_build,
        &[&buf_keys_lo, &buf_keys_hi, &buf_row_idx, &buf_offsets,
          &buf_r2lg, &buf_ng_per_part, &buf_overflow, &buf_n_rows],
        n_partitions as usize * tg_width,
        tg_width,
    ).expect("build");
    queue.wait_until_complete().expect("wait");
    times[5] = t0.elapsed().as_secs_f64() * 1000.0; // dispatch_build

    // 7. Readback per-partition counts.
    let t0 = Instant::now();
    let ngp_bytes = buf_ng_per_part.as_slice();
    let mut n_groups_per_part = vec![0u32; n_partitions as usize];
    for i in 0..n_partitions as usize {
        let b = &ngp_bytes[i * 4..(i + 1) * 4];
        n_groups_per_part[i] = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    let r2lg_bytes = buf_r2lg.as_slice();
    let mut row_to_local_group = vec![0u32; n_rows as usize];
    for i in 0..n_rows as usize {
        let b = &r2lg_bytes[i * 4..(i + 1) * 4];
        row_to_local_group[i] = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    times[6] = t0.elapsed().as_secs_f64() * 1000.0; // readback

    // 8. Final CPU row_to_group derivation.
    let t0 = Instant::now();
    let mut partition_group_offset = vec![0u32; n_partitions as usize + 1];
    for i in 0..n_partitions as usize {
        partition_group_offset[i + 1] = partition_group_offset[i] + n_groups_per_part[i];
    }
    let n_groups = partition_group_offset[n_partitions as usize];
    let mut row_to_group = vec![0u32; n_rows as usize];
    let mut first_row_per_group = vec![u32::MAX; n_groups as usize];
    for r in 0..n_rows as usize {
        let p = partition_id(keys[r], n_partitions) as usize;
        let local = row_to_local_group[r];
        let global = partition_group_offset[p] + local;
        row_to_group[r] = global;
        if first_row_per_group[global as usize] == u32::MAX {
            first_row_per_group[global as usize] = r as u32;
        }
    }
    times[7] = t0.elapsed().as_secs_f64() * 1000.0; // final_derive

    let total = t_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  A1 total: {total:.2}ms (sum: {:.2}ms)", times.iter().sum::<f64>());
    times
}

// ============================================================
//  A2 instrumented (one-pass breakdown per lane)
// ============================================================

fn profile_a2(device: &MetalDevice, keys: &[u128]) -> Vec<f64> {
    use polars_metal_kernels::groupby_build_sort::gpu::run_radix_lane;

    let mut times = vec![0.0f64; 6];
    let t_start = Instant::now();

    // 0. Key copy / setup.
    let t0 = Instant::now();
    let n_rows: u32 = keys.len() as u32;
    let mut current_keys: Vec<u128> = keys.to_vec();
    let mut current_idx: Vec<u32> = (0..n_rows).collect();
    times[0] = t0.elapsed().as_secs_f64() * 1000.0; // setup

    // 1-3. Per-lane: time first lane fully, then collect 15-lane total.
    let t0 = Instant::now();
    let (next_keys, next_idx) = run_radix_lane(device, &current_keys, &current_idx, 0)
        .expect("lane0");
    current_keys = next_keys;
    current_idx = next_idx;
    times[1] = t0.elapsed().as_secs_f64() * 1000.0; // one_lane

    let t0 = Instant::now();
    for lane in 1u32..16 {
        let (k, i) = run_radix_lane(device, &current_keys, &current_idx, lane)
            .expect("lane");
        current_keys = k;
        current_idx = i;
    }
    times[2] = t0.elapsed().as_secs_f64() * 1000.0; // remaining_15_lanes

    // 4. Segment kernel.
    let t0 = Instant::now();
    let sorted_lo: Vec<u64> = current_keys.iter().map(|k| *k as u64).collect();
    let sorted_hi: Vec<u64> = current_keys.iter().map(|k| (*k >> 64) as u64).collect();
    let u64_bytes = |s: &[u64]| unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, size_of_val(s))
    };
    let lib = shared_library(device).expect("lib");
    let pso = lib.pipeline("segment_starts").expect("pso");
    let starts_size_bytes = ((keys.len() + 7) >> 3).next_multiple_of(4).max(4);
    let buf_lo = device.new_buffer_from_bytes(u64_bytes(&sorted_lo)).expect("blo");
    let buf_hi = device.new_buffer_from_bytes(u64_bytes(&sorted_hi)).expect("bhi");
    let buf_starts = device.new_buffer_zeroed(starts_size_bytes).expect("bs");
    let buf_n = device.new_buffer_from_bytes(&n_rows.to_le_bytes()).expect("bn");
    let mut queue = CommandQueue::new(device).expect("queue");
    queue.dispatch_1d(&pso, &[&buf_lo, &buf_hi, &buf_starts, &buf_n], n_rows as usize)
        .expect("segment");
    queue.wait_until_complete().expect("wait");
    times[3] = t0.elapsed().as_secs_f64() * 1000.0; // segment

    // 5. CPU scan.
    let t0 = Instant::now();
    let starts = buf_starts.as_slice();
    let mut row_to_group = vec![0u32; keys.len()];
    let mut cur_group: u32 = 0;
    let sorted_idx = &current_idx;
    for i in 0..keys.len() {
        let bit = (starts[i >> 3] >> (i & 7)) & 1u8;
        if i > 0 && bit == 1 {
            cur_group += 1;
        }
        row_to_group[sorted_idx[i] as usize] = cur_group;
    }
    times[4] = t0.elapsed().as_secs_f64() * 1000.0; // cpu_scan

    let total = t_start.elapsed().as_secs_f64() * 1000.0;
    times[5] = total - times[..5].iter().sum::<f64>(); // misc
    eprintln!("  A2 total: {total:.2}ms");
    times
}

// ============================================================
//  CPU HashMap instrumented
// ============================================================

fn profile_cpu(_device: &MetalDevice, keys: &[u128]) -> Vec<f64> {
    let mut times = vec![0.0f64; 1];
    let t_start = Instant::now();

    let mut group_for_key: std::collections::HashMap<u128, u32> =
        std::collections::HashMap::with_capacity(keys.len().min(1 << 20));
    let mut next_gid: u32 = 0;
    let mut row_to_group = Vec::with_capacity(keys.len());
    let mut first_row_per_group: Vec<u32> = Vec::new();
    for (row, &key) in keys.iter().enumerate() {
        let gid = *group_for_key.entry(key).or_insert_with(|| {
            let g = next_gid;
            next_gid += 1;
            first_row_per_group.push(row as u32);
            g
        });
        row_to_group.push(gid);
    }
    let _ = row_to_group;
    let _ = first_row_per_group;

    times[0] = t_start.elapsed().as_secs_f64() * 1000.0;
    times
}

// ============================================================
//  Test entry
// ============================================================

#[test]
#[ignore = "perf data collector"]
fn profile_breakdown_all_paths() {
    let device = MetalDevice::system_default().expect("metal device");

    let cases: &[(usize, u32)] = &[
        (1_000_000, 4),
        (1_000_000, 1024),
        (10_000_000, 4),
        (10_000_000, 1024),
    ];

    for &(n_rows, n_groups) in cases {
        eprintln!("\n=== n_rows={n_rows} n_groups={n_groups} ===");
        let keys: Vec<u128> = (0..n_rows).map(|i| (i % n_groups as usize) as u128).collect();

        eprintln!("-- A1 breakdown (median over {MEASURE_ITERS} iters)");
        let a1 = run_times(|| profile_a1(&device, &keys, 16));
        let a1_labels = [
            "key_split", "alloc_scatter", "dispatch_count",
            "cpu_scan_1", "dispatch_scatter", "dispatch_build",
            "readback", "final_derive",
        ];
        for (l, t) in a1_labels.iter().zip(a1.iter()) {
            println!("  A1 [{l:>16}]  {t:>7.2} ms");
        }
        println!("  A1 sum: {:.2} ms", a1.iter().sum::<f64>());

        // A2 (skip if 10M to save time — known 2s/run, do 1 iter only).
        if n_rows >= 10_000_000 {
            eprintln!("-- A2 (single iter at 10M)");
            let a2 = profile_a2(&device, &keys);
            let a2_labels = ["setup", "lane0", "lanes_1_15", "segment", "cpu_scan", "misc"];
            for (l, t) in a2_labels.iter().zip(a2.iter()) {
                println!("  A2 [{l:>16}]  {t:>7.2} ms");
            }
            println!("  A2 sum: {:.2} ms", a2.iter().sum::<f64>());
        } else {
            eprintln!("-- A2 breakdown (median over {MEASURE_ITERS} iters)");
            let a2 = run_times(|| profile_a2(&device, &keys));
            let a2_labels = ["setup", "lane0", "lanes_1_15", "segment", "cpu_scan", "misc"];
            for (l, t) in a2_labels.iter().zip(a2.iter()) {
                println!("  A2 [{l:>16}]  {t:>7.2} ms");
            }
            println!("  A2 sum: {:.2} ms", a2.iter().sum::<f64>());
        }

        eprintln!("-- CPU breakdown");
        let cpu = run_times(|| profile_cpu(&device, &keys));
        println!("  CPU [hashmap_insert]  {:>7.2} ms", cpu[0]);
    }
}
