//! CPU-side dictionary encoder for Utf8 columns (capability D phase 1).
//!
//! Build a `Vec<String>` of distinct values + a parallel `Vec<u32>` of
//! codes (offsets into the dictionary). The kernel layer then treats
//! the codes as if they were a u32 column for composite-key encoding;
//! decode reverses the mapping via the stored dictionary.
//!
//! First-seen-wins ordering keeps the encoding deterministic for a
//! given input row order, which the engine's result-decoding step
//! relies on for correctness against the CPU baseline.

use std::collections::HashMap;

/// Encode `strings` as `(dict, codes)`.
///
/// - `dict`: distinct values in first-seen order.
/// - `codes[i]`: the dictionary index for `strings[i]`.
pub fn build_dict(strings: &[&str]) -> (Vec<String>, Vec<u32>) {
    let mut dict: Vec<String> = Vec::new();
    let mut seen: HashMap<String, u32> = HashMap::new();
    let mut codes = Vec::with_capacity(strings.len());
    for &s in strings {
        let code = if let Some(c) = seen.get(s) {
            *c
        } else {
            let idx = dict.len() as u32;
            dict.push(s.to_string());
            seen.insert(s.to_string(), idx);
            idx
        };
        codes.push(code);
    }
    (dict, codes)
}

/// Null-aware variant. Returns `(dict, codes, valid)`:
///
/// - `valid[i] = false` ⇒ row i is null; `codes[i]` is a sentinel zero
///   and must be ignored by callers.
/// - Empty string `""` is a valid distinct value, treated like any
///   other string.
pub fn build_dict_nullable(strings: &[Option<&str>]) -> (Vec<String>, Vec<u32>, Vec<bool>) {
    let mut dict: Vec<String> = Vec::new();
    let mut seen: HashMap<String, u32> = HashMap::new();
    let mut codes = Vec::with_capacity(strings.len());
    let mut valid = Vec::with_capacity(strings.len());
    for opt in strings {
        match opt {
            Some(s) => {
                valid.push(true);
                let code = if let Some(c) = seen.get(*s) {
                    *c
                } else {
                    let idx = dict.len() as u32;
                    dict.push(s.to_string());
                    seen.insert(s.to_string(), idx);
                    idx
                };
                codes.push(code);
            }
            None => {
                valid.push(false);
                codes.push(0);
            }
        }
    }
    (dict, codes, valid)
}

/// Decode `codes` back to strings using `dict`. Panics if any code is
/// out of range; callers should validate first.
pub fn decode_dict(dict: &[String], codes: &[u32]) -> Vec<String> {
    codes.iter().map(|&c| dict[c as usize].clone()).collect()
}
