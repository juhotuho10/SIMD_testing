#![feature(portable_simd)]

use std::simd::{Simd, num::SimdUint};

fn main() {
    const V_SIZE: usize = 16000000;
    let a: Vec<u64> = (0..V_SIZE as u64).map(|i| i + 5).collect();
    let b: Vec<u64> = vec![10; V_SIZE];

    let sum = add_sum_simd(&a, &b);
    dbg!(sum);
}

#[inline(never)]
fn add_sum_simd(a: &[u64], b: &[u64]) -> u64 {
    assert_eq!(a.len(), b.len());

    const LANES: usize = 4;

    let a_chunks = a.chunks(LANES);
    let b_chunks = b.chunks(LANES);
    let mut v_out: Simd<u64, LANES> = Simd::splat(0);

    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        let va: Simd<u64, LANES> = Simd::load_or_default(a_chunk);
        let vb: Simd<u64, LANES> = Simd::load_or_default(b_chunk);
        v_out += va + vb;
    }

    v_out.reduce_sum()
}

fn add_sum_autovectorized(a: &[u64], b: &[u64]) -> u64 {
    let mut sum = 0;
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b.iter()) {
        sum += a + b;
    }
    sum
}

#[inline(never)]
fn add_sum_normal(a: &[u64], b: &[u64]) -> u64 {
    let mut sum = 0;
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b.iter()) {
        sum += add_nums(*a, *b);
    }
    sum
}

#[inline(never)] // inline never to make sure that there is no autovectorization of the add
fn add_nums(a: u64, b: u64) -> u64 {
    a + b
}
