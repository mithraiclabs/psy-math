use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;

fn div_version(n: i128) -> i128 {
    n / 10_000_000_000
}

fn shift_version(n: i128) -> i128 {
    (n >> 10) / (9_765_625_i128)
}

fn bench_div(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let dataset: Vec<i128> = (0..100000).map(|_| rng.gen()).collect();
    let mut group = c.benchmark_group("Div");
    group.measurement_time(std::time::Duration::new(30, 0)); // Increase this as necessary
    group.bench_with_input(BenchmarkId::new("div version", ""), &dataset, |b, i| {
        b.iter(|| {
            for &n in i {
                black_box(div_version(n));
            }
        })
    });
    group.finish();
}

fn bench_shift(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let dataset: Vec<i128> = (0..100000).map(|_| rng.gen()).collect();
    let mut group = c.benchmark_group("Shift");
    group.measurement_time(std::time::Duration::new(30, 0)); // Increase this as necessary
    group.bench_with_input(BenchmarkId::new("shift version", ""), &dataset, |b, i| {
        b.iter(|| {
            for &n in i {
                black_box(shift_version(n));
            }
        })
    });
    group.finish();
}

criterion_group!(benches, bench_div, bench_shift);
criterion_main!(benches);
