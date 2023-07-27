To run unit tests: `cargo test --release`
Note that `release` flag is important as we want to test production behavior where certain overflow checks in non-checked math is disabled.
