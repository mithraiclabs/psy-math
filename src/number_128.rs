use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use bytemuck::{Pod, Zeroable};

const PRECISION: i32 = 10;
const ONE: i128 = 10_000_000_000;

const POWERS_OF_TEN: &[i128] = &[
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
];

/// A fixed-point decimal number 128 bits wide
#[derive(Pod, Zeroable, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
pub struct Number128(i128);

impl Number128 {
    pub const ONE: Self = Self(ONE);
    pub const ZERO: Self = Self(0i128);
    pub const MAX: Self = Self(i128::MAX);
    pub const MIN: Self = Self(i128::MIN);
    pub const BITS: u32 = i128::BITS;

    /// Convert this number to fit in a u64
    ///
    /// The precision of the number in the u64 is based on the
    /// exponent provided.
    pub fn as_u64(&self, exponent: impl Into<i32>) -> u64 {
        let extra_precision = PRECISION + exponent.into();
        let prec_value = POWERS_OF_TEN[extra_precision.unsigned_abs() as usize];

        let target_value = if extra_precision < 0 {
            self.0 * prec_value
        } else {
            self.0 / prec_value
        };

        if target_value > std::u64::MAX as i128 {
            panic!("cannot convert to u64 due to overflow");
        }

        if target_value < 0 {
            panic!("cannot convert to u64 because value < 0");
        }

        target_value as u64
    }

    /// Convert this number to a f64
    pub fn as_f64(&self) -> f64 {
        // i128::{MAX|MIN} fits within f64
        self.to_i128() as f64 / 10_000_000_000.0
    }

    /// Convert another integer
    pub fn from_decimal(value: impl Into<i128>, exponent: impl Into<i32>) -> Self {
        let extra_precision = PRECISION + exponent.into();
        let prec_value = POWERS_OF_TEN[extra_precision.unsigned_abs() as usize];

        if extra_precision < 0 {
            Self(value.into() / prec_value)
        } else {
            Self(value.into() * prec_value)
        }
    }

    /// Convert from basis points
    pub fn from_bps(basis_points: u16) -> Self {
        Self::from_decimal(basis_points, crate::BPS_EXPONENT)
    }

    /// Get the underlying 128-bit representation in bytes.
    /// Uses the target endianness of the caller
    pub fn into_bits(self) -> [u8; 16] {
        self.0.to_ne_bytes()
    }

    /// Read a number from a raw 128-bit representation, which was previously
    /// returned by a call to `into_bits`.
    /// Uses the target endianness of the caller
    pub fn from_bits(bits: [u8; 16]) -> Self {
        Self(i128::from_ne_bytes(bits))
    }

    /// Get the underlying i128 value
    pub fn to_i128(self) -> i128 {
        self.0
    }

    /// Create `Number128` from an `i128`
    pub fn from_i128(value: i128) -> Self {
        Self(value)
    }
}

impl std::fmt::Debug for Number128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Display>::fmt(self, f)
    }
}

impl std::fmt::Display for Number128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // todo optimize
        let rem = self.0 % ONE;
        let decimal_digits = PRECISION as usize;
        // convert to abs to remove sign
        let rem_str = rem.checked_abs().unwrap().to_string();
        // regular padding like {:010} doesn't work with i128
        let decimals = "0".repeat(decimal_digits - rem_str.len()) + &*rem_str;
        let stripped_decimals = decimals.trim_end_matches('0');
        let pretty_decimals = if stripped_decimals.is_empty() {
            "0"
        } else {
            stripped_decimals
        };
        if self.0 < -ONE {
            let int = self.0 / ONE;
            write!(f, "{}.{}", int, pretty_decimals)?;
        } else if self.0 < 0 {
            write!(f, "-0.{}", pretty_decimals)?;
        } else if self.0 < ONE {
            write!(f, "0.{}", pretty_decimals)?;
        } else {
            let int = self.0 / ONE;
            write!(f, "{}.{}", int, pretty_decimals)?;
        }
        Ok(())
    }
}

impl Add<Number128> for Number128 {
    type Output = Self;

    fn add(self, rhs: Number128) -> Self::Output {
        Self(self.0.checked_add(rhs.0).unwrap())
    }
}

impl AddAssign<Number128> for Number128 {
    fn add_assign(&mut self, rhs: Number128) {
        self.0 = self.0.checked_add(rhs.0).unwrap();
    }
}

impl Sub<Number128> for Number128 {
    type Output = Self;

    fn sub(self, rhs: Number128) -> Self::Output {
        Self(self.0.checked_sub(rhs.0).unwrap())
    }
}

impl SubAssign<Number128> for Number128 {
    fn sub_assign(&mut self, rhs: Number128) {
        self.0 = self.0.checked_sub(rhs.0).unwrap();
    }
}

impl Mul<Number128> for Number128 {
    type Output = Number128;

    fn mul(self, rhs: Number128) -> Self::Output {
        // Product is divided by ONE as RHS is also a fixed point number.
        Self(div_by_one(fast_checked_mul(self.0, rhs.0).unwrap()))
    }
}

impl MulAssign<Number128> for Number128 {
    fn mul_assign(&mut self, rhs: Number128) {
        // Product is divided by ONE as RHS is also a fixed point number.
        self.0 = div_by_one(fast_checked_mul(self.0, rhs.0).unwrap())
    }
}

impl Div<Number128> for Number128 {
    type Output = Number128;

    fn div(self, rhs: Number128) -> Self::Output {
        // Both div and checked_div panic on overflow or zero division,
        // so we use div directly.
        // print!("div: {} / {} = ", mul_by_one(self.0), rhs.0);
        Self(mul_by_one(self.0).div(rhs.0))
    }
}

impl DivAssign<Number128> for Number128 {
    fn div_assign(&mut self, rhs: Number128) {
        // Both div and checked_div panic on overflow or zero division,
        // so we use div directly.
        self.0 = mul_by_one(self.0).div(rhs.0);
    }
}

impl<T: Into<i128>> Mul<T> for Number128 {
    type Output = Number128;

    fn mul(self, rhs: T) -> Self::Output {
        Self(fast_checked_mul(self.0, rhs.into()).unwrap())
    }
}

impl<T: Into<i128>> Div<T> for Number128 {
    type Output = Number128;

    fn div(self, rhs: T) -> Self::Output {
        // Both div and checked_div panic on overflow or zero division,
        // so we use div directly.
        Self(self.0.div(rhs.into()))
    }
}

impl<T: Into<i128>> From<T> for Number128 {
    fn from(n: T) -> Self {
        Self::from_i128(n.into())
    }
}

impl Neg for Number128 {
    type Output = Number128;

    fn neg(self) -> Self::Output {
        Number128(-self.0)
    }
}

/// Divides value by ONE, which is `10_000_000_000_i128`. This is implemented
/// as a right bit-shift (on absolute value) by 10, followed by a division by
/// `9_765_625` (which is `5^10`), as this is faster than a division by `10_000_000_000`.
/// The sign is then restored before returning the result.
///
/// Works for all i128 inputs except `i128::MIN`
fn div_by_one(value: i128) -> i128 {
    // abs_result is expected to be positive unless
    // value.abs() has overflowed when value == i128::MIN
    let abs_result = (value.abs() >> 10) / (9_765_625_i128);

    // Return result with sign of value.
    // For abs_result < 0, return result without sign bit change.
    if value > 0 || abs_result < 0 {
        abs_result
    } else {
        -abs_result
    }
}

const ONE_REPR_BITS: u32 = 34; // bits needed to represent ONE (excluding sign bit)

/// Multiplies value by ONE, which is `10_000_000_000_i128`.
/// This is implemented as multiplication by `9_765_625` (which is `5^10`), followed by
/// a left bit-shift by 10, as this is faster than a multiplication by `10_000_000_000`.
///
/// Largest supported input: `i128::MAX >> 34 = 2^93 ~= 9.9^27`
///
/// Smallest supported input: `i128::MIN >> 35 = -2^93 ~= -9.9^27`
fn mul_by_one(value: i128) -> i128 {
    // Check that sum of bits required to represent product does not exceed
    // 128 bits. This is a conservative estimate, so it may return false positives.
    // Note that checked_abs is not used here, since the overflow case would
    // be caught by the following check.
    let left_bits = 128 - value.abs().leading_zeros();
    if (left_bits + ONE_REPR_BITS + 1) > 128 {
        panic!("Overflow in mul by one")
    }
    (value * 9_765_625_i128) << 10
}

/// Checks if the multiplication of two i128 values will overflow, This is
/// a conservative estimate, so it may return false positives
/// (detecting overflow when there is none).
fn fast_checked_mul(left: i128, right: i128) -> Option<i128> {
    if right == 0 || left == 0 {
        return Some(0);
    }

    // Convert values to positive first, as negative value always have no leading zeros.
    // Gets bits required to represent the absolute value, excluding the sign bit.
    // Note that checked_abs is not used here, since the overflow case (for i128::MIN)
    // would be caught by the following bit check.
    let left_bits = 128 - left.abs().leading_zeros();
    let right_bits = 128 - right.abs().leading_zeros();

    // Assume that a conservative case that both right and left value have
    // ones for left_bits and right_bits respectively. Therefore, the product
    // of the two values will require left_bits + right_bits bits to represent,
    // plus one sign bit.z
    if (left_bits + right_bits + 1) > 128 {
        return None;
    }
    Some(left * right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_equals_zero() {
        assert_eq!(Number128::ZERO, Number128::from_decimal(0, 0));
    }

    #[test]
    fn one_equals_one() {
        assert_eq!(Number128::ONE, Number128::from_decimal(1, 0));
    }

    #[test]
    fn negative_one_equals_negative_one() {
        assert_eq!(-Number128::ONE, Number128::from_decimal(-1, 0));
    }

    #[test]
    fn one_plus_one_equals_two() {
        assert_eq!(
            Number128::from_decimal(2, 0),
            Number128::ONE + Number128::ONE
        );
    }

    #[test]
    fn one_minus_one_equals_zero() {
        assert_eq!(Number128::ONE - Number128::ONE, Number128::ZERO);
    }

    #[test]
    fn one_times_one_equals_one() {
        // Mul
        assert_eq!(Number128::ONE, Number128::ONE * Number128::ONE);

        // MulAssign
        let mut x = Number128::ONE;
        x *= Number128::ONE;
        assert_eq!(Number128::ONE, x);

        // Mul Into
        assert_eq!(Number128::ONE, Number128::ONE * 1);
    }

    #[test]
    fn one_divided_by_one_equals_one() {
        // Div
        assert_eq!(Number128::ONE, Number128::ONE / Number128::ONE);

        // DivAssign
        let mut x = Number128::ONE;
        x /= Number128::ONE;
        assert_eq!(Number128::ONE, x);

        // Div Into
        assert_eq!(Number128::ONE, Number128::ONE / 1);
    }

    #[test]
    fn test_mul_by_one() {
        let one = 10_000_000_000_i128;

        // Multiple of ONE or -ONE by ONE
        assert_eq!(mul_by_one(one), one * one);
        assert_eq!(mul_by_one(-one), -one * one);

        // Multiple of (abs) values smaller than ONE by ONE.
        assert_eq!(mul_by_one(9_999_999_999_i128), 9_999_999_999_i128 * one);
        assert_eq!(mul_by_one(1), one);
        assert_eq!(mul_by_one(0), 0);
        assert_eq!(mul_by_one(-1), -one);
        assert_eq!(mul_by_one(-9_999_999_999_i128), -9_999_999_999_i128 * one);

        // Multiple of (abs) values larger than ONE by ONE.
        assert_eq!(mul_by_one(10_000_000_001_i128), 10_000_000_001_i128 * one);
        assert_eq!(mul_by_one(-10_000_000_001_i128), -10_000_000_001_i128 * one);
        assert_eq!(mul_by_one(123_456_000_000_000), 123_456_000_000_000 * one);
        assert_eq!(mul_by_one(-123_456_000_000_000), -123_456_000_000_000 * one);

        // The largest supported value is an i128 where the first 33 bits following
        // the sign bit are 0 (i128::MAX >> 34)
        let big_value = i128::MAX >> 34;
        assert_eq!(mul_by_one(big_value), (big_value) * one);
        // The smallest supported value is similar, except one more bit is needed.
        let small_value = i128::MIN >> 35;
        assert_eq!(mul_by_one(small_value), (small_value) * one);
    }

    #[test]
    #[should_panic = "Overflow in mul by one"]
    fn test_mul_by_one_overflow_high() {
        let one = 10_000_000_000_i128;
        let big_value = i128::MAX >> 33;
        let answer = mul_by_one(big_value);
        assert_eq!(answer, big_value * one);
    }

    #[test]
    #[should_panic = "Overflow in mul by one"]
    fn test_mul_by_one_overflow_low() {
        let one = 10_000_000_000_i128;
        let small_value = i128::MIN >> 34;
        let answer = mul_by_one(small_value);
        assert_eq!(answer, small_value * one);
    }

    #[test]
    #[should_panic = "Overflow in mul by one"]
    fn test_mul_by_one_overflow_min_i128() {
        let one = 10_000_000_000_i128;
        assert_eq!(mul_by_one(i128::MIN), i128::MIN * one);
    }

    #[test]
    fn test_div_by_one() {
        let one = 10_000_000_000_i128;

        // Division of ONE or -ONE by ONE
        assert_eq!(div_by_one(one), one.checked_div(one).unwrap());
        assert_eq!(div_by_one(-one), -one.checked_div(one).unwrap());

        // Division of (abs) values smaller than ONE by ONE.
        assert_eq!(
            div_by_one(9_999_999_999_i128),
            9_999_999_999_i128.checked_div(one).unwrap()
        );
        assert_eq!(div_by_one(1), 1_i128.checked_div(one).unwrap());
        assert_eq!(div_by_one(0), 0);
        assert_eq!(div_by_one(-1), -1_i128.checked_div(one).unwrap());
        assert_eq!(
            div_by_one(-9_999_999_999_i128),
            -9_999_999_999_i128.checked_div(one).unwrap()
        );

        // Division of (abs) values larger than ONE by ONE.
        assert_eq!(
            div_by_one(10_000_000_001_i128),
            10_000_000_001_i128.checked_div(one).unwrap()
        );
        assert_eq!(
            div_by_one(-10_000_000_001_i128),
            (-10_000_000_001_i128).checked_div(one).unwrap()
        );
        assert_eq!(
            div_by_one(123_456_000_000_000),
            123_456_000_000_000_i128.checked_div(one).unwrap()
        );
        assert_eq!(
            div_by_one(-123_456_000_000_000),
            (-123_456_000_000_000_i128).checked_div(one).unwrap()
        );

        // No overflow on MAX value, or values down to MIN + 1
        assert_eq!(div_by_one(i128::MAX), i128::MAX.checked_div(one).unwrap());
        assert_eq!(div_by_one(i128::MIN + 1), (i128::MIN + 1) / one);

        // No overflow on MIN value.
        assert_eq!(div_by_one(i128::MIN), i128::MIN.checked_div(one).unwrap());
    }

    #[test]
    fn test_fast_checked_mul() {
        let test_cases = [
            (10, 10),
            (0, 10),
            (10, 0),
            (-10, 10),
            (10, -10),
            (-10, -10),
            (1_000_000, 1_000_000),
            (2_000_000, 2_000_000),
            (i128::MAX >> 1, 1),
            (i128::MAX >> 2, 2),
            (1, i128::MAX >> 1),
            (2, i128::MAX >> 2),
            (3_000_000_000, 3_000_000_000), // both overflow
            (i128::MAX, 2),                 // both overflow
            (2, i128::MAX),                 // both overflow
            (i128::MIN, -1),                // both overflow
            (-1, i128::MIN),                // both overflow
            (i128::MIN, i128::MIN),         // both overflow
        ];

        for &(left, right) in &test_cases {
            let answer = fast_checked_mul(left, right);
            let expected = left.checked_mul(right);
            assert_eq!(answer, expected);
        }
    }

    #[test]
    fn test_fast_checked_failures() {
        // Test cases when fast_checked_mul detects false positives
        // and expects overflow, even when checked_mul does not.
        let test_cases = [
            (i128::MAX, 1),
            (i128::MAX >> 1, 2),
            (i128::MAX >> 2, 4),
            (i128::MAX >> 3, 8),
            (i128::MIN + 1, 1),
        ];

        for &(left, right) in &test_cases {
            let answer = fast_checked_mul(left, right);
            let expected = left.checked_mul(right);
            assert_ne!(answer, expected);
        }
    }

    #[test]
    fn ten_div_100_equals_point_1() {
        // Div
        assert_eq!(
            Number128::from_decimal(1, -1),
            Number128::from_decimal(1, 1) / Number128::from_decimal(100, 0)
        );

        // Div Assign
        let mut x = Number128::from_decimal(1, 1);
        x /= Number128::from_decimal(100, 0);
        assert_eq!(Number128::from_decimal(1, -1), x);

        // Div Into
        assert_eq!(
            Number128::from_decimal(1, -1),
            Number128::from_decimal(1, 1) / 100
        );
    }

    #[test]
    fn comparison() {
        let a = Number128::from_decimal(1000, -4);
        let b = Number128::from_decimal(10, -2);
        assert!(a >= b);

        let c = Number128::from_decimal(1001, -4);
        assert!(c > a);
        assert!(c > b);

        let d = Number128::from_decimal(9999999, -8);
        assert!(d < a);
        assert!(d < b);
        assert!(d < c);
        assert!(d <= d);

        assert_eq!(a.cmp(&b), std::cmp::Ordering::Equal);
        assert_eq!(a.cmp(&c), std::cmp::Ordering::Less);
        assert_eq!(a.cmp(&d), std::cmp::Ordering::Greater);
    }

    #[test]
    fn multiply_by_u64() {
        assert_eq!(
            Number128::from_decimal(3, 1),
            Number128::from_decimal(1, 1) * 3u64
        )
    }

    #[test]
    fn test_add_assign_101_2() {
        let mut a = Number128::from_decimal(101, 0);
        a += Number128::from_decimal(2, 0);
        assert_eq!(Number128::from_decimal(103, 0), a);
    }

    #[test]
    fn test_sub_assign_101_2() {
        let mut a = Number128::from_decimal(101, 0);
        a -= Number128::from_decimal(2, 0);
        assert_eq!(Number128::from_decimal(99, 0), a);
    }

    #[test]
    fn test_mul_assign_101_2() {
        let mut a = Number128::from_decimal(101, 0);
        a *= Number128::from_decimal(2, 0);
        assert_eq!(Number128::from_decimal(202, 0).0, a.0);
    }

    #[test]
    fn test_div_assign_101_2() {
        let mut a = Number128::from_decimal(101, 0);
        a /= Number128::from_decimal(2, 0);
        assert_eq!(Number128::from_decimal(505, -1), a);
    }

    #[test]
    fn test_div_assign_102_3() {
        let mut a = Number128::from_decimal(1, 1);
        a /= Number128::from_decimal(100, 0);
        assert_eq!(Number128::from_decimal(1, -1).0, a.0);
    }

    #[test]
    fn div_into_i128() {
        let a = Number128::from_decimal(1000, 0);
        let b = a / 500;
        assert_eq!(Number128::from_decimal(2, 0), b);

        let c = Number128::from_decimal(1000, -3);
        let d = c / 3;
        assert_eq!(Number128::from_decimal(3333333333i64, -10).0, d.0);
    }

    #[test]
    fn equality() {
        let a = Number128::from_decimal(1000, -4);
        let b = Number128::from_decimal(10, -2);
        assert_eq!(a, b);

        let c = Number128::from_decimal(-1000, -4);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn as_u64() {
        let u64in = 31455;
        let a = Number128::from_decimal(u64in, -3);
        let b = a.as_u64(-3);
        assert_eq!(b, u64in);
    }

    #[test]
    #[should_panic = "cannot convert to u64 because value < 0"]
    fn as_u64_panic_neg() {
        let a = Number128::from_decimal(-10000, -3);
        a.as_u64(-3);
    }

    #[test]
    #[should_panic = "cannot convert to u64 due to overflow"]
    fn as_u64_panic_big() {
        let a = Number128::from_decimal(u64::MAX as i128 + 1, -3);
        a.as_u64(-3);
    }

    #[test]
    fn as_f64() {
        let n = Number128::from_bps(15000);
        assert_eq!(1.5, n.as_f64());

        // Test that conversion is within bounds and doesn't lose precision for min
        let n = Number128::MIN; // -170141183460469231731687303715884105728
        assert_eq!(-17014118346046923173168730371.5884105728, n.as_f64());

        // Test that conversion is within bounds and doesn't lose precision for max
        let n = Number128::MAX; // 170141183460469231731687303715884105727
        assert_eq!(17014118346046923173168730371.5884105727, n.as_f64());

        // More cases
        let n = Number128::from_bps(0) - Number128::from_bps(15000);
        assert_eq!(-1.5, n.as_f64());

        let n = Number128::from_decimal(12345678901i128, -10);
        assert_eq!(1.2345678901, n.as_f64());

        let n = Number128::from_decimal(-12345678901i128, -10);
        assert_eq!(-1.2345678901, n.as_f64());

        let n = Number128::from_decimal(-12345678901i128, -9);
        assert_eq!(-12.345678901, n.as_f64());

        let n = Number128::from_decimal(12345678901i128, -9);
        assert_eq!(12.345678901, n.as_f64());

        let n = Number128::from_decimal(ONE - 1, 1);
        assert_eq!(99999999990.0, n.as_f64());

        let n = Number128::from_decimal(12345678901i128, -13);
        assert_eq!(0.0012345678, n.as_f64());

        let n = Number128::from_decimal(-12345678901i128, -13);
        assert_eq!(-0.0012345678, n.as_f64());
    }

    #[test]
    fn display() {
        let a = Number128::from_bps(15000);
        assert_eq!("1.5", a.to_string().as_str());

        let a = Number128::from_bps(0) - Number128::from_bps(15000);
        assert_eq!("-1.5", a.to_string().as_str());

        let b = Number128::from_decimal(12345678901i128, -10);
        assert_eq!("1.2345678901", b.to_string().as_str());

        let b = Number128::from_decimal(-12345678901i128, -10);
        assert_eq!("-1.2345678901", b.to_string().as_str());

        let c = Number128::from_decimal(-12345678901i128, -9);
        assert_eq!("-12.345678901", c.to_string().as_str());

        let c = Number128::from_decimal(12345678901i128, -9);
        assert_eq!("12.345678901", c.to_string().as_str());

        let d = Number128::from_decimal(ONE - 1, 1);
        assert_eq!("99999999990.0", d.to_string().as_str());

        let e = Number128::from_decimal(12345678901i128, -13);
        assert_eq!("0.0012345678", e.to_string().as_str());

        let e = Number128::from_decimal(-12345678901i128, -13);
        assert_eq!("-0.0012345678", e.to_string().as_str());
    }

    #[test]
    fn into_bits() {
        let bits = Number128::from_decimal(1242, -3).into_bits();
        let number = Number128::from_bits(bits);

        assert_eq!(Number128::from_decimal(1242, -3), number);
    }

    #[test]
    fn mul_overflow() {
        // Overflow when multiplying u64 max by u64 max.
        let x = Number128::from_decimal(u64::MAX, 0);
        assert!(std::panic::catch_unwind(|| x * x).is_err());

        // Overflow when multiplying i128 min by 2
        let x = Number128::from_i128(i128::MIN);
        let y = Number128::from_i128(2);
        assert!(std::panic::catch_unwind(|| x * y).is_err());

        // Overflow when multiplying ((0.5 * i128 max) + 1) by 2
        let mut x = Number128::from_i128(i128::MAX) / 2;
        x += Number128::from_i128(1);
        assert!(std::panic::catch_unwind(|| x * 2).is_err());
    }

    #[test]
    fn mul_assign_overflow() {
        // Overflow when multiplying u64 max by u64 max.
        let mut x = Number128::from_decimal(u64::MAX, 0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            x *= x;
        }));
        assert!(result.is_err());

        // Overflow when multiplying i128 min by 2
        let mut x = Number128::from_i128(i128::MIN);
        let y = Number128::from_i128(2);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            x *= y;
        }));
        assert!(result.is_err());

        // Overflow when multiplying ((0.5 * i128 max) + 1) by 2
        let mut x = Number128::from_i128(i128::MAX) / 2;
        x += Number128::from_i128(1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| x * 2));
        assert!(result.is_err());
    }

    #[test]
    fn div_overflow() {
        let x = Number128::from_decimal(u64::MAX, 0);
        assert!(std::panic::catch_unwind(|| x / x).is_err());
        let x = Number128::from_i128(i128::MIN);
        assert!(std::panic::catch_unwind(|| x / -1).is_err());
        assert!(std::panic::catch_unwind(|| x / 0).is_err());
    }

    #[test]
    fn div_assign_overflow() {
        let mut x = Number128::from_decimal(u64::MAX, 0);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            x /= x;
        }));
        assert!(result.is_err());

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            x /= Number128::from_i128(-1);
        }));
        assert!(result.is_err());

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            x /= Number128::from_i128(0);
        }));
        assert!(result.is_err());
    }

    #[test]
    fn div_into_overflow() {
        let x = Number128::from_i128(i128::MIN);
        assert!(std::panic::catch_unwind(|| x / -1).is_err());
        assert!(std::panic::catch_unwind(|| x / 0).is_err());
    }
}
