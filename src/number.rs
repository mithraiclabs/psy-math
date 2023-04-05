//! Yet another decimal library

use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use bytemuck::{Pod, Zeroable};
use std::fmt::{Display, Formatter};
use thiserror::Error;
use uint::construct_uint;

construct_uint! {
    #[derive(Pod, Zeroable)]
    pub struct U192(3);
}

pub const BPS_EXPONENT: i32 = -4;
const PRECISION: i32 = 15;
const ONE: U192 = U192([1_000_000_000_000_000, 0, 0]);
const U64_MAX: U192 = U192([0xffffffffffffffff, 0x0, 0x0]);

/// A large unsigned integer
#[derive(Pod, Zeroable, Default, Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
#[repr(transparent)]
pub struct Number(U192);

static_assertions::const_assert_eq!(24, std::mem::size_of::<Number>());
static_assertions::const_assert_eq!(0, std::mem::size_of::<Number>() % 8);
impl Number {
    pub const ONE: Number = Number(ONE);
    pub const ZERO: Number = Number(U192::zero());

    /// Convert this number to fit in a u64
    ///
    /// The precision of the number in the u64 is based on the
    /// exponent provided.
    pub fn as_u64(&self, exponent: impl Into<i32>) -> u64 {
        let extra_precision = PRECISION + exponent.into();
        let prec_value = Self::ten_pow(extra_precision.abs() as u32);

        let target_value = if extra_precision < 0 {
            self.0 * prec_value
        } else {
            self.0 / prec_value
        };

        if target_value > U64_MAX {
            panic!("cannot convert to u64 due to overflow");
        }

        target_value.as_u64()
    }

    /// Ceiling value of number, fit in a u64
    ///
    /// The precision of the number in the u64 is based on the
    /// exponent provided.
    ///
    /// The result is rounded up to the nearest one, based on the
    /// target precision.
    pub fn as_u64_ceil(&self, exponent: impl Into<i32>) -> u64 {
        let extra_precision = PRECISION + exponent.into();
        let prec_value = Self::ten_pow(extra_precision.abs() as u32);

        let target_rounded = prec_value - U192::from(1) + self.0;
        let target_value = if extra_precision < 0 {
            target_rounded * prec_value
        } else {
            target_rounded / prec_value
        };

        if target_value > U64_MAX {
            panic!("cannot convert to u64 due to overflow");
        }

        target_value.as_u64()
    }

    /// Convert this number to fit in a u64
    ///
    /// The precision of the number in the u64 is based on the
    /// exponent provided.
    ///
    /// The result is rounded to the nearest one, based on the
    /// target precision.
    pub fn as_u64_rounded(&self, exponent: impl Into<i32>) -> u64 {
        let extra_precision = PRECISION + exponent.into();
        let prec_value = Self::ten_pow(extra_precision.abs() as u32);

        let rounding = match extra_precision > 0 {
            true => U192::from(1) * prec_value / 2,
            false => U192::zero(),
        };

        let target_rounded = rounding + self.0;
        let target_value = if extra_precision < 0 {
            target_rounded * prec_value
        } else {
            target_rounded / prec_value
        };

        if target_value > U64_MAX {
            panic!("cannot convert to u64 due to overflow");
        }

        target_value.as_u64()
    }

    /// Convert another integer into a `Number`.
    pub fn from_decimal(value: impl Into<U192>, exponent: impl Into<i32>) -> Self {
        let extra_precision = PRECISION + exponent.into();
        let prec_value = Self::ten_pow(extra_precision.abs() as u32);

        if extra_precision < 0 {
            Self(value.into() / prec_value)
        } else {
            Self(value.into() * prec_value)
        }
    }

    /// Convert from basis points into a `Number`
    pub fn from_bps(basis_points: u16) -> Number {
        Number::from_decimal(basis_points, BPS_EXPONENT)
    }

    pub fn pow(&self, exp: impl Into<Number>) -> Number {
        let value = self.0.pow(exp.into().0);

        Self(value)
    }

    pub fn saturating_add(&self, n: Number) -> Number {
        Number(self.0.saturating_add(n.0))
    }

    pub fn saturating_sub(&self, n: Number) -> Number {
        Number(self.0.saturating_sub(n.0))
    }

    pub fn saturating_mul(&self, n: Number) -> Number {
        Number(self.0.saturating_mul(n.0))
    }

    pub fn ten_pow(exponent: u32) -> U192 {
        let value: u64 = match exponent {
            16 => 10_000_000_000_000_000,
            15 => 1_000_000_000_000_000,
            14 => 100_000_000_000_000,
            13 => 10_000_000_000_000,
            12 => 1_000_000_000_000,
            11 => 100_000_000_000,
            10 => 10_000_000_000,
            9 => 1_000_000_000,
            8 => 100_000_000,
            7 => 10_000_000,
            6 => 1_000_000,
            5 => 100_000,
            4 => 10_000,
            3 => 1_000,
            2 => 100,
            1 => 10,
            0 => 1,
            _ => panic!("no support for exponent: {}", exponent),
        };

        value.into()
    }

    /// Get the underlying representation in bits
    pub fn into_bits(self) -> [u8; 24] {
        unsafe { std::mem::transmute(self.0 .0) }
    }

    /// Read a number from a raw 196-bit representation, which was previously
    /// returned by a call to `into_bits`.
    pub fn from_bits(bits: [u8; 24]) -> Self {
        Self(U192(unsafe { std::mem::transmute(bits) }))
    }
}

impl<T: Into<U192>> From<T> for Number {
    fn from(n: T) -> Number {
        Number(n.into() * ONE)
    }
}

impl From<Number> for [u8; 24] {
    fn from(n: Number) -> Self {
        n.0.into()
    }
}

impl Display for Number {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // todo optimize
        let rem = self.0 % ONE;
        let decimal_digits = PRECISION as usize;
        let rem_str = rem.to_string();
        // regular padding like {:010} doesn't work with U192
        let decimals = "0".repeat(decimal_digits - rem_str.len()) + &*rem_str;
        let stripped_decimals = decimals.trim_end_matches('0');
        let pretty_decimals = if stripped_decimals.is_empty() {
            "0"
        } else {
            stripped_decimals
        };
        if self.0 < ONE {
            write!(f, "0.{}", pretty_decimals)?;
        } else {
            let int = self.0 / ONE;
            write!(f, "{}.{}", int, pretty_decimals)?;
        }
        Ok(())
    }
}

#[derive(Error, Debug, Clone, Eq, PartialEq)]
pub enum Error {
    #[error("An integer value overflowed")]
    Overflow(Number),

    #[error("Attempting to divide by zero")]
    DivideByZero,
}

impl Add<Number> for Number {
    type Output = Number;

    fn add(self, rhs: Number) -> Self::Output {
        Self(self.0.add(rhs.0))
    }
}

impl AddAssign<Number> for Number {
    fn add_assign(&mut self, rhs: Number) {
        self.0.add_assign(rhs.0)
    }
}

impl SubAssign<Number> for Number {
    fn sub_assign(&mut self, rhs: Number) {
        self.0.sub_assign(rhs.0)
    }
}

impl Sub<Number> for Number {
    type Output = Number;

    fn sub(self, rhs: Number) -> Self::Output {
        Self(self.0.sub(rhs.0))
    }
}

impl Mul<Number> for Number {
    type Output = Number;

    fn mul(self, rhs: Number) -> Self::Output {
        Self(self.0.mul(rhs.0).div(ONE))
    }
}

impl MulAssign<Number> for Number {
    fn mul_assign(&mut self, rhs: Number) {
        self.0.mul_assign(rhs.0);
        self.0.div_assign(ONE);
    }
}

impl Div<Number> for Number {
    type Output = Number;

    fn div(self, rhs: Number) -> Self::Output {
        Self(self.0.mul(ONE).div(rhs.0))
    }
}

impl<T: Into<U192>> Mul<T> for Number {
    type Output = Number;

    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0.mul(rhs.into()))
    }
}

impl<T: Into<U192>> Div<T> for Number {
    type Output = Number;

    fn div(self, rhs: T) -> Self::Output {
        Self(self.0.div(rhs.into()))
    }
}

impl Sum for Number {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| a + b).unwrap_or(Self::ZERO)
    }
}

/// Computes the Taylor expansion of exp(x) - 1, using the
/// indicated number of terms.
/// For example,
///     expm1_approx(x, 3) = x + x^2 / 2 + x^3 / 6
pub fn expm1_approx(x: Number, terms: usize) -> Number {
    if terms == 0 {
        return 0.into();
    }
    if terms == 1 {
        return x;
    }

    let mut z = x;
    let mut acc = x;
    let mut fac = 1u64;

    for k in 2..terms + 1 {
        z *= x;
        fac *= k as u64;
        acc += z / fac;
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    // For reference:
    // const SECONDS_PER_HOUR: u64 = 3600;
    // const SECONDS_PER_2H: u64 = SECONDS_PER_HOUR * 2;
    // const SECONDS_PER_12H: u64 = SECONDS_PER_HOUR * 12;
    // const SECONDS_PER_DAY: u64 = SECONDS_PER_HOUR * 24;
    // const SECONDS_PER_WEEK: u64 = SECONDS_PER_DAY * 7;
    // //Note: 365 days, does not account for leap-days
    // const SECONDS_PER_YEAR: u64 = 31_536_000;
    // const MAX_ACCRUAL_SECONDS: u64 = SECONDS_PER_WEEK;

    #[test]
    fn test_taylor_approx_point2ish() {
        /*
            x = .2
            e^.2 ~= 1.221402758160170
            e^.2 + 1 ~= 2.221402758160170
         */
        let expected: u64 = 221_402_758_160_170;
        let expected_number: Number = Number::from_decimal(expected, -15);
        // 221_402_666_666_665 <- actual result
        let answer = expm1_approx(Number::from_decimal(2, -1), 5);
        let tolerance = Number::from_decimal(10_000_000_000 as u64, -15);

        let diff = if expected_number.gt(&answer) {
            expected_number.sub(answer)
        }else{
            answer.sub(expected_number)
        };
        assert!(diff.lt(&tolerance));
    }

    #[test]
    fn test_taylor_approx_point3ish() {
        /*
            x = .3
            e^.2 ~= 1.349858807576000
            e^.2 + 1 ~= 2.349858807576000
         */
        let expected: u64 = 349_858_807_576_000;
        let expected_number: Number = Number::from_decimal(expected, -15);
        // 349_857_750_000_000 <- actual result
        let answer = expm1_approx(Number::from_decimal(3, -1), 5);
        let tolerance = Number::from_decimal(10_000_000_000 as u64, -15);

        let diff = if expected_number.gt(&answer) {
            expected_number.sub(answer)
        }else{
            answer.sub(expected_number)
        };
        
        assert!(diff.lt(&tolerance));
    }

    #[test]
    fn test_taylor_approx_maxish() {
        // assuming a max rate of 400%
        // max_rate * seconds_per_week / seconds_per_year = 4 * 604800 / 31536000 
        //    = 0.076712328767123 = 76712328767123 * 10^-15
        let max_x = Number::from_decimal(76712328767123 as u64, -15);

        /*
            x = .076712328767123
            e^x ~= 1.079731424041940
            e^x + 1 ~= 2.079731424041940
         */
        let expected: u64 = 079_731_424_041_940;
        let expected_number: Number = Number::from_decimal(expected, -15);
        // 079_731_423_755_760 <- actual result
        let answer = expm1_approx(max_x, 5);
        let tolerance = Number::from_decimal(10_000_000_000 as u64, -15);

        let diff = if expected_number.gt(&answer) {
            expected_number.sub(answer)
        }else{
            answer.sub(expected_number)
        };

        assert!(diff.lt(&tolerance));
    }

    #[test]
    fn test_taylor_approx_minish() {
        let min_x = Number::from_decimal(5 as u64, -15);

        /*
            x = 0.000000000000005
            e^x ~= 1.000000000000010
            e^x + 1 ~= 2.000000000000010
         */
        let expected: u64 = 000_000_000_000_010;
        let expected_number: Number = Number::from_decimal(expected, -15);
        // 000_000_000_000_005 <- actual result
        let answer = expm1_approx(min_x, 5);
        let tolerance = Number::from_decimal(100 as u64, -15);

        let diff = if expected_number.gt(&answer) {
            expected_number.sub(answer)
        }else{
            answer.sub(expected_number)
        };

        assert!(diff.lt(&tolerance));
    }


    #[test]
    fn zero_equals_zero() {
        assert_eq!(Number::ZERO, Number::from_decimal(0, 0));
        assert_eq!(Number::ZERO, Number::from(0u64));
    }

    #[test]
    fn one_equals_one() {
        assert_eq!(Number::ONE, Number::from_decimal(1, 0));
        assert_eq!(Number::ONE, Number::from(1u64));
    }

    #[test]
    fn one_plus_one_equals_two() {
        assert_eq!(Number::from_decimal(2, 0), Number::ONE + Number::ONE);
    }

    #[test]
    fn one_minus_one_equals_zero() {
        assert_eq!(Number::ONE - Number::ONE, Number::ZERO);
    }

    #[test]
    fn one_times_one_equals_one() {
        assert_eq!(Number::ONE, Number::ONE * Number::ONE);
    }

    #[test]
    fn one_divided_by_one_equals_one() {
        assert_eq!(Number::ONE, Number::ONE / Number::ONE);
    }

    #[test]
    fn ten_div_100_equals_point_1() {
        assert_eq!(
            Number::from_decimal(1, -1),
            Number::from_decimal(1, 1) / Number::from_decimal(100, 0)
        );
    }

    #[test]
    fn multiply_by_u64() {
        assert_eq!(
            Number::from_decimal(3, 1),
            Number::from_decimal(1, 1) * 3u64
        )
    }

    #[test]
    fn ceil_gt_one() {
        assert_eq!(Number::from_decimal(11, -1).as_u64_ceil(0), 2u64);
        assert_eq!(Number::from_decimal(19, -1).as_u64_ceil(0), 2u64);
    }

    #[test]
    fn ceil_lt_one() {
        assert_eq!(Number::from_decimal(1, -1).as_u64_ceil(0), 1u64);
        assert_eq!(Number::from_decimal(1, -10).as_u64_ceil(0), 1u64);
    }

    #[test]
    fn ceil_of_int() {
        assert_eq!(Number::from_decimal(1, 0).as_u64_ceil(0), 1u64);
        assert_eq!(
            Number::from_decimal(1_000_000u64, 0).as_u64_ceil(0),
            1_000_000u64
        );
    }

    #[test]
    fn to_string() {
        assert_eq!("1000.0", Number::from(1000).to_string());
        assert_eq!("1.0", Number::from(1).to_string());
        assert_eq!("0.001", Number::from_decimal(1, -3).to_string());
    }

    #[test]
    fn into_bits() {
        let bits = Number::from_decimal(1242, -3).into_bits();
        let number = Number::from_bits(bits);

        assert_eq!(Number::from_decimal(1242, -3), number);
    }
}