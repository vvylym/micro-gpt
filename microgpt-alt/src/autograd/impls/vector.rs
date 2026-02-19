//! Vector autograd: vectors of scalar nodes with element-wise ops and matrix-vector product.

use super::scalar::ValueRef;
use crate::autograd::ScalarNode;
use std::ops::{Add, Mul};

/// A vector of scalar autograd nodes (element-wise differentiable vector).
///
/// Supports element-wise add/mul and matrix-vector product. Backpropagation
/// is performed by building a scalar loss from the elements and calling
/// backward on it.
#[derive(Clone)]
pub struct VectorValue {
    /// One scalar node per element.
    refs: Vec<ValueRef>,
}

impl VectorValue {
    /// Creates a vector of leaf nodes with the given values and zero gradients.
    #[must_use]
    pub fn new(data: Vec<f64>) -> Self {
        VectorValue {
            refs: data.into_iter().map(ValueRef::new).collect(),
        }
    }

    /// Returns the forward pass values as a slice.
    #[must_use]
    pub fn data(&self) -> Vec<f64> {
        self.refs.iter().map(ValueRef::data).collect()
    }

    /// Returns the gradients (after backward) as a slice.
    #[must_use]
    pub fn grad(&self) -> Vec<f64> {
        self.refs.iter().map(ValueRef::grad).collect()
    }

    /// Matrix-vector product: treats `self` as row-major matrix with `rows` rows,
    /// multiplies by `x`, returns a vector of scalar nodes of length `rows`.
    #[must_use]
    pub fn matvec(&self, x: &VectorValue, rows: usize) -> Vec<ValueRef> {
        let cols = self.refs.len() / rows;
        assert_eq!(
            cols * rows,
            self.refs.len(),
            "matvec: self.len() must equal rows * cols"
        );
        assert_eq!(x.refs.len(), cols, "matvec: x.len() must equal cols");

        (0..rows)
            .map(|i| {
                let row_start = i * cols;
                let mut sum = ValueRef::new(0.0);
                for j in 0..cols {
                    sum = &sum + &(&self.refs[row_start + j] * &x.refs[j]);
                }
                sum
            })
            .collect()
    }

    /// Runs backprop from the sum of all elements (each element receives gradient 1).
    pub fn backward(&self) {
        let sum = self.refs.iter().fold(ValueRef::new(0.0), |acc, r| &acc + r);
        sum.backward();
    }

    /// Zeros the gradient at each element.
    pub fn zero_grad(&self) {
        for r in &self.refs {
            r.zero_grad();
        }
    }
}

// -----------------------------------------------------------------------------
// std::ops — element-wise + and *
// -----------------------------------------------------------------------------

impl Add for &VectorValue {
    type Output = VectorValue;

    fn add(self, rhs: Self) -> VectorValue {
        assert_eq!(
            self.refs.len(),
            rhs.refs.len(),
            "vector add: length mismatch"
        );
        VectorValue {
            refs: self
                .refs
                .iter()
                .zip(rhs.refs.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

impl Mul for &VectorValue {
    type Output = VectorValue;

    fn mul(self, rhs: Self) -> VectorValue {
        assert_eq!(
            self.refs.len(),
            rhs.refs.len(),
            "vector mul: length mismatch"
        );
        VectorValue {
            refs: self
                .refs
                .iter()
                .zip(rhs.refs.iter())
                .map(|(a, b)| a * b)
                .collect(),
        }
    }
}
