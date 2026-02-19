//! Autograd: scalar and vector computation graphs with automatic differentiation.
//!
//! This module provides a trait-based autograd engine. The graph is built during
//! forward operations; [`ScalarNode::backward`] propagates gradients from a loss
//! node to all leaves using the chain rule in topological order.

pub mod impls;
#[cfg(test)]
mod tests;

pub use impls::scalar::ValueRef;
pub use impls::vector::VectorValue;

/// Trait for raising a node to a scalar power (e.g. `(&a).pow(2.0)`).
pub trait Pow<Rhs> {
    /// Result of the power operation.
    type Output;

    /// Returns `self^exp` with gradient tracking.
    #[must_use]
    fn pow(self, exp: Rhs) -> Self::Output;
}

/// Trait for the exponential of a node (e.g. `(&a).exp()`).
pub trait Exp {
    /// Result of the exponential.
    type Output;

    /// Returns `exp(self)` with gradient tracking.
    #[must_use]
    fn exp(self) -> Self::Output;
}

/// Trait for the natural log of a node (e.g. `(&a).log()`).
pub trait Log {
    /// Result of the log.
    type Output;

    /// Returns `ln(self)` with gradient tracking.
    #[must_use]
    fn log(self) -> Self::Output;
}

/// A differentiable scalar node in the computation graph.
///
/// Implementations hold a single float value and an optional gradient, and
/// participate in backward propagation. Use [`ScalarNode::data`] for the
/// forward value and [`ScalarNode::grad`] after [`ScalarNode::backward`].
pub trait ScalarNode: Clone {
    /// Returns the forward pass value.
    fn data(&self) -> f64;

    /// Returns the gradient of the loss with respect to this node (set by backward).
    fn grad(&self) -> f64;

    /// Runs backpropagation from this node (e.g. the loss) to all leaves.
    fn backward(&self);

    /// Zeros the gradient at this node (e.g. after an optimizer step).
    fn zero_grad(&self);
}
