//! Scalar autograd: computation graph of single float values with backpropagation.

use crate::autograd::{Exp, Log, Pow, ScalarNode};
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// Internal scalar node: forward value, gradient, and graph edges for backprop.
struct Value {
    /// Forward pass value.
    data: f64,
    /// Gradient of the loss with respect to this node; set during backward.
    grad: f64,
    /// Child nodes in the computation graph.
    children: Vec<ValueRef>,
    /// Local partial derivatives (one per child) for the chain rule.
    local_grads: Vec<f64>,
}

/// Handle to a scalar node in the autograd computation graph.
///
/// Wraps the node state in `Rc<RefCell<_>>` so that the graph can be shared and
/// gradients can be accumulated during backward.
#[derive(Clone)]
pub struct ValueRef(Rc<RefCell<Value>>);

impl ValueRef {
    /// Creates a leaf node (no children) with the given value and zero gradient.
    #[must_use]
    pub fn new(data: f64) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children: Vec::new(),
            local_grads: Vec::new(),
        })))
    }

    /// Sets this node's gradient (e.g. to 1.0 at the loss).
    fn set_grad(&self, g: f64) {
        self.0.borrow_mut().grad = g;
    }

    /// Adds to this node's gradient (for accumulation when a value is used multiple times).
    fn add_grad(&self, g: f64) {
        self.0.borrow_mut().grad += g;
    }

    /// Creates a node that remembers its children and local grads for backprop.
    fn new_with_graph(data: f64, children: Vec<ValueRef>, local_grads: Vec<f64>) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children,
            local_grads,
        })))
    }

    /// ReLU: `max(0, self)`. Local grad is 1 if `self > 0`, else 0.
    #[must_use]
    pub fn relu(&self) -> ValueRef {
        let data = self.data().max(0.0);
        let local_grad = if self.data() > 0.0 { 1.0 } else { 0.0 };
        ValueRef::new_with_graph(data, vec![self.clone()], vec![local_grad])
    }
}

// -----------------------------------------------------------------------------
// std::ops — algebra: x + y, x - y, x * y, x / y, -x
// -----------------------------------------------------------------------------

impl Add for &ValueRef {
    type Output = ValueRef;

    fn add(self, rhs: Self) -> ValueRef {
        ValueRef::new_with_graph(
            self.data() + rhs.data(),
            vec![self.clone(), rhs.clone()],
            vec![1.0, 1.0],
        )
    }
}

impl Sub for &ValueRef {
    type Output = ValueRef;

    fn sub(self, rhs: Self) -> ValueRef {
        self + &(-rhs)
    }
}

impl Mul for &ValueRef {
    type Output = ValueRef;

    fn mul(self, rhs: Self) -> ValueRef {
        ValueRef::new_with_graph(
            self.data() * rhs.data(),
            vec![self.clone(), rhs.clone()],
            vec![rhs.data(), self.data()],
        )
    }
}

impl Div for &ValueRef {
    type Output = ValueRef;

    fn div(self, rhs: Self) -> ValueRef {
        self * &rhs.pow(-1.0)
    }
}

impl Neg for &ValueRef {
    type Output = ValueRef;

    fn neg(self) -> ValueRef {
        self * &ValueRef::new(-1.0)
    }
}

// -----------------------------------------------------------------------------
// Pow, Exp, Log — (&a).pow(exp), (&a).exp(), (&a).log()
// -----------------------------------------------------------------------------

impl Pow<f64> for &ValueRef {
    type Output = ValueRef;

    fn pow(self, exp: f64) -> ValueRef {
        let data = self.data().powf(exp);
        let local_grad = exp * self.data().powf(exp - 1.0);
        ValueRef::new_with_graph(data, vec![self.clone()], vec![local_grad])
    }
}

impl Exp for &ValueRef {
    type Output = ValueRef;

    fn exp(self) -> ValueRef {
        let data = self.data().exp();
        ValueRef::new_with_graph(data, vec![self.clone()], vec![data])
    }
}

impl Log for &ValueRef {
    type Output = ValueRef;

    fn log(self) -> ValueRef {
        let data = self.data().ln();
        let local_grad = 1.0 / self.data();
        ValueRef::new_with_graph(data, vec![self.clone()], vec![local_grad])
    }
}

impl ScalarNode for ValueRef {
    fn data(&self) -> f64 {
        self.0.borrow().data
    }

    fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();
        fn build_topo(
            v: &ValueRef,
            visited: &mut std::collections::HashSet<*const RefCell<Value>>,
            topo: &mut Vec<ValueRef>,
        ) {
            let ptr = Rc::as_ptr(&v.0);
            if !visited.insert(ptr) {
                return;
            }
            for child in &v.0.borrow().children {
                build_topo(child, visited, topo);
            }
            topo.push(v.clone());
        }
        build_topo(self, &mut visited, &mut topo);
        self.set_grad(1.0);
        for v in topo.iter().rev() {
            let v_grad = v.grad();
            let v_borrowed = v.0.borrow();
            for (child, &local_grad) in v_borrowed
                .children
                .iter()
                .zip(v_borrowed.local_grads.iter())
            {
                child.add_grad(local_grad * v_grad);
            }
        }
    }

    fn zero_grad(&self) {
        self.set_grad(0.0);
    }
}
