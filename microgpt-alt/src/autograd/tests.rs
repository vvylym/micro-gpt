//! Tests for scalar and vector autograd.
//!
//! Ensures backward pass correctness (add, mul, pow, log, exp, relu),
//! topological order and gradient accumulation, zero_grad, and vector ops.

use crate::autograd::{Exp, Log, Pow, ScalarNode, ValueRef, VectorValue};

#[test]
fn scalar_add_backward() {
    let a = ValueRef::new(2.0);
    let b = ValueRef::new(3.0);
    let c = &a + &b;
    assert_eq!(c.data(), 5.0);
    c.backward();
    assert_eq!(a.grad(), 1.0);
    assert_eq!(b.grad(), 1.0);
}

#[test]
fn scalar_mul_backward() {
    let a = ValueRef::new(2.0);
    let b = ValueRef::new(3.0);
    let c = &a * &b;
    assert_eq!(c.data(), 6.0);
    c.backward();
    assert_eq!(a.grad(), 3.0);
    assert_eq!(b.grad(), 2.0);
}

#[test]
fn scalar_pow_backward() {
    let a = ValueRef::new(2.0);
    let b = (&a).pow(3.0);
    assert!((b.data() - 8.0).abs() < 1e-10);
    b.backward();
    // d/dx x^3 = 3x^2 = 12 at x=2
    assert!((a.grad() - 12.0).abs() < 1e-10);
}

#[test]
fn scalar_log_backward() {
    let a = ValueRef::new(std::f64::consts::E);
    let b = (&a).log();
    assert!((b.data() - 1.0).abs() < 1e-10);
    b.backward();
    // d/dx ln(x) = 1/x = 1/e at x=e
    assert!((a.grad() - 1.0 / std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn scalar_exp_backward() {
    let a = ValueRef::new(1.0);
    let b = (&a).exp();
    assert!((b.data() - std::f64::consts::E).abs() < 1e-10);
    b.backward();
    assert!((a.grad() - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn scalar_relu_backward_positive() {
    let a = ValueRef::new(1.5);
    let b = a.relu();
    assert_eq!(b.data(), 1.5);
    b.backward();
    assert_eq!(a.grad(), 1.0);
}

#[test]
fn scalar_relu_backward_negative() {
    let a = ValueRef::new(-0.5);
    let b = a.relu();
    assert_eq!(b.data(), 0.0);
    b.backward();
    assert_eq!(a.grad(), 0.0);
}

#[test]
fn scalar_topo_and_gradient_accumulation() {
    // Use a value twice: c = a + a. dc/da = 2.
    let a = ValueRef::new(3.0);
    let c = &a + &a;
    assert_eq!(c.data(), 6.0);
    c.backward();
    assert_eq!(a.grad(), 2.0);
}

#[test]
fn scalar_zero_grad_after_step() {
    let a = ValueRef::new(2.0);
    let b = &a * &ValueRef::new(3.0);
    b.backward();
    assert_eq!(a.grad(), 3.0);
    a.zero_grad();
    assert_eq!(a.grad(), 0.0);
}

#[test]
fn scalar_neg_backward() {
    let a = ValueRef::new(3.0);
    let b = -&a;
    assert_eq!(b.data(), -3.0);
    b.backward();
    assert_eq!(a.grad(), -1.0);
}

#[test]
fn scalar_sub_backward() {
    let a = ValueRef::new(5.0);
    let b = ValueRef::new(2.0);
    let c = &a - &b;
    assert_eq!(c.data(), 3.0);
    c.backward();
    assert_eq!(a.grad(), 1.0);
    assert_eq!(b.grad(), -1.0);
}

#[test]
fn scalar_div_backward() {
    let a = ValueRef::new(6.0);
    let b = ValueRef::new(2.0);
    let c = &a / &b;
    assert_eq!(c.data(), 3.0);
    c.backward();
    assert_eq!(a.grad(), 0.5);
    assert_eq!(b.grad(), -1.5); // d/db (a/b) = -a/b^2 = -6/4 = -1.5
}

#[test]
fn scalar_chain_compound() {
    // loss = (a * b + c).relu(); a=1, b=2, c=-1 => loss = 1
    let a = ValueRef::new(1.0);
    let b = ValueRef::new(2.0);
    let c = ValueRef::new(-1.0);
    let loss = (&(&a * &b) + &c).relu();
    assert_eq!(loss.data(), 1.0);
    loss.backward();
    // d/da (ab+c) = b=2, d/db = a=1, d/dc = 1; relu grad 1
    assert!((a.grad() - 2.0).abs() < 1e-10);
    assert!((b.grad() - 1.0).abs() < 1e-10);
    assert!((c.grad() - 1.0).abs() < 1e-10);
}

#[test]
fn scalar_std_ops_add_sub_mul_div_neg() {
    // Use std::ops: &a + &b, &a - &b, &a * &b, &a / &b, -&a
    let a = ValueRef::new(3.0);
    let b = ValueRef::new(2.0);
    let sum = &a + &b;
    assert_eq!(sum.data(), 5.0);
    let diff = &a - &b;
    assert_eq!(diff.data(), 1.0);
    let prod = &a * &b;
    assert_eq!(prod.data(), 6.0);
    let quot = &a / &b;
    assert!((quot.data() - 1.5).abs() < 1e-10);
    let neg_a = -&a;
    assert_eq!(neg_a.data(), -3.0);
    // backward through an expression using ops
    let loss = &(&a * &b) + &a;
    loss.backward();
    assert!((a.grad() - (b.data() + 1.0)).abs() < 1e-10); // d/da (ab + a) = b + 1 = 3
    assert!((b.grad() - a.data()).abs() < 1e-10); // d/db = a = 3
}

// --- Vector autograd tests ---

#[test]
fn vector_add_backward() {
    let a = VectorValue::new(vec![1.0, 2.0]);
    let b = VectorValue::new(vec![3.0, 4.0]);
    let c = &a + &b;
    assert_eq!(c.data(), &[4.0, 6.0]);
    c.backward();
    assert_eq!(a.grad(), &[1.0, 1.0]);
    assert_eq!(b.grad(), &[1.0, 1.0]);
}

#[test]
fn vector_mul_backward() {
    let a = VectorValue::new(vec![2.0, 3.0]);
    let b = VectorValue::new(vec![4.0, 5.0]);
    let c = &a * &b;
    assert_eq!(c.data(), &[8.0, 15.0]);
    c.backward();
    assert_eq!(a.grad(), &[4.0, 5.0]);
    assert_eq!(b.grad(), &[2.0, 3.0]);
}

#[test]
fn vector_matvec_backward() {
    // y = W @ x; 2x2 @ 2x1
    let w = VectorValue::new(vec![1.0, 2.0, 3.0, 4.0]); // row-major 2x2
    let x = VectorValue::new(vec![1.0, 2.0]);
    let y = w.matvec(&x, 2);
    assert_eq!(y.len(), 2);
    assert!((y[0].data() - 5.0).abs() < 1e-10); // 1*1+2*2
    assert!((y[1].data() - 11.0).abs() < 1e-10); // 3*1+4*2
    let sum_out = &y[0] + &y[1];
    sum_out.backward();
    // d(sum)/d(y_i) = 1 => upstream to W and x
    assert_eq!(x.grad().len(), 2);
    assert_eq!(w.grad().len(), 4);
}

#[test]
fn vector_accumulation_when_used_twice() {
    let a = VectorValue::new(vec![1.0, 2.0]);
    let b = &a + &a;
    assert_eq!(b.data(), &[2.0, 4.0]);
    b.backward();
    assert_eq!(a.grad(), &[2.0, 2.0]);
}

#[test]
fn vector_zero_grad() {
    let a = VectorValue::new(vec![1.0, 2.0]);
    let b = &a + &VectorValue::new(vec![0.0, 0.0]);
    b.backward();
    a.zero_grad();
    assert_eq!(a.grad(), &[0.0, 0.0]);
}

#[test]
fn vector_std_ops_add_mul() {
    let a = VectorValue::new(vec![1.0, 2.0]);
    let b = VectorValue::new(vec![3.0, 4.0]);
    let sum = &a + &b;
    assert_eq!(sum.data(), &[4.0, 6.0]);
    let prod = &a * &b;
    assert_eq!(prod.data(), &[3.0, 8.0]);
    prod.backward();
    assert_eq!(a.grad(), &[3.0, 4.0]);
    assert_eq!(b.grad(), &[1.0, 2.0]);
}
