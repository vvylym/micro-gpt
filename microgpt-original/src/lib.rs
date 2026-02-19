//! # MicroGPT (Rust) — original implementation
//!
//! Minimal GPT: scalar autograd, transformer forward, training loop, and inference.
//! Single crate containing the full algorithm.

use rand::{prelude::*, rngs::StdRng};
use rand_distr::{weighted::WeightedIndex, Distribution, Normal};
use std::cell::RefCell;
use std::rc::Rc;

/// A fixed seed for reproducibility
const SEED: u64 = 42;
/// Default input dataset path (overridable via env or run argument).
pub const DEFAULT_INPUT_PATH: &str = "data/input.txt";
//=======================================//
//  Model parameters
//=======================================//
const N_EMBED: usize = 16;
const N_HEAD: usize = 4;
const N_LAYER: usize = 1;
const BLOCK_SIZE: usize = 16;
const HEAD_DIM: usize = N_EMBED / N_HEAD;
/// MLP hidden size = MLP_RATIO * N_EMBED (standard 4x in transformers)
const MLP_RATIO: usize = 4;
/// Weight init: Gaussian mean and std
const INIT_MEAN: f64 = 0.0;
const INIT_STD: f64 = 0.08;
/// Epsilon for RMSNorm numerical stability
const RMSNORM_EPS: f64 = 1e-5;
//=======================================//
//  Adam optimizer parameters
//=======================================//
const LEARNING_RATE: f64 = 0.01;
const BETA1: f64 = 0.85;
const BETA2: f64 = 0.99;
const EPSILON: f64 = 1e-8;
//=======================================//
//  Training parameters
//=======================================//
const NUM_STEPS: usize = 1000;
/// Print loss every this many steps (and at step 0)
const LOSS_LOG_EVERY: usize = 10;
//=======================================//
//  Inference parameters
//=======================================//
const TEMPERATURE: f64 = 0.5;
const SAMPLE_SIZE: usize = 20;

// =============================================================================
// AUTOGRAD — chain rule through a computation graph
// =============================================================================

/// Handle to a scalar node in the autograd computation graph.
///
/// Wraps [`Value`] in `Rc<RefCell<_>>` so that the graph can be shared and
/// gradients can be accumulated during backward. Use [`ValueRef::data`] for the
/// forward value and [`ValueRef::grad`] after [`ValueRef::backward`].
#[derive(Clone)]
pub(crate) struct ValueRef(Rc<RefCell<Value>>);

/// Internal scalar node: forward value, gradient, and graph edges for backprop.
pub(crate) struct Value {
    data: f64,
    grad: f64,
    children: Vec<ValueRef>,
    local_grads: Vec<f64>,
}

impl ValueRef {
    /// Creates a leaf node (no children) with the given value and zero gradient.
    pub(crate) fn new(data: f64) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children: Vec::new(),
            local_grads: Vec::new(),
        })))
    }

    /// Creates a node that remembers its `children` and `local_grads` for backprop.
    fn new_with_graph(data: f64, children: Vec<ValueRef>, local_grads: Vec<f64>) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children,
            local_grads,
        })))
    }

    /// Forward pass value (scalar).
    pub(crate) fn data(&self) -> f64 {
        self.0.borrow().data
    }

    /// Gradient of the loss with respect to this node; set by [`backward`](ValueRef::backward).
    pub(crate) fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    /// Sets this node's gradient (e.g. to 1.0 at the loss).
    fn set_grad(&self, g: f64) {
        self.0.borrow_mut().grad = g;
    }

    /// Adds `g` to this node's gradient (for accumulation when a value is used multiple times).
    fn add_grad(&self, g: f64) {
        self.0.borrow_mut().grad += g;
    }

    /// Addition: `self + other`. Local grads are 1 and 1.
    pub(crate) fn add(&self, other: &ValueRef) -> ValueRef {
        let data = self.data() + other.data();
        ValueRef::new_with_graph(data, vec![self.clone(), other.clone()], vec![1.0, 1.0])
    }

    /// Multiplication: `self * other`. Local grads are `other.data()` and `self.data()`.
    pub(crate) fn mul(&self, other: &ValueRef) -> ValueRef {
        let data = self.data() * other.data();
        ValueRef::new_with_graph(
            data,
            vec![self.clone(), other.clone()],
            vec![other.data(), self.data()],
        )
    }

    /// Power: `self^exp`. Local grad is `exp * self^(exp-1)`.
    pub(crate) fn pow(&self, exp: f64) -> ValueRef {
        let data = self.data().powf(exp);
        let local_grad = exp * self.data().powf(exp - 1.0);
        ValueRef::new_with_graph(data, vec![self.clone()], vec![local_grad])
    }

    /// Natural log. Local grad is `1/self`.
    pub(crate) fn log(&self) -> ValueRef {
        let data = self.data().ln();
        let local_grad = 1.0 / self.data();
        ValueRef::new_with_graph(data, vec![self.clone()], vec![local_grad])
    }

    /// Exponential. Local grad is `exp(self)`.
    pub(crate) fn exp(&self) -> ValueRef {
        let data = self.data().exp();
        let local_grad = data;
        ValueRef::new_with_graph(data, vec![self.clone()], vec![local_grad])
    }

    /// ReLU: `max(0, self)`. Local grad is 1 if `self > 0`, else 0.
    pub(crate) fn relu(&self) -> ValueRef {
        let data = self.data().max(0.0);
        let local_grad = if self.data() > 0.0 { 1.0 } else { 0.0 };
        ValueRef::new_with_graph(data, vec![self.clone()], vec![local_grad])
    }

    /// Negation: `-self`.
    pub(crate) fn neg(&self) -> ValueRef {
        let neg_one = ValueRef::new(-1.0);
        self.mul(&neg_one)
    }

    /// Subtraction: `self - other`.
    pub(crate) fn sub(&self, other: &ValueRef) -> ValueRef {
        self.add(&other.neg())
    }

    /// Division: `self / other` (via `self * other^(-1)`).
    pub(crate) fn div(&self, other: &ValueRef) -> ValueRef {
        self.mul(&other.pow(-1.0))
    }

    /// Runs backprop: topological sort, then chain rule from this node (e.g. loss) to all leaves.
    pub(crate) fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = std::collections::HashSet::new();

        fn build_topo(
            v: &ValueRef,
            visited: &mut std::collections::HashSet<*const RefCell<Value>>,
            topo: &mut Vec<ValueRef>,
        ) {
            let ptr = Rc::as_ptr(&v.0);
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                for child in &v.0.borrow().children {
                    build_topo(child, visited, topo);
                }
                topo.push(v.clone());
            }
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

    /// Sets this node's gradient to 0 (e.g. after an optimizer step).
    pub(crate) fn zero_grad(&self) {
        self.set_grad(0.0);
    }
}

/// Model parameters: embeddings (wte, wpe), lm_head, and per-layer attention and MLP.
pub(crate) struct StateDict {
    wte: Vec<Vec<ValueRef>>,
    wpe: Vec<Vec<ValueRef>>,
    lm_head: Vec<Vec<ValueRef>>,
    attn_wq: Vec<Vec<Vec<ValueRef>>>,
    attn_wk: Vec<Vec<Vec<ValueRef>>>,
    attn_wv: Vec<Vec<Vec<ValueRef>>>,
    attn_wo: Vec<Vec<Vec<ValueRef>>>,
    mlp_fc1: Vec<Vec<Vec<ValueRef>>>,
    mlp_fc2: Vec<Vec<Vec<ValueRef>>>,
}

impl StateDict {
    /// Builds a new state dict with Gaussian(0, 0.08) weights for the given `vocab_size`.
    pub(crate) fn new(vocab_size: usize, rng: &mut StdRng) -> Self {
        let normal = Normal::new(INIT_MEAN, INIT_STD).unwrap();
        let mut matrix = |nout: usize, nin: usize| -> Vec<Vec<ValueRef>> {
            (0..nout)
                .map(|_| {
                    (0..nin)
                        .map(|_| ValueRef::new(normal.sample(rng)))
                        .collect()
                })
                .collect()
        };

        let attn_layers = (0..N_LAYER)
            .map(|_| matrix(N_EMBED, N_EMBED))
            .collect::<Vec<_>>();

        StateDict {
            wte: matrix(vocab_size, N_EMBED),
            wpe: matrix(BLOCK_SIZE, N_EMBED),
            lm_head: matrix(vocab_size, N_EMBED),
            attn_wq: attn_layers.clone(),
            attn_wk: attn_layers.clone(),
            attn_wv: attn_layers.clone(),
            attn_wo: attn_layers.clone(),
            mlp_fc1: (0..N_LAYER)
                .map(|_| matrix(MLP_RATIO * N_EMBED, N_EMBED))
                .collect(),
            mlp_fc2: (0..N_LAYER)
                .map(|_| matrix(N_EMBED, MLP_RATIO * N_EMBED))
                .collect(),
        }
    }

    /// Returns all parameters as a flat list (for Adam or other optimizers).
    pub(crate) fn params(&self) -> Vec<ValueRef> {
        let mut params = Vec::new();
        for row in &self.wte {
            params.extend(row.clone());
        }
        for row in &self.wpe {
            params.extend(row.clone());
        }
        for row in &self.lm_head {
            params.extend(row.clone());
        }
        for layer_weights in &self.attn_wq {
            for row in layer_weights {
                params.extend(row.clone());
            }
        }
        for layer_weights in &self.attn_wk {
            for row in layer_weights {
                params.extend(row.clone());
            }
        }
        for layer_weights in &self.attn_wv {
            for row in layer_weights {
                params.extend(row.clone());
            }
        }
        for layer_weights in &self.attn_wo {
            for row in layer_weights {
                params.extend(row.clone());
            }
        }
        for layer_weights in &self.mlp_fc1 {
            for row in layer_weights {
                params.extend(row.clone());
            }
        }
        for layer_weights in &self.mlp_fc2 {
            for row in layer_weights {
                params.extend(row.clone());
            }
        }
        params
    }
}

pub(crate) fn linear(x: &[ValueRef], w: &[Vec<ValueRef>]) -> Vec<ValueRef> {
    w.iter()
        .map(|wo| {
            let mut sum = ValueRef::new(0.0);
            for (wi, xi) in wo.iter().zip(x.iter()) {
                sum = sum.add(&wi.mul(xi));
            }
            sum
        })
        .collect()
}

pub(crate) fn softmax(logits: &[ValueRef]) -> Vec<ValueRef> {
    let max_val = logits
        .iter()
        .map(|v| v.data())
        .fold(f64::NEG_INFINITY, f64::max);
    let max_v = ValueRef::new(max_val);

    let exps: Vec<_> = logits.iter().map(|l| l.sub(&max_v).exp()).collect();
    let mut total = ValueRef::new(0.0);
    for e in &exps {
        total = total.add(e);
    }

    exps.iter().map(|e| e.div(&total)).collect()
}

pub(crate) fn rmsnorm(x: &[ValueRef]) -> Vec<ValueRef> {
    let n = x.len() as f64;
    let mut ms = ValueRef::new(0.0);
    for xi in x {
        ms = ms.add(&xi.mul(xi));
    }
    ms = ms.div(&ValueRef::new(n));

    let scale = ms.add(&ValueRef::new(RMSNORM_EPS)).pow(-0.5);
    x.iter().map(|xi| xi.mul(&scale)).collect()
}

pub(crate) fn micro_gpt(
    token_id: usize,
    pos_id: usize,
    state: &StateDict,
    keys: &mut [Vec<Vec<ValueRef>>],
    values: &mut [Vec<Vec<ValueRef>>],
) -> Vec<ValueRef> {
    let mut x = Vec::new();
    for j in 0..N_EMBED {
        x.push(state.wte[token_id][j].add(&state.wpe[pos_id][j]));
    }
    x = rmsnorm(&x);

    for li in 0..N_LAYER {
        let x_residual = x.clone();
        x = rmsnorm(&x);

        let q = linear(&x, &state.attn_wq[li]);
        let k = linear(&x, &state.attn_wk[li]);
        let v = linear(&x, &state.attn_wv[li]);

        keys[li].push(k);
        values[li].push(v);

        let mut x_attn = Vec::new();
        for h in 0..N_HEAD {
            let hs = h * HEAD_DIM;
            let q_h: Vec<_> = q[hs..hs + HEAD_DIM].to_vec();
            let k_h: Vec<Vec<_>> = keys[li]
                .iter()
                .map(|ki| ki[hs..hs + HEAD_DIM].to_vec())
                .collect();
            let v_h: Vec<Vec<_>> = values[li]
                .iter()
                .map(|vi| vi[hs..hs + HEAD_DIM].to_vec())
                .collect();

            let mut attn_logits = Vec::new();
            for k_t in &k_h {
                let mut score = ValueRef::new(0.0);
                for j in 0..HEAD_DIM {
                    score = score.add(&q_h[j].mul(&k_t[j]));
                }
                score = score.div(&ValueRef::new((HEAD_DIM as f64).sqrt()));
                attn_logits.push(score);
            }

            let attn_weights = softmax(&attn_logits);
            for j in 0..HEAD_DIM {
                let mut head_out = ValueRef::new(0.0);
                for (v_t, w_t) in v_h.iter().zip(attn_weights.iter()) {
                    head_out = head_out.add(&w_t.mul(&v_t[j]));
                }
                x_attn.push(head_out);
            }
        }

        x = linear(&x_attn, &state.attn_wo[li]);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| a.add(b))
            .collect();

        let x_residual = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &state.mlp_fc1[li]);
        x = x.iter().map(|xi| xi.relu()).collect();
        x = linear(&x, &state.mlp_fc2[li]);
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(a, b)| a.add(b))
            .collect();
    }

    linear(&x, &state.lm_head)
}

/// Runs the full pipeline: load data, train, then inference.
///
/// Uses `input_path` for the dataset (one document per line). Prints progress and samples to stdout.
pub fn run(input_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    run_impl(input_path, None)
}

/// Performs one training step: forward, backward, Adam update. Returns loss for logging.
#[allow(clippy::too_many_arguments)]
fn do_one_training_step(
    doc: &str,
    bos_token: usize,
    char_to_id: &std::collections::HashMap<char, usize>,
    state: &StateDict,
    params: &[ValueRef],
    m: &mut [f64],
    v: &mut [f64],
    step: usize,
) -> f64 {
    let mut tokens = vec![bos_token];
    for ch in doc.chars() {
        if let Some(&id) = char_to_id.get(&ch) {
            tokens.push(id);
        }
    }
    tokens.push(bos_token);
    let n = (tokens.len() - 1).min(BLOCK_SIZE);

    let mut keys = vec![Vec::new(); N_LAYER];
    let mut values = vec![Vec::new(); N_LAYER];
    let mut losses = Vec::new();
    for pos_id in 0..n {
        let token_id = tokens[pos_id];
        let target_id = tokens[pos_id + 1];
        let logits = micro_gpt(token_id, pos_id, state, &mut keys, &mut values);
        let probs = softmax(&logits);
        let loss_t = probs[target_id].log().neg();
        losses.push(loss_t);
    }
    let mut loss = ValueRef::new(0.0);
    for l in &losses {
        loss = loss.add(l);
    }
    loss = loss.div(&ValueRef::new(n as f64));
    loss.backward();

    let lr_t = LEARNING_RATE * (1.0 - step as f64 / NUM_STEPS as f64);
    for (i, p) in params.iter().enumerate() {
        let grad = p.grad();
        m[i] = BETA1 * m[i] + (1.0 - BETA1) * grad;
        v[i] = BETA2 * v[i] + (1.0 - BETA2) * grad * grad;
        let m_hat = m[i] / (1.0 - BETA1.powi(step as i32 + 1));
        let v_hat = v[i] / (1.0 - BETA2.powi(step as i32 + 1));
        let new_data = p.data() - lr_t * m_hat / (v_hat.sqrt() + EPSILON);
        p.0.borrow_mut().data = new_data;
        p.zero_grad();
    }
    loss.data()
}

/// Generates one inference sample (BOS -> sample until BOS or block size).
fn do_one_inference_sample(
    state: &StateDict,
    vocab: &[char],
    bos_token: usize,
    rng: &mut StdRng,
) -> String {
    let mut keys = vec![Vec::new(); N_LAYER];
    let mut values = vec![Vec::new(); N_LAYER];
    let mut token_id = bos_token;
    let mut sample_text = String::new();
    for pos_id in 0..BLOCK_SIZE {
        let logits = micro_gpt(token_id, pos_id, state, &mut keys, &mut values);
        let scaled_logits: Vec<ValueRef> = logits
            .iter()
            .map(|l| l.div(&ValueRef::new(TEMPERATURE)))
            .collect();
        let probs = softmax(&scaled_logits);
        let normalized: Vec<f64> = probs.iter().map(|p| p.data()).collect();
        token_id = WeightedIndex::new(&normalized)
            .ok()
            .map(|dist| dist.sample(rng))
            .unwrap_or(bos_token);
        if token_id == bos_token {
            break;
        }
        sample_text.push(vocab[token_id]);
    }
    sample_text
}

/// Internal implementation: when `max_steps` is `Some(n)`, training stops after n steps (for tests).
#[doc(hidden)]
pub(crate) fn run_impl(
    input_path: &str,
    max_steps: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let input = std::fs::read_to_string(input_path)?;
    let mut names: Vec<&str> = input.lines().filter(|l| !l.is_empty()).collect();
    names.shuffle(&mut rng);
    println!("num docs: {}", names.len());

    let unique_chars: std::collections::BTreeSet<char> =
        names.iter().flat_map(|n| n.chars()).collect();
    let vocab: Vec<char> = unique_chars.into_iter().collect();
    let bos_token = vocab.len();
    let vocab_size = vocab.len() + 1;
    println!("vocab size: {}", vocab_size);

    let char_to_id: std::collections::HashMap<char, usize> =
        vocab.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let state = StateDict::new(vocab_size, &mut rng);
    let params = state.params();
    println!("num params: {}", params.len());

    let mut m = vec![0.0; params.len()];
    let mut v = vec![0.0; params.len()];
    let steps = max_steps.unwrap_or(NUM_STEPS);

    for step in 0..steps {
        let doc = names[step % names.len()];
        let loss_val = do_one_training_step(
            doc,
            bos_token,
            &char_to_id,
            &state,
            &params,
            &mut m,
            &mut v,
            step,
        );
        if (step + 1) % LOSS_LOG_EVERY == 0 || step == 0 {
            println!("step {:4} / {:4} | loss {:.4}", step + 1, steps, loss_val);
        }
    }

    let samples = if max_steps.is_some() { 2 } else { SAMPLE_SIZE };
    println!("\n--- inference (new, hallucinated names) ---");
    for sample_idx in 0..samples {
        let sample_text = do_one_inference_sample(&state, &vocab, bos_token, &mut rng);
        println!("sample {:2}: {}", sample_idx + 1, sample_text);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_ref_add_backward() {
        let a = ValueRef::new(2.0);
        let b = ValueRef::new(3.0);
        let c = a.add(&b);
        assert!((c.data() - 5.0).abs() < 1e-10);
        c.backward();
        assert!((a.grad() - 1.0).abs() < 1e-10);
        assert!((b.grad() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn value_ref_mul_backward() {
        let a = ValueRef::new(2.0);
        let b = ValueRef::new(3.0);
        let c = a.mul(&b);
        assert!((c.data() - 6.0).abs() < 1e-10);
        c.backward();
        assert!((a.grad() - 3.0).abs() < 1e-10);
        assert!((b.grad() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn value_ref_chain_backward() {
        let a = ValueRef::new(2.0);
        let b = ValueRef::new(3.0);
        let c = a.mul(&b).add(&ValueRef::new(1.0));
        assert!((c.data() - 7.0).abs() < 1e-10);
        c.backward();
        assert!((a.grad() - 3.0).abs() < 1e-10);
        assert!((b.grad() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn value_ref_relu_backward() {
        let a = ValueRef::new(-1.0);
        let b = ValueRef::new(1.0);
        let c = a.relu().add(&b.relu());
        assert!((c.data() - 1.0).abs() < 1e-10);
        c.backward();
        assert!((a.grad() - 0.0).abs() < 1e-10);
        assert!((b.grad() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn run_limited_steps_covers_training_and_inference() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        let path = dir.join("microgpt_run_limited_test.txt");
        let mut f = std::fs::File::create(&path).expect("create temp file");
        writeln!(f, "ab").expect("write");
        writeln!(f, "cd").expect("write");
        f.sync_all().expect("sync");
        drop(f);
        // Run with 2 steps and 2 samples to exercise training and inference without slowing tests
        let result = run_impl(path.to_str().unwrap(), Some(2));
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_ok(),
            "run_impl(_, Some(2)) should succeed: {:?}",
            result
        );
    }

    #[test]
    fn run_succeeds_with_data_file() {
        // Only run when data file exists (e.g. local or CI with data checked out)
        if std::path::Path::new(DEFAULT_INPUT_PATH).exists() {
            let result = run(DEFAULT_INPUT_PATH);
            assert!(
                result.is_ok(),
                "run() should succeed with existing data file"
            );
        }
    }

    #[test]
    fn linear_output_shape() {
        let a = ValueRef::new(1.0);
        let b = ValueRef::new(2.0);
        let x = vec![a, b];
        let w = vec![
            vec![ValueRef::new(0.5), ValueRef::new(0.5)],
            vec![ValueRef::new(1.0), ValueRef::new(0.0)],
        ];
        let out = linear(&x, &w);
        assert_eq!(out.len(), 2);
        assert!((out[0].data() - 1.5).abs() < 1e-10);
        assert!((out[1].data() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![ValueRef::new(0.0), ValueRef::new(0.0), ValueRef::new(0.0)];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().map(|p| p.data()).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rmsnorm_preserves_direction() {
        let x = vec![ValueRef::new(1.0), ValueRef::new(2.0)];
        let out = rmsnorm(&x);
        assert_eq!(out.len(), 2);
        out[0].backward();
    }

    #[test]
    fn state_dict_params_count() {
        let mut rng = StdRng::seed_from_u64(SEED);
        let state = StateDict::new(10, &mut rng);
        let params = state.params();
        let expected = 10 * N_EMBED
            + BLOCK_SIZE * N_EMBED
            + 10 * N_EMBED
            + N_LAYER
                * (4 * N_EMBED * N_EMBED
                    + (MLP_RATIO * N_EMBED) * N_EMBED
                    + N_EMBED * (MLP_RATIO * N_EMBED));
        assert_eq!(params.len(), expected);
    }

    #[test]
    fn micro_gpt_forward_shape() {
        let mut rng = StdRng::seed_from_u64(SEED);
        let vocab_size = 5;
        let state = StateDict::new(vocab_size, &mut rng);
        let mut keys = vec![Vec::new(); N_LAYER];
        let mut values = vec![Vec::new(); N_LAYER];
        let logits = micro_gpt(0, 0, &state, &mut keys, &mut values);
        assert_eq!(logits.len(), vocab_size);
    }
}
