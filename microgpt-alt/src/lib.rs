//! # microgpt-alt
//!
//! Refactored minimal GPT: config-driven, trait-based, with optional vector autograd,
//! delta adapters, and BPE tokenizer.

pub mod autograd;
pub mod config;
pub mod data;
pub mod tokenizer;
