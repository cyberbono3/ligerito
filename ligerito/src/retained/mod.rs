#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::print_stdout,
    clippy::await_holding_lock,
    clippy::allow_attributes
)]

mod error;
mod params;
#[cfg(feature = "prover")]
mod state;
mod verify;

pub use error::Error;
#[cfg(feature = "prover")]
pub use params::ProverParameters;
pub use params::{Limits, Parameters};
#[cfg(feature = "prover")]
pub use state::{commit, open, CommitRequest, OpenRequest, OpenedProof, ProverState};
pub use verify::{verify, CommitmentRoot, OperationError, VerifyRequest};
