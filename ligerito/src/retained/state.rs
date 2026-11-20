use std::mem::size_of;

use binary_fields::BinaryElem128;

use crate::data_structures::RecursiveLigeroWitness;
use crate::eval_proof::EvalClaim;
use crate::opening_transcript::OpeningTranscript;
use crate::FinalizedLigeritoProof;

use super::{CommitmentRoot, Error, Limits, OperationError, ProverParameters};

/// An exclusively owned native input and original commitment, consumed by opening.
pub struct ProverState<'a> {
    parameters: &'a ProverParameters,
    input: Vec<BinaryElem128>,
    witness: RecursiveLigeroWitness<BinaryElem128>,
    root: CommitmentRoot,
}

impl ProverState<'_> {
    #[must_use]
    pub fn root(&self) -> CommitmentRoot {
        self.root
    }

    fn retained_bytes(&self) -> Result<usize, Error> {
        let parameter_bytes = self.parameters.total_retained_bytes()?;
        let fields = self
            .input
            .capacity()
            .checked_add(self.witness.cols.capacity())
            .and_then(|count| count.checked_mul(size_of::<BinaryElem128>()))
            .ok_or(Error::Overflow)?;
        let layers = self
            .witness
            .tree
            .layers
            .capacity()
            .checked_mul(size_of::<Vec<[u8; 32]>>())
            .ok_or(Error::Overflow)?;
        self.witness.tree.layers.iter().try_fold(
            fields
                .checked_add(layers)
                .and_then(|bytes| bytes.checked_add(parameter_bytes))
                .ok_or(Error::Overflow)?,
            |bytes, layer| {
                layer
                    .capacity()
                    .checked_mul(32)
                    .and_then(|n| bytes.checked_add(n))
                    .ok_or(Error::Overflow)
            },
        )
    }
}

/// A checked commit request transfers its input buffer into retained state.
///
/// Commitment returns `Error` for incompatible input length, arithmetic overflow,
/// exceeded current resource limits, allocation failure or a missing native root.
pub struct CommitRequest<'a> {
    pub parameters: &'a ProverParameters,
    pub input: Vec<BinaryElem128>,
    pub limits: &'a Limits,
}

#[expect(
    clippy::missing_errors_doc,
    reason = "CommitRequest documents commitment errors."
)]
#[tracing::instrument(skip_all)]
pub fn commit(request: CommitRequest<'_>) -> Result<ProverState<'_>, Error> {
    request.parameters.check_limits(request.limits)?;
    let parameters = request.parameters.parameters();
    if request.input.len() != parameters.message_len() {
        return Err(Error::Length {
            expected: parameters.message_len(),
            actual: request.input.len(),
        });
    }
    parameters.check_prover_limits(request.input.capacity(), 0, request.limits)?;
    let config = request.parameters.config();
    let witness = crate::ligero::try_ligero_commit(
        &request.input,
        config.initial_dims.1,
        &config.initial_reed_solomon,
    )?;
    let root = witness
        .tree
        .get_root()
        .root
        .ok_or(Error::MissingRoot { level: 0 })?
        .into();
    let state = ProverState {
        parameters: request.parameters,
        input: request.input,
        witness,
        root,
    };
    parameters.check_prover_limits_with_retained(
        state.input.capacity(),
        0,
        state.retained_bytes()?,
        request.limits,
    )?;
    Ok(state)
}

/// Opening consumes state against independently supplied expected context.
///
/// Opening returns `OperationError::Input` for root or context mismatches, invalid
/// claims, arithmetic overflow or exceeded current limits. Native driver,
/// transcript, query and allocation failures retain their `OperationError::Driver`
/// source. Failure consumes the state and requires a fresh transcript on retry.
#[derive(Clone, Copy)]
pub struct OpenRequest<'a> {
    pub parameters: &'a ProverParameters,
    pub expected_root: CommitmentRoot,
    pub claims: &'a [EvalClaim<BinaryElem128>],
    pub limits: &'a Limits,
}

/// A proof and cleared input capacity released by the consuming native driver.
pub struct OpenedProof {
    pub proof: FinalizedLigeritoProof<BinaryElem128, BinaryElem128>,
    pub reusable_input: Vec<BinaryElem128>,
}

#[expect(
    clippy::missing_errors_doc,
    reason = "OpenRequest documents opening errors."
)]
#[tracing::instrument(skip_all)]
pub fn open<C>(
    state: ProverState<'_>,
    request: OpenRequest<'_>,
    transcript: &mut C,
) -> Result<OpenedProof, OperationError<C::Error>>
where
    C: OpeningTranscript<Message = BinaryElem128, Challenge = BinaryElem128>,
    C::Error: std::error::Error + 'static,
{
    let OpenRequest {
        parameters: prover_parameters,
        expected_root,
        claims,
        limits,
    } = request;
    if state.root != expected_root {
        return Err(Error::RootMismatch.into());
    }
    if !state
        .parameters
        .parameters()
        .same_geometry(prover_parameters.parameters())
    {
        return Err(Error::ContextMismatch.into());
    }
    prover_parameters.check_limits(limits)?;
    let parameters = prover_parameters.parameters();
    parameters.check_claims(claims, limits)?;
    let mut retained = state.retained_bytes()?;
    if !core::ptr::eq(state.parameters, prover_parameters) {
        retained = retained
            .checked_add(prover_parameters.total_retained_bytes()?)
            .ok_or(Error::Overflow)?;
    }
    parameters.check_prover_limits_with_retained(
        state.input.capacity(),
        claims.len(),
        retained,
        limits,
    )?;
    let driver = crate::prover::ProvingRequest {
        config: prover_parameters.config(),
        claims,
        max_query_draws: limits.max_query_draws,
    };
    let input = crate::prover::RetainedInput {
        table: crate::prover::MessageInput::Owned(state.input),
        witness: state.witness,
    };
    let output = crate::prover::prove_retained(driver, input, transcript)?;
    Ok(OpenedProof {
        proof: output.proof,
        reusable_input: output.reusable_input,
    })
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::retained::{verify, Parameters, VerifyRequest};
    use crate::{Sha256Transcript, VerifierConfig};

    #[test]
    fn b1_original_buffers_move_and_initial_commit_is_not_repeated(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let limits = Limits {
            max_allocation_bytes: 8_000_000,
            max_work: 20_000_000,
            max_retained_bytes: 4_000_000,
            max_claims: 4,
            max_query_draws: 4096,
        };
        let config = VerifierConfig {
            recursive_steps: 1,
            initial_dim: 8,
            log_dims: vec![7],
            initial_k: 1,
            ks: vec![1],
            num_queries: 160,
        };
        let parameters = ProverParameters::new(Parameters::new(config, &limits)?, &limits)?;
        let input: Vec<_> = (0..512u128).map(BinaryElem128::from_value).collect();
        let original_pointer = input.as_ptr();
        let original_capacity = input.capacity();
        let counts_before = crate::ligero::commitment_counts();
        let state = commit(CommitRequest {
            parameters: &parameters,
            input,
            limits: &limits,
        })?;
        assert_eq!(state.input.as_ptr(), original_pointer);
        assert_eq!(state.input.capacity(), original_capacity);
        for (row, expected) in [[0_u128, 1], [2, 2]].iter().enumerate() {
            let expected_fields: Vec<_> = expected
                .iter()
                .copied()
                .map(BinaryElem128::from_value)
                .collect();
            assert_eq!(state.witness.row(row), expected_fields);
            let mut bytes = 2_u32.to_le_bytes().to_vec();
            for value in expected {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let prehash: [u8; 32] = Sha256::digest(bytes).into();
            assert_eq!(
                state.witness.tree.layers[0][row],
                merkle_tree::hash_leaf(&prehash)
            );
        }
        let counts_committed = crate::ligero::commitment_counts();
        assert_eq!(counts_committed.0 - counts_before.0, 1);
        assert_eq!(counts_committed.1 - counts_before.1, 1);
        let retained = state.retained_bytes()?;
        let root = state.root();
        let claims = [EvalClaim {
            index: 257,
            value: BinaryElem128::from_value(257),
        }];
        let opened = open(
            state,
            OpenRequest {
                parameters: &parameters,
                expected_root: root,
                claims: &claims,
                limits: &limits,
            },
            &mut Sha256Transcript::new(7),
        )?;
        let counts_opened = crate::ligero::commitment_counts();
        assert_eq!(counts_opened.0 - counts_committed.0, 1);
        assert_eq!(counts_opened.1 - counts_committed.1, 1);
        assert_eq!(opened.reusable_input.as_ptr(), original_pointer);
        assert_eq!(opened.reusable_input.capacity(), original_capacity);
        assert!(opened.reusable_input.is_empty());
        assert_eq!(opened.proof.final_ligero_proof.yr.capacity(), 128);
        let proof = opened.proof;
        assert!(retained > original_capacity * size_of::<BinaryElem128>());
        verify(
            VerifyRequest {
                parameters: parameters.parameters(),
                expected_root: root,
                claims: &claims,
                proof: &proof,
                limits: &limits,
            },
            &mut Sha256Transcript::new(7),
        )?;
        Ok(())
    }
}
