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

use ligerito::eval_proof::EvalClaim;
use ligerito::opening_transcript::{
    DriverError, OpeningTranscript, QueryError, QueryRequest, TranscriptError,
};
use ligerito::retained::{
    commit, open, verify, CommitRequest, CommitmentRoot, Error, Limits, OpenRequest,
    OperationError, Parameters, ProverParameters, VerifyRequest,
};
use ligerito::{FinalizedLigeritoProof, Sha256Transcript, VerifierConfig};
use ligerito_binary_fields::{BinaryElem128, BinaryFieldElement};
use ligerito_merkle::MerkleRoot;

type Proof = FinalizedLigeritoProof<BinaryElem128, BinaryElem128>;
type TestResult = Result<(), Box<dyn std::error::Error>>;

fn limits() -> Limits {
    Limits {
        max_allocation_bytes: 16 * 1024 * 1024,
        max_work: 10_000_000,
        max_retained_bytes: 8 * 1024 * 1024,
        max_claims: 16,
        max_query_draws: 4096,
    }
}

fn parameters(limits: &Limits) -> Result<ProverParameters, Error> {
    let config = VerifierConfig {
        initial_dim: 8,
        initial_k: 1,
        recursive_steps: 1,
        log_dims: vec![7],
        ks: vec![1],
        num_queries: 160,
    };
    ProverParameters::new(Parameters::new(config, limits)?, limits)
}

fn multiple_level_parameters(limits: &Limits) -> Result<ProverParameters, Error> {
    let config = VerifierConfig {
        recursive_steps: 2,
        initial_dim: 8,
        initial_k: 2,
        log_dims: vec![7, 6],
        ks: vec![1, 1],
        num_queries: 160,
    };
    ProverParameters::new(Parameters::new(config, limits)?, limits)
}

fn value(index: u128, salt: u128) -> BinaryElem128 {
    BinaryElem128::from_value((index << 96) ^ (index * 0x1_0001) ^ salt)
}

fn input(salt: u128) -> Vec<BinaryElem128> {
    (0..512).map(|index| value(index, salt)).collect()
}

struct Control {
    root: CommitmentRoot,
    claims: Vec<EvalClaim<BinaryElem128>>,
    proof: Proof,
}

fn control(
    parameters: &ProverParameters,
    limits: &Limits,
    salt: u128,
) -> Result<Control, Box<dyn std::error::Error>> {
    control_with_claims(
        parameters,
        limits,
        salt,
        vec![EvalClaim {
            index: 257,
            value: value(257, salt),
        }],
    )
}

fn control_with_claims(
    parameters: &ProverParameters,
    limits: &Limits,
    salt: u128,
    claims: Vec<EvalClaim<BinaryElem128>>,
) -> Result<Control, Box<dyn std::error::Error>> {
    let state = commit(CommitRequest {
        parameters,
        input: (0_u128..)
            .take(parameters.parameters().message_len())
            .map(|index| value(index, salt))
            .collect(),
        limits,
    })?;
    let root = state.root();
    let proof = open(
        state,
        OpenRequest {
            parameters,
            expected_root: root,
            claims: &claims,
            limits,
        },
        &mut Sha256Transcript::new(7),
    )?
    .proof;
    let control = Control {
        root,
        claims,
        proof,
    };
    verify_control(parameters, limits, &control)?;
    Ok(control)
}

fn verify_control(
    parameters: &ProverParameters,
    limits: &Limits,
    control: &Control,
) -> Result<(), OperationError<TranscriptError>> {
    verify(
        VerifyRequest {
            parameters: parameters.parameters(),
            expected_root: control.root,
            claims: &control.claims,
            proof: &control.proof,
            limits,
        },
        &mut Sha256Transcript::new(7),
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Callback {
    Root,
    Bytes,
    Message,
    ChallengeObservation,
    ChallengeSlice,
    ChallengeSample,
    Queries,
}

#[derive(Clone, Copy)]
enum QueryCorruption {
    Short,
    Duplicate,
    OutOfRange,
    Descending,
}

#[derive(Debug, thiserror::Error)]
enum ProbeError {
    #[error("injected failure at callback {operation}")]
    Injected { operation: usize },
    #[error("native transcript operation failed")]
    Native(#[from] TranscriptError),
}

struct TranscriptProbe {
    inner: Sha256Transcript,
    callbacks: Vec<Callback>,
    fail_at: Option<usize>,
    corrupt_queries: Option<QueryCorruption>,
}

impl TranscriptProbe {
    fn new(fail_at: Option<usize>) -> Self {
        Self {
            inner: Sha256Transcript::new(7),
            callbacks: Vec::new(),
            fail_at,
            corrupt_queries: None,
        }
    }

    fn before(&mut self, callback: Callback) -> Result<(), ProbeError> {
        self.callbacks.push(callback);
        let operation = self.callbacks.len();
        if self.fail_at == Some(operation) {
            return Err(ProbeError::Injected { operation });
        }
        Ok(())
    }
}

impl OpeningTranscript for TranscriptProbe {
    type Message = BinaryElem128;
    type Challenge = BinaryElem128;
    type Error = ProbeError;

    fn observe_root(&mut self, root: &MerkleRoot) -> Result<(), Self::Error> {
        self.before(Callback::Root)?;
        self.inner.observe_root(root).map_err(ProbeError::Native)
    }

    fn observe_bytes(&mut self, label: &[u8], bytes: &[u8]) -> Result<(), Self::Error> {
        self.before(Callback::Bytes)?;
        self.inner
            .observe_bytes(label, bytes)
            .map_err(ProbeError::Native)
    }

    fn observe_message(&mut self, value: BinaryElem128) -> Result<(), Self::Error> {
        self.before(Callback::Message)?;
        self.inner
            .observe_message(value)
            .map_err(ProbeError::Native)
    }

    fn observe_challenge(&mut self, value: BinaryElem128) -> Result<(), Self::Error> {
        self.before(Callback::ChallengeObservation)?;
        self.inner
            .observe_challenge(value)
            .map_err(ProbeError::Native)
    }

    fn observe_challenges(&mut self, values: &[BinaryElem128]) -> Result<(), Self::Error> {
        self.before(Callback::ChallengeSlice)?;
        self.inner
            .observe_challenges(values)
            .map_err(ProbeError::Native)
    }

    fn sample_challenge(&mut self) -> Result<BinaryElem128, Self::Error> {
        self.before(Callback::ChallengeSample)?;
        self.inner.sample_challenge().map_err(ProbeError::Native)
    }

    fn sample_queries(&mut self, request: QueryRequest) -> Result<Vec<usize>, Self::Error> {
        self.before(Callback::Queries)?;
        let mut queries = self.inner.sample_queries(request)?;
        match self.corrupt_queries {
            None => {}
            Some(QueryCorruption::Short) => queries.truncate(queries.len() - 1),
            Some(QueryCorruption::Duplicate) => queries[1] = queries[0],
            Some(QueryCorruption::OutOfRange) => queries[0] = request.domain(),
            Some(QueryCorruption::Descending) => queries.swap(0, 1),
        }
        Ok(queries)
    }
}

#[test]
fn retained_open_uses_the_committed_root_and_exact_query_count() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let control = control(&parameters, &limits, 0)?;
    assert_eq!(control.proof.initial_ligero_proof.opened_rows.len(), 160);
    assert_eq!(control.proof.final_ligero_proof.opened_rows.len(), 160);
    assert_eq!(control.proof.final_ligero_proof.yr.len(), 128);
    assert_eq!(control.proof.eval_rounds.len(), 1);
    assert_eq!(control.proof.sumcheck_transcript.transcript.len(), 1);
    Ok(())
}

#[test]
fn changed_data_has_its_own_accepted_root_and_claim() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let first = control(&parameters, &limits, 0)?;
    let mut second = control(&parameters, &limits, 1 << 120)?;
    assert_ne!(first.root, second.root);
    second.root = first.root;
    assert!(matches!(
        verify_control(&parameters, &limits, &second),
        Err(OperationError::Input(Error::RootMismatch))
    ));
    Ok(())
}

#[test]
fn mismatched_root_rejects_before_transcript_observation() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let other = control(&parameters, &limits, 1)?;
    let state = commit(CommitRequest {
        parameters: &parameters,
        input: input(0),
        limits: &limits,
    })?;
    assert_ne!(state.root(), other.root);
    let mut transcript = TranscriptProbe::new(None);
    let result = open(
        state,
        OpenRequest {
            parameters: &parameters,
            expected_root: other.root,
            claims: &other.claims,
            limits: &limits,
        },
        &mut transcript,
    );
    assert!(matches!(
        result,
        Err(OperationError::Input(Error::RootMismatch))
    ));
    assert!(transcript.callbacks.is_empty());
    Ok(())
}

#[test]
fn false_claim_rejects_after_valid_shape() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let mut control = control(&parameters, &limits, 0)?;
    let state = commit(CommitRequest {
        parameters: &parameters,
        input: input(0),
        limits: &limits,
    })?;
    assert_eq!(state.root(), control.root);
    control.claims[0].value = control.claims[0].value.add(&BinaryElem128::one());
    control.proof = open(
        state,
        OpenRequest {
            parameters: &parameters,
            expected_root: control.root,
            claims: &control.claims,
            limits: &limits,
        },
        &mut Sha256Transcript::new(7),
    )?
    .proof;
    assert!(matches!(
        verify_control(&parameters, &limits, &control),
        Err(OperationError::Input(Error::ProtocolFailure))
    ));
    Ok(())
}

#[test]
fn incompatible_parameters_reject_before_callbacks() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let accepted = control(&parameters, &limits, 0)?;
    let other = ProverParameters::new(
        Parameters::new(
            VerifierConfig {
                initial_dim: 8,
                initial_k: 2,
                recursive_steps: 1,
                log_dims: vec![7],
                ks: vec![1],
                num_queries: 160,
            },
            &limits,
        )?,
        &limits,
    )?;
    let state = commit(CommitRequest {
        parameters: &parameters,
        input: input(0),
        limits: &limits,
    })?;
    let mut transcript = TranscriptProbe::new(None);
    let result = open(
        state,
        OpenRequest {
            parameters: &other,
            expected_root: accepted.root,
            claims: &accepted.claims,
            limits: &limits,
        },
        &mut transcript,
    );
    assert!(matches!(
        result,
        Err(OperationError::Input(Error::ContextMismatch))
    ));
    assert!(transcript.callbacks.is_empty());
    Ok(())
}

#[test]
fn invalid_input_length_and_claim_index_reject() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let mut short = input(0);
    short.truncate(511);
    assert!(matches!(
        commit(CommitRequest {
            parameters: &parameters,
            input: short,
            limits: &limits,
        }),
        Err(Error::Length {
            expected: 512,
            actual: 511
        })
    ));
    let accepted = control(&parameters, &limits, 0)?;
    let state = commit(CommitRequest {
        parameters: &parameters,
        input: input(0),
        limits: &limits,
    })?;
    let claims = [EvalClaim {
        index: 512,
        value: BinaryElem128::zero(),
    }];
    let mut transcript = TranscriptProbe::new(None);
    let result = open(
        state,
        OpenRequest {
            parameters: &parameters,
            expected_root: accepted.root,
            claims: &claims,
            limits: &limits,
        },
        &mut transcript,
    );
    assert!(matches!(
        result,
        Err(OperationError::Input(Error::ClaimIndex {
            index: 512,
            length: 512
        }))
    ));
    assert!(transcript.callbacks.is_empty());
    Ok(())
}

#[test]
fn malformed_round_and_row_counts_reject_before_callbacks() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let control = control(&parameters, &limits, 0)?;
    let mut short_rounds = control.proof.clone();
    short_rounds.eval_rounds.clear();
    let mut short_rows = control.proof.clone();
    short_rows.initial_ligero_proof.opened_rows.truncate(159);
    for proof in [&short_rounds, &short_rows] {
        let mut transcript = TranscriptProbe::new(None);
        let result = verify(
            VerifyRequest {
                parameters: parameters.parameters(),
                expected_root: control.root,
                claims: &control.claims,
                proof,
                limits: &limits,
            },
            &mut transcript,
        );
        assert!(matches!(
            result,
            Err(OperationError::Input(
                Error::ProofShape { .. } | Error::Length { .. }
            ))
        ));
        assert!(transcript.callbacks.is_empty());
    }
    Ok(())
}

#[test]
fn authenticated_row_mutation_reaches_protocol_rejection() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let mut control = control(&parameters, &limits, 0)?;
    let entry = &mut control.proof.initial_ligero_proof.opened_rows[0][0];
    *entry = entry.add(&BinaryElem128::one());
    assert!(matches!(
        verify_control(&parameters, &limits, &control),
        Err(OperationError::Input(Error::ProtocolFailure))
    ));
    Ok(())
}

#[test]
fn recursive_root_mutation_reaches_protocol_rejection() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let mut control = control(&parameters, &limits, 0)?;
    let mut root = control.proof.recursive_commitments[0]
        .root
        .root
        .ok_or(Error::MissingRoot { level: 1 })?;
    root[0] ^= 1;
    control.proof.recursive_commitments[0].root.root = Some(root);
    assert!(matches!(
        verify_control(&parameters, &limits, &control),
        Err(OperationError::Input(Error::ProtocolFailure))
    ));
    Ok(())
}

#[test]
fn tightened_retained_limit_rejects_and_a_fresh_attempt_succeeds() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let first = control(&parameters, &limits, 0)?;
    let state = commit(CommitRequest {
        parameters: &parameters,
        input: input(0),
        limits: &limits,
    })?;
    let tightened = Limits {
        max_retained_bytes: 0,
        ..limits
    };
    let mut transcript = TranscriptProbe::new(None);
    let result = open(
        state,
        OpenRequest {
            parameters: &parameters,
            expected_root: first.root,
            claims: &first.claims,
            limits: &tightened,
        },
        &mut transcript,
    );
    assert!(matches!(
        result,
        Err(OperationError::Input(Error::RetainedLimit { limit: 0, .. }))
    ));
    assert!(transcript.callbacks.is_empty());
    let last = control(&parameters, &limits, 2)?;
    let fresh = control(&parameters, &limits, 2)?;
    assert_eq!(last.root, fresh.root);
    assert_eq!(
        last.proof.final_ligero_proof.yr,
        fresh.proof.final_ligero_proof.yr
    );
    assert_ne!(first.root, last.root);
    Ok(())
}

#[test]
fn compact_compatible_parameters_cannot_hide_original_retained_capacity() -> TestResult {
    // Parallel verifier scratch exceeds two MiB for this compact control.
    const TIGHT_CAP: usize = 4 * 1024 * 1024;
    let roomy = Limits {
        max_allocation_bytes: 64 * 1024 * 1024,
        max_work: 20_000_000,
        max_retained_bytes: 32 * 1024 * 1024,
        ..limits()
    };
    let mut log_dims = Vec::new();
    let mut ks = Vec::new();
    log_dims.try_reserve_exact(1_000_000)?;
    ks.try_reserve_exact(1_000_000)?;
    log_dims.push(7);
    ks.push(1);
    let original = ProverParameters::new(
        Parameters::new(
            VerifierConfig {
                initial_dim: 8,
                initial_k: 1,
                recursive_steps: 1,
                log_dims,
                ks,
                num_queries: 160,
            },
            &roomy,
        )?,
        &roomy,
    )?;
    let accepted = control(&original, &roomy, 0)?;
    let compact = parameters(&roomy)?;
    assert!(original.parameters().same_geometry(compact.parameters()));
    for retained_bound in [true, false] {
        let tightened = if retained_bound {
            Limits {
                max_retained_bytes: TIGHT_CAP,
                ..roomy
            }
        } else {
            Limits {
                max_allocation_bytes: TIGHT_CAP,
                ..roomy
            }
        };
        let compact_control = control(&compact, &tightened, 0)?;
        assert_eq!(accepted.root, compact_control.root);
        let state = commit(CommitRequest {
            parameters: &original,
            input: input(0),
            limits: &roomy,
        })?;
        let mut transcript = TranscriptProbe::new(None);
        let result = open(
            state,
            OpenRequest {
                parameters: &compact,
                expected_root: accepted.root,
                claims: &accepted.claims,
                limits: &tightened,
            },
            &mut transcript,
        );
        assert!(matches!(
            (retained_bound, result),
            (
                true,
                Err(OperationError::Input(Error::RetainedLimit {
                    limit: TIGHT_CAP,
                    ..
                }))
            ) | (
                false,
                Err(OperationError::Input(Error::AllocationLimit {
                    limit: TIGHT_CAP,
                    ..
                }))
            )
        ));
        assert!(transcript.callbacks.is_empty());
    }
    Ok(())
}

fn open_with_probe(
    parameters: &ProverParameters,
    limits: &Limits,
    claims: &[EvalClaim<BinaryElem128>],
    transcript: &mut TranscriptProbe,
) -> Result<Proof, OperationError<ProbeError>> {
    let state = commit(CommitRequest {
        parameters,
        input: (0_u128..)
            .take(parameters.parameters().message_len())
            .map(|index| value(index, 0))
            .collect(),
        limits,
    })
    .map_err(OperationError::Input)?;
    let root = state.root();
    let opened = open(
        state,
        OpenRequest {
            parameters,
            expected_root: root,
            claims,
            limits,
        },
        transcript,
    )?;
    Ok(opened.proof)
}

fn assert_all_callbacks(callbacks: &[Callback], with_claims: bool) {
    assert_eq!(callbacks.contains(&Callback::Message), with_claims);
    for expected in [
        Callback::Root,
        Callback::Bytes,
        Callback::ChallengeObservation,
        Callback::ChallengeSlice,
        Callback::ChallengeSample,
        Callback::Queries,
    ] {
        assert!(callbacks.contains(&expected));
    }
}

fn assert_query_rejection<T>(
    result: &Result<T, OperationError<ProbeError>>,
    corruption: QueryCorruption,
) {
    match corruption {
        QueryCorruption::Short => assert!(matches!(
            result,
            Err(OperationError::Driver(DriverError::Query(
                QueryError::OutputCount {
                    expected: 160,
                    actual: 159
                }
            )))
        )),
        QueryCorruption::Duplicate | QueryCorruption::Descending => {
            assert!(matches!(
                result,
                Err(OperationError::Driver(DriverError::Query(
                    QueryError::Order
                )))
            ));
        }
        QueryCorruption::OutOfRange => assert!(matches!(
            result,
            Err(OperationError::Driver(DriverError::Query(
                QueryError::Index {
                    index: 1024,
                    domain: 1024
                }
            )))
        )),
    }
}

#[test]
fn every_open_callback_failure_propagates_without_continuing() -> TestResult {
    let limits = limits();
    let parameters = multiple_level_parameters(&limits)?;
    for claims in [
        vec![],
        vec![EvalClaim {
            index: 257,
            value: value(257, 0),
        }],
    ] {
        let control = control_with_claims(&parameters, &limits, 0, claims)?;
        let mut accepted = TranscriptProbe::new(None);
        let proof = open_with_probe(&parameters, &limits, &control.claims, &mut accepted)?;
        verify(
            VerifyRequest {
                parameters: parameters.parameters(),
                expected_root: control.root,
                claims: &control.claims,
                proof: &proof,
                limits: &limits,
            },
            &mut Sha256Transcript::new(7),
        )?;
        assert_all_callbacks(&accepted.callbacks, !control.claims.is_empty());
        for operation in 1..=accepted.callbacks.len() {
            let mut transcript = TranscriptProbe::new(Some(operation));
            let result = open_with_probe(&parameters, &limits, &control.claims, &mut transcript);
            assert!(matches!(result,
                Err(OperationError::Driver(DriverError::Transcript(ProbeError::Injected {
                    operation: actual
                }))) if actual == operation
            ));
            assert_eq!(transcript.callbacks.len(), operation);
        }
    }
    Ok(())
}

#[test]
fn every_verify_callback_failure_propagates_without_continuing() -> TestResult {
    let limits = limits();
    let parameters = multiple_level_parameters(&limits)?;
    for claims in [
        vec![],
        vec![EvalClaim {
            index: 257,
            value: value(257, 0),
        }],
    ] {
        let control = control_with_claims(&parameters, &limits, 0, claims)?;
        let request = || VerifyRequest {
            parameters: parameters.parameters(),
            expected_root: control.root,
            claims: &control.claims,
            proof: &control.proof,
            limits: &limits,
        };
        let mut accepted = TranscriptProbe::new(None);
        verify(request(), &mut accepted)?;
        assert_all_callbacks(&accepted.callbacks, !control.claims.is_empty());
        for operation in 1..=accepted.callbacks.len() {
            let mut transcript = TranscriptProbe::new(Some(operation));
            let result = verify(request(), &mut transcript);
            assert!(matches!(result,
                Err(OperationError::Driver(DriverError::Transcript(ProbeError::Injected {
                    operation: actual
                }))) if actual == operation
            ));
            assert_eq!(transcript.callbacks.len(), operation);
        }
    }
    Ok(())
}

#[test]
fn custom_query_outputs_cannot_bypass_driver_validation() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let accepted = control(&parameters, &limits, 0)?;
    for corruption in [
        QueryCorruption::Short,
        QueryCorruption::Duplicate,
        QueryCorruption::OutOfRange,
        QueryCorruption::Descending,
    ] {
        let mut transcript = TranscriptProbe::new(None);
        transcript.corrupt_queries = Some(corruption);
        let result = open_with_probe(&parameters, &limits, &accepted.claims, &mut transcript);
        assert_query_rejection(&result, corruption);
        let mut transcript = TranscriptProbe::new(None);
        transcript.corrupt_queries = Some(corruption);
        let result = verify(
            VerifyRequest {
                parameters: parameters.parameters(),
                expected_root: accepted.root,
                claims: &accepted.claims,
                proof: &accepted.proof,
                limits: &limits,
            },
            &mut transcript,
        );
        assert_query_rejection(&result, corruption);
    }
    Ok(())
}

#[test]
fn released_input_reuses_capacity_while_prior_proof_remains_live() -> TestResult {
    let limits = limits();
    let parameters = parameters(&limits)?;
    let state = commit(CommitRequest {
        parameters: &parameters,
        input: input(0),
        limits: &limits,
    })?;
    let first_root = state.root();
    let first_claims = [EvalClaim {
        index: 257,
        value: value(257, 0),
    }];
    let first = open(
        state,
        OpenRequest {
            parameters: &parameters,
            expected_root: first_root,
            claims: &first_claims,
            limits: &limits,
        },
        &mut Sha256Transcript::new(7),
    )?;
    let mut recycled = first.reusable_input;
    let pointer = recycled.as_ptr();
    let capacity = recycled.capacity();
    assert!(recycled.is_empty());
    assert_eq!(first.proof.final_ligero_proof.yr.capacity(), 128);
    recycled.extend((0..512).map(|index| value(index, 9)));
    assert_eq!(recycled.as_ptr(), pointer);
    assert_eq!(recycled.capacity(), capacity);
    let state = commit(CommitRequest {
        parameters: &parameters,
        input: recycled,
        limits: &limits,
    })?;
    let second_root = state.root();
    assert_ne!(first_root, second_root);
    let second_claims = [EvalClaim {
        index: 257,
        value: value(257, 9),
    }];
    let second = open(
        state,
        OpenRequest {
            parameters: &parameters,
            expected_root: second_root,
            claims: &second_claims,
            limits: &limits,
        },
        &mut Sha256Transcript::new(7),
    )?;
    assert_eq!(second.reusable_input.as_ptr(), pointer);
    assert_eq!(second.reusable_input.capacity(), capacity);
    for (expected_root, claims, proof) in [
        (first_root, &first_claims, &first.proof),
        (second_root, &second_claims, &second.proof),
    ] {
        verify(
            VerifyRequest {
                parameters: parameters.parameters(),
                expected_root,
                claims,
                proof,
                limits: &limits,
            },
            &mut Sha256Transcript::new(7),
        )?;
    }
    let fresh = control(&parameters, &limits, 9)?;
    assert_eq!(second_root, fresh.root);
    assert_eq!(
        second.proof.final_ligero_proof.yr,
        fresh.proof.final_ligero_proof.yr
    );
    Ok(())
}

#[test]
fn multiple_recursive_levels_preserve_indexed_and_no_claim_controls() -> TestResult {
    let mut limits = limits();
    limits.max_work = 50_000_000;
    let parameters = multiple_level_parameters(&limits)?;
    for claims in [
        vec![],
        vec![EvalClaim {
            index: 777,
            value: value(777, 3),
        }],
    ] {
        let state = commit(CommitRequest {
            parameters: &parameters,
            input: (0..1024).map(|index| value(index, 3)).collect(),
            limits: &limits,
        })?;
        let root = state.root();
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
        assert_eq!(opened.proof.recursive_commitments.len(), 2);
        assert_eq!(opened.proof.recursive_proofs.len(), 1);
        assert_eq!(opened.proof.final_ligero_proof.yr.len(), 64);
        verify(
            VerifyRequest {
                parameters: parameters.parameters(),
                expected_root: root,
                claims: &claims,
                proof: &opened.proof,
                limits: &limits,
            },
            &mut Sha256Transcript::new(7),
        )?;
    }
    Ok(())
}
