#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::opening_transcript::{
    sample_queries, DriverError, LegacyTranscript, OpeningTranscript, QueryRequest,
};
use crate::{
    eval_proof::{absorb_claims, EvalClaim},
    sumcheck_polys::{eval_round_poly, precompute_alpha_powers, tensorized_dot_product},
    transcript::{FiatShamir, Transcript},
    utils::{eval_sk_at_vks, hash_row},
    FinalizedLigeritoProof, VerifierConfig,
};
use binary_fields::BinaryFieldElement;
use merkle_tree::{BatchedMerkleProof, Hash, MerkleRoot};

use crate::security::LOG_INV_RATE;

/// `s_j(x)` for `j < k` (see `utils::evaluate_scaled_basis_inplace`).
fn subspace_evals<F: BinaryFieldElement>(x: F, k: usize, sks_vks: &[F]) -> Vec<F> {
    let mut s = Vec::with_capacity(k);
    if k > 0 {
        s.push(x);
        for j in 1..k {
            let p = s[j - 1];
            s.push(p.mul(&p).add(&sks_vks[j - 1].mul(&p)));
        }
    }
    s
}

/// `⟨y, ⊗_j (1, a_j)⟩` with `a_0` on the lsb of the index.
#[forbid(unsafe_code)]
#[warn(clippy::all, clippy::pedantic)]
#[deny(
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
fn tensor_inner<U: BinaryFieldElement>(y: &[U], a: &[U]) -> U {
    debug_assert_eq!(y.len(), 1 << a.len());
    let mut cur: Vec<U> = y
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| p[0].add(&a[0].mul(&p[1])))
        .collect();
    for &aj in &a[1..] {
        cur = cur
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| p[0].add(&aj.mul(&p[1])))
            .collect();
    }
    cur[0]
}

/// The verifier's running batched basis `B`, kept factored instead of as a
/// length-`m` vector: a sum of scaled generator rows `c · ⊗_{j≥off}(1, a_j)`
/// plus a sparse part from evaluation claims.
///
/// Binding the lsb variable of `c · (1, a_off) ⊗ rest` to `r` gives
/// `c · (1 + r·(a_off + 1)) · rest`, so each fold is O(#terms) and the final
/// `⟨yr, B⟩` is O(#terms · |yr|), independent of the initial message length.
struct FactoredBasis<U> {
    rows: Vec<(U, Vec<U>, usize)>,
    sparse: Vec<(usize, U)>,
}

impl<U: BinaryFieldElement> FactoredBasis<U> {
    /// Add `scale · Σ_k α^k · G[q_k]` for a code with message length `2^log_m`
    /// over field `F`, and return `Σ_k α^k · ⟨row_k, eq(rs)⟩` (the claim it
    /// enforces on the next message).
    fn add_queries<F>(
        &mut self,
        log_m: usize,
        rows: &[Vec<F>],
        queries: &[usize],
        rs: &[U],
        alpha: U,
        scale: U,
    ) -> U
    where
        F: BinaryFieldElement,
        U: From<F>,
    {
        let sks_vks: Vec<F> = eval_sk_at_vks(1 << log_m);
        let alpha_pows = precompute_alpha_powers(alpha, queries.len());
        let mut sum = U::zero();
        for ((row, &q), &ak) in rows.iter().zip(queries).zip(&alpha_pows) {
            sum = sum.add(&ak.mul(&tensorized_dot_product(row, rs)));
            let a = subspace_evals(F::from_bits(q as u64), log_m, &sks_vks)
                .into_iter()
                .map(U::from)
                .collect();
            self.rows.push((scale.mul(&ak), a, 0));
        }
        sum
    }

    fn fold(&mut self, r: U) {
        let one = U::one();
        for (c, a, off) in &mut self.rows {
            *c = c.mul(&one.add(&r.mul(&a[*off].add(&one))));
            *off += 1;
        }
        for (i, w) in &mut self.sparse {
            *w = w.mul(&if *i & 1 == 1 { r } else { one.add(&r) });
            *i >>= 1;
        }
    }

    fn scale(&mut self, beta: U) {
        for (c, _, _) in &mut self.rows {
            *c = c.mul(&beta);
        }
        for (_, w) in &mut self.sparse {
            *w = w.mul(&beta);
        }
    }

    fn inner(&self, y: &[U]) -> U
    where
        U: Send + Sync,
    {
        let term = |(c, a, off): &(U, Vec<U>, usize)| c.mul(&tensor_inner(y, &a[*off..]));
        #[cfg(feature = "parallel")]
        let rows = {
            use rayon::prelude::*;
            self.rows
                .par_iter()
                .map(term)
                .reduce(U::zero, |x, z| x.add(&z))
        };
        #[cfg(not(feature = "parallel"))]
        let rows = self.rows.iter().map(term).fold(U::zero(), |x, z| x.add(&z));
        self.sparse
            .iter()
            .fold(rows, |acc, &(i, w)| acc.add(&w.mul(&y[i])))
    }
}

/// Verify a Ligerito proof (default Merlin transcript)
pub fn verify<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    #[cfg(feature = "transcript-merlin")]
    let fs = FiatShamir::new_merlin();

    #[cfg(not(feature = "transcript-merlin"))]
    let fs = FiatShamir::new_sha256(0);

    verify_with_transcript(config, proof, fs)
}

#[derive(Clone, Copy)]
struct RowOpening<'a, F> {
    root: &'a MerkleRoot,
    merkle_proof: &'a BatchedMerkleProof,
    opened_rows: &'a [Vec<F>],
    log_rows: usize,
    row_len: usize,
}

#[forbid(unsafe_code)]
#[warn(clippy::all, clippy::pedantic)]
#[deny(
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
fn open_round<F: BinaryFieldElement, C: OpeningTranscript>(
    fs: &mut C,
    opening: RowOpening<'_, F>,
    request: QueryRequest,
) -> core::result::Result<Option<Vec<usize>>, DriverError<C::Error>> {
    let RowOpening {
        root,
        merkle_proof,
        opened_rows,
        log_rows,
        row_len,
    } = opening;
    let queries = sample_queries(fs, request)?;
    if opened_rows.len() != queries.len() || opened_rows.iter().any(|r| r.len() != row_len) {
        return Ok(None);
    }
    let hashed: Vec<Hash> = opened_rows.iter().map(|row| hash_row(row)).collect();
    Ok(ligerito_merkle::verify(root, merkle_proof, log_rows, &hashed, &queries).then_some(queries))
}

#[derive(Clone, Copy)]
pub(crate) struct VerificationRequest<'a, T: BinaryFieldElement, U: BinaryFieldElement> {
    pub(crate) config: &'a VerifierConfig,
    pub(crate) proof: &'a FinalizedLigeritoProof<T, U>,
    pub(crate) claims: &'a [EvalClaim<T>],
    pub(crate) max_query_draws: usize,
}

#[forbid(unsafe_code)]
#[warn(clippy::all, clippy::pedantic)]
#[deny(
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
fn verify_inner<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
    claims: &[EvalClaim<T>],
    fs: &mut impl Transcript,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let request = VerificationRequest {
        config,
        proof,
        claims,
        max_query_draws: 1_048_576,
    };
    let mut transcript = LegacyTranscript::<T, U, _>::new(fs);
    verify_retained(request, &mut transcript).map_err(DriverError::into_legacy)
}

struct RunningClaim<U> {
    sum: U,
    basis: FactoredBasis<U>,
}

struct InitialEvaluation<U> {
    sum: U,
    weights: Vec<U>,
    challenges: Vec<U>,
}

#[forbid(unsafe_code)]
#[warn(clippy::all, clippy::pedantic)]
#[deny(
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
fn verify_sumcheck_round<C: OpeningTranscript>(
    fs: &mut C,
    coefficients: (C::Challenge, C::Challenge, C::Challenge),
    sum: &mut C::Challenge,
) -> core::result::Result<Option<C::Challenge>, DriverError<C::Error>> {
    if coefficients.0.add(&coefficients.1) != *sum {
        return Ok(None);
    }
    fs.observe_challenge(coefficients.0)
        .map_err(DriverError::Transcript)?;
    fs.observe_challenge(coefficients.1)
        .map_err(DriverError::Transcript)?;
    fs.observe_challenge(coefficients.2)
        .map_err(DriverError::Transcript)?;
    let challenge = fs.sample_challenge().map_err(DriverError::Transcript)?;
    *sum = eval_round_poly(coefficients, challenge);
    Ok(Some(challenge))
}

#[forbid(unsafe_code)]
#[warn(clippy::all, clippy::pedantic)]
#[deny(
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
impl<U: BinaryFieldElement> RunningClaim<U> {
    fn fold_rounds<C: OpeningTranscript<Challenge = U>>(
        &mut self,
        rounds: &mut core::slice::Iter<'_, (U, U, U)>,
        count: usize,
        fs: &mut C,
    ) -> core::result::Result<Option<Vec<U>>, DriverError<C::Error>> {
        let mut challenges = Vec::with_capacity(count);
        for _ in 0..count {
            let Some(&coefficients) = rounds.next() else {
                return Ok(None);
            };
            let Some(challenge) = verify_sumcheck_round(fs, coefficients, &mut self.sum)? else {
                return Ok(None);
            };
            self.basis.fold(challenge);
            challenges.push(challenge);
        }
        Ok(Some(challenges))
    }
}

#[forbid(unsafe_code)]
#[warn(clippy::all, clippy::pedantic)]
#[deny(
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
impl<U: BinaryFieldElement> InitialEvaluation<U> {
    fn glue<T: BinaryFieldElement, C: OpeningTranscript<Challenge = U>>(
        self,
        mut basis: FactoredBasis<U>,
        row_sum: U,
        claims: &[EvalClaim<T>],
        initial_k: usize,
        fs: &mut C,
    ) -> core::result::Result<RunningClaim<U>, DriverError<C::Error>> {
        if claims.is_empty() {
            return Ok(RunningClaim {
                sum: row_sum,
                basis,
            });
        }
        let beta = fs.sample_challenge().map_err(DriverError::Transcript)?;
        basis.scale(beta);
        for (claim, &alpha) in claims.iter().zip(&self.weights) {
            let mut weight = alpha;
            for (bit, &challenge) in self.challenges.iter().enumerate() {
                let factor = if (claim.index >> bit) & 1 == 1 {
                    challenge
                } else {
                    U::one().add(&challenge)
                };
                weight = weight.mul(&factor);
            }
            basis.sparse.push((claim.index >> initial_k, weight));
        }
        Ok(RunningClaim {
            sum: self.sum.add(&beta.mul(&row_sum)),
            basis,
        })
    }
}

#[forbid(unsafe_code)]
#[warn(clippy::all, clippy::pedantic)]
#[deny(
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
impl<T, U> VerificationRequest<'_, T, U>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    fn has_expected_shape(&self) -> bool {
        let steps = self.config.recursive_steps;
        let last = steps - 1;
        let expected_eval_rounds = if self.claims.is_empty() {
            0
        } else {
            self.config.initial_k
        };
        self.proof.recursive_commitments.len() == steps
            && self.proof.recursive_proofs.len() == last
            && self.proof.sumcheck_transcript.transcript.len()
                == self.config.ks.iter().sum::<usize>()
            && self.proof.eval_rounds.len() == expected_eval_rounds
            && self.proof.final_ligero_proof.yr.len() == 1 << self.config.log_dims[last]
            && self
                .claims
                .iter()
                .all(|claim| claim.index >> self.config.poly_log_size() == 0)
    }

    fn initial_evaluation<C: OpeningTranscript<Message = T, Challenge = U>>(
        &self,
        fs: &mut C,
    ) -> core::result::Result<Option<InitialEvaluation<U>>, DriverError<C::Error>> {
        let mut evaluation = InitialEvaluation {
            sum: U::zero(),
            weights: Vec::new(),
            challenges: Vec::with_capacity(self.config.initial_k),
        };
        if self.claims.is_empty() {
            for _ in 0..self.config.initial_k {
                evaluation
                    .challenges
                    .push(fs.sample_challenge().map_err(DriverError::Transcript)?);
            }
            return Ok(Some(evaluation));
        }
        evaluation.weights = (0..self.claims.len())
            .map(|_| fs.sample_challenge())
            .collect::<core::result::Result<_, _>>()
            .map_err(DriverError::Transcript)?;
        for (claim, &alpha) in self.claims.iter().zip(&evaluation.weights) {
            evaluation.sum = evaluation.sum.add(&alpha.mul(&U::from(claim.value)));
        }
        for round in &self.proof.eval_rounds {
            let coefficients = (round.s0, round.s1, round.s2);
            let Some(challenge) = verify_sumcheck_round(fs, coefficients, &mut evaluation.sum)?
            else {
                return Ok(None);
            };
            evaluation.challenges.push(challenge);
        }
        Ok(Some(evaluation))
    }

    fn initial_state<C: OpeningTranscript<Message = T, Challenge = U>>(
        &self,
        fs: &mut C,
    ) -> core::result::Result<Option<RunningClaim<U>>, DriverError<C::Error>> {
        self.config.absorb_into(fs)?;
        fs.observe_root(&self.proof.initial_ligero_cm.root)
            .map_err(DriverError::Transcript)?;
        absorb_claims(fs, self.claims)?;
        let Some(evaluation) = self.initial_evaluation(fs)? else {
            return Ok(None);
        };
        fs.observe_root(&self.proof.recursive_commitments[0].root)
            .map_err(DriverError::Transcript)?;
        let initial = &self.proof.initial_ligero_proof;
        let log_rows = self.config.initial_dim + LOG_INV_RATE;
        let query_request =
            QueryRequest::new(1 << log_rows, self.config.num_queries, self.max_query_draws)
                .map_err(DriverError::Query)?;
        let Some(queries) = open_round(
            fs,
            RowOpening {
                root: &self.proof.initial_ligero_cm.root,
                merkle_proof: &initial.merkle_proof,
                opened_rows: &initial.opened_rows,
                log_rows,
                row_len: 1 << self.config.initial_k,
            },
            query_request,
        )?
        else {
            return Ok(None);
        };
        let alpha = fs.sample_challenge().map_err(DriverError::Transcript)?;
        let capacity = queries
            .len()
            .checked_mul(self.config.recursive_steps)
            .ok_or(DriverError::Native(crate::LigeritoError::InvalidProof))?;
        let mut basis = FactoredBasis {
            rows: Vec::new(),
            sparse: Vec::new(),
        };
        basis
            .rows
            .try_reserve_exact(capacity)
            .map_err(DriverError::Allocation)?;
        let row_sum = basis.add_queries::<T>(
            self.config.initial_dim,
            &initial.opened_rows,
            &queries,
            &evaluation.challenges,
            alpha,
            U::one(),
        );
        fs.observe_challenge(row_sum)
            .map_err(DriverError::Transcript)?;
        Ok(Some(evaluation.glue(
            basis,
            row_sum,
            self.claims,
            self.config.initial_k,
            fs,
        )?))
    }

    fn verify_nonfinal_level<C: OpeningTranscript<Message = T, Challenge = U>>(
        &self,
        level: usize,
        challenges: &[U],
        running: &mut RunningClaim<U>,
        fs: &mut C,
    ) -> core::result::Result<bool, DriverError<C::Error>> {
        fs.observe_root(&self.proof.recursive_commitments[level + 1].root)
            .map_err(DriverError::Transcript)?;
        let proof = &self.proof.recursive_proofs[level];
        let log_rows = self.config.log_dims[level] + LOG_INV_RATE;
        let query_request =
            QueryRequest::new(1 << log_rows, self.config.num_queries, self.max_query_draws)
                .map_err(DriverError::Query)?;
        let Some(queries) = open_round(
            fs,
            RowOpening {
                root: &self.proof.recursive_commitments[level].root,
                merkle_proof: &proof.merkle_proof,
                opened_rows: &proof.opened_rows,
                log_rows,
                row_len: 1 << self.config.ks[level],
            },
            query_request,
        )?
        else {
            return Ok(false);
        };
        let alpha = fs.sample_challenge().map_err(DriverError::Transcript)?;
        let start = running.basis.rows.len();
        let row_sum = running.basis.add_queries::<U>(
            self.config.log_dims[level],
            &proof.opened_rows,
            &queries,
            challenges,
            alpha,
            U::one(),
        );
        fs.observe_challenge(row_sum)
            .map_err(DriverError::Transcript)?;
        let beta = fs.sample_challenge().map_err(DriverError::Transcript)?;
        running.sum = running.sum.add(&beta.mul(&row_sum));
        for (coefficient, _, _) in &mut running.basis.rows[start..] {
            *coefficient = coefficient.mul(&beta);
        }
        Ok(true)
    }

    fn verify_final_level<C: OpeningTranscript<Message = T, Challenge = U>>(
        &self,
        challenges: &[U],
        running: &RunningClaim<U>,
        fs: &mut C,
    ) -> core::result::Result<bool, DriverError<C::Error>> {
        let last = self.config.recursive_steps - 1;
        let proof = &self.proof.final_ligero_proof;
        fs.observe_challenges(&proof.yr)
            .map_err(DriverError::Transcript)?;
        let log_rows = self.config.log_dims[last] + LOG_INV_RATE;
        let query_request =
            QueryRequest::new(1 << log_rows, self.config.num_queries, self.max_query_draws)
                .map_err(DriverError::Query)?;
        let Some(queries) = open_round(
            fs,
            RowOpening {
                root: &self.proof.recursive_commitments[last].root,
                merkle_proof: &proof.merkle_proof,
                opened_rows: &proof.opened_rows,
                log_rows,
                row_len: 1 << self.config.ks[last],
            },
            query_request,
        )?
        else {
            return Ok(false);
        };
        if running.basis.inner(&proof.yr) != running.sum {
            return Ok(false);
        }
        Ok(final_rows_match(
            proof,
            &queries,
            challenges,
            self.config.log_dims[last],
        ))
    }
}

#[forbid(unsafe_code)]
#[warn(clippy::all, clippy::pedantic)]
#[deny(
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
fn final_rows_match<U: BinaryFieldElement + Send + Sync>(
    proof: &crate::FinalLigeroProof<U>,
    queries: &[usize],
    challenges: &[U],
    log_dimension: usize,
) -> bool {
    let sks_vks: Vec<U> = eval_sk_at_vks(1 << log_dimension);
    let row_ok = |(&query, row): (&usize, &Vec<U>)| {
        let factors = subspace_evals(U::from_bits(query as u64), log_dimension, &sks_vks);
        tensorized_dot_product(row, challenges) == tensor_inner(&proof.yr, &factors)
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        queries
            .par_iter()
            .zip(proof.opened_rows.par_iter())
            .all(row_ok)
    }
    #[cfg(not(feature = "parallel"))]
    queries.iter().zip(&proof.opened_rows).all(row_ok)
}

#[cfg_attr(feature = "std", tracing::instrument(skip_all))]
#[forbid(unsafe_code)]
#[warn(clippy::all, clippy::pedantic)]
#[deny(
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
pub(crate) fn verify_retained<T, U, C>(
    request: VerificationRequest<'_, T, U>,
    fs: &mut C,
) -> core::result::Result<bool, DriverError<C::Error>>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
    C: OpeningTranscript<Message = T, Challenge = U>,
{
    if !request.config.is_well_formed() {
        return Err(DriverError::Native(crate::LigeritoError::InvalidProof));
    }
    if !request.has_expected_shape() {
        return Ok(false);
    }
    let Some(mut running) = request.initial_state(fs)? else {
        return Ok(false);
    };
    let mut rounds = request.proof.sumcheck_transcript.transcript.iter();
    for level in 0..request.config.recursive_steps {
        let Some(challenges) = running.fold_rounds(&mut rounds, request.config.ks[level], fs)?
        else {
            return Ok(false);
        };
        if level == request.config.recursive_steps - 1 {
            return request.verify_final_level(&challenges, &running, fs);
        }
        if !request.verify_nonfinal_level(level, &challenges, &mut running, fs)? {
            return Ok(false);
        }
    }
    Err(DriverError::Native(crate::LigeritoError::InvalidProof))
}

/// Verify with custom transcript implementation
pub fn verify_with_transcript<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
    mut fs: impl Transcript,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    verify_inner(config, proof, &[], &mut fs)
}

/// Verify a proof from `prove_with_evaluations`: the commitment opens to a
/// polynomial with `P(z_k) = v_k` for every claim.
pub fn verify_with_evaluations<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
    claims: &[EvalClaim<T>],
    mut fs: impl Transcript,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    if claims.is_empty() {
        return Err(crate::LigeritoError::InvalidProof);
    }
    verify_inner(config, proof, claims, &mut fs)
}

/// SHA256-based verification for compatibility
pub fn verify_sha256<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_sha256(1234);
    verify_with_transcript(config, proof, fs)
}

/// Same as `verify` (kept for API compatibility).
pub fn verify_debug<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    verify(config, proof)
}

/// Same as `verify_with_transcript` (kept for API compatibility).
pub fn verify_complete_with_transcript<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
    fs: impl Transcript,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    verify_with_transcript(config, proof, fs)
}

/// complete verifier with default transcript
#[cfg(feature = "transcript-merlin")]
pub fn verify_complete<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_merlin();
    verify_complete_with_transcript(config, proof, fs)
}

/// complete verifier with default transcript (SHA256 when Merlin not available)
#[cfg(not(feature = "transcript-merlin"))]
pub fn verify_complete<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_sha256(0);
    verify_complete_with_transcript(config, proof, fs)
}

/// complete verifier with the SHA-256 transcript
pub fn verify_complete_sha256<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_sha256(1234);
    verify_complete_with_transcript(config, proof, fs)
}

/// complete verifier with BLAKE2b transcript (optimized for Substrate runtimes)
///
/// Uses `sp_io::hashing::blake2_256` in no_std for efficient host function calls.
/// Proofs must be generated with `prove_blake2b()` to be compatible.
#[cfg(feature = "transcript-blake2b")]
pub fn verify_blake2b<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_blake2b();
    verify_with_transcript(config, proof, fs)
}

/// complete verifier with BLAKE2b transcript
#[cfg(feature = "transcript-blake2b")]
pub fn verify_complete_blake2b<T, U>(
    config: &VerifierConfig,
    proof: &FinalizedLigeritoProof<T, U>,
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync,
    U: BinaryFieldElement + Send + Sync + From<T>,
{
    let fs = FiatShamir::new_blake2b();
    verify_complete_with_transcript(config, proof, fs)
}

// ============================================================================
// bytes-based verification (no_std, no atomics, no SCALE codec)
// ============================================================================

/// Verify a proof from serialized bytes using postcard (no_std serde format).
///
/// This function handles deserialization internally, avoiding the need for
/// parity-scale-codec (which requires Arc/atomics). Suitable for on-chain
/// verification on targets like riscv32em (PolkaVM/JAM).
///
/// The proof bytes must be serialized with `postcard::to_allocvec()` by the prover.
/// Public outputs bytes are absorbed into the transcript before verification.
///
/// Returns Ok(true) if the proof is valid, Ok(false) if invalid, Err on deserialize failure.
#[cfg(feature = "postcard-serde")]
pub fn verify_bytes<T, U>(
    config: &VerifierConfig,
    proof_bytes: &[u8],
    public_outputs_bytes: &[u8],
) -> crate::Result<bool>
where
    T: BinaryFieldElement + Send + Sync + serde::de::DeserializeOwned,
    U: BinaryFieldElement + Send + Sync + From<T> + serde::de::DeserializeOwned,
{
    let proof: FinalizedLigeritoProof<T, U> =
        postcard::from_bytes(proof_bytes).map_err(|_| crate::LigeritoError::InvalidProof)?;

    let mut fs = FiatShamir::new_sha256(0);
    fs.absorb_bytes(b"public_outputs", public_outputs_bytes);

    verify_with_transcript(config, &proof, fs)
}

/// Serialize a proof to bytes using postcard (for prover side).
///
/// The resulting bytes can be verified with `verify_bytes()`.
#[cfg(feature = "postcard-serde")]
#[forbid(unsafe_code)]
#[warn(clippy::all, clippy::pedantic)]
#[deny(
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
#[expect(
    clippy::missing_errors_doc,
    reason = "FinalizedLigeritoProof documents legacy serialization errors."
)]
pub fn proof_to_bytes<T, U>(proof: &FinalizedLigeritoProof<T, U>) -> crate::Result<Vec<u8>>
where
    T: BinaryFieldElement + Send + Sync + serde::Serialize,
    U: BinaryFieldElement + Send + Sync + serde::Serialize,
{
    postcard::to_allocvec(proof).map_err(|_| crate::LigeritoError::InvalidProof)
}

#[cfg(all(test, feature = "prover"))]
mod tests {
    use super::*;
    use crate::configs::{hardcoded_config_12, hardcoded_config_12_verifier};
    use crate::prover::prove;
    use ligerito_binary_fields::{BinaryElem128, BinaryElem32};
    use std::marker::PhantomData;

    #[test]
    fn test_verify_simple_proof() {
        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        // Test with simple polynomial
        let poly = vec![BinaryElem32::one(); 1 << 12];

        let proof = prove(&prover_config, &poly).expect("Proof generation failed");
        let result = verify(&verifier_config, &proof).expect("Verification failed");

        assert!(result, "Verification should succeed for valid proof");
    }

    #[test]
    fn test_verify_zero_polynomial() {
        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        // Test with zero polynomial
        let poly = vec![BinaryElem32::zero(); 1 << 12];

        let proof = prove(&prover_config, &poly).expect("Proof generation failed");
        let result = verify(&verifier_config, &proof).expect("Verification failed");

        assert!(result, "Verification should succeed for zero polynomial");
    }

    #[test]
    fn test_verify_with_sha256() {
        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        // Test with patterned polynomial
        let mut poly = vec![BinaryElem32::zero(); 1 << 12];
        poly[0] = BinaryElem32::one();
        poly[1] = BinaryElem32::from(2);

        let proof = crate::prover::prove_sha256(&prover_config, &poly)
            .expect("SHA256 proof generation failed");
        let result = verify_sha256(&verifier_config, &proof).expect("SHA256 verification failed");

        assert!(result, "SHA256 verification should succeed");
    }

    #[test]
    fn test_debug_verification() {
        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        // Use a non-constant polynomial to avoid degenerate case
        let poly: Vec<BinaryElem32> = (0..(1 << 12))
            .map(|i| BinaryElem32::from((i * 7 + 13) as u32)) // Some pattern to ensure diversity
            .collect();

        let proof = prove(&prover_config, &poly).expect("Proof generation failed");
        let result = verify_debug(&verifier_config, &proof).expect("Debug verification failed");

        assert!(result, "Debug verification should succeed");
    }

    #[test]
    #[cfg(feature = "transcript-blake2b")]
    fn test_blake2b_transcript_compatibility() {
        use crate::prover::prove_blake2b;

        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        // Test with patterned polynomial
        let poly: Vec<BinaryElem32> = (0..(1 << 12))
            .map(|i| BinaryElem32::from((i * 7 + 13) as u32))
            .collect();

        // Prove with Blake2b
        let proof = prove_blake2b(&prover_config, &poly).expect("Blake2b proof generation failed");

        // Verify with Blake2b
        let result = verify_blake2b(&verifier_config, &proof).expect("Blake2b verification failed");

        assert!(
            result,
            "Blake2b verification should succeed for valid proof"
        );

        // Also test complete verifier
        let proof2 = prove_blake2b(&prover_config, &poly).expect("Blake2b proof generation failed");
        let result2 = verify_complete_blake2b(&verifier_config, &proof2)
            .expect("Blake2b complete verification failed");

        assert!(result2, "Blake2b complete verification should succeed");
    }

    #[test]
    #[cfg(feature = "transcript-blake2b")]
    fn test_sha256_transcript_compatibility() {
        // This test verifies that SHA256 proofs cannot be verified with Blake2b
        // (and vice versa) - they are NOT compatible
        use crate::prover::prove_sha256;

        let prover_config =
            hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = hardcoded_config_12_verifier();

        let poly: Vec<BinaryElem32> = (0..(1 << 12))
            .map(|i| BinaryElem32::from((i * 7 + 13) as u32))
            .collect();

        // Prove with SHA256
        let sha_proof =
            prove_sha256(&prover_config, &poly).expect("SHA256 proof generation failed");

        // Trying to verify SHA256 proof with Blake2b should fail
        let result = verify_blake2b(&verifier_config, &sha_proof)
            .expect("Verification call should not panic");

        assert!(
            !result,
            "SHA256 proof should NOT verify with Blake2b transcript"
        );
    }

    /// Commit uniformly random (non-codeword) matrices and follow the
    /// transcript exactly, patching every round message so `g(0) + g(1)`
    /// matches the running claim. This forgery verified before the fix.
    #[test]
    #[forbid(unsafe_code)]
    #[warn(clippy::all, clippy::pedantic)]
    #[deny(
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
    fn test_random_matrix_forgery_rejected() -> core::result::Result<(), Box<dyn std::error::Error>>
    {
        use crate::data_structures::{finalize, RecursiveLigeroWitness};
        use crate::sumcheck_polys::{fold_lsb, induce_sumcheck_poly, product_round_coeffs};
        use crate::{
            FinalLigeroProof, LigeritoProof, RecursiveLigeroCommitment, RecursiveLigeroProof,
            SumcheckTranscript,
        };
        type T = BinaryElem32;
        type U = BinaryElem128;

        fn garbage<F: BinaryFieldElement + bytemuck::Pod + Send + Sync>(
            rows: usize,
            cols: usize,
            rnd: &mut impl FnMut() -> u64,
        ) -> RecursiveLigeroWitness<F> {
            let data: Vec<F> = (0..rows * cols).map(|_| F::from_bits(rnd())).collect();
            RecursiveLigeroWitness::from_columns(data, cols)
        }

        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let pc = hardcoded_config_12(PhantomData::<T>, PhantomData::<U>);
        let vc = hardcoded_config_12_verifier();
        let mut fs = FiatShamir::new_sha256(0);

        vc.absorb_into(&mut LegacyTranscript::<T, U, _>::new(&mut fs))?;
        let w0 = garbage::<T>(pc.initial_dims.0 * 4, pc.initial_dims.1, &mut rnd);
        let cm0 = RecursiveLigeroCommitment {
            root: w0.tree.get_root(),
        };
        fs.absorb_root(&cm0.root);
        absorb_claims(&mut LegacyTranscript::<T, U, _>::new(&mut fs), &[])?;
        let rho0: Vec<U> = (0..vc.initial_k).map(|_| fs.get_challenge()).collect();

        let w1 = garbage::<U>(pc.dims[0].0 * 4, pc.dims[0].1, &mut rnd);
        let cm1 = RecursiveLigeroCommitment {
            root: w1.tree.get_root(),
        };
        fs.absorb_root(&cm1.root);

        let queries = fs.get_distinct_queries(w0.num_rows(), vc.num_queries);
        let alpha = fs.get_challenge::<U>();
        let rows: Vec<Vec<T>> = queries.iter().map(|&q| w0.row(q)).collect();
        let sks: Vec<T> = eval_sk_at_vks(1 << vc.initial_dim);
        let (mut basis, mut sum) =
            induce_sumcheck_poly(vc.initial_dim, &sks, &rows, &rho0, &queries, alpha);
        fs.absorb_elem(sum);

        // the forger's message: anything of the right length
        let mut v: Vec<U> = (0..1 << vc.initial_dim)
            .map(|_| U::from_bits(rnd()))
            .collect();
        let mut transcript = vec![];
        for _ in 0..vc.ks[0] {
            let (s0, _, s2) = product_round_coeffs(&v, &basis);
            let c = (s0, s0.add(&sum), s2); // force g(0) + g(1) = sum
            fs.absorb_elem(c.0);
            fs.absorb_elem(c.1);
            fs.absorb_elem(c.2);
            let r = fs.get_challenge::<U>();
            sum = eval_round_poly(c, r);
            v = fold_lsb(&v, r);
            basis = fold_lsb(&basis, r);
            transcript.push(c);
        }
        fs.absorb_elems(&v);
        let fq = fs.get_distinct_queries(w1.num_rows(), vc.num_queries);

        let mut proof = LigeritoProof::<T, U>::new();
        proof.initial_ligero_cm = Some(cm0);
        proof.initial_ligero_proof = Some(RecursiveLigeroProof {
            opened_rows: rows,
            merkle_proof: w0.tree.prove(&queries),
        });
        proof.recursive_commitments.push(cm1);
        proof.final_ligero_proof = Some(FinalLigeroProof {
            yr: v,
            opened_rows: fq.iter().map(|&q| w1.row(q)).collect(),
            merkle_proof: w1.tree.prove(&fq),
        });
        proof.sumcheck_transcript = Some(SumcheckTranscript { transcript });
        let proof = finalize(proof)?;

        assert!(!verify_with_transcript(
            &vc,
            &proof,
            FiatShamir::new_sha256(0)
        )?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "std")]
    #[forbid(unsafe_code)]
    #[warn(clippy::all, clippy::pedantic)]
    #[deny(
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
    fn b1_authenticated_rows_reject_wrong_count_and_width(
    ) -> core::result::Result<(), Box<dyn std::error::Error>> {
        use crate::opening_transcript::TranscriptError;
        use crate::retained::{Limits, Parameters, ProverParameters};
        use crate::Sha256Transcript;

        let limits = Limits {
            max_allocation_bytes: 16 * 1024 * 1024,
            max_work: 10_000_000,
            max_retained_bytes: 8 * 1024 * 1024,
            max_claims: 16,
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
        let config = parameters.config();
        let message: Vec<BinaryElem128> = (0_u128..512).map(BinaryElem128::from_value).collect();
        let witness = crate::ligero::ligero_commit(
            &message,
            config.initial_dims.0,
            config.initial_dims.1,
            &config.initial_reed_solomon,
        );
        assert_eq!(witness.num_rows(), 1024);
        assert_eq!(witness.num_cols(), 2);
        let root = witness.tree.get_root();
        let request = QueryRequest::new(1024, 160, 4096)?;
        let mut transcript = Sha256Transcript::new(7);
        transcript.observe_root(&root)?;
        let queries = transcript.sample_queries(request)?;
        let merkle_proof = witness.tree.prove(&queries);
        let mut rows: Vec<Vec<BinaryElem128>> =
            queries.iter().map(|&index| witness.row(index)).collect();
        let open = |rows: &[Vec<BinaryElem128>]| -> core::result::Result<
            Option<Vec<usize>>,
            DriverError<TranscriptError>,
        > {
            let mut transcript = Sha256Transcript::new(7);
            transcript
                .observe_root(&root)
                .map_err(DriverError::Transcript)?;
            open_round(
                &mut transcript,
                RowOpening {
                    root: &root,
                    merkle_proof: &merkle_proof,
                    opened_rows: rows,
                    log_rows: 10,
                    row_len: 2,
                },
                request,
            )
        };
        assert_eq!(open(&rows)?, Some(queries));
        // Isolate row-shape checks otherwise rejected by public preflight.
        assert_eq!(open(&rows[..159])?, None);
        rows[0].truncate(1);
        assert_eq!(open(&rows)?, None);
        Ok(())
    }
}
