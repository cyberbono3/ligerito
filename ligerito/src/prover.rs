use crate::data_structures::RecursiveLigeroWitness;
use crate::opening_transcript::{
    sample_queries, DriverError, LegacyTranscript, OpeningTranscript, QueryRequest,
};
use crate::{
    data_structures::finalize,
    eval_proof::{absorb_claims, compute_batched_eq, EvalClaim, EvalSumcheckRound},
    ligero::ligero_commit,
    sumcheck_polys::product_round_coeffs,
    transcript::{FiatShamir, Transcript},
    utils::{eval_sk_at_vks, partial_eval_lift, partial_eval_multilinear},
    FinalLigeroProof, FinalizedLigeritoProof, LigeritoProof, ProverConfig,
    RecursiveLigeroCommitment, RecursiveLigeroProof, SumcheckTranscript,
};
use binary_fields::BinaryFieldElement;

#[cfg(not(feature = "parallel"))]
use crate::sumcheck_polys::induce_sumcheck_poly as induce_sumcheck_poly_auto;
#[cfg(feature = "parallel")]
use crate::sumcheck_polys::induce_sumcheck_poly_parallel as induce_sumcheck_poly_auto;

type ProductRound<U> = ((U, U, U), U);

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
fn product_sumcheck_round<U, C>(
    f: &mut Vec<U>,
    g: &mut Vec<U>,
    fs: &mut C,
) -> core::result::Result<ProductRound<U>, DriverError<C::Error>>
where
    U: BinaryFieldElement,
    C: OpeningTranscript<Challenge = U>,
{
    let coeffs = product_round_coeffs(f, g);
    fs.observe_challenge(coeffs.0)
        .map_err(DriverError::Transcript)?;
    fs.observe_challenge(coeffs.1)
        .map_err(DriverError::Transcript)?;
    fs.observe_challenge(coeffs.2)
        .map_err(DriverError::Transcript)?;
    let r = fs.sample_challenge().map_err(DriverError::Transcript)?;
    partial_eval_multilinear(f, &[r]);
    partial_eval_multilinear(g, &[r]);
    Ok((coeffs, r))
}

pub(crate) enum MessageInput<'a, T, U> {
    Borrowed(&'a [T]),
    Owned(Vec<U>),
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
impl<T, U> MessageInput<'_, T, U>
where
    T: BinaryFieldElement,
    U: BinaryFieldElement + From<T>,
{
    fn into_full(self) -> Vec<U> {
        match self {
            Self::Borrowed(values) => values.iter().copied().map(U::from).collect(),
            Self::Owned(values) => values,
        }
    }

    fn fold(self, point: &[U]) -> Vec<U> {
        match self {
            Self::Borrowed(values) => partial_eval_lift(values, point),
            Self::Owned(mut values) => {
                partial_eval_multilinear(&mut values, point);
                values
            }
        }
    }
}

pub(crate) struct ProvingOutput<T: BinaryFieldElement, U: BinaryFieldElement> {
    pub(crate) proof: FinalizedLigeritoProof<T, U>,
    pub(crate) reusable_input: Vec<U>,
}

pub(crate) struct RetainedInput<'a, T: BinaryFieldElement, U> {
    pub(crate) table: MessageInput<'a, T, U>,
    pub(crate) witness: RecursiveLigeroWitness<T>,
}

#[derive(Clone, Copy)]
pub(crate) struct ProvingRequest<'a, T: BinaryFieldElement, U: BinaryFieldElement> {
    pub(crate) config: &'a ProverConfig<T, U>,
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
fn prove_inner<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
    claims: &[EvalClaim<T>],
    fs: &mut impl Transcript,
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + bytemuck::Pod + 'static,
    U: BinaryFieldElement + From<T> + bytemuck::Pod + 'static,
{
    config.validate()?;
    let vcfg = config.verifier_config();
    if poly.len() != 1 << vcfg.poly_log_size() || claims.iter().any(|c| c.index >= poly.len()) {
        return Err(crate::LigeritoError::InvalidProof);
    }
    let witness = ligero_commit(
        poly,
        config.initial_dims.0,
        config.initial_dims.1,
        &config.initial_reed_solomon,
    );
    let state = RetainedInput {
        table: MessageInput::Borrowed(poly),
        witness,
    };
    let request = ProvingRequest {
        config,
        claims,
        max_query_draws: 1_048_576,
    };
    let mut transcript = LegacyTranscript::<T, U, _>::new(fs);
    let output =
        prove_retained(request, state, &mut transcript).map_err(DriverError::into_legacy)?;
    Ok(output.proof)
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
#[expect(
    clippy::too_many_lines,
    reason = "Keep the reviewed sequential native transcript schedule in one driver."
)]
pub(crate) fn prove_retained<T, U, C>(
    request: ProvingRequest<'_, T, U>,
    input: RetainedInput<'_, T, U>,
    fs: &mut C,
) -> core::result::Result<ProvingOutput<T, U>, DriverError<C::Error>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
    C: OpeningTranscript<Message = T, Challenge = U>,
{
    let ProvingRequest {
        config,
        claims,
        max_query_draws,
    } = request;
    let RetainedInput {
        table: poly,
        witness: wtns_0,
    } = input;
    let vcfg = config.verifier_config();
    vcfg.absorb_into(fs)?;
    let cm_0 = RecursiveLigeroCommitment {
        root: wtns_0.tree.get_root(),
    };
    fs.observe_root(&cm_0.root)
        .map_err(DriverError::Transcript)?;
    absorb_claims(fs, claims)?;

    let mut proof = LigeritoProof::<T, U>::new();
    proof.initial_ligero_cm = Some(cm_0);

    // Bind the initial_k column variables of v_0.
    let mut eval_rounds = Vec::new();
    let (mut v, rho_0, eval_residual): (Vec<U>, Vec<U>, _) = if claims.is_empty() {
        let rho: Vec<U> = (0..config.initial_k)
            .map(|_| fs.sample_challenge())
            .collect::<core::result::Result<_, _>>()
            .map_err(DriverError::Transcript)?;
        (poly.fold(&rho), rho, None)
    } else {
        let mut v = poly.into_full();
        let alphas: Vec<U> = (0..claims.len())
            .map(|_| fs.sample_challenge())
            .collect::<core::result::Result<_, _>>()
            .map_err(DriverError::Transcript)?;
        let mut q = compute_batched_eq::<T, U>(claims, &alphas, vcfg.poly_log_size());
        let mut rho = Vec::with_capacity(config.initial_k);
        for _ in 0..config.initial_k {
            let ((s0, s1, s2), r) = product_sumcheck_round(&mut v, &mut q, fs)?;
            eval_rounds.push(EvalSumcheckRound { s0, s1, s2 });
            rho.push(r);
        }
        (v, rho, Some(q))
    };

    // v_1 = v_0(ρ_0, ·)
    let wtns_1 = ligero_commit(
        &v,
        config.dims[0].0,
        config.dims[0].1,
        &config.reed_solomon_codes[0],
    );
    let cm_1 = RecursiveLigeroCommitment {
        root: wtns_1.tree.get_root(),
    };
    proof.recursive_commitments.push(cm_1.clone());
    fs.observe_root(&cm_1.root)
        .map_err(DriverError::Transcript)?;

    let queries = sample_queries(
        fs,
        QueryRequest::new(wtns_0.num_rows(), config.num_queries, max_query_draws)
            .map_err(DriverError::Query)?,
    )?;
    let alpha = fs.sample_challenge().map_err(DriverError::Transcript)?;
    let opened_rows: Vec<Vec<T>> = queries.iter().map(|&q| wtns_0.row(q)).collect();
    let sks_vks: Vec<T> = eval_sk_at_vks(config.initial_dims.0);
    let (b_0, s_0) = induce_sumcheck_poly_auto(
        vcfg.initial_dim,
        &sks_vks,
        &opened_rows,
        &rho_0,
        &queries,
        alpha,
    );
    fs.observe_challenge(s_0).map_err(DriverError::Transcript)?;
    proof.initial_ligero_proof = Some(RecursiveLigeroProof {
        opened_rows,
        merkle_proof: wtns_0.tree.prove(&queries),
    });

    let mut basis = match eval_residual {
        None => b_0,
        Some(q) => {
            let beta = fs.sample_challenge().map_err(DriverError::Transcript)?;
            q.iter()
                .zip(&b_0)
                .map(|(qi, bi)| qi.add(&beta.mul(bi)))
                .collect()
        }
    };

    let mut sumcheck_transcript = Vec::new();
    let mut wtns_prev = wtns_1;

    for i in 0..config.recursive_steps {
        let mut rs = Vec::with_capacity(config.ks[i]);
        for _ in 0..config.ks[i] {
            let (coeffs, r) = product_sumcheck_round(&mut v, &mut basis, fs)?;
            sumcheck_transcript.push(coeffs);
            rs.push(r);
        }

        let rows = wtns_prev.num_rows();

        if i == config.recursive_steps - 1 {
            fs.observe_challenges(&v).map_err(DriverError::Transcript)?;
            let queries = sample_queries(
                fs,
                QueryRequest::new(rows, config.num_queries, max_query_draws)
                    .map_err(DriverError::Query)?,
            )?;
            let mut terminal = Vec::new();
            terminal
                .try_reserve_exact(v.len())
                .map_err(DriverError::Allocation)?;
            terminal.extend_from_slice(&v);
            v.clear();
            proof.final_ligero_proof = Some(FinalLigeroProof {
                yr: terminal,
                opened_rows: queries.iter().map(|&q| wtns_prev.row(q)).collect(),
                merkle_proof: wtns_prev.tree.prove(&queries),
            });
            proof.sumcheck_transcript = Some(SumcheckTranscript {
                transcript: sumcheck_transcript,
            });
            let mut proof = finalize(proof).map_err(DriverError::Native)?;
            proof.eval_rounds = eval_rounds;
            return Ok(ProvingOutput {
                proof,
                reusable_input: v,
            });
        }

        let wtns_next = ligero_commit(
            &v,
            config.dims[i + 1].0,
            config.dims[i + 1].1,
            &config.reed_solomon_codes[i + 1],
        );
        let cm_next = RecursiveLigeroCommitment {
            root: wtns_next.tree.get_root(),
        };
        proof.recursive_commitments.push(cm_next.clone());
        fs.observe_root(&cm_next.root)
            .map_err(DriverError::Transcript)?;

        let queries = sample_queries(
            fs,
            QueryRequest::new(rows, config.num_queries, max_query_draws)
                .map_err(DriverError::Query)?,
        )?;
        let alpha = fs.sample_challenge().map_err(DriverError::Transcript)?;
        let opened_rows: Vec<Vec<U>> = queries.iter().map(|&q| wtns_prev.row(q)).collect();
        let sks_vks: Vec<U> = eval_sk_at_vks(config.dims[i].0);
        let (b, s) = induce_sumcheck_poly_auto(
            vcfg.log_dims[i],
            &sks_vks,
            &opened_rows,
            &rs,
            &queries,
            alpha,
        );
        fs.observe_challenge(s).map_err(DriverError::Transcript)?;
        proof.recursive_proofs.push(RecursiveLigeroProof {
            opened_rows,
            merkle_proof: wtns_prev.tree.prove(&queries),
        });

        let beta = fs.sample_challenge().map_err(DriverError::Transcript)?;
        basis = basis
            .iter()
            .zip(&b)
            .map(|(x, y)| x.add(&beta.mul(y)))
            .collect();
        wtns_prev = wtns_next;
    }

    Err(DriverError::Native(crate::LigeritoError::InvalidProof))
}

/// Prove with a caller-supplied transcript (no evaluation claims).
pub fn prove_with_transcript<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
    mut fs: impl Transcript,
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    prove_inner(config, poly, &[], &mut fs)
}

/// Prove the commitment together with evaluation claims `P(z_k) = v_k`.
///
/// The claims are bound to the committed polynomial: verify with
/// `verify_with_evaluations` using the same claims.
pub fn prove_with_evaluations<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
    claims: &[EvalClaim<T>],
    mut fs: impl Transcript,
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    prove_inner(config, poly, claims, &mut fs)
}

/// Main prover function using default Merlin transcript
pub fn prove<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    #[cfg(feature = "transcript-merlin")]
    let fs = FiatShamir::new_merlin();

    #[cfg(not(feature = "transcript-merlin"))]
    let fs = FiatShamir::new_sha256(0);

    prove_with_transcript(config, poly, fs)
}

/// Prover using the SHA-256 transcript (verify with `verify_sha256`)
pub fn prove_sha256<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    let fs = FiatShamir::new_sha256(1234);
    prove_with_transcript(config, poly, fs)
}

/// Prover function using BLAKE2b transcript (for Substrate runtime compatibility)
///
/// Proofs generated with this function must be verified using `verify_blake2b()`
/// or `verify_complete_blake2b()`.
#[cfg(feature = "transcript-blake2b")]
pub fn prove_blake2b<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    let fs = FiatShamir::new_blake2b();
    prove_with_transcript(config, poly, fs)
}

/// Same as `prove` (kept for API compatibility).
pub fn prove_debug<T, U>(
    config: &ProverConfig<T, U>,
    poly: &[T],
) -> crate::Result<FinalizedLigeritoProof<T, U>>
where
    T: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
    U: BinaryFieldElement + Send + Sync + From<T> + bytemuck::Pod + 'static,
{
    prove(config, poly)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::hardcoded_config_12;
    use ligerito_binary_fields::{BinaryElem128, BinaryElem32};
    use std::marker::PhantomData;

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
    fn test_product_sumcheck_round() -> Result<(), Box<dyn std::error::Error>> {
        use crate::sumcheck_polys::{eval_round_poly, inner_product};
        type U = BinaryElem128;
        let mut f: Vec<U> = (0..8).map(|i| U::from_bits(3 * i + 1)).collect();
        let mut g: Vec<U> = (0..8).map(|i| U::from_bits(5 * i + 2)).collect();
        let claim = inner_product(&f, &g);
        let mut fs = FiatShamir::new_sha256(0);
        let ((s0, s1, s2), r) = product_sumcheck_round(
            &mut f,
            &mut g,
            &mut LegacyTranscript::<U, U, _>::new(&mut fs),
        )?;
        assert_eq!(s0.add(&s1), claim);
        assert_eq!(eval_round_poly((s0, s1, s2), r), inner_product(&f, &g));
        Ok(())
    }

    #[test]
    fn test_simple_prove() {
        let config = hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);

        // Test with all ones polynomial
        let poly = vec![BinaryElem32::one(); 1 << 12];

        // This should not panic
        let proof = prove(&config, &poly);
        assert!(proof.is_ok(), "Simple proof generation should succeed");
    }

    #[test]
    fn test_sumcheck_consistency_in_prover() {
        let config = hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);

        let poly = vec![BinaryElem32::zero(); 1 << 12];
        let proof = prove(&config, &poly);
        assert!(proof.is_ok(), "Zero polynomial proof should succeed");

        let mut poly = vec![BinaryElem32::zero(); 1 << 12];
        poly[0] = BinaryElem32::one();
        poly[1] = BinaryElem32::from(2);
        let proof = prove(&config, &poly);
        assert!(proof.is_ok(), "Simple pattern proof should succeed");
    }

    #[test]
    fn test_prove_with_evaluations() {
        use crate::eval_proof::EvalClaim;

        let config = hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = crate::hardcoded_config_12_verifier();

        // polynomial with known values
        let mut poly = vec![BinaryElem32::zero(); 1 << 12];
        poly[0] = BinaryElem32::from(42);
        poly[7] = BinaryElem32::from(99);
        poly[100] = BinaryElem32::from(255);

        // claim: P(0) = 42, P(7) = 99, P(100) = 255
        let claims = vec![
            EvalClaim {
                index: 0,
                value: BinaryElem32::from(42),
            },
            EvalClaim {
                index: 7,
                value: BinaryElem32::from(99),
            },
            EvalClaim {
                index: 100,
                value: BinaryElem32::from(255),
            },
        ];

        // prove
        let fs = FiatShamir::new_sha256(0);
        let proof = prove_with_evaluations(&config, &poly, &claims, fs);
        assert!(proof.is_ok(), "prove_with_evaluations should succeed");
        let proof = proof.unwrap();
        assert_eq!(
            proof.eval_rounds.len(),
            verifier_config.initial_k,
            "eval sumcheck runs over the initial_k column variables"
        );

        // verify
        let fs = FiatShamir::new_sha256(0);
        let result = crate::verifier::verify_with_evaluations::<BinaryElem32, BinaryElem128>(
            &verifier_config,
            &proof,
            &claims,
            fs,
        );
        assert!(result.unwrap(), "honest evaluation proof should verify");
    }

    #[test]
    fn test_prove_with_evaluations_wrong_claim_fails() {
        use crate::eval_proof::EvalClaim;

        let config = hardcoded_config_12(PhantomData::<BinaryElem32>, PhantomData::<BinaryElem128>);
        let verifier_config = crate::hardcoded_config_12_verifier();

        let mut poly = vec![BinaryElem32::zero(); 1 << 12];
        poly[5] = BinaryElem32::from(77);

        // wrong claim: P(5) = 88 (actual is 77)
        let claims = vec![EvalClaim {
            index: 5,
            value: BinaryElem32::from(88),
        }];

        // prove (prover uses actual polynomial, eval sumcheck computes from actual values)
        let fs = FiatShamir::new_sha256(0);
        let proof = prove_with_evaluations(&config, &poly, &claims, fs).unwrap();

        // verify with wrong claim — eval sumcheck should fail
        let fs = FiatShamir::new_sha256(0);
        let result = crate::verifier::verify_with_evaluations::<BinaryElem32, BinaryElem128>(
            &verifier_config,
            &proof,
            &claims,
            fs,
        )
        .unwrap();

        // target (α·88) ≠ actual sum (α·77)
        assert!(!result, "wrong eval claim should fail verification");
    }
}
