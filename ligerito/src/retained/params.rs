use std::mem::size_of;

use binary_fields::BinaryElem128;

use crate::eval_proof::EvalClaim;
use crate::{FinalizedLigeritoProof, VerifierConfig};

use super::error::Error;

const QUERIES: usize = 160;
const MAX_LOG_SIZE: usize = 28;

/// Caller-owned bounds for one retained native operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    pub max_allocation_bytes: usize,
    pub max_work: usize,
    pub max_retained_bytes: usize,
    pub max_claims: usize,
    pub max_query_draws: usize,
}

impl Limits {
    fn check_allocation(&self, requested: usize) -> Result<(), Error> {
        if requested > self.max_allocation_bytes {
            return Err(Error::AllocationLimit {
                requested,
                limit: self.max_allocation_bytes,
            });
        }
        Ok(())
    }

    fn check_work(&self, requested: usize) -> Result<(), Error> {
        if requested > self.max_work {
            return Err(Error::WorkLimit {
                requested,
                limit: self.max_work,
            });
        }
        Ok(())
    }

    fn check_claim_count(&self, requested: usize) -> Result<(), Error> {
        if requested > self.max_claims {
            return Err(Error::ClaimLimit {
                requested,
                limit: self.max_claims,
            });
        }
        Ok(())
    }

    fn check_draws(&self) -> Result<(), Error> {
        if self.max_query_draws < QUERIES {
            return Err(Error::QueryDrawLimit {
                requested: QUERIES,
                limit: self.max_query_draws,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Level {
    message_rows: usize,
    columns: usize,
    encoded_rows: usize,
    depth: usize,
}

impl Level {
    fn new(log_rows: usize, log_columns: usize, level: usize) -> Result<Self, Error> {
        if log_columns == 0 || log_rows == 0 {
            return Err(Error::Geometry { level });
        }
        let depth = add(log_rows, 2)?;
        let message_rows = power_of_two(log_rows)?;
        let columns = power_of_two(log_columns)?;
        let encoded_rows = power_of_two(depth)?;
        u32::try_from(columns)?;
        if encoded_rows < QUERIES {
            return Err(Error::Geometry { level });
        }
        Ok(Self {
            message_rows,
            columns,
            encoded_rows,
            depth,
        })
    }

    #[cfg(feature = "prover")]
    fn encoded_elements(self) -> Result<usize, Error> {
        multiply(self.encoded_rows, self.columns)
    }

    #[cfg(feature = "prover")]
    fn tree_bytes(self) -> Result<usize, Error> {
        let nodes = multiply(multiply(self.encoded_rows, 2)?, 32)?;
        let layers = multiply(add(self.depth, 1)?, size_of::<Vec<[u8; 32]>>())?;
        add(nodes, layers)
    }

    #[cfg(feature = "prover")]
    fn encoder_bytes(self) -> Result<usize, Error> {
        multiply(add(multiply(self.encoded_rows, 3)?, self.message_rows)?, 16)
    }

    #[cfg(feature = "prover")]
    fn commitment_bytes(self) -> Result<usize, Error> {
        let encoded = multiply(self.encoded_elements()?, 16)?;
        let hashes = multiply(self.encoded_rows, 32)?;
        let tile_rows = if cfg!(feature = "parallel") {
            self.encoded_rows
        } else {
            self.encoded_rows.min(256)
        };
        let tile = multiply(multiply(tile_rows, self.columns)?, 16)?;
        add(add(encoded, self.tree_bytes()?)?, add(hashes, tile)?)
    }

    #[cfg(feature = "prover")]
    fn row_bytes(self) -> Result<usize, Error> {
        let fields = multiply(multiply(QUERIES, self.columns)?, 16)?;
        // The native sibling vector grows by pushing, so charge spare capacity.
        let paths = multiply(multiply(QUERIES, self.depth)?, 64)?;
        let metadata = multiply(QUERIES, size_of::<Vec<BinaryElem128>>())?;
        add(add(fields, paths)?, metadata)
    }

    fn verifier_bytes(self) -> Result<usize, Error> {
        let generators = multiply(multiply(QUERIES, self.depth)?, 16)?;
        let temporary = multiply(QUERIES, add(160, size_of::<Vec<BinaryElem128>>())?)?;
        let concurrent_rows = if cfg!(feature = "parallel") {
            QUERIES
        } else {
            1
        };
        let row_fold = multiply(multiply(self.columns, concurrent_rows)?, 16)?;
        add(add(generators, temporary)?, row_fold)
    }

    #[cfg(feature = "prover")]
    fn parallel_prover_bytes(self) -> Result<usize, Error> {
        if !cfg!(feature = "parallel") {
            return Ok(0);
        }
        // At every possible split, 2^split + 2^(log_rows-split) <= rows + 1.
        let tensors = add(add(self.message_rows, 1)?, multiply(self.depth, 2)?)?;
        let fields = multiply(multiply(QUERIES, add(tensors, self.columns)?)?, 16)?;
        let headers = multiply(multiply(QUERIES, 4)?, size_of::<Vec<BinaryElem128>>())?;
        add(fields, headers)
    }

    #[cfg(feature = "prover")]
    fn prover_work(self) -> Result<usize, Error> {
        let encoding = multiply(self.encoded_elements()?, add(self.depth, 1)?)?;
        let hashing = add(self.encoded_elements()?, multiply(self.encoded_rows, 4)?)?;
        let row_batch = multiply(QUERIES, add(self.message_rows, self.columns)?)?;
        multiply(add(add(encoding, hashing)?, row_batch)?, 16)
    }
}

/// Checked recursive native geometry; this type grants no application profile approval.
///
/// Construction returns `Error` for unsupported layouts or sizes, invalid geometry
/// or query counts, arithmetic overflow, exceeded caller limits or allocation failure.
#[derive(Debug)]
pub struct Parameters {
    config: VerifierConfig,
    levels: Vec<Level>,
    message_len: usize,
    rounds: usize,
    metadata_bytes: usize,
}

impl Parameters {
    #[expect(
        clippy::missing_errors_doc,
        reason = "Parameters documents construction errors."
    )]
    #[tracing::instrument(skip_all)]
    pub fn new(config: VerifierConfig, limits: &Limits) -> Result<Self, Error> {
        validate_header(&config, limits)?;
        let log_size = add(config.initial_dim, config.initial_k)?;
        if !(9..=MAX_LOG_SIZE).contains(&log_size) {
            return Err(Error::UnsupportedSize { log_size });
        }
        let metadata_bytes = metadata_bytes(&config)?;
        limits.check_allocation(metadata_bytes)?;
        check_retained(metadata_bytes, limits)?;
        let mut levels = Vec::new();
        levels.try_reserve_exact(add(config.recursive_steps, 1)?)?;
        levels.push(Level::new(config.initial_dim, config.initial_k, 0)?);
        let mut previous = config.initial_dim;
        let mut rounds = 0;
        for (index, (&log_rows, &folds)) in config.log_dims.iter().zip(&config.ks).enumerate() {
            if add(log_rows, folds)? != previous {
                return Err(Error::Geometry { level: index + 1 });
            }
            levels.push(Level::new(log_rows, folds, index + 1)?);
            rounds = add(rounds, folds)?;
            previous = log_rows;
        }
        if previous > 8 {
            return Err(Error::Geometry {
                level: config.recursive_steps,
            });
        }
        let result = Self {
            config,
            levels,
            message_len: power_of_two(log_size)?,
            rounds,
            metadata_bytes,
        };
        result.check_verifier_limits(0, limits)?;
        Ok(result)
    }

    #[must_use]
    pub fn message_len(&self) -> usize {
        self.message_len
    }

    #[must_use]
    pub fn same_geometry(&self, other: &Self) -> bool {
        self.levels == other.levels
            && self.config.num_queries == other.config.num_queries
            && self.config.initial_k == other.config.initial_k
    }

    pub(crate) fn config(&self) -> &VerifierConfig {
        &self.config
    }

    pub(crate) fn check_claims(
        &self,
        claims: &[EvalClaim<BinaryElem128>],
        limits: &Limits,
    ) -> Result<(), Error> {
        limits.check_claim_count(claims.len())?;
        limits.check_work(claims.len())?;
        for claim in claims {
            if claim.index >= self.message_len {
                return Err(Error::ClaimIndex {
                    index: claim.index,
                    length: self.message_len,
                });
            }
        }
        Ok(())
    }

    pub(crate) fn check_proof(
        &self,
        proof: &FinalizedLigeritoProof<BinaryElem128, BinaryElem128>,
        claim_count: usize,
        limits: &Limits,
    ) -> Result<(), Error> {
        self.check_verifier_limits(claim_count, limits)?;
        self.check_proof_header(proof, claim_count)?;
        let initial = &proof.initial_ligero_proof;
        let mut bytes = check_rows(
            &initial.opened_rows,
            &initial.merkle_proof.siblings,
            self.levels[0],
            0,
        )?;
        for (index, row_proof) in proof.recursive_proofs.iter().enumerate() {
            bytes = add(
                bytes,
                check_rows(
                    &row_proof.opened_rows,
                    &row_proof.merkle_proof.siblings,
                    self.levels[index + 1],
                    index + 1,
                )?,
            )?;
        }
        let final_level = self.config.recursive_steps;
        let final_proof = &proof.final_ligero_proof;
        bytes = add(
            bytes,
            check_rows(
                &final_proof.opened_rows,
                &final_proof.merkle_proof.siblings,
                self.levels[final_level],
                final_level,
            )?,
        )?;
        bytes = add(bytes, proof_metadata_bytes(proof)?)?;
        limits.check_allocation(add(bytes, self.verifier_bytes(claim_count)?)?)
    }

    pub(crate) fn check_retained(&self, bytes: usize, limits: &Limits) -> Result<(), Error> {
        check_retained(add(self.metadata_bytes, bytes)?, limits)
    }

    #[cfg(feature = "prover")]
    pub(crate) fn check_prover_limits(
        &self,
        input_capacity: usize,
        claim_count: usize,
        limits: &Limits,
    ) -> Result<(), Error> {
        self.check_prover_limits_with_retained(input_capacity, claim_count, 0, limits)
    }

    /// `actual_retained` includes the original state and each distinct parameter owner.
    #[cfg(feature = "prover")]
    pub(crate) fn check_prover_limits_with_retained(
        &self,
        input_capacity: usize,
        claim_count: usize,
        actual_retained: usize,
        limits: &Limits,
    ) -> Result<(), Error> {
        limits.check_claim_count(claim_count)?;
        limits.check_draws()?;
        if input_capacity < self.message_len {
            return Err(Error::Length {
                expected: self.message_len,
                actual: input_capacity,
            });
        }
        let input_bytes = multiply(input_capacity, 16)?;
        let mut bytes = add(input_bytes, multiply(self.message_len, 64)?)?;
        bytes = add(bytes, multiply(claim_count, 128)?)?;
        bytes = add(
            bytes,
            multiply(add(self.rounds, self.config.initial_k)?, 96)?,
        )?;
        let mut work = multiply(
            multiply(self.message_len, add(claim_count, self.rounds)?)?,
            16,
        )?;
        for level in &self.levels {
            bytes = add(
                bytes,
                add(level.commitment_bytes()?, level.encoder_bytes()?)?,
            )?;
            bytes = add(bytes, level.row_bytes()?)?;
            bytes = add(bytes, level.parallel_prover_bytes()?)?;
            work = add(work, level.prover_work()?)?;
        }
        let sampling = multiply(
            multiply(limits.max_query_draws, QUERIES)?,
            self.levels.len(),
        )?;
        let state = add(
            input_bytes,
            add(
                multiply(self.levels[0].encoded_elements()?, 16)?,
                self.levels[0].tree_bytes()?,
            )?,
        )?;
        let estimated_retained = add(
            add(state, self.encoder_retained_bytes()?)?,
            self.metadata_bytes,
        )?;
        let retained = estimated_retained.max(actual_retained);
        let extra_retained = retained - estimated_retained;
        bytes = add(
            bytes,
            add(
                self.metadata_bytes,
                encoder_header_bytes(self.config.recursive_steps)?,
            )?,
        )?;
        limits.check_work(add(work, sampling)?)?;
        limits.check_allocation(add(bytes, extra_retained)?)?;
        check_retained(retained, limits)
    }

    fn check_verifier_limits(&self, claim_count: usize, limits: &Limits) -> Result<(), Error> {
        limits.check_claim_count(claim_count)?;
        limits.check_draws()?;
        limits.check_allocation(self.verifier_bytes(claim_count)?)?;
        let query_work = multiply(
            multiply(limits.max_query_draws, QUERIES)?,
            self.levels.len(),
        )?;
        let rounds = add(self.rounds, self.config.initial_k)?;
        let terminal = self.levels[self.config.recursive_steps].message_rows;
        let basis = multiply(
            multiply(QUERIES, self.levels.len())?,
            add(terminal, rounds)?,
        )?;
        let claims = multiply(claim_count, add(terminal, rounds)?)?;
        let mut work = multiply(add(basis, claims)?, 32)?;
        for level in &self.levels {
            work = add(work, multiply(multiply(QUERIES, level.columns)?, 16)?)?;
        }
        limits.check_work(add(work, query_work)?)?;
        self.check_retained(0, limits)
    }

    fn verifier_bytes(&self, claim_count: usize) -> Result<usize, Error> {
        // Sparse claim vectors grow by pushing; include their spare capacity.
        let mut bytes = add(self.metadata_bytes, multiply(claim_count, 128)?)?;
        bytes = add(bytes, 256)?;
        bytes = add(
            bytes,
            multiply(add(self.rounds, self.config.initial_k)?, 96)?,
        )?;
        for level in &self.levels {
            bytes = add(bytes, level.verifier_bytes()?)?;
        }
        let concurrent_terms = if cfg!(feature = "parallel") {
            multiply(QUERIES, self.levels.len())?
        } else {
            1
        };
        let terminal = self.levels[self.config.recursive_steps].message_rows;
        add(bytes, multiply(multiply(terminal, concurrent_terms)?, 64)?)
    }

    fn check_proof_header(
        &self,
        proof: &FinalizedLigeritoProof<BinaryElem128, BinaryElem128>,
        claim_count: usize,
    ) -> Result<(), Error> {
        let steps = self.config.recursive_steps;
        check_length(proof.recursive_commitments.len(), steps)?;
        check_length(proof.recursive_proofs.len(), steps - 1)?;
        check_length(proof.sumcheck_transcript.transcript.len(), self.rounds)?;
        let eval_rounds = if claim_count == 0 {
            0
        } else {
            self.config.initial_k
        };
        check_length(proof.eval_rounds.len(), eval_rounds)?;
        check_length(
            proof.final_ligero_proof.yr.len(),
            self.levels[steps].message_rows,
        )?;
        if proof.initial_ligero_cm.root.root.is_none() {
            return Err(Error::MissingRoot { level: 0 });
        }
        for (index, commitment) in proof.recursive_commitments.iter().enumerate() {
            if commitment.root.root.is_none() {
                return Err(Error::MissingRoot { level: index + 1 });
            }
        }
        Ok(())
    }

    #[cfg(feature = "prover")]
    fn encoder_retained_bytes(&self) -> Result<usize, Error> {
        let mut bytes = 0;
        for level in &self.levels {
            bytes = add(
                bytes,
                multiply(add(level.encoded_rows, level.message_rows)?, 16)?,
            )?;
        }
        add(bytes, encoder_header_bytes(self.config.recursive_steps)?)
    }

    #[cfg(feature = "prover")]
    fn check_encoder_limits(&self, limits: &Limits) -> Result<(), Error> {
        let mut bytes = self.metadata_bytes;
        let mut work = 0;
        for level in &self.levels {
            bytes = add(bytes, level.encoder_bytes()?)?;
            work = add(
                work,
                multiply(multiply(level.encoded_rows, level.depth)?, 16)?,
            )?;
        }
        bytes = add(bytes, encoder_header_bytes(self.config.recursive_steps)?)?;
        limits.check_allocation(bytes)?;
        limits.check_work(work)?;
        self.check_retained(self.encoder_retained_bytes()?, limits)
    }
}

/// Private encoder precomputations tied to checked native geometry.
///
/// Construction returns `Error` for arithmetic overflow, exceeded allocation, work
/// or retained-capacity limits, or failed explicit buffer reservation.
#[cfg(feature = "prover")]
pub struct ProverParameters {
    parameters: Parameters,
    config: crate::ProverConfig<BinaryElem128, BinaryElem128>,
    retained_bytes: usize,
}

#[cfg(feature = "prover")]
impl ProverParameters {
    #[expect(
        clippy::missing_errors_doc,
        reason = "ProverParameters documents construction errors."
    )]
    #[tracing::instrument(skip_all)]
    pub fn new(parameters: Parameters, limits: &Limits) -> Result<Self, Error> {
        parameters.check_encoder_limits(limits)?;
        let initial = parameters.levels[0];
        let mut dims = Vec::new();
        let mut ks = Vec::new();
        let mut codes = Vec::new();
        let steps = parameters.config.recursive_steps;
        dims.try_reserve_exact(steps)?;
        ks.try_reserve_exact(steps)?;
        codes.try_reserve_exact(steps)?;
        for (level, &folds) in parameters.levels[1..].iter().zip(&parameters.config.ks) {
            dims.push((level.message_rows, level.columns));
            ks.push(folds);
            codes.push(reed_solomon::ReedSolomon::new(
                level.message_rows,
                level.encoded_rows,
            ));
        }
        let config = crate::ProverConfig {
            recursive_steps: steps,
            initial_dims: (initial.message_rows, initial.columns),
            dims,
            initial_k: parameters.config.initial_k,
            ks,
            initial_reed_solomon: reed_solomon::ReedSolomon::new(
                initial.message_rows,
                initial.encoded_rows,
            ),
            reed_solomon_codes: codes,
            num_queries: QUERIES,
        };
        let retained_bytes = prover_parameter_bytes(&config)?;
        parameters.check_retained(retained_bytes, limits)?;
        Ok(Self {
            parameters,
            config,
            retained_bytes,
        })
    }

    #[must_use]
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    pub(crate) fn config(&self) -> &crate::ProverConfig<BinaryElem128, BinaryElem128> {
        &self.config
    }

    #[cfg(test)]
    fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(crate) fn total_retained_bytes(&self) -> Result<usize, Error> {
        add(self.parameters.metadata_bytes, self.retained_bytes)
    }

    pub(crate) fn check_limits(&self, limits: &Limits) -> Result<(), Error> {
        self.parameters.check_retained(self.retained_bytes, limits)
    }
}

fn validate_header(config: &VerifierConfig, limits: &Limits) -> Result<(), Error> {
    if !cfg!(target_endian = "little") || size_of::<BinaryElem128>() != 16 {
        return Err(Error::UnsupportedLayout);
    }
    if config.recursive_steps == 0 || config.recursive_steps > MAX_LOG_SIZE {
        return Err(Error::Geometry { level: 0 });
    }
    check_length(config.log_dims.len(), config.recursive_steps)?;
    check_length(config.ks.len(), config.recursive_steps)?;
    if config.num_queries != QUERIES {
        return Err(Error::QueryCount {
            expected: QUERIES,
            actual: config.num_queries,
        });
    }
    limits.check_work(config.recursive_steps)?;
    limits.check_draws()
}

fn metadata_bytes(config: &VerifierConfig) -> Result<usize, Error> {
    let capacities = add(config.log_dims.capacity(), config.ks.capacity())?;
    let levels = multiply(add(config.recursive_steps, 1)?, size_of::<Level>())?;
    add(
        add(multiply(capacities, size_of::<usize>())?, levels)?,
        size_of::<Parameters>(),
    )
}

#[cfg(feature = "prover")]
fn encoder_header_bytes(steps: usize) -> Result<usize, Error> {
    // Each private code stores two vector headers and two logarithms.
    let code = add(
        multiply(2, size_of::<Vec<BinaryElem128>>())?,
        multiply(2, size_of::<usize>())?,
    )?;
    let geometry = add(size_of::<(usize, usize)>(), size_of::<usize>())?;
    multiply(steps, add(code, geometry)?)
}

fn check_rows(
    rows: &[Vec<BinaryElem128>],
    siblings: &[[u8; 32]],
    geometry: Level,
    level: usize,
) -> Result<usize, Error> {
    if rows.len() != QUERIES || siblings.len() > multiply(QUERIES, geometry.depth)? {
        return Err(Error::ProofShape { level });
    }
    let mut bytes = 0;
    for row in rows {
        if row.len() != geometry.columns {
            return Err(Error::ProofShape { level });
        }
        bytes = add(bytes, multiply(row.capacity(), 16)?)?;
    }
    Ok(bytes)
}

fn proof_metadata_bytes(
    proof: &FinalizedLigeritoProof<BinaryElem128, BinaryElem128>,
) -> Result<usize, Error> {
    let mut bytes = multiply(
        proof.recursive_commitments.capacity(),
        size_of::<crate::RecursiveLigeroCommitment>(),
    )?;
    bytes = add(
        bytes,
        multiply(
            proof.recursive_proofs.capacity(),
            size_of::<crate::RecursiveLigeroProof<BinaryElem128>>(),
        )?,
    )?;
    bytes = add(
        bytes,
        multiply(proof.sumcheck_transcript.transcript.capacity(), 48)?,
    )?;
    bytes = add(bytes, multiply(proof.eval_rounds.capacity(), 48)?)?;
    bytes = add(bytes, multiply(proof.final_ligero_proof.yr.capacity(), 16)?)?;
    let first = &proof.initial_ligero_proof;
    let last = &proof.final_ligero_proof;
    bytes = add(
        bytes,
        row_metadata_bytes(
            first.opened_rows.capacity(),
            first.merkle_proof.siblings.capacity(),
        )?,
    )?;
    bytes = add(
        bytes,
        row_metadata_bytes(
            last.opened_rows.capacity(),
            last.merkle_proof.siblings.capacity(),
        )?,
    )?;
    for row_proof in &proof.recursive_proofs {
        bytes = add(
            bytes,
            row_metadata_bytes(
                row_proof.opened_rows.capacity(),
                row_proof.merkle_proof.siblings.capacity(),
            )?,
        )?;
    }
    Ok(bytes)
}

fn row_metadata_bytes(rows_capacity: usize, siblings_capacity: usize) -> Result<usize, Error> {
    add(
        multiply(rows_capacity, size_of::<Vec<BinaryElem128>>())?,
        multiply(siblings_capacity, 32)?,
    )
}

#[cfg(feature = "prover")]
fn prover_parameter_bytes(
    config: &crate::ProverConfig<BinaryElem128, BinaryElem128>,
) -> Result<usize, Error> {
    let mut elements = add(
        config.initial_reed_solomon.twiddles.capacity(),
        config.initial_reed_solomon.pis.capacity(),
    )?;
    for rs in &config.reed_solomon_codes {
        elements = add(elements, add(rs.twiddles.capacity(), rs.pis.capacity())?)?;
    }
    let mut bytes = multiply(elements, 16)?;
    bytes = add(
        bytes,
        multiply(config.dims.capacity(), size_of::<(usize, usize)>())?,
    )?;
    bytes = add(bytes, multiply(config.ks.capacity(), size_of::<usize>())?)?;
    add(
        bytes,
        multiply(
            config.reed_solomon_codes.capacity(),
            size_of::<reed_solomon::ReedSolomon<BinaryElem128>>(),
        )?,
    )
}

fn check_retained(requested: usize, limits: &Limits) -> Result<(), Error> {
    if requested > limits.max_retained_bytes {
        return Err(Error::RetainedLimit {
            requested,
            limit: limits.max_retained_bytes,
        });
    }
    Ok(())
}

fn check_length(actual: usize, expected: usize) -> Result<(), Error> {
    if actual != expected {
        return Err(Error::Length { expected, actual });
    }
    Ok(())
}

fn power_of_two(log_size: usize) -> Result<usize, Error> {
    let shift = u32::try_from(log_size)?;
    1usize.checked_shl(shift).ok_or(Error::Overflow)
}

fn add(left: usize, right: usize) -> Result<usize, Error> {
    left.checked_add(right).ok_or(Error::Overflow)
}

fn multiply(left: usize, right: usize) -> Result<usize, Error> {
    left.checked_mul(right).ok_or(Error::Overflow)
}

#[cfg(test)]
mod tests {
    use binary_fields::BinaryFieldElement;
    use merkle_tree::{BatchedMerkleProof, MerkleRoot};

    use super::*;
    use crate::{
        eval_proof::EvalSumcheckRound, FinalLigeroProof, RecursiveLigeroCommitment,
        RecursiveLigeroProof, SumcheckTranscript,
    };

    fn limits() -> Limits {
        Limits {
            max_allocation_bytes: 64 * 1024 * 1024,
            max_work: 100_000_000,
            max_retained_bytes: 64 * 1024 * 1024,
            max_claims: 16,
            max_query_draws: 4096,
        }
    }

    fn config() -> VerifierConfig {
        VerifierConfig {
            recursive_steps: 1,
            initial_dim: 8,
            log_dims: vec![7],
            initial_k: 1,
            ks: vec![1],
            num_queries: QUERIES,
        }
    }

    fn shape_proof() -> FinalizedLigeritoProof<BinaryElem128, BinaryElem128> {
        let zero = BinaryElem128::zero();
        FinalizedLigeritoProof {
            initial_ligero_cm: RecursiveLigeroCommitment {
                root: MerkleRoot {
                    root: Some([0; 32]),
                },
            },
            initial_ligero_proof: RecursiveLigeroProof {
                opened_rows: vec![vec![zero; 2]; QUERIES],
                merkle_proof: BatchedMerkleProof { siblings: vec![] },
            },
            recursive_commitments: vec![RecursiveLigeroCommitment {
                root: MerkleRoot {
                    root: Some([0; 32]),
                },
            }],
            recursive_proofs: vec![],
            final_ligero_proof: FinalLigeroProof {
                yr: vec![zero; 128],
                opened_rows: vec![vec![zero; 2]; QUERIES],
                merkle_proof: BatchedMerkleProof { siblings: vec![] },
            },
            sumcheck_transcript: SumcheckTranscript {
                transcript: vec![(zero, zero, zero)],
            },
            eval_rounds: vec![EvalSumcheckRound {
                s0: zero,
                s1: zero,
                s2: zero,
            }],
        }
    }

    #[test]
    fn b1_parameters_preserve_complete_geometry() -> Result<(), Error> {
        let a = Parameters::new(config(), &limits())?;
        let b = Parameters::new(config(), &limits())?;
        assert_eq!(a.message_len(), 512);
        assert!(a.same_geometry(&b));
        assert_eq!(a.config().num_queries, 160);
        let mut changed = config();
        changed.initial_dim = 9;
        changed.log_dims[0] = 8;
        let c = Parameters::new(changed, &limits())?;
        assert!(!a.same_geometry(&c));
        Ok(())
    }

    #[test]
    fn b1_parameters_reject_empty_and_mismatched_lengths() -> Result<(), Error> {
        Parameters::new(config(), &limits())?;
        let mut bad = config();
        bad.recursive_steps = 0;
        assert!(matches!(
            Parameters::new(bad, &limits()),
            Err(Error::Geometry { level: 0 })
        ));
        let mut bad = config();
        bad.ks.clear();
        assert!(matches!(
            Parameters::new(bad, &limits()),
            Err(Error::Length {
                expected: 1,
                actual: 0
            })
        ));
        Ok(())
    }

    #[test]
    fn b1_parameters_reject_bad_folds_and_domains() -> Result<(), Error> {
        Parameters::new(config(), &limits())?;
        for folds in [0, 2] {
            let mut bad = config();
            bad.ks[0] = folds;
            assert!(matches!(
                Parameters::new(bad, &limits()),
                Err(Error::Geometry { level: 1 })
            ));
        }
        let mut bad = config();
        bad.log_dims[0] = 5;
        bad.ks[0] = 3;
        assert!(matches!(
            Parameters::new(bad, &limits()),
            Err(Error::Geometry { level: 1 })
        ));
        let mut bad = config();
        bad.initial_dim = 10;
        bad.log_dims[0] = 9;
        assert!(matches!(
            Parameters::new(bad, &limits()),
            Err(Error::Geometry { level: 1 })
        ));
        Ok(())
    }

    #[test]
    fn b1_parameters_reject_query_changes() -> Result<(), Error> {
        Parameters::new(config(), &limits())?;
        for actual in [0, 148, 159, 161, usize::MAX] {
            let mut bad = config();
            bad.num_queries = actual;
            assert!(
                matches!(Parameters::new(bad, &limits()), Err(Error::QueryCount { expected: 160, actual: found }) if found == actual)
            );
        }
        Ok(())
    }

    #[test]
    fn b1_parameters_reject_sizes_and_overflow() -> Result<(), Error> {
        Parameters::new(config(), &limits())?;
        for (initial_dim, expected) in [(7, 8), (28, 29)] {
            let mut bad = config();
            bad.initial_dim = initial_dim;
            assert!(
                matches!(Parameters::new(bad, &limits()), Err(Error::UnsupportedSize { log_size }) if log_size == expected)
            );
        }
        let mut bad = config();
        bad.initial_dim = usize::MAX;
        assert!(matches!(
            Parameters::new(bad, &limits()),
            Err(Error::Overflow)
        ));
        assert!(matches!(power_of_two(usize::MAX), Err(Error::Overflow)));
        assert!(matches!(
            power_of_two(usize::BITS as usize),
            Err(Error::Overflow)
        ));
        assert!(matches!(multiply(usize::MAX, 2), Err(Error::Overflow)));
        Ok(())
    }

    #[test]
    fn b1_parameters_bound_metadata_and_work() -> Result<(), Error> {
        let accepted = Parameters::new(config(), &limits())?;
        let mut cap = limits();
        cap.max_allocation_bytes = 0;
        assert!(matches!(
            Parameters::new(config(), &cap),
            Err(Error::AllocationLimit { .. })
        ));
        cap = limits();
        cap.max_work = 0;
        assert!(matches!(
            Parameters::new(config(), &cap),
            Err(Error::WorkLimit { .. })
        ));
        cap = limits();
        cap.max_query_draws = 159;
        assert!(matches!(
            Parameters::new(config(), &cap),
            Err(Error::QueryDrawLimit {
                requested: 160,
                limit: 159
            })
        ));
        cap = limits();
        cap.max_retained_bytes = accepted.metadata_bytes;
        accepted.check_retained(0, &cap)?;
        cap.max_retained_bytes -= 1;
        assert!(matches!(
            accepted.check_retained(0, &cap),
            Err(Error::RetainedLimit { .. })
        ));
        Ok(())
    }

    #[test]
    fn b1_claim_checks_use_current_limits() -> Result<(), Error> {
        let p = Parameters::new(config(), &limits())?;
        let mut claims = vec![EvalClaim {
            index: 511,
            value: BinaryElem128::zero(),
        }];
        p.check_claims(&claims, &limits())?;
        let mut cap = limits();
        cap.max_claims = 0;
        assert!(matches!(
            p.check_claims(&claims, &cap),
            Err(Error::ClaimLimit {
                requested: 1,
                limit: 0
            })
        ));
        claims[0].index = 512;
        assert!(matches!(
            p.check_claims(&claims, &limits()),
            Err(Error::ClaimIndex {
                index: 512,
                length: 512
            })
        ));
        Ok(())
    }

    #[test]
    fn b1_proof_preflight_checks_counts_and_roots() -> Result<(), Error> {
        let p = Parameters::new(config(), &limits())?;
        let mut proof = shape_proof();
        p.check_proof(&proof, 1, &limits())?;
        proof.initial_ligero_cm.root.root = None;
        assert!(matches!(
            p.check_proof(&proof, 1, &limits()),
            Err(Error::MissingRoot { level: 0 })
        ));
        proof = shape_proof();
        proof.recursive_commitments[0].root.root = None;
        assert!(matches!(
            p.check_proof(&proof, 1, &limits()),
            Err(Error::MissingRoot { level: 1 })
        ));
        proof = shape_proof();
        proof.eval_rounds.clear();
        assert!(matches!(
            p.check_proof(&proof, 1, &limits()),
            Err(Error::Length {
                expected: 1,
                actual: 0
            })
        ));
        p.check_proof(&proof, 0, &limits())?;
        Ok(())
    }

    #[test]
    fn b1_proof_preflight_checks_rows_and_sibling_bounds() -> Result<(), Error> {
        let p = Parameters::new(config(), &limits())?;
        let mut proof = shape_proof();
        p.check_proof(&proof, 1, &limits())?;
        proof.initial_ligero_proof.opened_rows[0].clear();
        assert!(matches!(
            p.check_proof(&proof, 1, &limits()),
            Err(Error::ProofShape { level: 0 })
        ));
        proof = shape_proof();
        proof.final_ligero_proof.opened_rows.pop();
        assert!(matches!(
            p.check_proof(&proof, 1, &limits()),
            Err(Error::ProofShape { level: 1 })
        ));
        proof = shape_proof();
        proof.final_ligero_proof.merkle_proof.siblings = vec![[0; 32]; 160 * 9 + 1];
        assert!(matches!(
            p.check_proof(&proof, 1, &limits()),
            Err(Error::ProofShape { level: 1 })
        ));
        Ok(())
    }

    #[test]
    fn b1_proof_preflight_charges_live_capacities() -> Result<(), Error> {
        let p = Parameters::new(config(), &limits())?;
        let mut proof = shape_proof();
        p.check_proof(&proof, 1, &limits())?;
        proof.initial_ligero_proof.opened_rows[0].try_reserve_exact(1_000_000)?;
        let mut cap = limits();
        cap.max_allocation_bytes = 8_000_000;
        assert!(matches!(
            p.check_proof(&proof, 1, &cap),
            Err(Error::AllocationLimit { .. })
        ));
        Ok(())
    }

    #[cfg(feature = "prover")]
    #[test]
    fn b1_prover_bounds_charge_input_capacity_and_claims() -> Result<(), Error> {
        let p = Parameters::new(config(), &limits())?;
        p.check_prover_limits(512, 1, &limits())?;
        assert!(matches!(
            p.check_prover_limits(511, 1, &limits()),
            Err(Error::Length {
                expected: 512,
                actual: 511
            })
        ));
        let mut cap = limits();
        cap.max_retained_bytes = 100_000;
        assert!(matches!(
            p.check_prover_limits(100_000, 1, &cap),
            Err(Error::RetainedLimit { .. })
        ));
        cap = limits();
        cap.max_claims = 0;
        assert!(matches!(
            p.check_prover_limits(512, 1, &cap),
            Err(Error::ClaimLimit {
                requested: 1,
                limit: 0
            })
        ));
        Ok(())
    }

    #[cfg(feature = "prover")]
    #[test]
    fn b1_prover_parameters_keep_checked_encoders_private() -> Result<(), Box<dyn std::error::Error>>
    {
        let p = ProverParameters::new(Parameters::new(config(), &limits())?, &limits())?;
        p.config().validate()?;
        assert_eq!(p.parameters().message_len(), 512);
        assert_eq!(p.config().initial_reed_solomon.message_length(), 256);
        assert_eq!(p.config().initial_reed_solomon.block_length(), 1024);
        assert_eq!(p.config().reed_solomon_codes[0].message_length(), 128);
        assert_eq!(p.config().reed_solomon_codes[0].block_length(), 512);
        p.check_limits(&limits())?;
        let mut cap = limits();
        cap.max_retained_bytes = p.retained_bytes();
        assert!(matches!(
            p.check_limits(&cap),
            Err(Error::RetainedLimit { .. })
        ));
        Ok(())
    }
}
