use binary_fields::BinaryElem128;

use crate::eval_proof::EvalClaim;
use crate::opening_transcript::{DriverError, OpeningTranscript};
use crate::FinalizedLigeritoProof;

use super::{Error, Limits, Parameters};

/// A caller-owned original native commitment root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommitmentRoot([u8; 32]);

impl From<[u8; 32]> for CommitmentRoot {
    fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl CommitmentRoot {
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub(crate) fn check(&self, root: &merkle_tree::MerkleRoot) -> Result<(), Error> {
        if root.root != Some(self.0) {
            return Err(Error::RootMismatch);
        }
        Ok(())
    }
}

/// A retained native operation preserves validation and transcript error sources.
#[derive(Debug, thiserror::Error)]
pub enum OperationError<E: std::error::Error + 'static> {
    #[error("invalid retained operation")]
    Input(#[from] Error),
    #[error("native opening driver failed")]
    Driver(#[from] DriverError<E>),
}

/// Trusted verifier context and a borrowed native proof.
///
/// Verification returns `OperationError::Input` for root mismatches, malformed
/// claims or proof shape, arithmetic overflow, exceeded limits or failed native
/// verification. Native driver, transcript, query and allocation failures retain their
/// `OperationError::Driver` source.
#[derive(Clone, Copy)]
pub struct VerifyRequest<'a> {
    pub parameters: &'a Parameters,
    pub expected_root: CommitmentRoot,
    pub claims: &'a [EvalClaim<BinaryElem128>],
    pub proof: &'a FinalizedLigeritoProof<BinaryElem128, BinaryElem128>,
    pub limits: &'a Limits,
}

#[expect(
    clippy::missing_errors_doc,
    reason = "VerifyRequest documents verification errors."
)]
#[tracing::instrument(skip_all)]
pub fn verify<C>(
    request: VerifyRequest<'_>,
    transcript: &mut C,
) -> Result<(), OperationError<C::Error>>
where
    C: OpeningTranscript<Message = BinaryElem128, Challenge = BinaryElem128>,
    C::Error: std::error::Error + 'static,
{
    let VerifyRequest {
        parameters,
        expected_root,
        claims,
        proof,
        limits,
    } = request;
    expected_root.check(&proof.initial_ligero_cm.root)?;
    parameters.check_claims(claims, limits)?;
    parameters.check_proof(proof, claims.len(), limits)?;
    let driver = crate::verifier::VerificationRequest {
        config: parameters.config(),
        proof,
        claims,
        max_query_draws: limits.max_query_draws,
    };
    if !crate::verifier::verify_retained(driver, transcript)? {
        return Err(Error::ProtocolFailure.into());
    }
    Ok(())
}
