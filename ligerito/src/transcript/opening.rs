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

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use binary_fields::{BinaryElem128, BinaryFieldElement};
use merkle_tree::MerkleRoot;
use sha2::Digest;

use super::Sha256Transcript;
use crate::opening_transcript::{OpeningTranscript, QueryRequest, TranscriptError};

impl Sha256Transcript {
    fn opening_digest(&mut self) -> Result<[u8; 32], TranscriptError> {
        let next = self
            .counter
            .checked_add(1)
            .ok_or(TranscriptError::CounterExhausted)?;
        let digest = self.absorb_counter_digest();
        self.counter = next;
        Ok(digest)
    }

    fn opening_field(&mut self, value: BinaryElem128) {
        let bytes = value.poly().value().to_le_bytes();
        if self.domain_separated {
            self.hasher.update(b"field_element");
            self.hasher.update(16_u64.to_le_bytes());
        }
        self.hasher.update(bytes);
    }

    fn opening_query(&mut self, domain: usize) -> Result<usize, TranscriptError> {
        let mask = u64::try_from(domain - 1)?;
        let digest = self.opening_digest()?;
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&digest[..8]);
        Ok(usize::try_from(u64::from_le_bytes(bytes) & mask)?)
    }
}

impl OpeningTranscript for Sha256Transcript {
    type Message = BinaryElem128;
    type Challenge = BinaryElem128;
    type Error = TranscriptError;

    fn observe_root(&mut self, root: &MerkleRoot) -> Result<(), Self::Error> {
        let hash = root.root.as_ref().ok_or(TranscriptError::MissingRoot)?;
        if self.domain_separated {
            self.hasher.update(b"merkle_root");
            self.hasher.update(32_u64.to_le_bytes());
        }
        self.hasher.update(hash);
        Ok(())
    }

    fn observe_bytes(&mut self, label: &[u8], data: &[u8]) -> Result<(), Self::Error> {
        let length = u64::try_from(data.len())?;
        self.hasher.update(label);
        self.hasher.update(length.to_le_bytes());
        self.hasher.update(data);
        Ok(())
    }

    fn observe_message(&mut self, value: Self::Message) -> Result<(), Self::Error> {
        self.opening_field(value);
        Ok(())
    }

    fn observe_challenge(&mut self, value: Self::Challenge) -> Result<(), Self::Error> {
        self.opening_field(value);
        Ok(())
    }

    fn observe_challenges(&mut self, values: &[Self::Challenge]) -> Result<(), Self::Error> {
        let count = u64::try_from(values.len())?;
        let length = count
            .checked_mul(16)
            .ok_or(TranscriptError::LengthOverflow)?;
        if self.domain_separated {
            self.hasher.update(b"field_elements");
            self.hasher.update(length.to_le_bytes());
        }
        for value in values {
            self.hasher.update(value.poly().value().to_le_bytes());
        }
        Ok(())
    }

    fn sample_challenge(&mut self) -> Result<Self::Challenge, Self::Error> {
        let digest = self.opening_digest()?;
        let mut bytes = [0; 16];
        bytes.copy_from_slice(&digest[..16]);
        Ok(BinaryElem128::from(u128::from_le_bytes(bytes)))
    }

    fn sample_queries(&mut self, request: QueryRequest) -> Result<Vec<usize>, Self::Error> {
        request.sample(|| self.opening_query(request.domain()))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use binary_fields::{BinaryElem128, BinaryFieldElement};
    use merkle_tree::MerkleRoot;

    use crate::opening_transcript::{OpeningTranscript, QueryRequest, TranscriptError};
    use crate::transcript::{Sha256Transcript, Transcript};

    #[test]
    fn b1_typed_sha_preserves_native_observations() -> Result<(), Box<dyn std::error::Error>> {
        let mut typed = Sha256Transcript::new(7);
        let mut legacy = Sha256Transcript::new(7);
        let root = MerkleRoot {
            root: Some([0x5a; 32]),
        };
        let value = BinaryElem128::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100);
        let values = [BinaryElem128::zero(), BinaryElem128::one()];
        typed.observe_root(&root)?;
        legacy.absorb_root(&root);
        typed.observe_message(value)?;
        legacy.absorb_elem(value);
        typed.observe_challenge(BinaryElem128::one())?;
        legacy.absorb_elem(BinaryElem128::one());
        typed.observe_challenges(&values)?;
        legacy.absorb_elems(&values);
        typed.observe_bytes(b"context", &[4, 5, 6])?;
        legacy.absorb_bytes(b"context", &[4, 5, 6]);
        let challenge = typed.sample_challenge()?;
        assert_eq!(
            challenge.poly().value(),
            0xda03_0780_fcbf_cd3f_7187_fee8_1d7c_4e97
        );
        assert_eq!(challenge, legacy.get_challenge::<BinaryElem128>());
        assert_ne!(challenge.poly().value() >> 64, 0);
        let queries = typed.sample_queries(QueryRequest::new(1024, 160, 4096)?)?;
        assert_eq!(queries, legacy.get_distinct_queries(1024, 160));
        Ok(())
    }

    #[test]
    fn b1_typed_sha_rejects_missing_root_without_mutation() -> Result<(), TranscriptError> {
        let mut transcript = Sha256Transcript::new(7);
        let result = transcript.observe_root(&MerkleRoot { root: None });
        assert!(matches!(result, Err(TranscriptError::MissingRoot)));
        assert_eq!(
            transcript.sample_challenge()?,
            Sha256Transcript::new(7).sample_challenge()?
        );
        Ok(())
    }

    #[test]
    fn b1_typed_sha_counter_exhaustion_precedes_mutation() -> Result<(), TranscriptError> {
        use sha2::Digest;

        let mut transcript = Sha256Transcript::new(7);
        transcript.counter = u32::MAX - 1;
        transcript.sample_challenge()?;
        let before = transcript.hasher.clone();
        assert!(matches!(
            transcript.sample_challenge(),
            Err(TranscriptError::CounterExhausted)
        ));
        assert_eq!(transcript.counter, u32::MAX);
        assert_eq!(transcript.hasher.finalize(), before.finalize());
        Ok(())
    }

    #[test]
    fn b1_typed_sha_raw_mode_preserves_stream() -> Result<(), Box<dyn std::error::Error>> {
        let mut typed = Sha256Transcript::new_raw(-7);
        let mut legacy = Sha256Transcript::new_raw(-7);
        let root = MerkleRoot {
            root: Some([9; 32]),
        };
        typed.observe_root(&root)?;
        legacy.absorb_root(&root);
        typed.observe_message(BinaryElem128::from(u128::MAX))?;
        legacy.absorb_elem(BinaryElem128::from(u128::MAX));
        typed.observe_challenges(&[])?;
        legacy.absorb_elems::<BinaryElem128>(&[]);
        assert_eq!(
            typed.sample_challenge()?,
            legacy.get_challenge::<BinaryElem128>()
        );
        let queries = typed.sample_queries(QueryRequest::new(8, 1, 1)?)?;
        assert_eq!(queries, legacy.get_distinct_queries(8, 1));
        Ok(())
    }

    #[test]
    fn b1_typed_sha_query_counter_exhaustion_is_fallible() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut transcript = Sha256Transcript::new(7);
        transcript.counter = u32::MAX;
        let request = QueryRequest::new(8, 1, 1)?;
        assert!(matches!(
            transcript.sample_queries(request),
            Err(TranscriptError::CounterExhausted)
        ));
        assert_eq!(transcript.counter, u32::MAX);
        Ok(())
    }

    #[test]
    #[expect(
        clippy::unnecessary_wraps,
        reason = "Repository tests uniformly return Result."
    )]
    fn b1_legacy_sha_multiblock_squeeze_matches_fixed_vector() -> Result<(), TranscriptError> {
        let expected = [
            0xaa, 0xe8, 0x9f, 0xc0, 0xf0, 0x3e, 0x29, 0x59, 0xae, 0x4d, 0x70, 0x1a, 0x80, 0xcc,
            0x39, 0x15, 0x91, 0x8c, 0x95, 0x0b, 0x15, 0x9f, 0x6a, 0xbb, 0x6c, 0x92, 0xc1, 0x43,
            0x3b, 0x1a, 0x85, 0x34, 0x0c, 0x7e, 0xf1, 0xd3, 0x79, 0x86, 0x66, 0x54, 0x39, 0x7b,
            0xdc, 0xc7, 0xa4, 0xa8, 0xc2, 0x0a, 0x0a, 0x2b, 0x86, 0x9c, 0x60, 0x28, 0x00, 0x83,
            0x3e, 0xdf, 0x86, 0xc0, 0xf4, 0x28, 0x82, 0x6d, 0xcf,
        ];
        let mut transcript = Sha256Transcript::new(7);
        assert_eq!(transcript.squeeze_bytes(65), expected);
        assert_eq!(transcript.counter, 3);
        assert_eq!(transcript.squeeze_bytes(1), [0xdf]);
        let mut empty = Sha256Transcript::new(7);
        assert!(empty.squeeze_bytes(0).is_empty());
        assert_eq!(empty.counter, 1);
        Ok(())
    }
}
