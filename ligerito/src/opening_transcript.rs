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
use alloc::{collections::TryReserveError, vec::Vec};
#[cfg(feature = "std")]
use std::collections::TryReserveError;

use core::marker::PhantomData;

use binary_fields::BinaryFieldElement;
use merkle_tree::MerkleRoot;

use crate::transcript::Transcript;

/// A validated request for distinct transcript-derived row indices.
///
/// Construction rejects invalid domains, counts and draw budgets with `QueryError`.
/// Validation rejects a wrong count, an out-of-domain index or unsorted duplicates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryRequest {
    domain: usize,
    count: usize,
    max_draws: usize,
}

impl QueryRequest {
    #[expect(
        clippy::missing_errors_doc,
        reason = "QueryRequest documents construction errors."
    )]
    pub fn new(domain: usize, count: usize, max_draws: usize) -> Result<Self, QueryError> {
        if !domain.is_power_of_two() {
            return Err(QueryError::Domain { domain });
        }
        if count == 0 || count > domain {
            return Err(QueryError::Count { count, domain });
        }
        if max_draws < count {
            return Err(QueryError::DrawBudget { max_draws, count });
        }
        Ok(Self {
            domain,
            count,
            max_draws,
        })
    }

    #[must_use]
    pub const fn domain(self) -> usize {
        self.domain
    }

    #[must_use]
    pub const fn count(self) -> usize {
        self.count
    }

    #[must_use]
    pub const fn max_draws(self) -> usize {
        self.max_draws
    }

    #[expect(
        clippy::missing_errors_doc,
        reason = "QueryRequest documents validation errors."
    )]
    pub fn validate(self, queries: &[usize]) -> Result<(), QueryError> {
        if queries.len() != self.count {
            return Err(QueryError::OutputCount {
                expected: self.count,
                actual: queries.len(),
            });
        }
        for &index in queries {
            if index >= self.domain {
                return Err(QueryError::Index {
                    index,
                    domain: self.domain,
                });
            }
        }
        if queries.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(QueryError::Order);
        }
        Ok(())
    }

    pub(crate) fn sample(
        self,
        mut draw: impl FnMut() -> Result<usize, TranscriptError>,
    ) -> Result<Vec<usize>, TranscriptError> {
        let mut queries = Vec::new();
        queries
            .try_reserve_exact(self.count)
            .map_err(QueryError::Allocation)
            .map_err(TranscriptError::Query)?;
        for _ in 0..self.max_draws {
            let index = draw()?;
            if index >= self.domain {
                return Err(TranscriptError::Query(QueryError::Index {
                    index,
                    domain: self.domain,
                }));
            }
            if !queries.contains(&index) {
                queries.push(index);
            }
            if queries.len() == self.count {
                queries.sort_unstable();
                return Ok(queries);
            }
        }
        Err(TranscriptError::Query(QueryError::Exhausted {
            requested: self.count,
            accepted: queries.len(),
            draws: self.max_draws,
        }))
    }
}

/// Query request, allocation or output validation failure.
#[derive(Debug)]
#[cfg_attr(feature = "std", derive(thiserror::Error))]
pub enum QueryError {
    #[cfg_attr(
        feature = "std",
        error("query domain {domain} is not a nonzero power of two")
    )]
    Domain { domain: usize },
    #[cfg_attr(
        feature = "std",
        error("query count {count} is invalid for domain {domain}")
    )]
    Count { count: usize, domain: usize },
    #[cfg_attr(
        feature = "std",
        error("draw budget {max_draws} is smaller than query count {count}")
    )]
    DrawBudget { max_draws: usize, count: usize },
    #[cfg_attr(feature = "std", error("query count is {actual}, expected {expected}"))]
    OutputCount { expected: usize, actual: usize },
    #[cfg_attr(
        feature = "std",
        error("query index {index} is outside domain {domain}")
    )]
    Index { index: usize, domain: usize },
    #[cfg_attr(feature = "std", error("query indices are not strictly increasing"))]
    Order,
    #[cfg_attr(
        feature = "std",
        error("collected {accepted} of {requested} queries in {draws} draws")
    )]
    Exhausted {
        requested: usize,
        accepted: usize,
        draws: usize,
    },
    #[cfg_attr(feature = "std", error("query allocation failed"))]
    Allocation(#[cfg_attr(feature = "std", source)] TryReserveError),
}

/// A failure in a concrete native or legacy transcript adapter.
#[derive(Debug)]
#[cfg_attr(feature = "std", derive(thiserror::Error))]
pub enum TranscriptError {
    #[cfg_attr(feature = "std", error("commitment root is missing"))]
    MissingRoot,
    #[cfg_attr(feature = "std", error("transcript length cannot be represented"))]
    LengthOverflow,
    #[cfg_attr(feature = "std", error("transcript counter is exhausted"))]
    CounterExhausted,
    #[cfg_attr(feature = "std", error("query operation failed"))]
    Query(#[cfg_attr(feature = "std", source)] QueryError),
}

impl From<core::num::TryFromIntError> for TranscriptError {
    fn from(_: core::num::TryFromIntError) -> Self {
        Self::LengthOverflow
    }
}

/// Failure propagated by a native driver borrowing a fallible transcript.
#[derive(Debug)]
#[cfg_attr(feature = "std", derive(thiserror::Error))]
pub enum DriverError<E> {
    #[cfg_attr(feature = "std", error("native operation failed"))]
    Native(#[cfg_attr(feature = "std", source)] crate::LigeritoError),
    #[cfg_attr(feature = "std", error("transcript operation failed"))]
    Transcript(#[cfg_attr(feature = "std", source)] E),
    #[cfg_attr(feature = "std", error("query operation failed"))]
    Query(#[cfg_attr(feature = "std", source)] QueryError),
    #[cfg_attr(feature = "std", error("native driver allocation failed"))]
    Allocation(#[cfg_attr(feature = "std", source)] TryReserveError),
}

impl<E> DriverError<E> {
    pub(crate) fn into_legacy(self) -> crate::LigeritoError {
        match self {
            Self::Native(error) => error,
            Self::Transcript(_) | Self::Query(_) | Self::Allocation(_) => {
                crate::LigeritoError::VerificationFailed
            }
        }
    }
}

/// Typed observations and fallible challenges for the native opening driver.
///
/// Each callback returns the implementation's `Error` without replacement.
/// The concrete SHA256 adapter reports missing roots, encoding overflow, exhausted
/// counters and failed query sampling. Other adapters define their own callback
/// errors. Callers propagate errors immediately; failure does not authorize reuse.
pub trait OpeningTranscript {
    type Message: BinaryFieldElement;
    type Challenge: BinaryFieldElement;
    type Error;

    #[expect(
        clippy::missing_errors_doc,
        reason = "OpeningTranscript documents callback errors."
    )]
    fn observe_root(&mut self, root: &MerkleRoot) -> Result<(), Self::Error>;
    #[expect(
        clippy::missing_errors_doc,
        reason = "OpeningTranscript documents callback errors."
    )]
    fn observe_bytes(&mut self, label: &[u8], data: &[u8]) -> Result<(), Self::Error>;
    #[expect(
        clippy::missing_errors_doc,
        reason = "OpeningTranscript documents callback errors."
    )]
    fn observe_message(&mut self, value: Self::Message) -> Result<(), Self::Error>;
    #[expect(
        clippy::missing_errors_doc,
        reason = "OpeningTranscript documents callback errors."
    )]
    fn observe_challenge(&mut self, value: Self::Challenge) -> Result<(), Self::Error>;
    #[expect(
        clippy::missing_errors_doc,
        reason = "OpeningTranscript documents callback errors."
    )]
    fn observe_challenges(&mut self, values: &[Self::Challenge]) -> Result<(), Self::Error>;
    #[expect(
        clippy::missing_errors_doc,
        reason = "OpeningTranscript documents callback errors."
    )]
    fn sample_challenge(&mut self) -> Result<Self::Challenge, Self::Error>;
    #[expect(
        clippy::missing_errors_doc,
        reason = "OpeningTranscript documents callback errors."
    )]
    fn sample_queries(&mut self, request: QueryRequest) -> Result<Vec<usize>, Self::Error>;
}

pub(crate) fn sample_queries<C: OpeningTranscript>(
    transcript: &mut C,
    request: QueryRequest,
) -> Result<Vec<usize>, DriverError<C::Error>> {
    let queries = transcript
        .sample_queries(request)
        .map_err(DriverError::Transcript)?;
    request.validate(&queries).map_err(DriverError::Query)?;
    Ok(queries)
}

pub(crate) struct LegacyTranscript<'a, T, U, C> {
    transcript: &'a mut C,
    fields: PhantomData<(T, U)>,
}

impl<'a, T, U, C> LegacyTranscript<'a, T, U, C> {
    pub(crate) fn new(transcript: &'a mut C) -> Self {
        Self {
            transcript,
            fields: PhantomData,
        }
    }
}

impl<T, U, C> OpeningTranscript for LegacyTranscript<'_, T, U, C>
where
    T: BinaryFieldElement,
    U: BinaryFieldElement,
    C: Transcript,
{
    type Message = T;
    type Challenge = U;
    type Error = TranscriptError;

    fn observe_root(&mut self, root: &MerkleRoot) -> Result<(), Self::Error> {
        if root.root.is_none() {
            return Err(TranscriptError::MissingRoot);
        }
        self.transcript.absorb_root(root);
        Ok(())
    }

    fn observe_bytes(&mut self, label: &[u8], data: &[u8]) -> Result<(), Self::Error> {
        self.transcript.absorb_bytes(label, data);
        Ok(())
    }

    fn observe_message(&mut self, value: T) -> Result<(), Self::Error> {
        self.transcript.absorb_elem(value);
        Ok(())
    }

    fn observe_challenge(&mut self, value: U) -> Result<(), Self::Error> {
        self.transcript.absorb_elem(value);
        Ok(())
    }

    fn observe_challenges(&mut self, values: &[U]) -> Result<(), Self::Error> {
        self.transcript.absorb_elems(values);
        Ok(())
    }

    fn sample_challenge(&mut self) -> Result<U, Self::Error> {
        Ok(self.transcript.get_challenge())
    }

    fn sample_queries(&mut self, request: QueryRequest) -> Result<Vec<usize>, Self::Error> {
        request.sample(|| Ok(self.transcript.get_query(request.domain())))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use binary_fields::{BinaryElem128, BinaryElem32};
    use merkle_tree::MerkleRoot;

    use super::{LegacyTranscript, OpeningTranscript, QueryError, QueryRequest, TranscriptError};
    use crate::transcript::{Sha256Transcript, Transcript};

    #[test]
    fn b1_query_request_rejects_invalid_bounds() -> Result<(), QueryError> {
        assert!(matches!(
            QueryRequest::new(0, 1, 1),
            Err(QueryError::Domain { domain: 0 })
        ));
        assert!(matches!(
            QueryRequest::new(3, 1, 1),
            Err(QueryError::Domain { domain: 3 })
        ));
        assert!(matches!(
            QueryRequest::new(2, 0, 1),
            Err(QueryError::Count {
                count: 0,
                domain: 2
            })
        ));
        assert!(matches!(
            QueryRequest::new(2, 3, 3),
            Err(QueryError::Count {
                count: 3,
                domain: 2
            })
        ));
        assert!(matches!(
            QueryRequest::new(2, 2, 1),
            Err(QueryError::DrawBudget {
                max_draws: 1,
                count: 2
            })
        ));
        let request = QueryRequest::new(2, 2, 2)?;
        assert_eq!(
            (request.domain(), request.count(), request.max_draws()),
            (2, 2, 2)
        );
        Ok(())
    }

    #[test]
    fn b1_query_output_rejects_count_range_and_order() -> Result<(), QueryError> {
        let request = QueryRequest::new(4, 2, 8)?;
        request.validate(&[0, 3])?;
        assert!(matches!(
            request.validate(&[0]),
            Err(QueryError::OutputCount {
                expected: 2,
                actual: 1
            })
        ));
        assert!(matches!(
            request.validate(&[0, 4]),
            Err(QueryError::Index {
                index: 4,
                domain: 4
            })
        ));
        assert!(matches!(request.validate(&[1, 1]), Err(QueryError::Order)));
        assert!(matches!(request.validate(&[2, 1]), Err(QueryError::Order)));
        Ok(())
    }

    #[test]
    fn b1_query_sampling_succeeds_on_last_draw() -> Result<(), Box<dyn std::error::Error>> {
        let request = QueryRequest::new(4, 3, 5)?;
        let mut draws = [3, 3, 1, 1, 2].into_iter();
        let queries = request.sample(|| draws.next().ok_or(TranscriptError::CounterExhausted))?;
        assert_eq!(queries, [1, 2, 3]);
        assert_eq!(draws.next(), None);
        Ok(())
    }

    #[test]
    fn b1_query_sampling_reports_exhaustion() -> Result<(), QueryError> {
        let request = QueryRequest::new(4, 2, 7)?;
        let mut calls = 0;
        let result = request.sample(|| {
            calls += 1;
            Ok(0)
        });
        assert_eq!(calls, 7);
        assert!(matches!(
            result,
            Err(TranscriptError::Query(QueryError::Exhausted {
                requested: 2,
                accepted: 1,
                draws: 7
            }))
        ));
        Ok(())
    }

    #[test]
    fn b1_query_sampling_propagates_failure_before_later_draws() -> Result<(), QueryError> {
        let request = QueryRequest::new(4, 2, 7)?;
        let mut calls = 0;
        let result = request.sample(|| {
            calls += 1;
            Err(TranscriptError::CounterExhausted)
        });
        assert_eq!(calls, 1);
        assert!(matches!(result, Err(TranscriptError::CounterExhausted)));
        assert!(matches!(
            request.sample(|| Ok(4)),
            Err(TranscriptError::Query(QueryError::Index {
                index: 4,
                domain: 4
            }))
        ));
        Ok(())
    }

    #[test]
    fn b1_query_sampling_reports_capacity_overflow() -> Result<(), QueryError> {
        let count = 1_usize << (usize::BITS - 1);
        let request = QueryRequest::new(count, count, count)?;
        let mut calls = 0;
        let result = request.sample(|| {
            calls += 1;
            Ok(0)
        });
        assert!(matches!(
            result,
            Err(TranscriptError::Query(QueryError::Allocation(_)))
        ));
        assert_eq!(calls, 0);
        Ok(())
    }

    #[test]
    fn b1_legacy_adapter_preserves_distinct_field_roles() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut actual = Sha256Transcript::new(11);
        let mut expected = Sha256Transcript::new(11);
        let mut adapter = LegacyTranscript::<BinaryElem32, BinaryElem128, _>::new(&mut actual);
        let root = MerkleRoot {
            root: Some([0xa3; 32]),
        };
        let message = BinaryElem32::from(0xfedc_ba98);
        let challenge = BinaryElem128::from(0xfedc_ba98_7654_3210_0123_4567_89ab_cdef);
        adapter.observe_root(&root)?;
        expected.absorb_root(&root);
        adapter.observe_bytes(b"example", &[1, 2])?;
        expected.absorb_bytes(b"example", &[1, 2]);
        adapter.observe_message(message)?;
        expected.absorb_elem(message);
        adapter.observe_challenge(challenge)?;
        expected.absorb_elem(challenge);
        adapter.observe_challenges(&[challenge])?;
        expected.absorb_elems(&[challenge]);
        assert_eq!(
            adapter.sample_challenge()?,
            expected.get_challenge::<BinaryElem128>()
        );
        assert_eq!(
            adapter.sample_queries(QueryRequest::new(1024, 160, 4096)?)?,
            expected.get_distinct_queries(1024, 160)
        );
        assert!(matches!(
            adapter.observe_root(&MerkleRoot { root: None }),
            Err(TranscriptError::MissingRoot)
        ));
        Ok(())
    }
}
