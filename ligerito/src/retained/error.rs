use std::collections::TryReserveError;

/// Validation and resource failures at the retained native boundary.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid native geometry at level {level}")]
    Geometry { level: usize },
    #[error("unsupported polynomial log size {log_size}")]
    UnsupportedSize { log_size: usize },
    #[error("expected {expected} queries, received {actual}")]
    QueryCount { expected: usize, actual: usize },
    #[error("native size arithmetic overflow")]
    Overflow,
    #[error("expected length {expected}, received {actual}")]
    Length { expected: usize, actual: usize },
    #[error("invalid proof shape at level {level}")]
    ProofShape { level: usize },
    #[error("claim index {index} exceeds table length {length}")]
    ClaimIndex { index: usize, length: usize },
    #[error("missing commitment root at level {level}")]
    MissingRoot { level: usize },
    #[error("allocation charge {requested} exceeds limit {limit}")]
    AllocationLimit { requested: usize, limit: usize },
    #[error("work charge {requested} exceeds limit {limit}")]
    WorkLimit { requested: usize, limit: usize },
    #[error("retained charge {requested} exceeds limit {limit}")]
    RetainedLimit { requested: usize, limit: usize },
    #[error("claim count {requested} exceeds limit {limit}")]
    ClaimLimit { requested: usize, limit: usize },
    #[error("query draw requirement {requested} exceeds limit {limit}")]
    QueryDrawLimit { requested: usize, limit: usize },
    #[error("unsupported native field layout")]
    UnsupportedLayout,
    #[error("retained state and expected root differ")]
    RootMismatch,
    #[error("retained state and expected context differ")]
    ContextMismatch,
    #[error("native proof verification failed")]
    ProtocolFailure,
    #[error("native buffer reservation failed")]
    Allocation(#[from] TryReserveError),
}

impl From<core::num::TryFromIntError> for Error {
    fn from(_: core::num::TryFromIntError) -> Self {
        Self::Overflow
    }
}
