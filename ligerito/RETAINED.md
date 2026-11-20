# Retained native indexed openings

`ligerito::retained` adds checked native GF(2^128) commit/open/verify operations.
This is an experimental, non-zero-knowledge component API. It does not implement
arbitrary-point openings, approve an application profile, or establish a composed
security bound. Existing indexed claims are the supported regression contract.

Use `default-features = false`. Verification requires `std`; commitment and
opening additionally require `prover`. The selected sequential component lane is
`std,prover`. Native defaults also activate CLI, parallelism and acceleration.

## Ownership and call sequence

1. Construct `Parameters` from trusted `VerifierConfig` and current `Limits`.
   Checked recursive geometry uses 160 queries and at most 256 terminal values.
   It is not an approval token for the complete O2 branch schedule.
2. Construct `ProverParameters`, which owns private native encoder precomputations.
3. Call `commit(CommitRequest { parameters, input, limits })`. The owned input
   moves into opaque, non-cloneable `ProverState` with original encoded columns
   and Merkle tree. Save `state.root()` before choosing indexed claims.
4. Call `open(state, OpenRequest { parameters, expected_root, claims, limits },
   &mut transcript)`. Opening consumes state and reuses the original commitment.
   Necessary later recursive commitments still execute normally.
5. `OpenedProof` contains the proof and cleared `reusable_input`. Only bounded
   terminal values are copied into the proof. Refill returned capacity completely
   before a new commitment. A live proof never borrows this returned storage.
   Clearing does not erase bytes.
6. Call `verify(VerifyRequest { parameters, expected_root, claims, proof, limits },
   &mut transcript)` with independently supplied expected context and a fresh
   verifier transcript reproducing the same observations. No witness is needed.

Opening checks expected root, geometry, claims and current limits before its
first transcript callback. Compatible replacement parameters are allowed, but
both original and replacement parameter capacities count while live. Errors
consume and drop per-call state. A new attempt uses fresh input and transcript;
successful input-capacity reuse is explicit through the returned buffer.

## Transcript and compatibility

`OpeningTranscript` separates message fields, challenge fields, roots, bytes and
query operations. Every callback returns an error. The concrete native
`Sha256Transcript` implementation uses safe GF128 value conversion and checks its
counter before mutation. A driver never restarts or seeds a transcript.

`QueryRequest` requires a nonzero power-of-two domain, a positive count no greater
than that domain, and a finite draw budget. Native sampling masks uniform 64-bit
draws into that domain, removes duplicates, and returns exactly the requested
sorted indices or an error. Success on the last permitted draw is valid. The
driver validates custom transcript outputs too. The core transcript mapping and
cross-layer grammar remain separate work.

Legacy public functions delegate to the same engine and preserve successful
native observations and proof bytes. They use `LegacyTranscript` with a finite
1,048,576-draw cap. Oversubscribed queries fail instead of being clamped. Errors
that the old public error type cannot represent map to its existing verification
failure; the retained API preserves typed transcript/query/allocation sources.
Legacy generic transcript implementations and inherited overflow behavior remain
distinct from the safe concrete GF128 implementation.

## Resources and layout

Limits cover conservative vector capacity, field/hash work, claims, query
attempts and actual retained capacities. Proof shape and native assertion
preconditions are checked before verifier calls. Oversized input capacity counts
even when its length is correct. Byte charges include live output and scratch;
parallel-feature bounds include concurrent native scratch. They do not measure
allocator metadata, thread stacks or total process RSS.

New explicit reservations are fallible. Reused native encoder, Merkle and driver
routines still contain internal infallible allocations within preflighted bounds;
recovery from every allocator failure is not promised. Native memory hashing
requires validated little-endian GF128 layouts. Row prehash and Merkle hashing
remain SHA256 over the little-endian u32 row length and native field bytes, then
BLAKE3 leaf/ordered-child hashing.

The verifier-only surface omits `commit`, `open`, `ProverParameters`, `ProverState`
and `OpenedProof`. First-party field conversion, shared-core transcript adaptation,
bounded wire codecs and application profile admission remain later stages.
