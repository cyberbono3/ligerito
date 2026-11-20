# Changelog

## Unreleased

- Added checked native GF128 retained commitment state, consuming indexed opening,
  expected-root verification, and explicit reuse of cleared input capacity.
  ([#2](https://github.com/cyberbono3/ligerito/pull/2))
- Added a typed fallible transcript boundary and bounded exact query sampling.
  Legacy proving and verification wrappers share the native driver; oversubscribed
  queries now fail instead of being clamped.
- Preserved native commitment hashing and successful known-answer proof bytes.
  Arbitrary-point openings and application profile approval remain separate work.
