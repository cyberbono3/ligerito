# Changelog

## Unreleased

- Add checked native GF128 retained commitment state, consuming indexed opening,
  expected-root verification, and explicit reuse of cleared input capacity.
- Add a typed fallible transcript boundary and bounded exact query sampling.
  Legacy proving and verification wrappers share the native driver; oversubscribed
  queries now fail instead of being clamped.
- Preserve native commitment hashing and successful known-answer proof bytes.
  Arbitrary-point openings and application profile approval remain separate work.
