#[cfg(feature = "prover")]
use crate::data_structures::RecursiveLigeroWitness;
use binary_fields::BinaryFieldElement;
use merkle_tree::{build_merkle_tree, Hash};
#[cfg(all(feature = "prover", feature = "parallel"))]
use rayon::prelude::*;
#[cfg(feature = "prover")]
use reed_solomon::ReedSolomon;
use sha2::{Digest, Sha256};

/// Lay out `poly` as an `m × n` message matrix padded to `m · inv_rate` rows.
///
/// Row-contiguous: `mat[i][c] = poly[i * n + c]`. The column index `c` is the
/// low `log n` bits of the coefficient index, so combining a row with
/// `eq(ρ, ·)` equals `partial_eval_multilinear(poly, ρ)` (lsb-first binding).
#[cfg(feature = "prover")]
pub fn poly2mat<F: BinaryFieldElement>(
    poly: &[F],
    m: usize,
    n: usize,
    inv_rate: usize,
) -> Vec<Vec<F>> {
    assert_eq!(poly.len(), m * n, "poly length must equal m * n");
    let mut mat = vec![vec![F::zero(); n]; m * inv_rate];
    for (row, chunk) in mat.iter_mut().zip(poly.chunks_exact(n)) {
        row.copy_from_slice(chunk);
    }
    mat
}

/// Reed-Solomon encode every column of `poly_mat` (non-systematic encoding,
/// so the generator row for query `q` is `X_i(from_bits(q))`; see
/// `utils::evaluate_scaled_basis_inplace`).
#[cfg(feature = "prover")]
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
    clippy::ptr_arg,
    reason = "Preserve the existing public function-pointer signature."
)]
pub fn encode_cols<F: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static>(
    poly_mat: &mut Vec<Vec<F>>,
    rs: &ReedSolomon<F>,
    parallel: bool,
) {
    let n = poly_mat[0].len();
    let encode_col = |j: usize| {
        let mut col: Vec<F> = poly_mat.iter().map(|row| row[j]).collect();
        // sequential FFT within each column task to avoid nested parallelism
        reed_solomon::encode_non_systematic_with_parallel(rs, &mut col, false);
        col
    };

    #[cfg(feature = "parallel")]
    let cols: Vec<Vec<F>> = if parallel {
        (0..n).into_par_iter().map(encode_col).collect()
    } else {
        (0..n).map(encode_col).collect()
    };

    #[cfg(not(feature = "parallel"))]
    let cols: Vec<Vec<F>> = {
        let _ = parallel;
        (0..n).map(encode_col).collect()
    };

    for (i, row) in poly_mat.iter_mut().enumerate() {
        for (j, col) in cols.iter().enumerate() {
            row[j] = col[i];
        }
    }
}

/// Hash a row of field elements with deterministic serialization
/// Optimized: hashes entire row at once instead of element-by-element
#[inline(always)]
pub fn hash_row<F: BinaryFieldElement>(row: &[F]) -> Hash {
    #[cfg(feature = "op-count")]
    crate::op_count::bump_row_hash();
    let mut hasher = Sha256::new();

    // Hash row length for domain separation
    hasher.update((row.len() as u32).to_le_bytes());

    // Hash entire row at once using bytemuck for zero-copy byte slice
    // This is much faster than per-element hashing
    let row_bytes = unsafe {
        core::slice::from_raw_parts(row.as_ptr() as *const u8, core::mem::size_of_val(row))
    };
    hasher.update(row_bytes);

    hasher.finalize().into()
}

/// Commit to `poly` laid out as an `m × n` message matrix (see `poly2mat`):
/// RS-encode every column to length `4m` and Merkle-commit the rows.
#[cfg(feature = "prover")]
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
    clippy::missing_panics_doc,
    reason = "RecursiveLigeroWitness documents legacy geometry assertions."
)]
pub fn ligero_commit<F: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static>(
    poly: &[F],
    m: usize,
    n: usize,
    rs: &ReedSolomon<F>,
) -> RecursiveLigeroWitness<F> {
    assert_eq!(poly.len(), m * n, "poly length must equal m * n");
    let block = rs.block_length();
    assert_eq!(rs.message_length(), m);

    let cols: Vec<F> = bytemuck::zeroed_vec(n * block);
    encode_owned_columns(poly, n, rs, cols)
}

#[cfg(all(feature = "std", feature = "prover"))]
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
pub(crate) fn try_ligero_commit<F>(
    poly: &[F],
    columns: usize,
    rs: &ReedSolomon<F>,
) -> Result<RecursiveLigeroWitness<F>, crate::retained::Error>
where
    F: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
{
    let mut encoded = Vec::new();
    let length = columns
        .checked_mul(rs.block_length())
        .ok_or(crate::retained::Error::Overflow)?;
    encoded.try_reserve_exact(length)?;
    encoded.resize(length, F::zero());
    Ok(encode_owned_columns(poly, columns, rs, encoded))
}

#[cfg(feature = "prover")]
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
fn encode_owned_columns<F>(
    poly: &[F],
    columns: usize,
    rs: &ReedSolomon<F>,
    mut encoded: Vec<F>,
) -> RecursiveLigeroWitness<F>
where
    F: BinaryFieldElement + Send + Sync + bytemuck::Pod + 'static,
{
    #[cfg(test)]
    COMMIT_COUNTS.with(|counts| {
        let (encodings, trees) = counts.get();
        counts.set((encodings + 1, trees));
    });
    let block = rs.block_length();
    #[cfg(feature = "parallel")]
    let inner = columns < rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let inner = false;
    let encode_column = |(column, data): (usize, &mut [F])| {
        for (row, value) in data.iter_mut().take(rs.message_length()).enumerate() {
            *value = poly[row * columns + column];
        }
        reed_solomon::encode_non_systematic_with_parallel(rs, data, inner);
    };
    #[cfg(feature = "parallel")]
    encoded
        .par_chunks_mut(block)
        .enumerate()
        .for_each(encode_column);
    #[cfg(not(feature = "parallel"))]
    encoded
        .chunks_mut(block)
        .enumerate()
        .for_each(encode_column);
    RecursiveLigeroWitness::from_columns(encoded, columns)
}

#[cfg(all(test, feature = "prover"))]
std::thread_local! {
    static COMMIT_COUNTS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

#[cfg(all(test, feature = "prover"))]
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
pub(crate) fn commitment_counts() -> (usize, usize) {
    COMMIT_COUNTS.with(std::cell::Cell::get)
}

#[cfg(feature = "prover")]
impl<F: BinaryFieldElement + Send + Sync + bytemuck::Pod> RecursiveLigeroWitness<F> {
    /// Merkle-commit the rows of a column-major matrix with `num_cols` columns.
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
        clippy::missing_panics_doc,
        reason = "RecursiveLigeroWitness documents legacy geometry assertions."
    )]
    #[must_use]
    pub fn from_columns(cols: Vec<F>, num_cols: usize) -> Self {
        const TILE: usize = 256;
        assert!(num_cols > 0 && cols.len().is_multiple_of(num_cols));
        let num_rows = cols.len() / num_cols;

        // Hash rows in tiles: gather a tile of rows into a small buffer
        // (contiguous reads from every column), then hash each row.
        let hash_tile = |(t, out): (usize, &mut [Hash])| {
            let r0 = t * TILE;
            let rows = out.len();
            let mut buf = vec![F::zero(); rows * num_cols];
            for c in 0..num_cols {
                let col = &cols[c * num_rows + r0..c * num_rows + r0 + rows];
                for (i, &v) in col.iter().enumerate() {
                    buf[i * num_cols + c] = v;
                }
            }
            for (h, row) in out.iter_mut().zip(buf.chunks_exact(num_cols)) {
                *h = hash_row(row);
            }
        };
        let mut hashes = vec![Hash::default(); num_rows];
        #[cfg(feature = "parallel")]
        hashes.par_chunks_mut(TILE).enumerate().for_each(hash_tile);
        #[cfg(not(feature = "parallel"))]
        hashes.chunks_mut(TILE).enumerate().for_each(hash_tile);

        #[cfg(test)]
        COMMIT_COUNTS.with(|counts| {
            let (encodings, trees) = counts.get();
            counts.set((encodings, trees + 1));
        });
        let tree = build_merkle_tree(&hashes);
        RecursiveLigeroWitness {
            cols,
            num_rows,
            num_cols,
            tree,
        }
    }
}

#[cfg(test)]
mod tests {
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

    use binary_fields::{BinaryElem128, BinaryFieldElement};

    use super::encode_cols;

    #[test]
    fn b1_legacy_column_encoding_matches_degree_one_native_oracle(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let encoder = reed_solomon::reed_solomon::<BinaryElem128>(2, 8);
        for parallel in [false, true] {
            let mut matrix = vec![vec![BinaryElem128::zero(); 2]; 8];
            matrix[0][1] = BinaryElem128::one();
            matrix[1].fill(BinaryElem128::one());
            encode_cols(&mut matrix, &encoder, parallel);
            assert_eq!(matrix.len(), 8);
            for (row, q) in matrix.iter().zip(0_u64..8) {
                assert_eq!(
                    row.as_slice(),
                    [BinaryElem128::from_bits(q), BinaryElem128::from_bits(q ^ 1)]
                );
            }
        }
        Ok(())
    }
}
