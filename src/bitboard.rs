use core::fmt;

use crate::types::Square;

/// A 64-bit set of squares. Bit `n` corresponds to `Square::from_index(n)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bitboard(u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);

    pub const fn from_square(square: Square) -> Bitboard {
        Bitboard(1u64 << square.index())
    }

    pub const fn contains(self, square: Square) -> bool {
        self.0 & (1u64 << square.index()) != 0
    }

    pub const fn with(self, square: Square) -> Bitboard {
        Bitboard(self.0 | (1u64 << square.index()))
    }

    pub const fn without(self, square: Square) -> Bitboard {
        Bitboard(self.0 & !(1u64 << square.index()))
    }

    pub const fn union(self, other: Bitboard) -> Bitboard {
        Bitboard(self.0 | other.0)
    }

    pub const fn intersect(self, other: Bitboard) -> Bitboard {
        Bitboard(self.0 & other.0)
    }

    pub const fn complement(self) -> Bitboard {
        Bitboard(!self.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn any(self) -> bool {
        self.0 != 0
    }

    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Remove and return the least-significant set square.
    pub fn pop_lsb(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            let index = self.0.trailing_zeros() as u8;
            self.0 &= self.0 - 1;
            Some(Square::from_index(index))
        }
    }

    pub fn iter(self) -> BitboardIter {
        BitboardIter(self)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Bitboard({:#018x})", self.0)?;
        for rank in (0..8).rev() {
            for file in 0..8 {
                let square = Square::from_file_rank(file, rank);
                f.write_str(if self.contains(square) { "1" } else { "." })?;
            }
            if rank != 0 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

pub struct BitboardIter(Bitboard);

impl Iterator for BitboardIter {
    type Item = Square;

    fn next(&mut self) -> Option<Square> {
        self.0.pop_lsb()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.0.count() as usize;
        (count, Some(count))
    }
}

impl ExactSizeIterator for BitboardIter {}
