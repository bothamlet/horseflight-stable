use core::fmt;

/// Side to move.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const ALL: [Color; 2] = [Color::White, Color::Black];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    pub const fn is_white(self) -> bool {
        matches!(self, Color::White)
    }

    /// Home rank for the king, rooks and other back-rank pieces.
    pub const fn back_rank(self) -> u8 {
        match self {
            Color::White => 0,
            Color::Black => 7,
        }
    }

    /// Rank on which this colour's pawns start.
    pub const fn pawn_start_rank(self) -> u8 {
        match self {
            Color::White => 1,
            Color::Black => 6,
        }
    }

    /// Rank on which this colour's pawns promote.
    pub const fn promotion_rank(self) -> u8 {
        match self {
            Color::White => 7,
            Color::Black => 0,
        }
    }

    /// Single-step forward direction on the rank axis.
    pub const fn forward(self) -> i8 {
        match self {
            Color::White => 1,
            Color::Black => -1,
        }
    }

    /// The named FEN character for the side to move.
    pub const fn fen_char(self) -> char {
        match self {
            Color::White => 'w',
            Color::Black => 'b',
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Color::White => "white",
            Color::Black => "black",
        })
    }
}

/// A piece type without colour.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Role {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Role {
    pub const ALL: [Role; 6] = [
        Role::Pawn,
        Role::Knight,
        Role::Bishop,
        Role::Rook,
        Role::Queen,
        Role::King,
    ];

    /// Roles a pawn may promote to, in deterministic generation order.
    pub const PROMOTION_TARGETS: [Role; 4] = [Role::Knight, Role::Bishop, Role::Rook, Role::Queen];

    pub const fn index(self) -> usize {
        self as usize
    }

    /// Uppercase FEN character.
    pub const fn fen_char(self) -> char {
        match self {
            Role::Pawn => 'P',
            Role::Knight => 'N',
            Role::Bishop => 'B',
            Role::Rook => 'R',
            Role::Queen => 'Q',
            Role::King => 'K',
        }
    }

    /// Lowercase UCI character, used for promotion suffixes.
    pub const fn uci_char(self) -> char {
        match self {
            Role::Pawn => 'p',
            Role::Knight => 'n',
            Role::Bishop => 'b',
            Role::Rook => 'r',
            Role::Queen => 'q',
            Role::King => 'k',
        }
    }

    pub fn from_fen_char(ch: char) -> Option<Role> {
        match ch.to_ascii_uppercase() {
            'P' => Some(Role::Pawn),
            'N' => Some(Role::Knight),
            'B' => Some(Role::Bishop),
            'R' => Some(Role::Rook),
            'Q' => Some(Role::Queen),
            'K' => Some(Role::King),
            _ => None,
        }
    }

    pub fn from_uci_char(ch: char) -> Option<Role> {
        match ch {
            'n' => Some(Role::Knight),
            'b' => Some(Role::Bishop),
            'r' => Some(Role::Rook),
            'q' => Some(Role::Queen),
            _ => None,
        }
    }

    pub const fn is_minor(self) -> bool {
        matches!(self, Role::Knight | Role::Bishop)
    }

    pub const fn is_slider(self) -> bool {
        matches!(self, Role::Bishop | Role::Rook | Role::Queen)
    }
}

/// A coloured piece.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Piece {
    pub color: Color,
    pub role: Role,
}

impl Piece {
    pub const fn new(color: Color, role: Role) -> Piece {
        Piece { color, role }
    }

    /// Index in `0..12`, ordered by colour and then role.
    pub const fn index(self) -> usize {
        self.color.index() * 6 + self.role.index()
    }

    pub fn from_fen_char(ch: char) -> Option<Piece> {
        let role = Role::from_fen_char(ch)?;
        let color = if ch.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        Some(Piece::new(color, role))
    }

    pub const fn fen_char(self) -> char {
        let ch = self.role.fen_char();
        match self.color {
            Color::White => ch,
            Color::Black => ch.to_ascii_lowercase(),
        }
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.fen_char().to_string())
    }
}

/// A square on the 8x8 board, indexed from `a1 = 0` to `h8 = 63`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Square(u8);

impl Square {
    pub const A1: Square = Square(0);
    pub const B1: Square = Square(1);
    pub const C1: Square = Square(2);
    pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);
    pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);
    pub const A8: Square = Square(56);
    pub const B8: Square = Square(57);
    pub const C8: Square = Square(58);
    pub const D8: Square = Square(59);
    pub const E8: Square = Square(60);
    pub const F8: Square = Square(61);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);

    /// Construct from a `0..64` index. Panics for any out-of-range runtime
    /// value as well as out-of-range constants; use [`Square::from_ascii`] for
    /// external text.
    pub const fn from_index(index: u8) -> Square {
        assert!(index < 64, "square index out of range");
        Square(index)
    }

    pub const fn from_file_rank(file: u8, rank: u8) -> Square {
        assert!(file < 8 && rank < 8, "file or rank out of range");
        Square(rank * 8 + file)
    }

    pub fn from_ascii(text: &str) -> Option<Square> {
        let bytes = text.as_bytes();
        if bytes.len() != 2
            || !(b'a'..=b'h').contains(&bytes[0])
            || !(b'1'..=b'8').contains(&bytes[1])
        {
            return None;
        }
        Some(Square::from_file_rank(bytes[0] - b'a', bytes[1] - b'1'))
    }

    pub const fn index(self) -> u8 {
        self.0
    }

    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    /// Offset by file and rank deltas, returning `None` when off the board.
    /// The intermediate arithmetic is widened so every `i8` delta is safe.
    pub const fn offset(self, file_delta: i8, rank_delta: i8) -> Option<Square> {
        let file = self.file() as i16 + file_delta as i16;
        let rank = self.rank() as i16 + rank_delta as i16;
        if file < 0 || file >= 8 || rank < 0 || rank >= 8 {
            None
        } else {
            Some(Square::from_file_rank(file as u8, rank as u8))
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{file}{rank}")
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CastlingSide {
    KingSide,
    QueenSide,
}

impl CastlingSide {
    pub const fn king_square(color: Color) -> Square {
        match color {
            Color::White => Square::E1,
            Color::Black => Square::E8,
        }
    }

    pub const fn rook_square(self, color: Color) -> Square {
        match (color, self) {
            (Color::White, CastlingSide::KingSide) => Square::H1,
            (Color::White, CastlingSide::QueenSide) => Square::A1,
            (Color::Black, CastlingSide::KingSide) => Square::H8,
            (Color::Black, CastlingSide::QueenSide) => Square::A8,
        }
    }

    pub const fn king_destination(self, color: Color) -> Square {
        match (color, self) {
            (Color::White, CastlingSide::KingSide) => Square::G1,
            (Color::White, CastlingSide::QueenSide) => Square::C1,
            (Color::Black, CastlingSide::KingSide) => Square::G8,
            (Color::Black, CastlingSide::QueenSide) => Square::C8,
        }
    }

    pub const fn rook_destination(self, color: Color) -> Square {
        match (color, self) {
            (Color::White, CastlingSide::KingSide) => Square::F1,
            (Color::White, CastlingSide::QueenSide) => Square::D1,
            (Color::Black, CastlingSide::KingSide) => Square::F8,
            (Color::Black, CastlingSide::QueenSide) => Square::D8,
        }
    }
}

/// Four standard-chess castling rights.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CastlingRights(u8);

impl CastlingRights {
    pub const NONE: CastlingRights = CastlingRights(0);
    pub const ALL: CastlingRights = CastlingRights(0b1111);

    const WHITE_KING: u8 = 1 << 0;
    const WHITE_QUEEN: u8 = 1 << 1;
    const BLACK_KING: u8 = 1 << 2;
    const BLACK_QUEEN: u8 = 1 << 3;

    const fn bit(color: Color, side: CastlingSide) -> u8 {
        match (color, side) {
            (Color::White, CastlingSide::KingSide) => Self::WHITE_KING,
            (Color::White, CastlingSide::QueenSide) => Self::WHITE_QUEEN,
            (Color::Black, CastlingSide::KingSide) => Self::BLACK_KING,
            (Color::Black, CastlingSide::QueenSide) => Self::BLACK_QUEEN,
        }
    }

    pub const fn has(self, color: Color, side: CastlingSide) -> bool {
        self.0 & Self::bit(color, side) != 0
    }

    pub const fn with(self, color: Color, side: CastlingSide) -> CastlingRights {
        CastlingRights(self.0 | Self::bit(color, side))
    }

    pub const fn without(self, color: Color, side: CastlingSide) -> CastlingRights {
        CastlingRights(self.0 & !Self::bit(color, side))
    }

    pub const fn without_color(self, color: Color) -> CastlingRights {
        CastlingRights(
            self.0
                & !Self::bit(color, CastlingSide::KingSide)
                & !Self::bit(color, CastlingSide::QueenSide),
        )
    }

    /// Remove the right associated with a rook on `square`, if any.
    pub const fn without_rook_square(self, square: Square) -> CastlingRights {
        match square {
            Square::A1 => self.without(Color::White, CastlingSide::QueenSide),
            Square::H1 => self.without(Color::White, CastlingSide::KingSide),
            Square::A8 => self.without(Color::Black, CastlingSide::QueenSide),
            Square::H8 => self.without(Color::Black, CastlingSide::KingSide),
            _ => self,
        }
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Rights in canonical FEN order, or `-`.
    pub fn to_fen(self) -> String {
        if self.is_empty() {
            return "-".to_string();
        }
        let mut text = String::with_capacity(4);
        for (color, side, ch) in [
            (Color::White, CastlingSide::KingSide, 'K'),
            (Color::White, CastlingSide::QueenSide, 'Q'),
            (Color::Black, CastlingSide::KingSide, 'k'),
            (Color::Black, CastlingSide::QueenSide, 'q'),
        ] {
            if self.has(color, side) {
                text.push(ch);
            }
        }
        text
    }

    /// Table index in `0..4`, ordered `KQkq`.
    pub const fn index_of(color: Color, side: CastlingSide) -> usize {
        match (color, side) {
            (Color::White, CastlingSide::KingSide) => 0,
            (Color::White, CastlingSide::QueenSide) => 1,
            (Color::Black, CastlingSide::KingSide) => 2,
            (Color::Black, CastlingSide::QueenSide) => 3,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MoveKind {
    Normal,
    DoublePawnPush,
    EnPassant,
    Castle,
}

/// A coordinate move. Castling stores the king's origin and destination;
/// en passant stores the capturing pawn's origin and the target square.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Move {
    from: Square,
    to: Square,
    promotion: Option<Role>,
    kind: MoveKind,
}

impl Move {
    pub const fn normal(from: Square, to: Square) -> Move {
        Move {
            from,
            to,
            promotion: None,
            kind: MoveKind::Normal,
        }
    }

    pub const fn double_pawn_push(from: Square, to: Square) -> Move {
        Move {
            from,
            to,
            promotion: None,
            kind: MoveKind::DoublePawnPush,
        }
    }

    pub const fn en_passant(from: Square, to: Square) -> Move {
        Move {
            from,
            to,
            promotion: None,
            kind: MoveKind::EnPassant,
        }
    }

    pub const fn castle(king_from: Square, king_to: Square) -> Move {
        Move {
            from: king_from,
            to: king_to,
            promotion: None,
            kind: MoveKind::Castle,
        }
    }

    pub const fn promotion(from: Square, to: Square, role: Role) -> Move {
        Move {
            from,
            to,
            promotion: Some(role),
            kind: MoveKind::Normal,
        }
    }

    pub const fn from(self) -> Square {
        self.from
    }

    pub const fn to(self) -> Square {
        self.to
    }

    pub const fn promotion_role(self) -> Option<Role> {
        self.promotion
    }

    pub const fn kind(self) -> MoveKind {
        self.kind
    }

    pub const fn is_castle(self) -> bool {
        matches!(self.kind, MoveKind::Castle)
    }

    pub const fn is_en_passant(self) -> bool {
        matches!(self.kind, MoveKind::EnPassant)
    }

    pub const fn is_promotion(self) -> bool {
        self.promotion.is_some()
    }

    pub const fn is_double_pawn_push(self) -> bool {
        matches!(self.kind, MoveKind::DoublePawnPush)
    }

    /// Long algebraic / UCI text. Standard castling is the king's two-square
    /// move (`e1g1`, `e1c1`).
    pub fn to_uci(self) -> String {
        let mut text = String::with_capacity(5);
        text.push((b'a' + self.from.file()) as char);
        text.push((b'1' + self.from.rank()) as char);
        text.push((b'a' + self.to.file()) as char);
        text.push((b'1' + self.to.rank()) as char);
        if let Some(role) = self.promotion {
            text.push(role.uci_char());
        }
        text
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_uci())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_offset(square: Square, file_delta: i8, rank_delta: i8) -> Option<Square> {
        let file = square.file() as i16 + file_delta as i16;
        let rank = square.rank() as i16 + rank_delta as i16;
        if (0..8).contains(&file) && (0..8).contains(&rank) {
            Some(Square::from_file_rank(file as u8, rank as u8))
        } else {
            None
        }
    }

    #[test]
    fn offset_is_total_for_every_i8_delta_on_both_axes() {
        for index in 0..64 {
            let square = Square::from_index(index);
            for delta in i8::MIN..=i8::MAX {
                assert_eq!(
                    square.offset(delta, 0),
                    expected_offset(square, delta, 0),
                    "file offset {delta} from {square}"
                );
                assert_eq!(
                    square.offset(0, delta),
                    expected_offset(square, 0, delta),
                    "rank offset {delta} from {square}"
                );
            }
        }
    }

    #[test]
    fn offset_normal_edges_and_extremes_are_none_not_panics() {
        assert_eq!(Square::A1.offset(0, 0), Some(Square::A1));
        assert_eq!(Square::A1.offset(1, 0), Some(Square::B1));
        assert_eq!(Square::A1.offset(-1, 0), None);
        assert_eq!(Square::H1.offset(1, 0), None);
        assert_eq!(
            Square::A1.offset(0, 1),
            Some(Square::from_ascii("a2").unwrap())
        );
        assert_eq!(Square::A1.offset(0, -1), None);
        assert_eq!(Square::H8.offset(-1, 0), Some(Square::G8));
        assert_eq!(Square::H8.offset(0, 1), None);
        assert_eq!(
            Square::from_ascii("d4").unwrap().offset(-3, -3),
            Some(Square::A1)
        );
        assert_eq!(
            Square::from_ascii("d4").unwrap().offset(3, 3),
            Some(Square::from_ascii("g7").unwrap())
        );
        assert_eq!(Square::H8.offset(127, 0), None);
        assert_eq!(Square::H8.offset(0, 127), None);
        assert_eq!(Square::A1.offset(-128, 0), None);
        assert_eq!(Square::A1.offset(0, -128), None);
        assert_eq!(Square::from_ascii("d4").unwrap().offset(127, 127), None);
        assert_eq!(Square::from_ascii("d4").unwrap().offset(-128, -128), None);
    }
}
