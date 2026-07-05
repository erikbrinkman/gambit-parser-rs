//! A library for parsing [gambit extensive form
//! game](https://gambitproject.readthedocs.io/en/v16.0.2/formats.html) (`.efg`) files
//!
//! This library produces an [`ExtensiveFormGame`], which can then be easily used to model an
//! extensive form game.
//!
//! In order to minimize memory consumption, this stores references to the underlying string where
//! possible. One side effect is that this is a borrowed struct, and any quoted labels will still
//! have escape sequences in them in the form of [`EscapedStr`]s.
#![warn(missing_docs, clippy::pedantic)]

mod unescaped;

use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit0, digit1, multispace0, multispace1, none_of, one_of, u64},
    combinator::{map, opt, recognize},
    error::{ErrorKind, ParseError},
    multi::{many0, separated_list1},
    sequence::{delimited, pair, preceded, separated_pair},
};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::error::Error as StdError;
use std::fmt::{Display, Error as FmtError, Formatter};
pub use unescaped::{EscapedStr, Unescaped};

/// A chance infoset's label paired with its ordered actions and their probabilities
type ChanceInfoset<'a> = (&'a EscapedStr, Box<[(&'a EscapedStr, BigRational)]>);
/// A player infoset's label paired with its ordered action labels
type PlayerInfoset<'a> = (&'a EscapedStr, Box<[&'a EscapedStr]>);

/// Every outcome defined while parsing, keyed by id. Each entry holds the outcome's name and its
/// payoffs, which the tree nodes reference by id. The null (0) outcome is never stored.
type Outcomes<'a> = HashMap<u64, (&'a EscapedStr, Box<[BigRational]>)>;

/// Every infoset seen while parsing, keyed by id. Player infosets are split per player (index =
/// `player_num - 1`); chance is its own namespace. Each entry holds the infoset's label and ordered
/// actions, which the tree nodes reference by id.
#[derive(Debug, PartialEq, Clone)]
struct Infosets<'a> {
    player: Box<[HashMap<u64, PlayerInfoset<'a>>]>,
    chance: HashMap<u64, ChanceInfoset<'a>>,
}

/// An index into the game's flat node arena ([`ExtensiveFormGame`]'s `nodes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NodeId(usize);

/// A node in the raw, id-referenced game tree. Infoset payloads live on the game, not here.
#[derive(Debug, PartialEq, Clone)]
enum RawNode<'a> {
    Chance(RawChance<'a>),
    Player(RawPlayer<'a>),
    Terminal(RawTerminal<'a>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct RawChance<'a> {
    name: &'a EscapedStr,
    infoset: u64,
    // did THIS node write the infoset block, or inherit it by omission?
    declared: bool,
    children: Box<[NodeId]>,
    outcome: u64,
    // did THIS node write the outcome's name and payoffs, or reference it by id?
    outcome_declared: bool,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct RawPlayer<'a> {
    name: &'a EscapedStr,
    player_num: usize,
    infoset: u64,
    // did THIS node write the infoset block, or inherit it by omission?
    declared: bool,
    children: Box<[NodeId]>,
    outcome: u64,
    // did THIS node write the outcome's name and payoffs, or reference it by id?
    outcome_declared: bool,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct RawTerminal<'a> {
    name: &'a EscapedStr,
    outcome: u64,
    // did THIS node write the outcome's name and payoffs, or reference it by id?
    outcome_declared: bool,
}

/// A full extensive form game
///
/// This can be parsed from a [str] reference using [`ExtensiveFormGame::try_from_str`] or using the
/// [`TryFrom`] / [`TryInto`] traits. It implements [Display] for formatting.
///
/// # Example
///
/// ```
/// # use gambit_parser::ExtensiveFormGame;
/// let gambit = r#"EFG 2 R "" { "1" "2" } t "" 1 "" { 1 2 }"#;
/// let game: ExtensiveFormGame<'_> = gambit.try_into().unwrap();
/// let output = game.to_string();
/// ```
#[derive(Debug, PartialEq, Clone)]
pub struct ExtensiveFormGame<'a> {
    name: &'a EscapedStr,
    player_names: Box<[&'a EscapedStr]>,
    comment: Option<&'a EscapedStr>,
    infosets: Infosets<'a>,
    outcomes: Outcomes<'a>,
    nodes: Box<[RawNode<'a>]>,
    root: NodeId,
}

impl<'a> ExtensiveFormGame<'a> {
    /// The name of the game
    #[must_use]
    pub fn name(&self) -> &'a EscapedStr {
        self.name
    }

    /// Names for every player, in order
    #[must_use]
    pub fn player_names(&self) -> &[&'a EscapedStr] {
        &self.player_names
    }

    /// An optional game comment
    #[must_use]
    pub fn comment(&self) -> Option<&'a EscapedStr> {
        self.comment
    }

    /// The root node of the game tree
    #[must_use]
    pub fn root<'g>(&'g self) -> Node<'a, 'g> {
        self.wrap(self.root)
    }

    /// Adapt this game for writing in a given [`WriteMode`]; the result implements [`Display`].
    ///
    /// Plain `Display` (and `to_string`) uses [`WriteMode::Faithful`].
    #[must_use]
    pub fn display<'g>(&'g self, mode: WriteMode) -> GameDisplay<'a, 'g> {
        GameDisplay { game: self, mode }
    }

    fn wrap<'g>(&'g self, id: NodeId) -> Node<'a, 'g> {
        match &self.nodes[id.0] {
            RawNode::Chance(raw) => Node::Chance(Chance { game: self, raw }),
            RawNode::Player(raw) => Node::Player(Player { game: self, raw }),
            RawNode::Terminal(raw) => Node::Terminal(Terminal { game: self, raw }),
        }
    }

    /// The outcome's name, or `None` for the null (0) outcome
    fn outcome_name(&self, outcome: u64) -> Option<&'a EscapedStr> {
        self.outcomes.get(&outcome).map(|(name, _)| *name)
    }

    /// The outcome's payoffs, or `None` for the null (0) outcome
    fn outcome_payoffs(&self, outcome: u64) -> Option<&[BigRational]> {
        self.outcomes.get(&outcome).map(|(_, payoffs)| &payoffs[..])
    }
}

impl Display for ExtensiveFormGame<'_> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        self.display(WriteMode::Faithful).fmt(out)
    }
}

/// How much of each shared infoset and outcome to write when serializing a game.
///
/// Both are declared once and referenced by id, so a given node may or may not repeat the block.
/// This selects which nodes write it. See [`ExtensiveFormGame::display`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteMode {
    /// Declare each infoset and outcome only on its first appearance; reference it by id after.
    Minimal,
    /// Reproduce the parsed input: write a block exactly where the source node wrote one.
    Faithful,
    /// Write every infoset and outcome in full on every node, as Gambit's own writer does.
    Exhaustive,
}

/// A [`Display`] adapter that writes a game in a chosen [`WriteMode`], returned by
/// [`ExtensiveFormGame::display`].
#[derive(Clone, Copy)]
pub struct GameDisplay<'a, 'g> {
    game: &'g ExtensiveFormGame<'a>,
    mode: WriteMode,
}

impl GameDisplay<'_, '_> {
    /// Whether a node should write its outcome's full definition rather than reference it by id
    fn declares_outcome(self, outcome: u64, declared: bool, seen: &mut HashSet<u64>) -> bool {
        match self.mode {
            WriteMode::Minimal => outcome != 0 && seen.insert(outcome),
            WriteMode::Faithful => declared,
            WriteMode::Exhaustive => outcome != 0,
        }
    }
}

impl Display for GameDisplay<'_, '_> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        let game = self.game;
        write!(out, "EFG 2 R \"{}\" {{ ", game.name.escape())?;
        for name in &game.player_names {
            write!(out, "\"{}\" ", name.escape())?;
        }
        writeln!(out, "}}")?;
        if let Some(comment) = game.comment {
            writeln!(out, "\"{}\"", comment.escape())?;
        }

        // Minimal writes each block the first time its id is seen and references it afterward; only
        // it reads these sets, so reserve their exact block counts for Minimal and leave the other
        // modes' sets unallocated
        let mut chance_seen = HashSet::new();
        let mut player_seen = HashSet::new();
        let mut outcome_seen = HashSet::new();
        if self.mode == WriteMode::Minimal {
            chance_seen.reserve(game.infosets.chance.len());
            player_seen.reserve(game.infosets.player.iter().map(HashMap::len).sum());
            outcome_seen.reserve(game.outcomes.len());
        }
        let mut stack = vec![game.root];
        while let Some(id) = stack.pop() {
            match &game.nodes[id.0] {
                RawNode::Chance(raw) => {
                    write!(out, "\nc \"{}\" {}", raw.name.escape(), raw.infoset)?;
                    let block = match self.mode {
                        WriteMode::Minimal => chance_seen.insert(raw.infoset),
                        WriteMode::Faithful => raw.declared,
                        WriteMode::Exhaustive => true,
                    };
                    if block {
                        let (label, actions) = &game.infosets.chance[&raw.infoset];
                        write!(out, " \"{}\" {{ ", label.escape())?;
                        for (action, prob) in actions {
                            write!(out, "\"{}\" {} ", action.escape(), prob)?;
                        }
                        write!(out, "}}")?;
                    }
                    let declared =
                        self.declares_outcome(raw.outcome, raw.outcome_declared, &mut outcome_seen);
                    write_outcome(out, game, raw.outcome, declared)?;
                    stack.extend(raw.children.iter().rev().copied());
                }
                RawNode::Player(raw) => {
                    write!(
                        out,
                        "\np \"{}\" {} {}",
                        raw.name.escape(),
                        raw.player_num,
                        raw.infoset
                    )?;
                    let block = match self.mode {
                        WriteMode::Minimal => player_seen.insert((raw.player_num, raw.infoset)),
                        WriteMode::Faithful => raw.declared,
                        WriteMode::Exhaustive => true,
                    };
                    if block {
                        let (label, actions) =
                            &game.infosets.player[raw.player_num - 1][&raw.infoset];
                        write!(out, " \"{}\" {{ ", label.escape())?;
                        for action in actions {
                            write!(out, "\"{}\" ", action.escape())?;
                        }
                        write!(out, "}}")?;
                    }
                    let declared =
                        self.declares_outcome(raw.outcome, raw.outcome_declared, &mut outcome_seen);
                    write_outcome(out, game, raw.outcome, declared)?;
                    stack.extend(raw.children.iter().rev().copied());
                }
                RawNode::Terminal(raw) => {
                    write!(out, "\nt \"{}\"", raw.name.escape())?;
                    let declared =
                        self.declares_outcome(raw.outcome, raw.outcome_declared, &mut outcome_seen);
                    write_outcome(out, game, raw.outcome, declared)?;
                }
            }
        }
        writeln!(out)
    }
}

/// An error that happens while trying to turn a string into an [`ExtensiveFormGame`]
#[derive(Debug)]
#[non_exhaustive]
pub enum Error<'a> {
    /// A problem with parsing
    ///
    /// This will show the remainder of the string where the parse error occurred
    Parse(&'a str),
    /// A well-formed line that makes the game inconsistent (a mismatched infoset, an undefined
    /// outcome, and so on)
    Validation(ValidationError),
}

impl Display for Error<'_> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Error::Parse(rem) => write!(fmt, "error parsing game at: '{rem}'"),
            Error::Validation(err) => write!(fmt, "invalid efg: {err}"),
        }
    }
}

impl StdError for Error<'_> {}

/// An error that results from something invalid about the parsed extensive form game
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidationError {
    /// A player's number wasn't between one and the number of players
    InvalidPlayerNum,
    /// An infoset had different names attached to it
    NonMatchingInfosetNames,
    /// An infoset had different sets of associated actions
    NonMatchingInfosetActions,
    /// A name or payoffs were attached to the null (0) outcome
    NullOutcomePayoffs,
    /// The number of specified payoffs did not match the number of players
    InvalidNumberOfPayoffs,
    /// An outcome was defined with a name that didn't match its first definition
    NonMatchingOutcomeNames,
    /// An outcome was defined with payoffs that didn't match its first definition
    NonMatchingOutcomePayoffs,
    /// A node referenced an outcome that was never defined
    UndefinedOutcome,
    /// A node omitted its action list for an infoset that was never declared
    UndeclaredInfoset,
}

impl Display for ValidationError {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(fmt, "{self:?}")
    }
}

impl From<ValidationError> for Error<'_> {
    fn from(err: ValidationError) -> Self {
        Error::Validation(err)
    }
}

impl<'a> From<nom::Err<nom::error::Error<&'a str>>> for Error<'a> {
    fn from(err: nom::Err<nom::error::Error<&'a str>>) -> Self {
        match err {
            nom::Err::Incomplete(_) => panic!("internal error: incomplete parsing"),
            nom::Err::Error(err) | nom::Err::Failure(err) => Error::Parse(err.input),
        }
    }
}

impl<'a> ExtensiveFormGame<'a> {
    /// Try to parse a game from a string
    ///
    /// This is identical to `ExtensiveFormGame::try_from` or `"...".try_into()`.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if the input isn't a syntactically valid and self-consistent game.
    pub fn try_from_str(input: &'a str) -> Result<Self, Error<'a>> {
        let (rest, game) = parse_game(input)?;
        let rest = rest.trim_start();
        if !rest.is_empty() {
            return Err(Error::Parse(rest));
        }
        Ok(game)
    }
}

impl<'a> TryFrom<&'a str> for ExtensiveFormGame<'a> {
    type Error = Error<'a>;

    fn try_from(input: &'a str) -> Result<Self, Self::Error> {
        Self::try_from_str(input)
    }
}

/// An arbitrary node in the game tree
///
/// A handle that pairs a node with its game so it can resolve its infoset. These are cheap to copy.
#[derive(Clone, Copy)]
pub enum Node<'a, 'g> {
    /// A chance node
    Chance(Chance<'a, 'g>),
    /// A player node
    Player(Player<'a, 'g>),
    /// A terminal node
    Terminal(Terminal<'a, 'g>),
}

/// Write a node's outcome: always the id, plus the name and payoffs when this node declared them
/// (a node that only referenced the outcome by id writes just the id, matching the file)
fn write_outcome(
    out: &mut Formatter<'_>,
    game: &ExtensiveFormGame<'_>,
    outcome: u64,
    declared: bool,
) -> Result<(), FmtError> {
    write!(out, " {outcome}")?;
    if declared {
        if let Some(name) = game.outcome_name(outcome) {
            write!(out, " \"{}\"", name.escape())?;
        }
        if let Some(payoffs) = game.outcome_payoffs(outcome) {
            write!(out, " {{ ")?;
            for payoff in payoffs {
                write!(out, "{payoff} ")?;
            }
            write!(out, "}}")?;
        }
    }
    Ok(())
}

/// A chance node
///
/// A chance node represents a point in the game where things advance randomly, or alternatively,
/// where "nature" takes a turn.
#[derive(Clone, Copy)]
pub struct Chance<'a, 'g> {
    game: &'g ExtensiveFormGame<'a>,
    raw: &'g RawChance<'a>,
}

impl<'a, 'g> Chance<'a, 'g> {
    fn entry(self) -> &'g ChanceInfoset<'a> {
        &self.game.infosets.chance[&self.raw.infoset]
    }

    /// The name of the node
    #[must_use]
    pub fn name(self) -> &'a EscapedStr {
        self.raw.name
    }

    /// The id of the node's infoset
    #[must_use]
    pub fn infoset(self) -> u64 {
        self.raw.infoset
    }

    /// The infoset's label
    #[must_use]
    pub fn infoset_name(self) -> &'a EscapedStr {
        self.entry().0
    }

    /// All possible actions with their names, probabilities, and resulting nodes
    pub fn actions(
        self,
    ) -> impl Iterator<Item = (&'a EscapedStr, &'g BigRational, Node<'a, 'g>)> + 'g {
        let (_, actions) = self.entry();
        let game = self.game;
        actions
            .iter()
            .zip(self.raw.children.iter())
            .map(move |((label, prob), &child)| (*label, prob, game.wrap(child)))
    }

    /// The probability and child for the action with the given label
    ///
    /// `label` is matched against the unescaped form of each action's label. Labels need not be
    /// unique within an infoset, so this returns the first match.
    #[must_use]
    pub fn action(self, label: &str) -> Option<(&'g BigRational, Node<'a, 'g>)> {
        self.actions()
            .find(|(name, _, _)| name.unescape().eq(label.chars()))
            .map(|(_, prob, next)| (prob, next))
    }

    /// The number of actions (always at least one)
    #[allow(clippy::len_without_is_empty)]
    #[must_use]
    pub fn len(self) -> usize {
        self.raw.children.len()
    }

    /// The name, probability, and child for the action at the given index
    #[must_use]
    pub fn action_at(
        self,
        index: usize,
    ) -> Option<(&'a EscapedStr, &'g BigRational, Node<'a, 'g>)> {
        let (_, actions) = self.entry();
        let (label, prob) = actions.get(index)?;
        let &child = self.raw.children.get(index)?;
        Some((*label, prob, self.game.wrap(child)))
    }

    /// The outcome id
    #[must_use]
    pub fn outcome(self) -> u64 {
        self.raw.outcome
    }

    /// The name of the outcome
    ///
    /// `None` for the null (0) outcome.
    #[must_use]
    pub fn outcome_name(self) -> Option<&'a EscapedStr> {
        self.game.outcome_name(self.raw.outcome)
    }

    /// Outcome payoffs for this node
    ///
    /// These are added to every player's payoff for traversing through this node. `None` only for
    /// the null (0) outcome.
    #[must_use]
    pub fn outcome_payoffs(self) -> Option<&'g [BigRational]> {
        self.game.outcome_payoffs(self.raw.outcome)
    }
}

/// A player node in the game tree
///
/// A player node represents a place where one of the players chooses what happens next.
#[derive(Clone, Copy)]
pub struct Player<'a, 'g> {
    game: &'g ExtensiveFormGame<'a>,
    raw: &'g RawPlayer<'a>,
}

impl<'a, 'g> Player<'a, 'g> {
    fn entry(self) -> &'g PlayerInfoset<'a> {
        &self.game.infosets.player[self.raw.player_num - 1][&self.raw.infoset]
    }

    /// The name of the node
    #[must_use]
    pub fn name(self) -> &'a EscapedStr {
        self.raw.name
    }

    /// The player acting at this node
    ///
    /// This will always be between 1 and the number of players.
    #[must_use]
    pub fn player_num(self) -> usize {
        self.raw.player_num
    }

    /// The infoset id for this node and player
    #[must_use]
    pub fn infoset(self) -> u64 {
        self.raw.infoset
    }

    /// The infoset's label
    #[must_use]
    pub fn infoset_name(self) -> &'a EscapedStr {
        self.entry().0
    }

    /// All the actions a player can take with their names and resulting nodes
    pub fn actions(self) -> impl Iterator<Item = (&'a EscapedStr, Node<'a, 'g>)> + 'g {
        let (_, labels) = self.entry();
        let game = self.game;
        labels
            .iter()
            .zip(self.raw.children.iter())
            .map(move |(label, &child)| (*label, game.wrap(child)))
    }

    /// The child reached by the action with the given label
    ///
    /// `label` is matched against the unescaped form of each action's label. Labels need not be
    /// unique within an infoset, so this returns the first match.
    #[must_use]
    pub fn action(self, label: &str) -> Option<Node<'a, 'g>> {
        self.actions()
            .find(|(name, _)| name.unescape().eq(label.chars()))
            .map(|(_, next)| next)
    }

    /// The number of actions (always at least one)
    #[allow(clippy::len_without_is_empty)]
    #[must_use]
    pub fn len(self) -> usize {
        self.raw.children.len()
    }

    /// The name and child for the action at the given index
    #[must_use]
    pub fn action_at(self, index: usize) -> Option<(&'a EscapedStr, Node<'a, 'g>)> {
        let (_, actions) = self.entry();
        let &label = actions.get(index)?;
        let &child = self.raw.children.get(index)?;
        Some((label, self.game.wrap(child)))
    }

    /// The outcome id
    #[must_use]
    pub fn outcome(self) -> u64 {
        self.raw.outcome
    }

    /// The name of the outcome
    ///
    /// `None` for the null (0) outcome.
    #[must_use]
    pub fn outcome_name(self) -> Option<&'a EscapedStr> {
        self.game.outcome_name(self.raw.outcome)
    }

    /// Payoffs associated with the outcome
    ///
    /// `None` only for the null (0) outcome.
    #[must_use]
    pub fn outcome_payoffs(self) -> Option<&'g [BigRational]> {
        self.game.outcome_payoffs(self.raw.outcome)
    }
}

/// A terminal node represents the end of a game
///
/// Terminal nodes simply assign payoffs to every player in the game
#[derive(Clone, Copy)]
pub struct Terminal<'a, 'g> {
    game: &'g ExtensiveFormGame<'a>,
    raw: &'g RawTerminal<'a>,
}

impl<'a, 'g> Terminal<'a, 'g> {
    /// The name of this node
    #[must_use]
    pub fn name(self) -> &'a EscapedStr {
        self.raw.name
    }

    /// The outcome id
    #[must_use]
    pub fn outcome(self) -> u64 {
        self.raw.outcome
    }

    /// The name of this outcome
    ///
    /// `None` for the null (0) outcome.
    #[must_use]
    pub fn outcome_name(self) -> Option<&'a EscapedStr> {
        self.game.outcome_name(self.raw.outcome)
    }

    /// The payoffs to every player
    ///
    /// `None` only for the null (0) outcome — a terminal with no outcome attached.
    #[must_use]
    pub fn outcome_payoffs(self) -> Option<&'g [BigRational]> {
        self.game.outcome_payoffs(self.raw.outcome)
    }
}

fn negate(input: &str) -> IResult<&str, bool> {
    let (input, res) = opt(one_of("+-")).parse(input)?;
    Ok((input, res == Some('-')))
}

fn fail(input: &str) -> nom::Err<nom::error::Error<&str>> {
    nom::Err::Error(nom::error::Error::new(input, ErrorKind::Fail))
}

/// The largest exponent magnitude accepted, bounding the cost of the exact `10^exp` below.
const MAX_ABS_EXPONENT: i32 = 10_000;

fn big_float(input: &str) -> IResult<&str, BigRational> {
    let (res_input, (main_neg, (int, dec), exp)) = (
        negate,
        alt((
            pair(
                digit1,
                map(opt(preceded(char('.'), digit0)), Option::unwrap_or_default),
            ),
            separated_pair(digit0, char('.'), digit1),
        )),
        opt(preceded(one_of("eE"), pair(negate, digit1))),
    )
        .parse(input)?;
    let mut res = if int.is_empty() {
        BigRational::zero()
    } else {
        BigRational::from_integer(int.parse().unwrap())
    };
    if !dec.is_empty() {
        let pow: u32 = dec.len().try_into().map_err(|_| fail(input))?;
        res += BigRational::new(dec.parse().unwrap(), BigInt::from(10).pow(pow));
    }
    if let Some((neg, exp)) = exp {
        let exp: i32 = exp.parse().map_err(|_| fail(input))?;
        if exp > MAX_ABS_EXPONENT {
            return Err(fail(input));
        }
        res *= BigRational::from_integer(10.into()).pow(if neg { -exp } else { exp });
    }
    if main_neg {
        res = -res;
    }
    Ok((res_input, res))
}

fn big_rational(input: &str) -> IResult<&str, BigRational> {
    let (rest, (num, denom)) = pair(big_float, opt(preceded(char('/'), big_float))).parse(input)?;
    match denom {
        // a zero denominator would panic in num-rational's `Div`; reject it as a parse error
        Some(denom) if denom.is_zero() => Err(fail(input)),
        Some(denom) => Ok((rest, num / denom)),
        None => Ok((rest, num)),
    }
}

fn label(input: &str) -> IResult<&str, &EscapedStr> {
    map(
        delimited(
            char('"'),
            // the body is any mix of the `\"` escape and ordinary non-quote characters; matching
            // `\"` first means a lone `\` is ordinary and only a `"` after a `\` is escaped
            recognize(many0(alt((tag(r#"\""#), recognize(none_of("\"")))))),
            char('"'),
        ),
        EscapedStr::new,
    )
    .parse(input)
}

fn spacelist<'a, O, E, F>(f: F) -> impl Parser<&'a str, Output = Vec<O>, Error = E>
where
    F: Parser<&'a str, Output = O, Error = E>,
    E: ParseError<&'a str>,
{
    delimited(
        pair(char('{'), multispace0),
        separated_list1(multispace1, f),
        pair(multispace0, char('}')),
    )
}

fn commalist<'a, O, E, F>(f: F) -> impl Parser<&'a str, Output = Vec<O>, Error = E>
where
    F: Parser<&'a str, Output = O, Error = E>,
    E: ParseError<&'a str>,
{
    delimited(
        pair(char('{'), multispace0),
        separated_list1((multispace0, opt(char(',')), multispace0), f),
        pair(multispace0, char('}')),
    )
}

/// A parent node whose header is parsed but whose children are still being collected.
struct PendingNode<'a> {
    node: RawNode<'a>,
    child_count: usize,
    children: Vec<NodeId>,
}

impl<'a> PendingNode<'a> {
    fn finish(self) -> RawNode<'a> {
        let PendingNode {
            mut node, children, ..
        } = self;
        match &mut node {
            RawNode::Chance(chance) => chance.children = children.into(),
            RawNode::Player(player) => player.children = children.into(),
            // only chance and player nodes gather children, so only they are ever pending
            RawNode::Terminal(_) => unreachable!("terminal nodes are never pending"),
        }
        node
    }
}

/// Record or check an infoset declaration, returning whether the block was written here and the
/// action count. A first declaration is inserted, a repeat must match exactly, an omission inherits.
fn resolve_infoset<'a, A: PartialEq>(
    map: &mut HashMap<u64, (&'a EscapedStr, Box<[A]>)>,
    infoset: u64,
    declared: Option<(&'a EscapedStr, Vec<A>)>,
) -> Result<(bool, usize), Error<'a>> {
    if let Some((name, actions)) = declared {
        match map.entry(infoset) {
            // a first declaration is stored (boxed), a repeat is only compared against the stored
            // one, so the parsed `Vec` is never boxed just to be dropped
            Entry::Vacant(ent) => {
                let count = actions.len();
                ent.insert((name, actions.into()));
                Ok((true, count))
            }
            Entry::Occupied(ent) => {
                let (stored_name, stored_actions) = ent.get();
                if *stored_name != name {
                    Err(ValidationError::NonMatchingInfosetNames.into())
                } else if **stored_actions != *actions {
                    Err(ValidationError::NonMatchingInfosetActions.into())
                } else {
                    Ok((true, actions.len()))
                }
            }
        }
    } else {
        let (_, actions) = map
            .get(&infoset)
            .ok_or(ValidationError::UndeclaredInfoset)?;
        Ok((false, actions.len()))
    }
}

/// Record or check a node's outcome, mirroring [`resolve_infoset`]. A node either defines the
/// outcome with a name and payoffs, or references it by bare id: the first definition is stored, a
/// repeat definition must match it exactly, and a reference must name an already-defined outcome.
/// The null (0) outcome carries no data and is never stored.
fn resolve_outcome<'a>(
    outcomes: &mut Outcomes<'a>,
    num_players: usize,
    outcome: u64,
    definition: Option<(&'a EscapedStr, Vec<BigRational>)>,
) -> Result<(), Error<'a>> {
    if let Some((name, payoffs)) = definition {
        if outcome == 0 {
            Err(ValidationError::NullOutcomePayoffs.into())
        } else if payoffs.len() != num_players {
            Err(ValidationError::InvalidNumberOfPayoffs.into())
        } else {
            match outcomes.entry(outcome) {
                // a first definition is stored (boxed), a repeat is only compared against the
                // stored one, so the parsed `Vec` is never boxed just to be dropped
                Entry::Vacant(ent) => {
                    ent.insert((name, payoffs.into()));
                    Ok(())
                }
                Entry::Occupied(ent) => {
                    let (stored_name, stored_payoffs) = ent.get();
                    if *stored_name != name {
                        Err(ValidationError::NonMatchingOutcomeNames.into())
                    } else if **stored_payoffs != *payoffs {
                        Err(ValidationError::NonMatchingOutcomePayoffs.into())
                    } else {
                        Ok(())
                    }
                }
            }
        }
    } else if outcome != 0 && !outcomes.contains_key(&outcome) {
        // a bare id references an outcome that must already be defined; the null (0) id is fine
        Err(ValidationError::UndefinedOutcome.into())
    } else {
        Ok(())
    }
}

/// Parse the whole game tree into a flat arena, returning the nodes and the root's id.
fn parse_tree<'a>(
    mut input: &'a str,
    infosets: &mut Infosets<'a>,
    outcomes: &mut Outcomes<'a>,
    num_players: usize,
) -> Result<(&'a str, Box<[RawNode<'a>]>, NodeId), Error<'a>> {
    // finished nodes, in the post-order they complete (every child precedes its parent)
    let mut nodes: Vec<RawNode<'a>> = Vec::new();
    // parents still gathering their children
    let mut stack: Vec<PendingNode<'a>> = Vec::new();

    loop {
        let (rest, style) = preceded(multispace1, one_of("cpt")).parse(input)?;
        input = rest;
        // a chance or player node opens a frame; a terminal completes immediately
        let mut completed = match style {
            'c' => {
                let (rest, chance, child_count) =
                    parse_chance(input, infosets, outcomes, num_players)?;
                input = rest;
                stack.push(PendingNode {
                    node: RawNode::Chance(chance),
                    child_count,
                    children: Vec::with_capacity(child_count),
                });
                continue;
            }
            'p' => {
                let (rest, player, child_count) =
                    parse_player(input, infosets, outcomes, num_players)?;
                input = rest;
                stack.push(PendingNode {
                    node: RawNode::Player(player),
                    child_count,
                    children: Vec::with_capacity(child_count),
                });
                continue;
            }
            't' => {
                let (rest, term) = parse_terminal(input, outcomes, num_players)?;
                input = rest;
                push_node(&mut nodes, RawNode::Terminal(term))
            }
            // `one_of("cpt")` only ever yields one of these three characters
            _ => unreachable!(),
        };

        // attach the finished node to its waiting parent, finishing parents that fill up in turn
        loop {
            let Some(pending) = stack.last_mut() else {
                // nothing is waiting, so this node is the root and the tree is complete
                return Ok((input, nodes.into(), completed));
            };
            pending.children.push(completed);
            if pending.children.len() < pending.child_count {
                break;
            }
            completed = push_node(&mut nodes, stack.pop().unwrap().finish());
        }
    }
}

/// Append a finished node to the arena and return its id
fn push_node<'a>(nodes: &mut Vec<RawNode<'a>>, node: RawNode<'a>) -> NodeId {
    let id = NodeId(nodes.len());
    nodes.push(node);
    id
}

/// Parse a chance node's header, resolving its outcome and returning the node (children still
/// empty) and its child count
fn parse_chance<'a>(
    input: &'a str,
    infosets: &mut Infosets<'a>,
    outcomes: &mut Outcomes<'a>,
    num_players: usize,
) -> Result<(&'a str, RawChance<'a>, usize), Error<'a>> {
    let (input, (name, infoset, declared, outcome, definition)) = (
        preceded(multispace1, label),
        preceded(multispace1, u64),
        opt((
            preceded(multispace1, label),
            preceded(
                multispace1,
                spacelist(separated_pair(label, multispace1, big_rational)),
            ),
        )),
        preceded(multispace1, u64),
        // an outcome is either a bare id or a name paired with payoffs (see resolve_outcome)
        opt((
            preceded(multispace1, label),
            preceded(multispace1, commalist(big_rational)),
        )),
    )
        .parse(input)?;
    let (declared, child_count) = resolve_infoset(&mut infosets.chance, infoset, declared)?;
    let outcome_declared = definition.is_some();
    resolve_outcome(outcomes, num_players, outcome, definition)?;
    Ok((
        input,
        RawChance {
            name,
            infoset,
            declared,
            // filled once the following child nodes are parsed (see PendingNode::finish)
            children: Box::default(),
            outcome,
            outcome_declared,
        },
        child_count,
    ))
}

/// Parse a player node's header, resolving its outcome and returning the node (children still
/// empty) and its child count
fn parse_player<'a>(
    input: &'a str,
    infosets: &mut Infosets<'a>,
    outcomes: &mut Outcomes<'a>,
    num_players: usize,
) -> Result<(&'a str, RawPlayer<'a>, usize), Error<'a>> {
    let (input, (name, player_num, infoset, declared, outcome, definition)) = (
        preceded(multispace1, label),
        preceded(multispace1, u64),
        preceded(multispace1, u64),
        opt((
            preceded(multispace1, label),
            preceded(multispace1, spacelist(label)),
        )),
        preceded(multispace1, u64),
        // an outcome is either a bare id or a name paired with payoffs (see resolve_outcome)
        opt((
            preceded(multispace1, label),
            preceded(multispace1, commalist(big_rational)),
        )),
    )
        .parse(input)?;
    let player_num: usize = player_num.try_into().map_err(|_| fail(input))?;
    // checked here, since the per-player infoset map is indexed by it
    if player_num == 0 || player_num > infosets.player.len() {
        return Err(ValidationError::InvalidPlayerNum.into());
    }
    let (declared, child_count) =
        resolve_infoset(&mut infosets.player[player_num - 1], infoset, declared)?;
    let outcome_declared = definition.is_some();
    resolve_outcome(outcomes, num_players, outcome, definition)?;
    Ok((
        input,
        RawPlayer {
            name,
            player_num,
            infoset,
            declared,
            // filled once the following child nodes are parsed (see PendingNode::finish)
            children: Box::default(),
            outcome,
            outcome_declared,
        },
        child_count,
    ))
}

/// Parse a terminal node, resolving its outcome
fn parse_terminal<'a>(
    input: &'a str,
    outcomes: &mut Outcomes<'a>,
    num_players: usize,
) -> Result<(&'a str, RawTerminal<'a>), Error<'a>> {
    let (input, (name, outcome, definition)) = (
        preceded(multispace1, label),
        preceded(multispace1, u64),
        // an outcome is either a bare id or a name paired with payoffs (see resolve_outcome)
        opt((
            preceded(multispace1, label),
            preceded(multispace1, commalist(big_rational)),
        )),
    )
        .parse(input)?;
    let outcome_declared = definition.is_some();
    resolve_outcome(outcomes, num_players, outcome, definition)?;
    Ok((
        input,
        RawTerminal {
            name,
            outcome,
            outcome_declared,
        },
    ))
}

fn parse_game(input: &str) -> Result<(&str, ExtensiveFormGame<'_>), Error<'_>> {
    let (input, (name, player_names, comment)) = (
        preceded(
            (
                multispace0,
                tag("EFG"),
                multispace1,
                tag("2"),
                multispace1,
                // Gambit accepts either data-type letter; `D` is legacy but still circulates
                one_of("RD"),
                multispace1,
            ),
            label,
        ),
        preceded(multispace1, spacelist(label)),
        opt(preceded(multispace1, label)),
    )
        .parse(input)?;
    let num_players = player_names.len();
    let mut infosets = Infosets {
        player: (0..num_players).map(|_| HashMap::new()).collect(),
        chance: HashMap::new(),
    };
    let mut outcomes = Outcomes::new();
    let (input, nodes, root) = parse_tree(input, &mut infosets, &mut outcomes, num_players)?;
    Ok((
        input,
        ExtensiveFormGame {
            name,
            player_names: player_names.into(),
            comment,
            infosets,
            outcomes,
            nodes,
            root,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::{Error, EscapedStr, ExtensiveFormGame, Node, ValidationError, WriteMode};
    use num_rational::BigRational;
    use num_traits::One;

    /// Parse a game expected to fail validation (or a parse-time infoset check) and return the error
    fn validation_err(game: &str) -> ValidationError {
        match ExtensiveFormGame::try_from_str(game) {
            Err(Error::Validation(err)) => err,
            other => panic!("expected a validation error, got {other:?}"),
        }
    }

    #[test]
    fn test_big_float() {
        let (input, num) = super::big_float("3 ").unwrap();
        assert_eq!(input, " ");
        assert_eq!(num, BigRational::from_integer(3.into()));

        let (input, num) = super::big_float("-2. ").unwrap();
        assert_eq!(input, " ");
        assert_eq!(num, BigRational::from_integer((-2).into()));

        let (input, num) = super::big_float("+.56 ").unwrap();
        assert_eq!(input, " ");
        assert_eq!(num, BigRational::new(56.into(), 100.into()));

        let (input, num) = super::big_float("3.14e-1 ").unwrap();
        assert_eq!(input, " ");
        assert_eq!(num, BigRational::new(314.into(), 1000.into()));
    }

    #[test]
    fn test_big_rational() {
        let (input, num) = super::big_rational("3 ").unwrap();
        assert_eq!(input, " ");
        assert_eq!(num, BigRational::from_integer(3.into()));

        let (input, num) = super::big_rational("99/100 ").unwrap();
        assert_eq!(input, " ");
        assert_eq!(num, BigRational::new(99.into(), 100.into()));

        let (input, num) = super::big_rational(".1e3/+1.e2 ").unwrap();
        assert_eq!(input, " ");
        assert_eq!(num, BigRational::one());
    }

    #[test]
    fn test_label() {
        let (input, label) = super::label(r#""" "#).unwrap();
        assert_eq!(input, " ");
        assert_eq!(label.escape(), "");

        let (input, label) = super::label(r#""normal" "#).unwrap();
        assert_eq!(input, " ");
        assert_eq!(label.escape(), "normal");

        // `\"` is an escaped quote and does not close the label
        let (input, label) = super::label(r#""esca\"ped" "#).unwrap();
        assert_eq!(input, " ");
        assert_eq!(label.escape(), r#"esca\"ped"#);

        // a backslash before a non-quote is kept; the final `"` (preceded by `h`) closes the label
        let (input, label) = super::label(r#""back\slash" "#).unwrap();
        assert_eq!(input, " ");
        assert_eq!(label.escape(), r"back\slash");

        // a `\` always escapes the immediately following `"`, so a label whose closing quote is
        // preceded by a backslash is unterminated (matching gambit)
        assert!(super::label(r#""pair\\" "#).is_err());
        assert!(super::label(r#""unterminated"#).is_err());
        assert!(super::label("noquote").is_err());
    }

    #[test]
    fn simple_test() {
        let game_str = r#"
        EFG 2 R "General Bayes game, one stage" { "Player 1" "Player 2" }
        "A single stage General Bayes Game"

        c "ROOT" 1 "(0,1)" { "1G" 0.500000 "1B" 0.500000 } 0
        p "" 1 1 "(1,1)" { "H" "L" } 0
        t "" 1 "Outcome 1" { 10.000000 2.000000 }
        t "" 2 "Outcome 2" { 0.000000 10.000000 }
        p "" 2 1 "(2,1)" { "h" "l" } 0
        t "" 3 "Outcome 3" { 2.000000 4.000000 }
        t "" 4 "Outcome 4" { 4.000000 0.000000 }
        "#;
        let game = ExtensiveFormGame::try_from_str(game_str).unwrap();
        assert_eq!(
            game.to_string(),
            r#"EFG 2 R "General Bayes game, one stage" { "Player 1" "Player 2" }
"A single stage General Bayes Game"

c "ROOT" 1 "(0,1)" { "1G" 1/2 "1B" 1/2 } 0
p "" 1 1 "(1,1)" { "H" "L" } 0
t "" 1 "Outcome 1" { 10 2 }
t "" 2 "Outcome 2" { 0 10 }
p "" 2 1 "(2,1)" { "h" "l" } 0
t "" 3 "Outcome 3" { 2 4 }
t "" 4 "Outcome 4" { 4 0 }
"#
        );

        // spot-check a few handle accessors
        assert_eq!(game.name().to_string(), "General Bayes game, one stage");
        assert_eq!(game.player_names().len(), 2);
        let Node::Chance(root) = game.root() else {
            panic!("expected a chance root");
        };
        let labels: Vec<_> = root.actions().map(|(label, _, _)| label.escape()).collect();
        assert_eq!(labels, ["1G", "1B"]);
    }

    #[test]
    fn navigates_handles() {
        let game_str = r#"EFG 2 R "g" { "Player 1" "Player 2" }
p "root" 1 1 "iset" { "L" "R" } 0
t "tl" 1 "o1" { 1 2 }
t "tr" 2 "o2" { 3 4 }
"#;
        let game = ExtensiveFormGame::try_from_str(game_str).unwrap();
        let Node::Player(root) = game.root() else {
            panic!("expected a player root");
        };
        assert_eq!(root.player_num(), 1);
        assert_eq!(root.infoset(), 1);
        assert_eq!(root.infoset_name().escape(), "iset");
        let labels: Vec<_> = root.actions().map(|(label, _)| label.escape()).collect();
        assert_eq!(labels, ["L", "R"]);

        let Some(Node::Terminal(left)) = root.action("L") else {
            panic!("expected a terminal after action L");
        };
        assert_eq!(left.name().escape(), "tl");
        assert_eq!(left.outcome(), 1);
        assert_eq!(left.outcome_name().map(EscapedStr::escape), Some("o1"));
        let payoffs: Vec<_> = left
            .outcome_payoffs()
            .unwrap()
            .iter()
            .map(BigRational::to_string)
            .collect();
        assert_eq!(payoffs, ["1", "2"]);
    }

    #[test]
    fn chance_probabilities_need_not_sum_to_one() {
        // matching Gambit, chance probabilities are kept as written and not checked as a distribution
        let game = "EFG 2 R \"\" { \"1\" \"2\" }
c \"\" 1 \"a\" { \"x\" 9/10 } 0
t \"\" 1 \"\" { 0 0 }
";
        let parsed = ExtensiveFormGame::try_from_str(game).unwrap();
        let Node::Chance(root) = parsed.root() else {
            panic!("expected a chance root");
        };
        let (prob, _) = root.action("x").unwrap();
        assert_eq!(prob.to_string(), "9/10");
    }

    #[test]
    fn invalid_player_num() {
        // a player number above the player count is rejected at parse time
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 3 1 \"a\" { \"x\" } 0
t \"\" 1 { 0 0 }
"
            ),
            ValidationError::InvalidPlayerNum
        );
    }

    #[test]
    fn invalid_infoset_names() {
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 1 1 \"a\" { \"x\" } 0
p \"\" 1 1 \"b\" { \"x\" } 0
t \"\" 1 { 0 0 }
"
            ),
            ValidationError::NonMatchingInfosetNames
        );
    }

    #[test]
    fn invalid_chance_infoset_names() {
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
c \"\" 1 \"a\" { \"x\" 1 } 0
c \"\" 1 \"b\" { \"x\" 1 } 0
t \"\" 1 { 0 0 }
"
            ),
            ValidationError::NonMatchingInfosetNames
        );
    }

    #[test]
    fn invalid_infoset_actions() {
        // a reordered list no longer matches the first declaration, since order is significant
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 1 1 \"a\" { \"L\" \"R\" } 0
t \"\" 1 \"\" { 0 0 }
p \"\" 1 1 \"a\" { \"R\" \"L\" } 0
t \"\" 2 \"\" { 0 0 }
t \"\" 3 \"\" { 0 0 }
"
            ),
            ValidationError::NonMatchingInfosetActions
        );
    }

    #[test]
    fn invalid_chance_infoset_actions() {
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
c \"\" 1 \"a\" { \"x\" 1 } 0
c \"\" 1 \"a\" { \"y\" 1 } 0
t \"\" 1 { 0 0 }
"
            ),
            ValidationError::NonMatchingInfosetActions
        );
    }

    #[test]
    fn null_outcome_payoffs() {
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 1 1 \"a\" { \"x\" } 0 \"n\" { 0 0 }
t \"\" 1 { 0 0 }
"
            ),
            ValidationError::NullOutcomePayoffs
        );
    }

    #[test]
    fn invalid_payoff_number() {
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
t \"\" 1 \"\" { 0 }
"
            ),
            ValidationError::InvalidNumberOfPayoffs
        );
    }

    #[test]
    fn non_matching_outcome_names() {
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 1 1 \"a\" { \"x\" } 1 \"b\" { 0 0 }
t \"\" 1 \"c\" { 0 0 }
"
            ),
            ValidationError::NonMatchingOutcomeNames
        );
    }

    #[test]
    fn non_matching_outcome_payoffs() {
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 1 1 \"a\" { \"x\" } 1 \"\" { 0 0 }
t \"\" 1 \"\" { 1 1 }
"
            ),
            ValidationError::NonMatchingOutcomePayoffs
        );
    }

    #[test]
    fn undefined_outcome() {
        // referencing an outcome by bare id that was never defined is an error
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
t \"\" 5
"
            ),
            ValidationError::UndefinedOutcome
        );
    }

    #[test]
    fn undeclared_infoset() {
        // omitting the action list before the infoset has ever been declared is an error
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 1 1 0
t \"\" 1 { 0 0 }
"
            ),
            ValidationError::UndeclaredInfoset
        );
    }

    #[test]
    fn fills_omitted_action_list() {
        let game_str = "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 1 1 \"a\" { \"L\" \"R\" } 0
t \"\" 1 \"\" { 0 0 }
p \"\" 1 1 0
t \"\" 2 \"\" { 0 0 }
t \"\" 3 \"\" { 0 0 }
";
        let game = ExtensiveFormGame::try_from_str(game_str).unwrap();
        let Node::Player(root) = game.root() else {
            panic!("expected a player root");
        };
        let Some(Node::Player(omitted)) = root.action("R") else {
            panic!("expected a player after action R");
        };
        // the omitted node inherits the declared label and actions
        assert_eq!(omitted.infoset_name().escape(), "a");
        let labels: Vec<_> = omitted.actions().map(|(label, _)| label.escape()).collect();
        assert_eq!(labels, ["L", "R"]);
        // and the omitted form round-trips (the omission is preserved)
        let written = game.to_string();
        let reparsed = ExtensiveFormGame::try_from_str(written.as_str()).unwrap();
        assert_eq!(game, reparsed);
    }

    #[test]
    fn handle_accessors() {
        let game_str = r#"EFG 2 R "game" { "P1" "P2" } "the comment"
c "chance" 1 "ci" { "a" 1/2 "b" 1/2 } 5 "co" { 1 2 }
p "pl1" 1 1 "pi1" { "x" "y" } 6 "po1" { 3 4 }
t "ta" 1 "oa" { 7 8 }
t "tb" 2 "ob" { 9 10 }
p "pl2" 2 2 "pi2" { "x" "y" } 7 "po2" { 5 6 }
t "tc" 3 "oc" { 11 12 }
t "td" 4 "od" { 13 14 }
"#;
        let game = ExtensiveFormGame::try_from_str(game_str).unwrap();
        assert_eq!(game.comment().map(EscapedStr::escape), Some("the comment"));
        // Display covers every node kind's formatting and round-trips
        let written = game.to_string();
        let reparsed = ExtensiveFormGame::try_from_str(written.as_str()).unwrap();
        assert_eq!(game, reparsed);

        let Node::Chance(chance) = game.root() else {
            panic!("expected a chance root");
        };
        assert_eq!(chance.name().escape(), "chance");
        assert_eq!(chance.infoset(), 1);
        assert_eq!(chance.infoset_name().escape(), "ci");
        assert_eq!(chance.len(), 2);
        assert_eq!(chance.outcome(), 5);
        let chance_payoffs: Vec<_> = chance
            .outcome_payoffs()
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(chance_payoffs, ["1", "2"]);
        let chance_labels: Vec<_> = chance
            .actions()
            .map(|(label, _, _)| label.escape())
            .collect();
        assert_eq!(chance_labels, ["a", "b"]);
        assert!(chance.action_at(0).is_some());
        assert!(chance.action_at(2).is_none());
        assert!(chance.action("none").is_none());
        let (prob, first_child) = chance.action("a").unwrap();
        assert_eq!(prob.to_string(), "1/2");

        let Node::Player(player) = first_child else {
            panic!("expected a player after chance action a");
        };
        assert_eq!(player.name().escape(), "pl1");
        assert_eq!(player.player_num(), 1);
        assert_eq!(player.infoset(), 1);
        assert_eq!(player.infoset_name().escape(), "pi1");
        assert_eq!(player.len(), 2);
        assert_eq!(player.outcome(), 6);
        assert_eq!(player.outcome_name().map(EscapedStr::escape), Some("po1"));
        let player_payoffs: Vec<_> = player
            .outcome_payoffs()
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(player_payoffs, ["3", "4"]);
        let player_labels: Vec<_> = player.actions().map(|(label, _)| label.escape()).collect();
        assert_eq!(player_labels, ["x", "y"]);
        assert!(player.action("none").is_none());
        assert!(player.action("y").is_some());
        let (label, leaf) = player.action_at(0).unwrap();
        assert_eq!(label.escape(), "x");
        assert!(player.action_at(2).is_none());

        let Node::Terminal(terminal) = leaf else {
            panic!("expected a terminal after player action x");
        };
        assert_eq!(terminal.name().escape(), "ta");
        assert_eq!(terminal.outcome(), 1);
        assert_eq!(terminal.outcome_name().map(EscapedStr::escape), Some("oa"));
        let terminal_payoffs: Vec<_> = terminal
            .outcome_payoffs()
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(terminal_payoffs, ["7", "8"]);
    }

    #[test]
    fn error_display() {
        let parse_err = ExtensiveFormGame::try_from_str("not an efg").unwrap_err();
        assert!(parse_err.to_string().starts_with("error parsing game at:"));

        let bad = "EFG 2 R \"\" { \"1\" \"2\" }\np \"\" 3 1 \"a\" { \"x\" } 0\nt \"\" 1 { 0 0 }\n";
        assert_eq!(
            ExtensiveFormGame::try_from_str(bad)
                .unwrap_err()
                .to_string(),
            "invalid efg: InvalidPlayerNum"
        );
        assert_eq!(
            ValidationError::UndefinedOutcome.to_string(),
            "UndefinedOutcome"
        );
    }

    #[test]
    fn accepts_d_data_type() {
        // Gambit reads either the R or legacy D data-type letter; Display normalizes to R
        let game = ExtensiveFormGame::try_from_str(
            "EFG 2 D \"\" { \"1\" \"2\" }\nt \"\" 1 \"\" { 1 2 }\n",
        )
        .unwrap();
        assert!(game.to_string().starts_with("EFG 2 R "));
    }

    #[test]
    fn trailing_input_is_rejected() {
        let game = r#"EFG 2 R "" { "1" "2" } t "" 1 "" { 1 2 } trailing"#;
        assert!(matches!(
            ExtensiveFormGame::try_from_str(game),
            Err(Error::Parse("trailing"))
        ));
    }

    #[test]
    fn rejects_overflowing_exponent() {
        // an exponent that doesn't fit an i32 fails the number parse
        assert!(super::big_float("1e99999999999 ").is_err());
    }

    #[test]
    fn rejects_huge_exponent() {
        // a huge but i32-valid exponent is capped, not materialized
        assert!(super::big_float("1e2000000000 ").is_err());
        assert!(super::big_float("1e-2000000000 ").is_err());
        // an exponent within the cap still parses
        assert!(super::big_float("1e100 ").is_ok());
    }

    #[test]
    fn rejects_zero_denominator() {
        // a zero denominator must surface as a parse error rather than panicking in `Div`
        assert!(super::big_rational("1/0 ").is_err());
        assert!(
            ExtensiveFormGame::try_from_str(
                "EFG 2 R \"\" { \"1\" \"2\" }\nt \"\" 1 \"\" { 1/0 2 }\n"
            )
            .is_err()
        );
    }

    #[test]
    fn outcome_defined_then_referenced() {
        // once an outcome is defined, later nodes may reference it by bare id
        let game = "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 1 1 \"i\" { \"x\" } 1 \"named\" { 3 4 }
t \"\" 1
";
        assert!(ExtensiveFormGame::try_from_str(game).is_ok());
    }

    #[test]
    fn chance_null_outcome_with_payoffs() {
        // outcome validation also runs for chance nodes
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
c \"\" 1 \"i\" { \"x\" 1 } 0 \"n\" { 1 2 }
t \"\" 1 { 0 0 }
"
            ),
            ValidationError::NullOutcomePayoffs
        );
    }

    #[test]
    fn deep_tree_parses_and_drops() {
        // a tree far deeper than any call stack could hold must parse, validate, navigate, and drop
        // without overflowing, now that the arena makes every path flat rather than recursive
        let depth = 200_000;
        let mut game = String::with_capacity(depth * 24 + 64);
        game.push_str("EFG 2 R \"\" { \"1\" \"2\" }\n");
        for _ in 0..depth {
            game.push_str("p \"\" 1 1 \"i\" { \"a\" } 0\n");
        }
        game.push_str("t \"\" 1 \"\" { 0 0 }\n");
        let parsed = ExtensiveFormGame::try_from_str(&game).unwrap();
        assert!(matches!(parsed.root(), Node::Player(_)));
        // dropping the deep tree is itself a flat pass over the arena
        drop(parsed);
    }

    #[test]
    fn tolerates_flexible_whitespace() {
        // whitespace is not significant: braces need no padding, and payoff commas need no space
        let game =
            ExtensiveFormGame::try_from_str("EFG 2 R \"\" {\"1\" \"2\"}\nt \"\" 1 \"\" {1,2}\n")
                .unwrap();
        assert_eq!(game.player_names().len(), 2);
        let Node::Terminal(root) = game.root() else {
            panic!("expected a terminal root");
        };
        let payoffs: Vec<_> = root
            .outcome_payoffs()
            .unwrap()
            .iter()
            .map(BigRational::to_string)
            .collect();
        assert_eq!(payoffs, ["1", "2"]);
        // a comma padded with spaces is equally acceptable
        assert!(
            ExtensiveFormGame::try_from_str(
                "EFG 2 R \"\" { \"1\" \"2\" }\nt \"\" 1 \"\" { 1 , 2 }\n"
            )
            .is_ok()
        );
    }

    #[test]
    fn chance_outcome_name() {
        // a chance node may carry an outcome name (Gambit writes and reads one)
        let game_str = "EFG 2 R \"\" { \"1\" \"2\" }
c \"\" 1 \"i\" { \"a\" 1/2 \"b\" 1/2 } 1 \"oname\" { 3 4 }
t \"\" 1
t \"\" 1
";
        let game = ExtensiveFormGame::try_from_str(game_str).unwrap();
        let Node::Chance(root) = game.root() else {
            panic!("expected a chance root");
        };
        assert_eq!(root.outcome_name().map(EscapedStr::escape), Some("oname"));
        // the name is preserved through a Display round-trip
        let written = game.to_string();
        let reparsed = ExtensiveFormGame::try_from_str(written.as_str()).unwrap();
        assert_eq!(game, reparsed);
    }

    #[test]
    fn terminal_null_and_referenced_outcomes() {
        // `t "" 0` (null outcome, no payoffs) and a terminal that only references an outcome
        // defined on another node both parse and round-trip
        let game_str = "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 1 1 \"i\" { \"L\" \"M\" \"R\" } 0
t \"a\" 0
t \"b\" 1 \"obname\" { 3 4 }
t \"c\" 1
";
        let game = ExtensiveFormGame::try_from_str(game_str).unwrap();
        let Node::Player(root) = game.root() else {
            panic!("expected a player root");
        };
        let Some(Node::Terminal(null_term)) = root.action("L") else {
            panic!("expected a terminal after action L");
        };
        // the null outcome resolves to no payoffs and no name
        assert_eq!(null_term.outcome(), 0);
        assert!(null_term.outcome_payoffs().is_none());
        assert!(null_term.outcome_name().is_none());
        let Some(Node::Terminal(referenced)) = root.action("R") else {
            panic!("expected a terminal after action R");
        };
        // the referencing terminal resolves through the shared outcome to the payoffs "b" defined
        assert_eq!(referenced.outcome(), 1);
        let payoffs: Vec<_> = referenced
            .outcome_payoffs()
            .unwrap()
            .iter()
            .map(BigRational::to_string)
            .collect();
        assert_eq!(payoffs, ["3", "4"]);
        // and the game round-trips
        let written = game.to_string();
        let reparsed = ExtensiveFormGame::try_from_str(written.as_str()).unwrap();
        assert_eq!(game, reparsed);
    }

    #[test]
    fn write_modes() {
        // infoset 1 and outcome 1 are each declared more than minimally (root+mid repeat the
        // infoset block, "a"+"b" repeat the outcome), and "c" references the outcome by id
        let input = "EFG 2 R \"\" { \"1\" \"2\" }
p \"root\" 1 1 \"iset\" { \"L\" \"R\" } 0
p \"mid\" 1 1 \"iset\" { \"L\" \"R\" } 0
t \"a\" 1 \"out\" { 1 2 }
t \"b\" 1 \"out\" { 1 2 }
t \"c\" 1
";
        let game = ExtensiveFormGame::try_from_str(input).unwrap();

        // Display defaults to Faithful, which reproduces the parsed declare/reference structure
        assert_eq!(
            game.to_string(),
            game.display(WriteMode::Faithful).to_string()
        );
        let faithful = game.display(WriteMode::Faithful).to_string();
        assert_eq!(ExtensiveFormGame::try_from_str(&faithful).unwrap(), game);

        // every mode is valid and resolves to the same game, so its exhaustive rendering matches
        let canonical = game.display(WriteMode::Exhaustive).to_string();
        for mode in [
            WriteMode::Minimal,
            WriteMode::Faithful,
            WriteMode::Exhaustive,
        ] {
            let out = game.display(mode).to_string();
            let reparsed = ExtensiveFormGame::try_from_str(&out).unwrap();
            assert_eq!(
                reparsed.display(WriteMode::Exhaustive).to_string(),
                canonical
            );
        }

        // Minimal declares each block once; Exhaustive writes them on every node
        let minimal = game.display(WriteMode::Minimal).to_string();
        assert!(minimal.len() < faithful.len());
        assert!(faithful.len() < canonical.len());
    }
}
