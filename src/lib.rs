//! A library for parsing [gambit extensive form
//! game](https://gambitproject.readthedocs.io/en/v16.0.2/formats.html) (`.efg`) files
//!
//! This library produces an [ExtensiveFormGame], which can then be easily used to model an
//! extensive form game.
//!
//! In order to minimize memory consumption, this stores references to the underlying string where
//! possible. One side effect is that this is a borrowed struct, and any quoted labels will still
//! have escape sequences in them in the form of [EscapedStr]s.
#![warn(missing_docs)]

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
use num_traits::{One, Zero};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::error::Error as StdError;
use std::fmt::{Display, Error as FmtError, Formatter};
pub use unescaped::{EscapedStr, Unescaped};

/// A chance infoset's label paired with its ordered actions and their probabilities
type ChanceInfoset<'a> = (&'a EscapedStr, Box<[(&'a EscapedStr, BigRational)]>);
/// A player infoset's label paired with its ordered action labels
type PlayerInfoset<'a> = (&'a EscapedStr, Box<[&'a EscapedStr]>);

/// Every infoset seen while parsing, keyed by id. Player infosets are split per player (index =
/// `player_num - 1`); chance is its own namespace. Each entry holds the infoset's label and ordered
/// actions, which the tree nodes reference by id.
#[derive(Debug, PartialEq, Clone)]
struct Infosets<'a> {
    player: Box<[HashMap<u64, PlayerInfoset<'a>>]>,
    chance: HashMap<u64, ChanceInfoset<'a>>,
}

/// A node in the raw, id-referenced game tree. Infoset payloads live on the game, not here.
#[derive(Debug, PartialEq, Clone)]
enum RawNode<'a> {
    Chance(RawChance<'a>),
    Player(RawPlayer<'a>),
    Terminal(RawTerminal<'a>),
}

#[derive(Debug, PartialEq, Clone)]
struct RawChance<'a> {
    name: &'a EscapedStr,
    infoset: u64,
    // did THIS node write the infoset block, or inherit it by omission?
    declared: bool,
    children: Box<[RawNode<'a>]>,
    outcome: u64,
    outcome_payoffs: Option<Box<[BigRational]>>,
}

#[derive(Debug, PartialEq, Clone)]
struct RawPlayer<'a> {
    name: &'a EscapedStr,
    player_num: usize,
    infoset: u64,
    // did THIS node write the infoset block, or inherit it by omission?
    declared: bool,
    children: Box<[RawNode<'a>]>,
    outcome: u64,
    outcome_name: Option<&'a EscapedStr>,
    outcome_payoffs: Option<Box<[BigRational]>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct RawTerminal<'a> {
    name: &'a EscapedStr,
    outcome: u64,
    outcome_name: Option<&'a EscapedStr>,
    outcome_payoffs: Box<[BigRational]>,
}

/// A full extensive form game
///
/// This can be parsed from a [str] reference using [ExtensiveFormGame::try_from_str] or using the
/// [TryFrom] / [TryInto] traits. It implements [Display] for formatting.
///
/// # Example
///
/// ```
/// # use gambit_parser::ExtensiveFormGame;
/// let gambit = r#"EFG 2 R "" { "1" "2" } t "" 1 { 1 2 }"#;
/// let game: ExtensiveFormGame<'_> = gambit.try_into().unwrap();
/// let output = game.to_string();
/// ```
#[derive(Debug, PartialEq, Clone)]
pub struct ExtensiveFormGame<'a> {
    name: &'a EscapedStr,
    player_names: Box<[&'a EscapedStr]>,
    comment: Option<&'a EscapedStr>,
    infosets: Infosets<'a>,
    root: RawNode<'a>,
}

impl<'a> ExtensiveFormGame<'a> {
    /// The name of the game
    pub fn name(&self) -> &'a EscapedStr {
        self.name
    }

    /// Names for every player, in order
    pub fn player_names(&self) -> &[&'a EscapedStr] {
        &self.player_names
    }

    /// An optional game comment
    pub fn comment(&self) -> Option<&'a EscapedStr> {
        self.comment
    }

    /// The root node of the game tree
    pub fn root<'g>(&'g self) -> Node<'a, 'g> {
        self.wrap(&self.root)
    }

    fn wrap<'g>(&'g self, raw: &'g RawNode<'a>) -> Node<'a, 'g> {
        match raw {
            RawNode::Chance(raw) => Node::Chance(Chance { game: self, raw }),
            RawNode::Player(raw) => Node::Player(Player { game: self, raw }),
            RawNode::Terminal(raw) => Node::Terminal(Terminal { raw }),
        }
    }
}

impl<'a> Display for ExtensiveFormGame<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(out, "EFG 2 R \"{}\" {{ ", self.name.escape())?;
        for name in self.player_names.iter() {
            write!(out, "\"{}\" ", name.escape())?;
        }
        writeln!(out, "}}")?;
        if let Some(comment) = self.comment {
            writeln!(out, "\"{}\"", comment.escape())?;
        }
        writeln!(out, "{}", self.root())
    }
}

/// An error that happens while trying to turn a string into an [ExtensiveFormGame]
#[derive(Debug)]
#[non_exhaustive]
pub enum Error<'a> {
    /// A problem with parsing
    ///
    /// This will show the remainder of the string where the parse error occurred
    Parse(&'a str),
    /// A problem validating the tree after parsing
    Validation(ValidationError),
}

impl<'a> Display for Error<'a> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Error::Parse(rem) => write!(fmt, "error parsing game at: '{}'", rem),
            Error::Validation(err) => write!(fmt, "invalid efg: {}", err),
        }
    }
}

impl<'a> StdError for Error<'a> {}

/// An error that results from something invalid about the parsed extensive form game
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValidationError {
    /// The probabilities of actions associated with a chance node don't sum to one
    ChanceNotDistribution,
    /// A player's number wasn't between one and the number of players
    InvalidPlayerNum,
    /// An infoset had different names attached to it
    NonMatchingInfosetNames,
    /// An infoset had different sets of associated actions
    NonMatchingInfosetActions,
    /// There was payoff data associated with the null (0) outcome
    NullOutcomePayoffs,
    /// The number of specified payoffs did not match the number of players
    InvalidNumberOfPayoffs,
    /// An outcome had different names attached to it
    NonMatchingOutcomeNames,
    /// An outcome had different associated payoffs
    NonMatchingOutcomePayoffs,
    /// An outcome was defined without payoffs
    NoOutcomePayoffs,
    /// A node omitted its action list for an infoset that was never declared
    UndeclaredInfoset,
}

impl Display for ValidationError {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(fmt, "{:?}", self)
    }
}

impl<'a> From<ValidationError> for Error<'a> {
    fn from(err: ValidationError) -> Self {
        Error::Validation(err)
    }
}

impl<'a> From<nom::Err<nom::error::Error<&'a str>>> for Error<'a> {
    fn from(err: nom::Err<nom::error::Error<&'a str>>) -> Self {
        match err {
            nom::Err::Incomplete(_) => panic!("internal error: incomplete parsing"),
            nom::Err::Error(err) => Error::Parse(err.input),
            nom::Err::Failure(err) => Error::Parse(err.input),
        }
    }
}

impl<'a> ExtensiveFormGame<'a> {
    /// Try to parse a game from a string
    ///
    /// This is identical to `ExtensiveFormGame::try_from` or `"...".try_into()`.
    pub fn try_from_str(input: &'a str) -> Result<Self, Error<'a>> {
        let (rest, game) = parse_game(input)?;
        let rest = rest.trim_start();
        if !rest.is_empty() {
            return Err(Error::Parse(rest));
        }
        game.validate()?;
        Ok(game)
    }

    /// Infoset consistency (matching names and actions across a shared id) and player number ranges
    /// are enforced while parsing, so this only covers what the tree shape can't: chance
    /// distributions and outcome agreement.
    fn validate(&self) -> Result<(), ValidationError> {
        for (_, actions) in self.infosets.chance.values() {
            let total: BigRational = actions.iter().map(|(_, prob)| prob).sum();
            if total != BigRational::one() {
                return Err(ValidationError::ChanceNotDistribution);
            }
        }

        let mut outcomes = HashMap::new();
        let mut queue = vec![&self.root];
        while let Some(node) = queue.pop() {
            match node {
                RawNode::Chance(chance) => {
                    self.validate_outcome(
                        chance.outcome,
                        None,
                        chance.outcome_payoffs.as_deref(),
                        &mut outcomes,
                    )?;
                    queue.extend(chance.children.iter());
                }
                RawNode::Player(player) => {
                    self.validate_outcome(
                        player.outcome,
                        player.outcome_name,
                        player.outcome_payoffs.as_deref(),
                        &mut outcomes,
                    )?;
                    queue.extend(player.children.iter());
                }
                RawNode::Terminal(term) => {
                    self.validate_outcome(
                        term.outcome,
                        term.outcome_name,
                        Some(&term.outcome_payoffs),
                        &mut outcomes,
                    )?;
                }
            }
        }

        for (_, (_, pays)) in outcomes {
            if pays.is_none() {
                return Err(ValidationError::NoOutcomePayoffs);
            }
        }

        Ok(())
    }

    fn validate_outcome<'b>(
        &self,
        outcome: u64,
        outcome_name: Option<&'a EscapedStr>,
        outcome_payoffs: Option<&'b [BigRational]>,
        outcomes: &mut HashMap<u64, (Option<&'a EscapedStr>, Option<&'b [BigRational]>)>,
    ) -> Result<(), ValidationError> {
        if outcome == 0 {
            if outcome_payoffs.is_some() {
                return Err(ValidationError::NullOutcomePayoffs);
            }
        } else {
            match outcomes.entry(outcome) {
                Entry::Vacant(ent) => match outcome_payoffs {
                    Some(pays) => {
                        if pays.len() == self.player_names.len() {
                            ent.insert((outcome_name, Some(pays)));
                        } else {
                            return Err(ValidationError::InvalidNumberOfPayoffs);
                        }
                    }
                    None => {
                        ent.insert((outcome_name, None));
                    }
                },
                Entry::Occupied(mut ent) => {
                    let (name, payoffs) = ent.get_mut();
                    match (name, outcome_name) {
                        (Some(old), Some(new)) if old != &new => {
                            return Err(ValidationError::NonMatchingOutcomeNames);
                        }
                        (old @ None, Some(new)) => {
                            *old = Some(new);
                        }
                        _ => (),
                    };
                    match (payoffs, outcome_payoffs) {
                        (Some(old), Some(new)) if old != &new => {
                            return Err(ValidationError::NonMatchingOutcomePayoffs);
                        }
                        (old @ None, Some(new)) => {
                            *old = Some(new);
                        }
                        _ => (),
                    }
                }
            }
        }
        Ok(())
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

impl<'a, 'g> Display for Node<'a, 'g> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        let mut queue = vec![*self];
        while let Some(node) = queue.pop() {
            match node {
                Node::Chance(chance) => {
                    queue.extend(
                        chance
                            .raw
                            .children
                            .iter()
                            .rev()
                            .map(|c| chance.game.wrap(c)),
                    );
                    write!(out, "\nc {}", chance)?
                }
                Node::Player(player) => {
                    queue.extend(
                        player
                            .raw
                            .children
                            .iter()
                            .rev()
                            .map(|c| player.game.wrap(c)),
                    );
                    write!(out, "\np {}", player)?
                }
                Node::Terminal(terminal) => write!(out, "\nt {}", terminal)?,
            }
        }
        Ok(())
    }
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
    pub fn name(self) -> &'a EscapedStr {
        self.raw.name
    }

    /// The id of the node's infoset
    pub fn infoset(self) -> u64 {
        self.raw.infoset
    }

    /// The infoset's label
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
            .map(move |((label, prob), child)| (*label, prob, game.wrap(child)))
    }

    /// The probability and child for the action with the given label
    ///
    /// Labels need not be unique within an infoset, so this returns the first match.
    pub fn action(self, label: &EscapedStr) -> Option<(&'g BigRational, Node<'a, 'g>)> {
        self.actions()
            .find(|(name, _, _)| *name == label)
            .map(|(_, prob, next)| (prob, next))
    }

    /// The number of actions (always at least one)
    #[allow(clippy::len_without_is_empty)]
    pub fn len(self) -> usize {
        self.raw.children.len()
    }

    /// The name, probability, and child for the action at the given index
    pub fn action_at(
        self,
        index: usize,
    ) -> Option<(&'a EscapedStr, &'g BigRational, Node<'a, 'g>)> {
        let (_, actions) = self.entry();
        let (label, prob) = actions.get(index)?;
        Some((*label, prob, self.game.wrap(self.raw.children.get(index)?)))
    }

    /// The outcome id
    pub fn outcome(self) -> u64 {
        self.raw.outcome
    }

    /// Outcome payoffs for this node
    ///
    /// Outcome payoffs are added to every players' payoffs for traversing through this node. Note
    /// that if these are missing, they be defined at another node sharing the same outcome.
    pub fn outcome_payoffs(self) -> Option<&'g [BigRational]> {
        self.raw.outcome_payoffs.as_deref()
    }
}

impl<'a, 'g> Display for Chance<'a, 'g> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(out, "\"{}\" {}", self.raw.name.escape(), self.raw.infoset)?;
        // the label and action list are written together, or both omitted, matching the file
        if self.raw.declared {
            let (label, actions) = self.entry();
            write!(out, " \"{}\" {{ ", label.escape())?;
            for (action, prob) in actions.iter() {
                write!(out, "\"{}\" {} ", action.escape(), prob)?;
            }
            write!(out, "}}")?;
        }
        write!(out, " {}", self.raw.outcome)?;
        if let Some(payoffs) = &self.raw.outcome_payoffs {
            write!(out, " {{ ")?;
            for payoff in payoffs.iter() {
                write!(out, "{} ", payoff)?;
            }
            write!(out, "}}")?;
        }
        Ok(())
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
    pub fn name(self) -> &'a EscapedStr {
        self.raw.name
    }

    /// The player acting at this node
    ///
    /// This will always be between 1 and the number of players.
    pub fn player_num(self) -> usize {
        self.raw.player_num
    }

    /// The infoset id for this node and player
    pub fn infoset(self) -> u64 {
        self.raw.infoset
    }

    /// The infoset's label
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
            .map(move |(label, child)| (*label, game.wrap(child)))
    }

    /// The child reached by the action with the given label
    ///
    /// Labels need not be unique within an infoset, so this returns the first match.
    pub fn action(self, label: &EscapedStr) -> Option<Node<'a, 'g>> {
        self.actions()
            .find(|(name, _)| *name == label)
            .map(|(_, next)| next)
    }

    /// The number of actions (always at least one)
    #[allow(clippy::len_without_is_empty)]
    pub fn len(self) -> usize {
        self.raw.children.len()
    }

    /// The name and child for the action at the given index
    pub fn action_at(self, index: usize) -> Option<(&'a EscapedStr, Node<'a, 'g>)> {
        let (_, actions) = self.entry();
        let &label = actions.get(index)?;
        Some((label, self.game.wrap(self.raw.children.get(index)?)))
    }

    /// The outcome id
    pub fn outcome(self) -> u64 {
        self.raw.outcome
    }

    /// The name of the outcome
    ///
    /// If omitted it may still be defined on another node.
    pub fn outcome_name(self) -> Option<&'a EscapedStr> {
        self.raw.outcome_name
    }

    /// Payoffs associated with the outcome
    ///
    /// If omitted they may be defined on another node.
    pub fn outcome_payoffs(self) -> Option<&'g [BigRational]> {
        self.raw.outcome_payoffs.as_deref()
    }
}

impl<'a, 'g> Display for Player<'a, 'g> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            out,
            "\"{}\" {} {}",
            self.raw.name.escape(),
            self.raw.player_num,
            self.raw.infoset
        )?;
        // the label and action list are written together, or both omitted, matching the file
        if self.raw.declared {
            let (label, actions) = self.entry();
            write!(out, " \"{}\" {{ ", label.escape())?;
            for action in actions.iter() {
                write!(out, "\"{}\" ", action.escape())?;
            }
            write!(out, "}}")?;
        }
        write!(out, " {}", self.raw.outcome)?;
        if let Some(name) = self.raw.outcome_name {
            write!(out, " \"{}\"", name.escape())?;
        }
        if let Some(payoffs) = &self.raw.outcome_payoffs {
            write!(out, " {{ ")?;
            for payoff in payoffs.iter() {
                write!(out, "{} ", payoff)?;
            }
            write!(out, "}}")?;
        }
        Ok(())
    }
}

/// A terminal node represents the end of a game
///
/// Terminal nodes simply assign payoffs to every player in the game
#[derive(Clone, Copy)]
pub struct Terminal<'a, 'g> {
    raw: &'g RawTerminal<'a>,
}

impl<'a, 'g> Terminal<'a, 'g> {
    /// The name of this node
    pub fn name(self) -> &'a EscapedStr {
        self.raw.name
    }

    /// The outcome id
    pub fn outcome(self) -> u64 {
        self.raw.outcome
    }

    /// The name of this outcome
    ///
    /// Note that if omitted it may be specified on a different node with the same outcome.
    pub fn outcome_name(self) -> Option<&'a EscapedStr> {
        self.raw.outcome_name
    }

    /// The payoffs to every player
    pub fn outcome_payoffs(self) -> &'g [BigRational] {
        &self.raw.outcome_payoffs
    }
}

impl<'a, 'g> Display for Terminal<'a, 'g> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(out, "\"{}\" {}", self.raw.name.escape(), self.raw.outcome)?;
        if let Some(name) = self.raw.outcome_name {
            write!(out, " \"{}\"", name.escape())?;
        }
        write!(out, " {{ ")?;
        for payoff in self.raw.outcome_payoffs.iter() {
            write!(out, "{} ", payoff)?;
        }
        write!(out, "}}")
    }
}

fn negate(input: &str) -> IResult<&str, bool> {
    let (input, res) = opt(one_of("+-")).parse(input)?;
    Ok((input, res == Some('-')))
}

fn fail(input: &str) -> nom::Err<nom::error::Error<&str>> {
    nom::Err::Error(nom::error::Error::new(input, ErrorKind::Fail))
}

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
    };
    if let Some((neg, exp)) = exp {
        let exp: i32 = exp.parse().map_err(|_| fail(input))?;
        res *= BigRational::from_integer(10.into()).pow(if neg { -exp } else { exp });
    };
    if main_neg {
        res = -res;
    };
    Ok((res_input, res))
}

fn big_rational(input: &str) -> IResult<&str, BigRational> {
    let (input, (num, denom)) =
        pair(big_float, opt(preceded(char('/'), big_float))).parse(input)?;
    Ok((
        input,
        match denom {
            Some(denom) => num / denom,
            None => num,
        },
    ))
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
        pair(char('{'), multispace1),
        separated_list1(multispace1, f),
        pair(multispace1, char('}')),
    )
}

fn commalist<'a, O, E, F>(f: F) -> impl Parser<&'a str, Output = Vec<O>, Error = E>
where
    F: Parser<&'a str, Output = O, Error = E>,
    E: ParseError<&'a str>,
{
    delimited(
        pair(char('{'), multispace1),
        separated_list1(pair(opt(char(',')), multispace1), f),
        pair(multispace1, char('}')),
    )
}

/// Parse `count` child nodes in sequence (they follow a node, one per action)
fn parse_children<'a>(
    mut input: &'a str,
    count: usize,
    infosets: &mut Infosets<'a>,
) -> Result<(&'a str, Box<[RawNode<'a>]>), Error<'a>> {
    let mut children = Vec::with_capacity(count);
    for _ in 0..count {
        let (rest, next) = parse_node(input, infosets)?;
        input = rest;
        children.push(next);
    }
    Ok((input, children.into()))
}

/// Record or check an infoset declaration, returning whether the block was written here and the
/// action count. A first declaration is inserted, a repeat must match exactly, an omission inherits.
fn resolve_infoset<'a, A: PartialEq>(
    map: &mut HashMap<u64, (&'a EscapedStr, Box<[A]>)>,
    infoset: u64,
    declared: Option<(&'a EscapedStr, Vec<A>)>,
) -> Result<(bool, usize), Error<'a>> {
    match declared {
        Some((name, actions)) => match map.entry(infoset) {
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
        },
        None => {
            let (_, actions) = map
                .get(&infoset)
                .ok_or(ValidationError::UndeclaredInfoset)?;
            Ok((false, actions.len()))
        }
    }
}

fn parse_node<'a>(
    input: &'a str,
    infosets: &mut Infosets<'a>,
) -> Result<(&'a str, RawNode<'a>), Error<'a>> {
    let (input, style) = preceded(multispace1, one_of("cpt")).parse(input)?;
    match style {
        'c' => {
            let (input, chance) = parse_chance(input, infosets)?;
            Ok((input, RawNode::Chance(chance)))
        }
        'p' => {
            let (input, player) = parse_player(input, infosets)?;
            Ok((input, RawNode::Player(player)))
        }
        't' => {
            let (input, term) = parse_terminal(input)?;
            Ok((input, RawNode::Terminal(term)))
        }
        // `one_of("cpt")` only ever yields one of these three characters
        _ => unreachable!(),
    }
}

fn parse_chance<'a>(
    input: &'a str,
    infosets: &mut Infosets<'a>,
) -> Result<(&'a str, RawChance<'a>), Error<'a>> {
    let (input, (name, infoset, declared, outcome, outcome_payoffs)) = (
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
        opt(preceded(multispace1, commalist(big_rational))),
    )
        .parse(input)?;
    let (declared, child_count) = resolve_infoset(&mut infosets.chance, infoset, declared)?;
    let (input, children) = parse_children(input, child_count, infosets)?;
    Ok((
        input,
        RawChance {
            name,
            infoset,
            declared,
            children,
            outcome,
            outcome_payoffs: outcome_payoffs.map(Into::into),
        },
    ))
}

fn parse_player<'a>(
    input: &'a str,
    infosets: &mut Infosets<'a>,
) -> Result<(&'a str, RawPlayer<'a>), Error<'a>> {
    let (input, (name, player_num, infoset, declared, outcome, outcome_name, outcome_payoffs)) = (
        preceded(multispace1, label),
        preceded(multispace1, u64),
        preceded(multispace1, u64),
        opt((
            preceded(multispace1, label),
            preceded(multispace1, spacelist(label)),
        )),
        preceded(multispace1, u64),
        opt(preceded(multispace1, label)),
        opt(preceded(multispace1, commalist(big_rational))),
    )
        .parse(input)?;
    let player_num: usize = player_num.try_into().map_err(|_| fail(input))?;
    // checked here, since the per-player infoset map is indexed by it
    if player_num == 0 || player_num > infosets.player.len() {
        return Err(ValidationError::InvalidPlayerNum.into());
    }
    let (declared, child_count) =
        resolve_infoset(&mut infosets.player[player_num - 1], infoset, declared)?;
    let (input, children) = parse_children(input, child_count, infosets)?;
    Ok((
        input,
        RawPlayer {
            name,
            player_num,
            infoset,
            declared,
            children,
            outcome,
            outcome_name,
            outcome_payoffs: outcome_payoffs.map(Into::into),
        },
    ))
}

fn parse_terminal(input: &str) -> IResult<&str, RawTerminal<'_>> {
    let (input, (name, outcome, outcome_name, payoffs)) = (
        preceded(multispace1, label),
        preceded(multispace1, u64),
        opt(preceded(multispace1, label)),
        preceded(multispace1, commalist(big_rational)),
    )
        .parse(input)?;
    Ok((
        input,
        RawTerminal {
            name,
            outcome,
            outcome_name,
            outcome_payoffs: payoffs.into(),
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
                tag("R"),
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
    let (input, root) = parse_node(input, &mut infosets)?;
    Ok((
        input,
        ExtensiveFormGame {
            name,
            player_names: player_names.into(),
            comment,
            infosets,
            root,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::{Error, EscapedStr, ExtensiveFormGame, Node, ValidationError};
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

        let Some(Node::Terminal(left)) = root.action(EscapedStr::new("L")) else {
            panic!("expected a terminal after action L");
        };
        assert_eq!(left.name().escape(), "tl");
        assert_eq!(left.outcome(), 1);
        assert_eq!(left.outcome_name().map(EscapedStr::escape), Some("o1"));
        let payoffs: Vec<_> = left
            .outcome_payoffs()
            .iter()
            .map(BigRational::to_string)
            .collect();
        assert_eq!(payoffs, ["1", "2"]);
    }

    #[test]
    fn not_distribution() {
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
c \"\" 1 \"a\" { \"x\" 9/10 } 0
t \"\" 1 { 0 0 }
"
            ),
            ValidationError::ChanceNotDistribution
        );
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
t \"\" 1 { 0 0 }
p \"\" 1 1 \"a\" { \"R\" \"L\" } 0
t \"\" 2 { 0 0 }
t \"\" 3 { 0 0 }
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
p \"\" 1 1 \"a\" { \"x\" } 0 { 0 0 }
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
t \"\" 1 { 0 }
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
p \"\" 1 1 \"a\" { \"x\" } 1 { 0 0 }
t \"\" 1 { 1 1 }
"
            ),
            ValidationError::NonMatchingOutcomePayoffs
        );
    }

    #[test]
    fn no_outcome_payoffs() {
        assert_eq!(
            validation_err(
                "EFG 2 R \"\" { \"1\" \"2\" }
p \"\" 1 1 \"a\" { \"x\" } 1
t \"\" 2 { 0 0 }
"
            ),
            ValidationError::NoOutcomePayoffs
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
t \"\" 1 { 0 0 }
p \"\" 1 1 0
t \"\" 2 { 0 0 }
t \"\" 3 { 0 0 }
";
        let game = ExtensiveFormGame::try_from_str(game_str).unwrap();
        let Node::Player(root) = game.root() else {
            panic!("expected a player root");
        };
        let Some(Node::Player(omitted)) = root.action(EscapedStr::new("R")) else {
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
}
