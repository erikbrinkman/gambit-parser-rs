//! A library for parsing [gambit extensive form
//! game](https://gambitproject.readthedocs.io/en/v16.0.2/formats.html) (`.efg`) files
//!
//! Ths library produces an [ExtensiveFormGame], which can then be easily used to model an
//! extensive form game.
//!
//! In order to minimize memory consumption, this stores references to the underlying string where
//! possible. One side effect is that this is a borrowed struct, and any quoted labels will still
//! have escape sequences in them in the form of [EscapedStr]s.
//!
//! This also tries to represent the file as it's structured, so if a name is attached to an
//! infoset on one node, this won't propogate the name to other nodes with the same infoset.
#![warn(missing_docs)]

mod multiset;
mod unescaped;

use multiset::BTreeMultiSet;
use nom::{
    branch::alt,
    bytes::complete::{escaped, tag},
    character::complete::{char, digit0, digit1, multispace0, multispace1, none_of, one_of, u64},
    combinator::{all_consuming, map, opt},
    error::{ErrorKind, ParseError},
    multi::separated_list1,
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    IResult, Parser,
};
use num::rational::BigRational;
use num::{BigInt, One, Zero};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};
pub use unescaped::{EscapedStr, Unescaped};

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
    root: Node<'a>,
}

impl<'a> ExtensiveFormGame<'a> {
    /// The name of the game
    pub fn name(&self) -> &'a EscapedStr {
        self.name
    }

    /// Names for every player, in order
    pub fn player_names(&self) -> &[&'a EscapedStr] {
        &*self.player_names
    }

    /// An optional game comment
    pub fn comment(&self) -> Option<&'a EscapedStr> {
        self.comment
    }

    /// The root node of the game tree
    pub fn root(&self) -> &Node<'a> {
        &self.root
    }
}

impl<'a> Display for ExtensiveFormGame<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        write!(out, "EFG 2 R \"{}\" {{ ", self.name.as_raw_str())?;
        for name in self.player_names.iter() {
            write!(out, "\"{}\" ", name.as_raw_str())?;
        }
        writeln!(out, "}}")?;
        if let Some(comment) = self.comment {
            writeln!(out, "\"{}\"", comment)?;
        }
        writeln!(out, "{}", self.root)
    }
}

/// An error that happens while trying to turn a string into an [ExtensiveFormGame]
#[derive(Debug)]
pub enum Error<'a> {
    /// A problem with parsing
    ///
    /// This will show the remainder of the string where the parse error occured
    ParseError(&'a str),
    /// A problem validating the tree after parsing
    ValidationError(ValidationError),
}

/// An error that results from something invalid about the parsed extensive form game
#[derive(Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// The probabilities of actions associated with a chance node don't sum to one
    ChanceNotDistribution,
    /// A players number wasn't between one and the number of players
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
    /// An outcomes was defined without payoffs
    NoOutcomePayoffs,
}

impl<'a> From<ValidationError> for Error<'a> {
    fn from(err: ValidationError) -> Self {
        Error::ValidationError(err)
    }
}

impl<'a> From<nom::Err<nom::error::Error<&'a str>>> for Error<'a> {
    fn from(err: nom::Err<nom::error::Error<&'a str>>) -> Self {
        match err {
            nom::Err::Incomplete(_) => panic!("internal error: incomplete parsing"),
            nom::Err::Error(err) => Error::ParseError(err.input),
            nom::Err::Failure(err) => Error::ParseError(err.input),
        }
    }
}

impl<'a> ExtensiveFormGame<'a> {
    /// Try to parse a game from a string
    ///
    /// This is identical to `ExtensiveFormGame::try_from` or `"...".try_into()`.
    pub fn try_from_str(input: &'a str) -> Result<Self, Error<'a>> {
        let (_, game) = all_consuming(efg)(input)?;
        game.validate()?;
        Ok(game)
    }

    fn validate(&self) -> Result<(), ValidationError> {
        let num_players = self.player_names.len();
        let mut chance_infosets = HashMap::new();
        let mut player_infosets = vec![HashMap::new(); num_players].into_boxed_slice();
        let mut outcomes = HashMap::new();
        let mut queue = vec![&self.root];
        while let Some(node) = queue.pop() {
            match node {
                Node::Chance(chance) => {
                    let total: BigRational = chance.actions.iter().map(|(_, prob, _)| prob).sum();
                    if total != BigRational::one() {
                        return Err(ValidationError::ChanceNotDistribution);
                    }

                    Self::validate_infoset(
                        chance.infoset,
                        chance.infoset_name,
                        chance.actions.iter().map(|(a, p, _)| (a, p)).collect(),
                        &mut chance_infosets,
                    )?;

                    self.validate_outcome(
                        chance.outcome,
                        None,
                        chance.outcome_payoffs.as_deref(),
                        &mut outcomes,
                    )?;

                    queue.extend(chance.actions.iter().map(|(_, _, next)| next));
                }
                Node::Player(player) => {
                    if !(1..=num_players).contains(&player.player_num) {
                        return Err(ValidationError::InvalidPlayerNum);
                    }

                    Self::validate_infoset(
                        player.infoset,
                        player.infoset_name,
                        player.actions.iter().map(|(action, _)| action).collect(),
                        &mut player_infosets[player.player_num - 1],
                    )?;

                    self.validate_outcome(
                        player.outcome,
                        player.outcome_name,
                        player.outcome_payoffs.as_deref(),
                        &mut outcomes,
                    )?;

                    queue.extend(player.actions.iter().map(|(_, next)| next));
                }
                Node::Terminal(term) => {
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

    fn validate_infoset<T: Eq>(
        infoset: u64,
        infoset_name: Option<&'a EscapedStr>,
        actions: BTreeMultiSet<T>,
        infosets: &mut HashMap<u64, (Option<&'a EscapedStr>, BTreeMultiSet<T>)>,
    ) -> Result<(), ValidationError> {
        match infosets.entry(infoset) {
            Entry::Vacant(ent) => {
                ent.insert((infoset_name, actions));
            }
            Entry::Occupied(mut ent) => {
                let (name, acts) = ent.get_mut();
                match (name, infoset_name) {
                    (Some(old), Some(new)) if old != &new => {
                        return Err(ValidationError::NonMatchingInfosetNames)
                    }
                    (old @ None, Some(new)) => {
                        *old = Some(new);
                    }
                    _ => (),
                };
                if acts != &actions {
                    return Err(ValidationError::NonMatchingInfosetActions);
                }
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
#[derive(Debug, PartialEq, Clone)]
pub enum Node<'a> {
    /// A chance node
    Chance(Chance<'a>),
    /// A player node
    Player(Player<'a>),
    /// A terminal node
    Terminal(Terminal<'a>),
}

impl<'a> Display for Node<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        let mut queue = vec![self];
        while let Some(node) = queue.pop() {
            match node {
                Node::Chance(chance) => {
                    queue.extend(chance.actions.iter().rev().map(|(_, _, next)| next));
                    write!(out, "\nc {}", chance)?
                }
                Node::Player(player) => {
                    queue.extend(player.actions.iter().rev().map(|(_, next)| next));
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
#[derive(Debug, PartialEq, Clone)]
pub struct Chance<'a> {
    name: &'a EscapedStr,
    infoset: u64,
    infoset_name: Option<&'a EscapedStr>,
    actions: Box<[(&'a EscapedStr, BigRational, Node<'a>)]>,
    outcome: u64,
    outcome_payoffs: Option<Box<[BigRational]>>,
}

impl<'a> Chance<'a> {
    /// The name of the node
    pub fn name(&self) -> &'a EscapedStr {
        self.name
    }

    /// The if of the node's infoset
    pub fn infoset(&self) -> u64 {
        self.infoset
    }

    /// The infoset's name
    ///
    /// Note that just because this infoset doesn't have a name attached, doesn't mean that the
    /// same id doesn't have a name attached at a different node.
    pub fn infoset_name(&self) -> Option<&'a EscapedStr> {
        self.infoset_name
    }

    /// All possible outcomes with names and probabilities
    pub fn actions(&self) -> &[(&'a EscapedStr, BigRational, Node<'a>)] {
        &*self.actions
    }

    /// The outcome id
    pub fn outcome(&self) -> u64 {
        self.outcome
    }

    /// Outcome payoffs for this node
    ///
    /// Outcome payoffs are added to every players' payoffs for traversing through this node. Note
    /// that if these are missing, they be defined at another node sharing the same outcome.
    pub fn outcome_payoffs(&self) -> Option<&[BigRational]> {
        self.outcome_payoffs.as_deref()
    }
}

impl<'a> Display for Chance<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        write!(out, "\"{}\" {}", self.name.as_raw_str(), self.infoset)?;
        if let Some(name) = self.infoset_name {
            write!(out, " \"{}\"", name.as_raw_str())?;
        }
        write!(out, " {{ ")?;
        for (action, prob, _) in self.actions.iter() {
            write!(out, "\"{}\" {} ", action, prob)?;
        }
        write!(out, "}} {}", self.outcome)?;
        if let Some(payoffs) = &self.outcome_payoffs {
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
#[derive(Debug, PartialEq, Clone)]
pub struct Player<'a> {
    name: &'a EscapedStr,
    player_num: usize,
    infoset: u64,
    infoset_name: Option<&'a EscapedStr>,
    actions: Box<[(&'a EscapedStr, Node<'a>)]>,
    outcome: u64,
    outcome_name: Option<&'a EscapedStr>,
    outcome_payoffs: Option<Box<[BigRational]>>,
}

impl<'a> Player<'a> {
    /// The name of the node
    pub fn name(&self) -> &'a EscapedStr {
        self.name
    }

    /// The player acting at this node
    ///
    /// This will always be between 1 and the number of players.
    pub fn player_num(&self) -> usize {
        self.player_num
    }

    /// The infoset id for this node and player
    pub fn infoset(&self) -> u64 {
        self.infoset
    }

    /// The infoset's name
    ///
    /// If the name is omitted, it may be defined on a different node.
    pub fn infoset_name(&self) -> Option<&'a EscapedStr> {
        self.infoset_name
    }

    /// All the actions a player can take with names
    pub fn actions(&self) -> &[(&'a EscapedStr, Node<'a>)] {
        &*self.actions
    }

    /// The outcome id
    pub fn outcome(&self) -> u64 {
        self.outcome
    }

    /// The name of the outcome
    ///
    /// If omitted it may still be defined on another node.
    pub fn outcome_name(&self) -> Option<&'a EscapedStr> {
        self.outcome_name
    }

    /// Payoffs associated with the outcome
    ///
    /// If ommited they may be defined on another node.
    pub fn outcome_payoffs(&self) -> Option<&[BigRational]> {
        self.outcome_payoffs.as_deref()
    }
}

impl<'a> Display for Player<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        write!(
            out,
            "\"{}\" {} {}",
            self.name.as_raw_str(),
            self.player_num,
            self.infoset
        )?;
        if let Some(name) = self.infoset_name {
            write!(out, " \"{}\"", name.as_raw_str())?;
        }
        write!(out, " {{ ")?;
        for (action, _) in self.actions.iter() {
            write!(out, "\"{}\" ", action)?;
        }
        write!(out, "}} {}", self.outcome)?;
        if let Some(name) = self.outcome_name {
            write!(out, " \"{}\"", name.as_raw_str())?;
        }
        if let Some(payoffs) = &self.outcome_payoffs {
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
/// Terminal nodes simple assign payoffs to every player in the game
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Terminal<'a> {
    name: &'a EscapedStr,
    outcome: u64,
    outcome_name: Option<&'a EscapedStr>,
    outcome_payoffs: Box<[BigRational]>,
}

impl<'a> Terminal<'a> {
    /// The name of this node
    pub fn name(&self) -> &'a EscapedStr {
        self.name
    }

    /// The outcome id
    pub fn outcome(&self) -> u64 {
        self.outcome
    }

    /// The name of this outcome
    ///
    /// Note that if omitted it may be specified on a different node with the same outcome.
    pub fn outcome_name(&self) -> Option<&'a EscapedStr> {
        self.outcome_name
    }

    /// The payoffs to every player
    pub fn outcome_payoffs(&self) -> &[BigRational] {
        &*self.outcome_payoffs
    }
}

impl<'a> Display for Terminal<'a> {
    fn fmt(&self, out: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        write!(out, "\"{}\" {}", self.name.as_raw_str(), self.outcome)?;
        if let Some(name) = self.outcome_name {
            write!(out, " \"{}\"", name.as_raw_str())?;
        }
        write!(out, " {{ ")?;
        for payoff in self.outcome_payoffs.iter() {
            write!(out, "{} ", payoff)?;
        }
        write!(out, "}}")
    }
}

fn negate(input: &str) -> IResult<&str, bool> {
    let (input, res) = opt(one_of("+-"))(input)?;
    Ok((input, res == Some('-')))
}

fn fail(input: &str) -> nom::Err<nom::error::Error<&str>> {
    nom::Err::Error(nom::error::Error::new(input, ErrorKind::Fail))
}

fn big_float(input: &str) -> IResult<&str, BigRational> {
    let (res_input, (main_neg, (int, dec), exp)) = tuple((
        negate,
        alt((
            pair(
                digit1,
                map(opt(preceded(char('.'), digit0)), Option::unwrap_or_default),
            ),
            separated_pair(digit0, char('.'), digit1),
        )),
        opt(preceded(one_of("eE"), pair(negate, digit1))),
    ))(input)?;
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
    let (input, (num, denom)) = pair(big_float, opt(preceded(char('/'), big_float)))(input)?;
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
            alt((escaped(none_of("\\\""), '\\', one_of("\\\"")), tag(""))),
            char('"'),
        ),
        EscapedStr::new,
    )(input)
}

fn spacelist<'a, O, E, F>(f: F) -> impl FnMut(&'a str) -> IResult<&'a str, Vec<O>, E>
where
    F: Parser<&'a str, O, E>,
    E: ParseError<&'a str>,
{
    delimited(
        pair(char('{'), multispace1),
        separated_list1(multispace1, f),
        pair(multispace1, char('}')),
    )
}

fn commalist<'a, O, E, F>(f: F) -> impl FnMut(&'a str) -> IResult<&'a str, Vec<O>, E>
where
    F: Parser<&'a str, O, E>,
    E: ParseError<&'a str>,
{
    delimited(
        pair(char('{'), multispace1),
        separated_list1(pair(opt(char(',')), multispace1), f),
        pair(multispace1, char('}')),
    )
}

fn node(input: &str) -> IResult<&str, Node<'_>> {
    let (input, style) = preceded(multispace1, one_of("cpt"))(input)?;
    match style {
        'c' => {
            let (input, chance) = chance(input)?;
            Ok((input, Node::Chance(chance)))
        }
        'p' => {
            let (input, play) = player(input)?;
            Ok((input, Node::Player(play)))
        }
        't' => {
            let (input, term) = terminal(input)?;
            Ok((input, Node::Terminal(term)))
        }
        _ => panic!(),
    }
}

fn chance(input: &str) -> IResult<&str, Chance<'_>> {
    let (mut input, (name, infoset, infoset_name, action_probs, outcome, outcome_payoffs)) =
        tuple((
            preceded(multispace1, label),
            preceded(multispace1, u64),
            opt(preceded(multispace1, label)),
            preceded(
                multispace1,
                spacelist(separated_pair(label, multispace1, big_rational)),
            ),
            preceded(multispace1, u64),
            opt(preceded(multispace1, commalist(big_rational))),
        ))(input)?;
    let mut actions = Vec::with_capacity(action_probs.len());
    for (name, prob) in action_probs {
        let (next_inp, next) = node(input)?;
        input = next_inp;
        actions.push((name, prob, next));
    }
    Ok((
        input,
        Chance {
            name,
            infoset,
            infoset_name,
            actions: actions.into(),
            outcome,
            outcome_payoffs: outcome_payoffs.map(|p| p.into()),
        },
    ))
}

fn player(input: &str) -> IResult<&str, Player<'_>> {
    let (
        mut res_input,
        (
            name,
            player_num,
            infoset,
            infoset_name,
            action_names,
            outcome,
            outcome_name,
            outcome_payoffs,
        ),
    ) = tuple((
        preceded(multispace1, label),
        preceded(multispace1, u64),
        preceded(multispace1, u64),
        opt(preceded(multispace1, label)),
        preceded(multispace1, spacelist(label)),
        preceded(multispace1, u64),
        opt(preceded(multispace1, label)),
        opt(preceded(multispace1, commalist(big_rational))),
    ))(input)?;
    let player_num = player_num.try_into().map_err(|_| fail(input))?;
    let mut actions = Vec::with_capacity(action_names.len());
    for name in action_names {
        let (next_inp, next) = node(res_input)?;
        res_input = next_inp;
        actions.push((name, next));
    }
    Ok((
        res_input,
        Player {
            name,
            player_num,
            infoset,
            infoset_name,
            actions: actions.into(),
            outcome,
            outcome_name,
            outcome_payoffs: outcome_payoffs.map(|p| p.into()),
        },
    ))
}

fn terminal(input: &str) -> IResult<&str, Terminal<'_>> {
    let (input, (name, outcome, outcome_name, payoffs)) = tuple((
        preceded(multispace1, label),
        preceded(multispace1, u64),
        opt(preceded(multispace1, label)),
        preceded(multispace1, commalist(big_rational)),
    ))(input)?;
    Ok((
        input,
        Terminal {
            name,
            outcome,
            outcome_name,
            outcome_payoffs: payoffs.into(),
        },
    ))
}

fn efg(input: &str) -> IResult<&str, ExtensiveFormGame<'_>> {
    let (input, (name, player_names, comment, root)) = tuple((
        preceded(
            tuple((
                multispace0,
                tag("EFG"),
                multispace1,
                tag("2"),
                multispace1,
                tag("R"),
                multispace1,
            )),
            label,
        ),
        preceded(multispace1, spacelist(label)),
        opt(preceded(multispace1, label)),
        terminated(node, multispace0),
    ))(input)?;
    Ok((
        input,
        ExtensiveFormGame {
            name,
            player_names: player_names.into(),
            comment,
            root,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::{Chance, EscapedStr, ExtensiveFormGame, Node, Player, Terminal, ValidationError};
    use num::rational::BigRational;
    use num::{One, Zero};

    fn new_game<'a>(
        name: &'a str,
        player_names: impl IntoIterator<Item = &'a str>,
        comment: Option<&'a str>,
        root: Node<'a>,
    ) -> ExtensiveFormGame<'a> {
        ExtensiveFormGame {
            name: EscapedStr::new(name),
            player_names: player_names.into_iter().map(EscapedStr::new).collect(),
            comment: comment.map(EscapedStr::new),
            root,
        }
    }

    fn new_chance<'a>(
        name: &'a str,
        infoset: u64,
        infoset_name: Option<&'a str>,
        actions: impl IntoIterator<Item = (&'a str, BigRational, Node<'a>)>,
        outcome: u64,
        outcome_payoffs: Option<Box<[BigRational]>>,
    ) -> Chance<'a> {
        Chance {
            name: EscapedStr::new(name),
            infoset,
            infoset_name: infoset_name.map(EscapedStr::new),
            actions: actions
                .into_iter()
                .map(|(name, prob, node)| (EscapedStr::new(name), prob, node))
                .collect(),
            outcome,
            outcome_payoffs,
        }
    }

    fn new_player<'a>(
        name: &'a str,
        player_num: usize,
        infoset: u64,
        infoset_name: Option<&'a str>,
        actions: impl IntoIterator<Item = (&'a str, Node<'a>)>,
        outcome: u64,
        outcome_name: Option<&'a str>,
        outcome_payoffs: Option<Box<[BigRational]>>,
    ) -> Player<'a> {
        Player {
            name: EscapedStr::new(name),
            player_num,
            infoset,
            infoset_name: infoset_name.map(EscapedStr::new),
            actions: actions
                .into_iter()
                .map(|(name, node)| (EscapedStr::new(name), node))
                .collect(),
            outcome,
            outcome_name: outcome_name.map(EscapedStr::new),
            outcome_payoffs,
        }
    }

    fn new_terminal<'a>(
        name: &'a str,
        outcome: u64,
        outcome_name: Option<&'a str>,
        outcome_payoffs: impl Into<Box<[BigRational]>>,
    ) -> Terminal<'a> {
        Terminal {
            name: EscapedStr::new(name),
            outcome,
            outcome_name: outcome_name.map(EscapedStr::new),
            outcome_payoffs: outcome_payoffs.into(),
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
        assert_eq!(label.as_raw_str(), "");

        let (input, label) = super::label(r#""normal" "#).unwrap();
        assert_eq!(input, " ");
        assert_eq!(label.as_raw_str(), "normal");

        let (input, label) = super::label(r#""esca\"ped" "#).unwrap();
        assert_eq!(input, " ");
        assert_eq!(label.as_raw_str(), "esca\\\"ped");
    }

    #[test]
    fn test_terminal() {
        let (input, term) =
            super::terminal(r#" "name" 1 "outcome name" { 10.000000 2.000000 }"#).unwrap();
        let expected = new_terminal(
            "name",
            1,
            Some("outcome name"),
            [
                BigRational::from_integer(10.into()),
                BigRational::from_integer(2.into()),
            ],
        );
        assert_eq!(input, "");
        assert_eq!(term, expected);
        assert_eq!(expected.to_string(), r#""name" 1 "outcome name" { 10 2 }"#);

        let (input, term) = super::terminal(r#" "" 2 { -1, 4/6 }"#).unwrap();
        let expected = new_terminal(
            "",
            2,
            None,
            [
                BigRational::from_integer((-1).into()),
                BigRational::new(4.into(), 6.into()),
            ],
        );
        assert_eq!(input, "");
        assert_eq!(term, expected);
        assert_eq!(expected.to_string(), r#""" 2 { -1 2/3 }"#);
    }

    #[test]
    fn test_player() {
        let (input, player) = super::player(
            r#" "name" 1 2 "infoset name" { "action" } 0
            t "" 1 { 10 }"#,
        )
        .unwrap();
        let expected = new_player(
            "name",
            1,
            2,
            Some("infoset name"),
            [(
                "action",
                Node::Terminal(new_terminal(
                    "",
                    1,
                    None,
                    [BigRational::from_integer(10.into())],
                )),
            )],
            0,
            None,
            None,
        );
        assert_eq!(input, "");
        assert_eq!(player, expected);
        assert_eq!(
            expected.to_string(),
            r#""name" 1 2 "infoset name" { "action" } 0"#
        );

        let (input, player) = super::player(
            r#" "" 2 3 { "" } 1 "outcome name" { -4.5, 6.7 }
            t "" 1 { 10 }"#,
        )
        .unwrap();
        let expected = new_player(
            "",
            2,
            3,
            None,
            [(
                "",
                Node::Terminal(new_terminal(
                    "",
                    1,
                    None,
                    [BigRational::from_integer(10.into())],
                )),
            )],
            1,
            Some("outcome name"),
            Some(
                [
                    BigRational::new((-45).into(), 10.into()),
                    BigRational::new(67.into(), 10.into()),
                ]
                .into(),
            ),
        );
        assert_eq!(input, "");
        assert_eq!(player, expected);
        assert_eq!(
            expected.to_string(),
            r#""" 2 3 { "" } 1 "outcome name" { -9/2 67/10 }"#
        );
    }

    #[test]
    fn test_chance() {
        let (input, chance) = super::chance(
            r#" "name" 1 "infoset name" { "action" 1/2 } 0
            t "" 1 { 10 }"#,
        )
        .unwrap();
        let expected = new_chance(
            "name",
            1,
            Some("infoset name"),
            [(
                "action",
                BigRational::new(1.into(), 2.into()),
                Node::Terminal(new_terminal(
                    "",
                    1,
                    None,
                    [BigRational::from_integer(10.into())],
                )),
            )],
            0,
            None,
        );
        assert_eq!(input, "");
        assert_eq!(chance, expected);
        assert_eq!(
            expected.to_string(),
            r#""name" 1 "infoset name" { "action" 1/2 } 0"#
        );

        let (input, chance) = super::chance(
            r#" "" 2 { "" 1 } 4 { 0.1, 4 }
            t "" 1 { 10 }"#,
        )
        .unwrap();
        let expected = new_chance(
            "",
            2,
            None,
            [(
                "",
                BigRational::from_integer(1.into()),
                Node::Terminal(new_terminal(
                    "",
                    1,
                    None,
                    [BigRational::from_integer(10.into())],
                )),
            )],
            4,
            Some(
                [
                    BigRational::new(1.into(), 10.into()),
                    BigRational::from_integer(4.into()),
                ]
                .into(),
            ),
        );
        assert_eq!(input, "");
        assert_eq!(chance, expected);
        assert_eq!(expected.to_string(), r#""" 2 { "" 1 } 4 { 1/10 4 }"#);
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
        let (input, efg) = super::efg(game_str).unwrap();
        let expected = new_game(
            "General Bayes game, one stage",
            ["Player 1", "Player 2"],
            Some("A single stage General Bayes Game"),
            Node::Chance(new_chance(
                "ROOT",
                1,
                Some("(0,1)"),
                [
                    (
                        "1G",
                        BigRational::new(1.into(), 2.into()),
                        Node::Player(new_player(
                            "",
                            1,
                            1,
                            Some("(1,1)"),
                            [
                                (
                                    "H",
                                    Node::Terminal(new_terminal(
                                        "",
                                        1,
                                        Some("Outcome 1"),
                                        [
                                            BigRational::from_integer(10.into()),
                                            BigRational::from_integer(2.into()),
                                        ],
                                    )),
                                ),
                                (
                                    "L",
                                    Node::Terminal(new_terminal(
                                        "",
                                        2,
                                        Some("Outcome 2"),
                                        [
                                            BigRational::from_integer(0.into()),
                                            BigRational::from_integer(10.into()),
                                        ],
                                    )),
                                ),
                            ],
                            0,
                            None,
                            None,
                        )),
                    ),
                    (
                        "1B",
                        BigRational::new(1.into(), 2.into()),
                        Node::Player(new_player(
                            "",
                            2,
                            1,
                            Some("(2,1)"),
                            [
                                (
                                    "h",
                                    Node::Terminal(new_terminal(
                                        "",
                                        3,
                                        Some("Outcome 3"),
                                        [
                                            BigRational::from_integer(2.into()),
                                            BigRational::from_integer(4.into()),
                                        ],
                                    )),
                                ),
                                (
                                    "l",
                                    Node::Terminal(new_terminal(
                                        "",
                                        4,
                                        Some("Outcome 4"),
                                        [
                                            BigRational::from_integer(4.into()),
                                            BigRational::from_integer(0.into()),
                                        ],
                                    )),
                                ),
                            ],
                            0,
                            None,
                            None,
                        )),
                    ),
                ],
                0,
                None,
            )),
        );
        assert_eq!(input, "");
        assert_eq!(efg, expected);
        assert_eq!(
            expected.to_string(),
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
    }

    fn empty_game<'a>(
        player_names: impl IntoIterator<Item = &'a str>,
        root: Node<'a>,
    ) -> ExtensiveFormGame<'a> {
        ExtensiveFormGame {
            name: EscapedStr::new(""),
            player_names: player_names.into_iter().map(EscapedStr::new).collect(),
            comment: None,
            root,
        }
    }

    #[test]
    fn not_distribution() {
        let game = empty_game(
            ["1", "2"],
            Node::Chance(new_chance(
                "",
                1,
                None,
                [(
                    "",
                    BigRational::new(9.into(), 10.into()),
                    Node::Terminal(new_terminal("", 1, None, vec![BigRational::zero(); 2])),
                )],
                0,
                None,
            )),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::ChanceNotDistribution
        );
    }

    #[test]
    fn invalid_player_num() {
        let game = empty_game(
            ["1", "2"],
            Node::Player(new_player("", 3, 1, None, [], 0, None, None)),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::InvalidPlayerNum
        );
    }

    #[test]
    fn invalid_infoset_names() {
        let game = empty_game(
            ["1", "2"],
            Node::Player(new_player(
                "",
                1,
                1,
                Some("a"),
                [(
                    "",
                    Node::Player(new_player("", 1, 1, Some("b"), [], 0, None, None)),
                )],
                0,
                None,
                None,
            )),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::NonMatchingInfosetNames
        );
    }

    #[test]
    fn invalid_chance_infoset_names() {
        let game = empty_game(
            ["1", "2"],
            Node::Chance(new_chance(
                "",
                1,
                Some("a"),
                [(
                    "",
                    BigRational::one(),
                    Node::Chance(new_chance(
                        "",
                        1,
                        Some("b"),
                        [(
                            "",
                            BigRational::one(),
                            Node::Terminal(new_terminal("", 1, None, vec![BigRational::zero(); 2])),
                        )],
                        0,
                        None,
                    )),
                )],
                0,
                None,
            )),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::NonMatchingInfosetNames
        );
    }

    #[test]
    fn invalid_infoset_actions() {
        let game = empty_game(
            ["1", "2"],
            Node::Player(new_player(
                "",
                1,
                1,
                None,
                [(
                    "",
                    Node::Player(new_player("", 1, 1, None, [], 0, None, None)),
                )],
                0,
                None,
                None,
            )),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::NonMatchingInfosetActions
        );
    }

    #[test]
    fn invalid_chance_infoset_actions() {
        let game = empty_game(
            ["1", "2"],
            Node::Chance(new_chance(
                "",
                1,
                None,
                [(
                    "",
                    BigRational::one(),
                    Node::Chance(new_chance(
                        "",
                        1,
                        None,
                        [(
                            "a",
                            BigRational::one(),
                            Node::Terminal(new_terminal("", 0, None, vec![BigRational::zero(); 2])),
                        )],
                        0,
                        None,
                    )),
                )],
                0,
                None,
            )),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::NonMatchingInfosetActions
        );
    }

    #[test]
    fn null_outcome_payoffs() {
        let game = empty_game(
            ["1", "2"],
            Node::Player(new_player(
                "",
                1,
                1,
                None,
                [],
                0,
                None,
                Some(vec![BigRational::zero(); 2].into()),
            )),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::NullOutcomePayoffs
        );
    }

    #[test]
    fn invalid_payoff_number() {
        let game = empty_game(
            ["1", "2"],
            Node::Terminal(new_terminal("", 1, None, [BigRational::zero()])),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::InvalidNumberOfPayoffs
        );
    }

    #[test]
    fn non_matching_outcome_names() {
        let game = empty_game(
            ["1", "2"],
            Node::Player(new_player(
                "",
                1,
                1,
                None,
                [(
                    "",
                    Node::Player(new_player("", 1, 2, None, [], 1, Some("a"), None)),
                )],
                1,
                Some("b"),
                None,
            )),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::NonMatchingOutcomeNames
        );
    }

    #[test]
    fn non_matching_outcome_payoffs() {
        let game = empty_game(
            ["1", "2"],
            Node::Player(new_player(
                "",
                1,
                1,
                None,
                [(
                    "",
                    Node::Player(new_player(
                        "",
                        1,
                        2,
                        None,
                        [],
                        1,
                        None,
                        Some(vec![BigRational::one(); 2].into()),
                    )),
                )],
                1,
                None,
                Some(vec![BigRational::zero(); 2].into()),
            )),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::NonMatchingOutcomePayoffs
        );
    }

    #[test]
    fn no_outcome_payoffs() {
        let game = empty_game(
            ["1", "2"],
            Node::Player(new_player("", 1, 1, None, [], 1, None, None)),
        );
        assert_eq!(
            game.validate().unwrap_err(),
            ValidationError::NoOutcomePayoffs
        );
    }
}
