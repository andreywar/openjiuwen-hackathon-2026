"""Zero-token solver for the warehouse arithmetic tasks.

The statement is a chain of simple steps applied to a stock of boxes. Every
content sentence must be matched *completely* by one of the step patterns; if
anything is left unexplained the solver answers UNKNOWN and the task goes to
the LLM tier instead of risking a wrong number.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from .base import NUM, Outcome, content_sentences, parse_int


class Infeasible(Exception):
    pass


class Unknown(Exception):
    pass


_SUBJ = r"(?:it|the warehouse|the stock|the inventory|the depot)"
_STOCK = r"(?:the |its )?(?:stock|inventory|it|total|amount|number of boxes)"
_UNITS = r"(?:warehouses|parts|sites|locations|depots|shares|groups|stores|branches|portions)"
_ROUND = r"(?:[ ,]*\(?(rounded down|rounded up|round down|round up|floor|ceiling)\)?)?"
_OUT = r"(?:ships out|ships|sends out|sends|sells|dispatches|loses|delivers|removes|discards|gives away|donates)"
_IN = r"(?:receives|gets|gains|adds|acquires|buys|takes in|accepts|restocks)"
_PER_DAY = r"(?:per|each|every|a) day"
_BATCH = r"(?:deliveries|delivery|shipments|shipment|batches|batch|loads|load|pallets|pallet|trucks|truck|orders|order|crates|crate)"
_MULT = {"doubled": 2, "tripled": 3, "quadrupled": 4, "doubles": 2, "triples": 3, "quadruples": 4}


def _divide(stock: int, n: int, rounding: Optional[str]) -> int:
    if n == 0:
        raise Infeasible("division by zero (split into 0 parts)")
    if n < 0:
        raise Infeasible("split into a negative number of parts")
    q, r = divmod(stock, n)
    if r == 0:
        return q
    if rounding is None:
        raise Unknown("non-integer share and no rounding rule")
    return q + 1 if ("up" in rounding or "ceil" in rounding) else q


def _rules() -> List[Tuple[re.Pattern, Callable]]:
    rules: List[Tuple[str, Callable]] = []

    def add(pattern: str, fn: Callable) -> None:
        rules.append((re.compile("^" + pattern + "$"), fn))

    # split equally into N warehouses, keep one share
    add(rf"{_STOCK} is (?:then )?(?:split|divided|shared) (?:equally|evenly) "
        rf"(?:into|among|between|across) {NUM} {_UNITS},? and {_SUBJ} keeps (?:only )?one (?:share|part|portion){_ROUND}",
        lambda s, m: _divide(s, parse_int(m.group(1)), m.group(2)))
    add(rf"{_SUBJ} (?:splits|divides) {_STOCK} (?:equally|evenly) (?:into|among|between|across) {NUM} {_UNITS},? "
        rf"and keeps (?:only )?one (?:share|part|portion){_ROUND}",
        lambda s, m: _divide(s, parse_int(m.group(1)), m.group(2)))
    # doubled / tripled
    add(rf"{_STOCK} is (?:then )?(doubled|tripled|quadrupled)(?: (?:by|after|through|when) [a-z ,'-]+)?",
        lambda s, m: s * _MULT[m.group(1)])
    add(rf"{_SUBJ} (doubles|triples|quadruples) {_STOCK}(?: (?:by|after|through) [a-z ,'-]+)?",
        lambda s, m: s * _MULT[m.group(1)])
    add(rf"{_STOCK} is (?:then )?halved{_ROUND}",
        lambda s, m: _divide(s, 2, m.group(1)))
    # X boxes per day for D days
    add(rf"(?:{_SUBJ} )?{_OUT} {NUM} boxes {_PER_DAY} for {NUM} days?",
        lambda s, m: s - parse_int(m.group(1)) * parse_int(m.group(2)))
    add(rf"(?:{_SUBJ} )?{_IN} {NUM} boxes {_PER_DAY} for {NUM} days?",
        lambda s, m: s + parse_int(m.group(1)) * parse_int(m.group(2)))
    # N deliveries of M boxes each
    add(rf"(?:{_SUBJ} )?{_IN} {NUM} {_BATCH} of {NUM} boxes(?: each| apiece)?",
        lambda s, m: s + parse_int(m.group(1)) * parse_int(m.group(2)))
    add(rf"(?:{_SUBJ} )?{_OUT} {NUM} {_BATCH} of {NUM} boxes(?: each| apiece)?",
        lambda s, m: s - parse_int(m.group(1)) * parse_int(m.group(2)))
    # plain +/- X boxes
    add(rf"(?:{_SUBJ} )?{_IN} (?:another |an additional )?{NUM} (?:more |new |additional |extra )?boxes",
        lambda s, m: s + parse_int(m.group(1)))
    add(rf"(?:{_SUBJ} )?{_OUT} (?:another |an additional )?{NUM} (?:more |additional |extra )?boxes",
        lambda s, m: s - parse_int(m.group(1)))
    add(rf"{NUM} (?:more |new |additional |extra )?boxes are (?:added|received|delivered|bought)",
        lambda s, m: s + parse_int(m.group(1)))
    add(rf"{NUM} (?:more |additional |extra )?boxes are (?:removed|shipped|sold|lost|discarded|sent out|shipped out)",
        lambda s, m: s - parse_int(m.group(1)))
    return rules


_RULES = _rules()

_START = [
    re.compile(rf"^(?:a|the|one|this|our) (?:warehouse|depot|storehouse|store) "
               rf"(?:starts|begins|opens|started|began|opened)(?: out| off)?(?: the (?:day|week|month))? with {NUM} boxes$"),
    re.compile(rf"^(?:a|the|one|this|our) (?:warehouse|depot|storehouse|store) (?:initially |currently |now )?"
               rf"(?:has|holds|contains|stores) {NUM} boxes(?: initially| at the start| at first| in stock)?$"),
    re.compile(rf"^(?:initially|at the start|at first),? (?:a|the) (?:warehouse|depot|storehouse|store) "
               rf"(?:has|holds|contains|stores) {NUM} boxes$"),
]

_QUESTION = [
    re.compile(r"^how many boxes (?:does|will|would|did|do) (?:it|the warehouse|the depot) (?:then |now |finally )?"
               r"(?:have|hold|contain|store)(?: left| remaining| in stock)?"
               r"(?: at the end| in the end| now| finally| after (?:all )?(?:this|that|these steps|of (?:this|that)))?$"),
    re.compile(r"^how many boxes (?:are|remain|are there|will there be|are left|remain there)(?: left| remaining)?"
               r"(?: in (?:it|the warehouse|stock))?(?: at the end| in the end| now| finally)?$"),
    re.compile(r"^what is the (?:final|resulting|remaining) (?:number of boxes|stock|box count)"
               r"(?: in the warehouse| at the end)?$"),
]

_LEAD = re.compile(
    r"^(?:(?:and )?(?:then|next|after that|afterwards|afterward|finally|later|subsequently|"
    r"first|second|third|lastly|after this|following that))[, ]+")


def _normalise(sentence: str) -> str:
    s = sentence.strip().lower()
    s = re.sub(r"[.?!]+$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    prev = None
    while prev != s:
        prev = s
        s = _LEAD.sub("", s)
    return s


def _apply_step(stock: int, clause: str) -> Optional[int]:
    for rx, fn in _RULES:
        m = rx.match(clause)
        if m:
            return fn(stock, m)
    return None


def _apply_sentence(stock: int, sentence: str) -> int:
    whole = _apply_step(stock, sentence)
    if whole is not None:
        return whole
    # Compound sentence: every clause must be a known step.
    clauses = [c.strip() for c in re.split(r"\s*(?:;|, and then|, then| and then|, and| and)\s+", sentence) if c.strip()]
    if len(clauses) < 2:
        raise Unknown("unrecognised step: %r" % sentence)
    for c in clauses:
        c = _normalise(c)
        nxt = _apply_step(stock, c)
        if nxt is None:
            raise Unknown("unrecognised step: %r" % c)
        stock = nxt
    return stock


_VAGUE = re.compile(
    r"\b(?:an? (?:unknown|unspecified|uncertain|undisclosed|certain) (?:number|amount|quantity) of|"
    r"some number of|a number of|some|several|a few|many|unknown|unspecified|[xnk])\b(?= (?:more |new )?"
    r"(?:boxes|days?|deliveries|shipments|batches|warehouses|parts|sites)\b)")


def _missing_quantity(sentence: str, opening: bool) -> bool:
    """True if the sentence is a known step whose number is replaced by "some" / "unknown"."""
    if not _VAGUE.search(sentence):
        return False
    probe = _VAGUE.sub("1", sentence)
    if opening:
        return any(rx.match(probe) for rx in _START)
    try:
        return _apply_step(7, probe) is not None
    except (Infeasible, Unknown):
        return True


def solve(instruction: str) -> Outcome:
    sentences = [_normalise(s) for s in content_sentences(instruction)]
    sentences = [s for s in sentences if s]
    if len(sentences) < 2:
        return Outcome.unknown("statement too short")

    stock: Optional[int] = None
    for rx in _START:
        m = rx.match(sentences[0])
        if m:
            stock = parse_int(m.group(1))
            break
    if stock is None:
        if _missing_quantity(sentences[0], True):
            return Outcome.infeasible("the initial number of boxes is not given")
        return Outcome.unknown("unrecognised opening: %r" % sentences[0])
    if not any(rx.match(sentences[-1]) for rx in _QUESTION):
        return Outcome.unknown("unrecognised question: %r" % sentences[-1])

    steps = sentences[1:-1]
    try:
        for s in steps:
            stock = _apply_sentence(stock, s)
    except Infeasible as e:
        return Outcome.infeasible(str(e))
    except Unknown as e:
        if any(_missing_quantity(s, False) for s in steps):
            return Outcome.infeasible("a quantity needed for the computation is not given")
        return Outcome.unknown(str(e))
    except Exception as e:  # noqa: BLE001
        return Outcome.unknown("math solver error: %s" % e)
    return Outcome.solved(stock, "parsed %d steps" % len(steps), steps=len(steps))
