"""Zero-token solver for the knowledge-base lookup tasks.

The knowledge base is a directory of small markdown cards::

    # Sergei Garcia
    Born: 1985
    Employer: Fjord Data
    Mentor: Dmitri Smith

A question names one entity and one or more relations ("the headquarters of
Sergei Garcia's employer"). The solver finds the entity, maps the words of the
question to relations and follows the chain through the cards. The cards are
typed implicitly (a person has no ``Headquarters``), so the order of a chain
is validated against the data itself. All words of the question must be
accounted for, otherwise the solver answers UNKNOWN.

Free infeasibility checks: the entity is not in the knowledge base at all, a
card or a fact needed by the chain is missing, the cards contradict each other.
"""
from __future__ import annotations

import itertools
import os
import re
from typing import Dict, List, Optional, Tuple

from .. import wsfiles
from .base import Outcome, content_sentences

_FACT = re.compile(
    r"^\s*(?:[-*+]\s+|\|\s*)?(?:\*\*|__)?([A-Za-z][A-Za-z0-9 _/()-]{0,40}?)(?:\*\*|__)?\s*(?::|\|)\s*(?:\*\*|__)?\s*(.+?)\s*\|?\s*$")

# Relation synonyms: canonical name -> (card keys, noun triggers, verb triggers)
_REL = {
    "born": (
        ["born", "birth year", "year of birth", "birth", "birthyear", "date of birth", "birthdate", "birth date"],
        [r"birth ?year", r"year of birth", r"birth ?date", r"date of birth", r"birthday"],
        [r"born"],
    ),
    "employer": (
        ["employer", "company", "works at", "works for", "workplace", "employed by", "employed at", "organization",
         "organisation", "employment"],
        [r"employers?", r"workplace"],
        [r"works?(?: for| at)?", r"working(?: for| at)?", r"employed(?: by| at)?", r"employs", r"worked(?: for| at)?"],
    ),
    "mentor": (
        ["mentor", "mentored by", "advisor", "adviser", "supervisor"],
        [r"mentors?"],
        [],
    ),
    "headquarters": (
        ["headquarters", "hq", "headquartered", "headquartered in", "head office", "based in", "location", "city",
         "headquarter"],
        [r"headquarters?", r"hq", r"head office", r"home city", r"home base"],
        [r"headquartered", r"based", r"located", r"situated"],
    ),
    "founded": (
        ["founded", "established", "founding year", "year founded", "founded in", "year of foundation", "founding"],
        [r"founding year", r"year of foundation", r"foundation year", r"founding"],
        [r"founded", r"established", r"set up", r"started"],
    ),
}

_STOP = set("""
the of is was were are in at a an s does do did that city year town place
company person name named for by to has have had their his her its whose currently now it be been on
according knowledge base kb firm organisation organization people individual
""".split())


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", s.strip().lower())).strip()


class Card:
    def __init__(self, name: str, kind: str, path: str):
        self.name = name
        self.kind = kind
        self.path = path
        self.facts: Dict[str, str] = {}      # normalised key -> value
        self.conflicts: set = set()

    def add(self, key: str, value: str) -> None:
        k = _norm(key)
        if k in self.facts and self.facts[k] != value:
            self.conflicts.add(k)
        self.facts.setdefault(k, value)


class KB:
    def __init__(self) -> None:
        self.cards: Dict[str, Card] = {}
        self.raw_text = ""
        self.files = 0
        self.parsed_files = 0  # files that produced a titled card with at least one fact
        self.dupes: set = set()

    @property
    def well_formed(self) -> bool:
        """Every file was understood, so "not found" really means "not there"."""
        return self.files > 0 and self.parsed_files == self.files

    def get(self, name: str) -> Optional[Card]:
        return self.cards.get(_norm(name))


def find_kb_dir(instruction: str, workspace: str) -> Optional[str]:
    m = re.search(r"directory\s+`?([\w./-]+?)/?`?\s", instruction or "")
    for rel in ([m.group(1)] if m else []) + ["kb"]:
        c = wsfiles.safe_dir(workspace, rel)  # never outside the workspace
        if c is not None:
            return c
    return None


def load_kb(root: str) -> KB:
    kb = KB()
    texts = []
    for _rel, path in wsfiles.list_files(root):
        if True:
            fn = os.path.basename(path)
            if not fn.lower().endswith((".md", ".txt", ".markdown")):
                continue
            try:
                with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            kb.files += 1
            texts.append(text)
            stem = os.path.splitext(fn)[0]
            kind = ""
            m = re.match(r"(person|people|company|companies|org|organization)[_-](.+)", stem, re.I)
            file_name = stem
            if m:
                kind = "person" if m.group(1).lower().startswith("pe") else "company"
                file_name = m.group(2)
            title = None
            facts: List[Tuple[str, str]] = []
            for line in text.splitlines():
                h = re.match(r"^\s*#+\s*(.+?)\s*#*\s*$", line)
                if h:
                    if title is None:
                        title = h.group(1).strip()
                    continue
                fm = _FACT.match(line)
                if fm and not set(fm.group(1)) <= set("-| "):
                    facts.append((fm.group(1).strip(), fm.group(2).strip()))
            for k, v in list(facts):
                if title is None and _norm(k) in ("name", "full name", "title"):
                    title = v
            if title and facts:
                kb.parsed_files += 1
            name = title or re.sub(r"[_-]+", " ", file_name).title()
            key = _norm(name)
            card = kb.cards.get(key)
            if card is None:
                card = Card(name, kind, path)
                kb.cards[key] = card
            else:
                kb.dupes.add(key)
            for k, v in facts:
                card.add(k, v)
    kb.raw_text = "\n".join(texts).lower()
    return kb


def _question(instruction: str) -> Optional[str]:
    sents = content_sentences(instruction)
    qs = [s for s in sents if not re.search(r"\b(?:directory|folder)\b.*\bcontains?\b|knowledge base of", s, re.I)]
    marked = [s for s in qs if s.rstrip().endswith("?")]
    if len(marked) == 1:
        return marked[0]
    if len(qs) == 1:
        return qs[0]
    return None


def _find_entities(q: str, kb: KB) -> List[Tuple[int, int, Card]]:
    low = q.lower()
    hits = []
    for key, card in kb.cards.items():
        pat = r"(?<![\w])" + r"[\s_-]+".join(re.escape(p) for p in key.split()) + r"(?![\w])"
        for m in re.finditer(pat, low):
            hits.append((m.start(), m.end(), card))
    # prefer longer matches, drop matches nested in a longer one
    hits.sort(key=lambda h: (-(h[1] - h[0]), h[0]))
    chosen: List[Tuple[int, int, Card]] = []
    for h in hits:
        if not any(h[0] < c[1] and c[0] < h[1] for c in chosen):
            chosen.append(h)
    return sorted(chosen, key=lambda h: h[0])


_WH_HEAD = re.compile(
    r"^(?:in|at|on|for|by|from)?\s*(?:which|what)\s+(year|city|town|place|country|company|firm|organi[sz]ation|person)\b"
    r"|^(who|whom|where|when)\b|^(?:what|which)\b", re.I)


def _answer_hint(q: str) -> str:
    m = _WH_HEAD.search(q.strip())
    if not m:
        return ""
    w = (m.group(1) or m.group(2) or "").lower()
    if w in ("year", "when"):
        return "year"
    if w in ("city", "town", "place", "where", "country"):
        return "place"
    if w == "person":
        return "person"
    if w in ("who", "whom"):
        return "name"   # a person or an organisation, but not a year
    if w in ("company", "firm", "organization", "organisation"):
        return "company"
    return ""


def _relation_mentions(text: str, kb_keys: List[str]) -> Tuple[List[Tuple[int, int, str, str]], str]:
    """Find relation words. Returns (mentions, text with the mentions blanked).

    A mention is (start, end, relation, 'noun'|'verb'). ``relation`` is a
    canonical name from ``_REL`` or ``key:<card key>`` for keys that only
    exist in this knowledge base.
    """
    spans: List[Tuple[int, int, str, str]] = []
    cands: List[Tuple[str, str, str]] = []
    for rel, (_keys, nouns, verbs) in _REL.items():
        for n in nouns:
            cands.append((n, rel, "noun"))
        for v in verbs:
            cands.append((v, rel, "verb"))
    known = {k for keys, _n, _v in _REL.values() for k in keys}
    for k in kb_keys:
        if k not in known and len(k) >= 3:
            cands.append((re.escape(k) + r"s?", "key:" + k, "noun"))
    found = []
    for pat, rel, kind in cands:
        for m in re.finditer(r"(?<![\w])(?:" + pat + r")(?![\w])", text):
            found.append((m.start(), m.end(), rel, kind))
    found.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    for s in found:
        if not any(s[0] < c[1] and c[0] < s[1] for c in spans):
            spans.append(s)
    spans.sort(key=lambda s: s[0])
    chars = list(text)
    for a, b, _r, _k in spans:
        for i in range(a, b):
            chars[i] = " "
    return spans, "".join(chars)


def _card_key(card: Card, rel: str) -> Optional[str]:
    if rel.startswith("key:"):
        k = rel[4:]
        return k if k in card.facts else None
    for k in _REL[rel][0]:
        if k in card.facts:
            return k
    return None


class ChainError(Exception):
    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind = kind  # 'missing' (data absent) or 'conflict'


def _follow(kb: KB, card: Card, chain: List[str]) -> str:
    value = card.name
    cur: Optional[Card] = card
    for i, rel in enumerate(chain):
        if cur is None:
            raise ChainError("missing", "no card for %r" % value)
        if _norm(cur.name) in kb.dupes and cur.conflicts:
            raise ChainError("conflict", "contradictory cards for %r" % cur.name)
        k = _card_key(cur, rel)
        if k is None:
            raise ChainError("missing", "%r has no %s" % (cur.name, rel))
        if k in cur.conflicts:
            raise ChainError("conflict", "contradictory %s for %r" % (rel, cur.name))
        value = cur.facts[k]
        if not value or value.lower() in ("unknown", "n/a", "none", "?", "-", "tbd", "null"):
            raise ChainError("missing", "%s of %r is not recorded" % (rel, cur.name))
        cur = kb.get(value)
    return value


def _hint_ok(hint: str, value: str, kb: KB) -> bool:
    is_year = bool(re.fullmatch(r"\d{3,4}", value.strip()))
    card = kb.get(value)
    if hint == "year":
        return is_year or bool(re.search(r"\b\d{4}\b", value))
    if hint == "place":
        return not is_year and (card is None)
    if hint == "person":
        return not is_year and (card is None or card.kind in ("", "person"))
    if hint == "name":
        return not is_year
    if hint == "company":
        return not is_year and (card is None or card.kind in ("", "company"))
    return True


_INVERSE_VERBS = {"mentor": "mentor", "advise": "mentor", "supervise": "mentor", "employ": "employer"}


def _inverse_relation(before: str, after: str) -> Optional[str]:
    """Relation asked "backwards" by the question, or None.

    Recognised: "Who does X mentor?", "Whom did X advise?", "X mentors whom?",
    "Who is X the mentor of?". ``before`` / ``after`` are the lower-cased parts
    of the question around the entity X.
    """
    b = re.sub(r"\s+", " ", before.strip())
    a = re.sub(r"\s+", " ", after.strip().rstrip("?.! ").strip())
    verbs = "|".join(_INVERSE_VERBS)
    who = r"(?:who|whom|which person|what person)"
    m = re.fullmatch(r"(%s)(?:e?s|ed|d)?(?: (?:now|currently))?" % verbs, a)
    if m and re.fullmatch(who + r" (?:does|do|did)", b):
        return _INVERSE_VERBS[m.group(1)]
    m = re.fullmatch(r"(%s)(?:e?s|ed|d) %s" % (verbs, who), a)
    if m and not b:
        return _INVERSE_VERBS[m.group(1)]
    if re.fullmatch(who + r" (?:is|was)", b) and re.fullmatch(r"(?:the |a )?mentor (?:of|to|for)", a):
        return "mentor"
    return None


def solve(instruction: str, workspace: str) -> Outcome:
    try:
        return _solve(instruction, workspace)
    except Exception as e:  # noqa: BLE001
        return Outcome.unknown("search solver error: %s: %s" % (type(e).__name__, e))


def _solve(instruction: str, workspace: str) -> Outcome:
    root = find_kb_dir(instruction, workspace)
    if root is None:
        has_docs = any(rel.lower().endswith((".md", ".txt")) for rel, _p in wsfiles.list_files(workspace))
        if not has_docs:
            return Outcome.infeasible("knowledge base directory is missing")
        root = workspace
    kb = load_kb(root)
    if kb.files == 0:
        return Outcome.infeasible("knowledge base is empty")
    if not kb.cards:
        return Outcome.unknown("knowledge base has an unknown format")

    q = _question(instruction)
    if not q:
        return Outcome.unknown("no single question found")
    hint = _answer_hint(q)

    ents = _find_entities(q, kb)
    if not ents:
        # A proper name that the knowledge base never mentions -> unsolvable.
        body = re.sub(r"^\W*\w+", "", q)  # skip the sentence-initial capital
        names = re.findall(r"[A-Z][\w.'-]*(?:\s+(?:[A-Z][\w.'-]*|de|van|von|al|bin|del|la))*", body)
        names = [re.sub(r"'s$", "", n).strip() for n in names]
        names = [n for n in names if n and n.lower() not in _STOP]
        if names and kb.well_formed and all(_norm(n) not in _norm(kb.raw_text) for n in names):
            return Outcome.infeasible("entity %r does not occur in the knowledge base" % names[0])
        return Outcome.unknown("no known entity in the question")
    if len(ents) > 1:
        return Outcome.unknown("several entities in the question")
    start, end, card = ents[0]

    low = q.lower()
    kb_keys = sorted({k for c in kb.cards.values() for k in c.facts})
    before, after = low[:start], low[end:]
    after = re.sub(r"^['’]s\b|^['’]", " ", after)
    wh = _WH_HEAD.search(before.strip() or after.strip())
    m_before, rest_before = _relation_mentions(before, kb_keys)
    m_after, rest_after = _relation_mentions(after, kb_keys)

    # Words of the answer-type head ("in which city") are not relations.
    def drop_head(mentions, text_before: str):
        m = _WH_HEAD.search(text_before.strip())
        if not m:
            return mentions
        head_end = len(text_before) - len(text_before.lstrip()) + m.end()
        return [x for x in mentions if x[0] >= head_end]

    m_before = drop_head(m_before, before)
    if not before.strip():
        m_after = drop_head(m_after, after)

    # Inverse questions: the entity is the *value* of the relation, not its owner
    # ("Who does X mentor?" asks for the card whose Mentor is X - not for X's mentor).
    inverse = _inverse_relation(before, after)
    if inverse is not None:
        owners = []
        for c in kb.cards.values():
            k = _card_key(c, inverse)
            if k is not None and _norm(c.facts[k]) == _norm(card.name):
                owners.append(c.name)
        if len(owners) == 1:
            return Outcome.solved(owners[0], "inverse %s of %s" % (inverse, card.name), chain=["~" + inverse],
                                  entity=card.name)
        # none or several: no single answer can be justified without guessing
        return Outcome.unknown("inverse relation with %d matching cards" % len(owners))
    if re.search(r"\bmentored\b|\bmentoring\b|\bmentees?\b|\bproteges?\b|\bemployees?\b", low):
        return Outcome.unknown("inverse relation wording")
    if hint in ("person", "name") and not m_before and all(k == "verb" for *_x, k in m_after) and card.kind == "company":
        return Outcome.unknown("inverse relation wording")

    # The leading question head ("in which year", "who") is consumed here; question
    # words anywhere else mean a compound question, which is left to the LLM tier.
    head_src = rest_before if before.strip() else rest_after
    hm = _WH_HEAD.search(head_src.strip())
    if hm:
        cut = len(head_src) - len(head_src.lstrip()) + hm.end()
        if before.strip():
            rest_before = " " * cut + rest_before[cut:]
        else:
            rest_after = " " * cut + rest_after[cut:]
    leftover = re.sub(r"[^a-z0-9' ]+", " ", rest_before + " " + rest_after)
    leftover = re.sub(r"'s\b|'", " ", leftover)
    extra = [w for w in leftover.split() if w not in _STOP]
    if extra:
        return Outcome.unknown("unexplained words in the question: %s" % " ".join(extra[:6]))

    nouns_after = [r for _a, _b, r, k in m_after if k == "noun"]
    nouns_before = [r for _a, _b, r, k in m_before if k == "noun"]
    verbs = [r for _a, _b, r, k in sorted(m_before + [(a + end, b + end, r, k) for a, b, r, k in m_after])
             if k == "verb"]
    chain = nouns_after + list(reversed(nouns_before)) + verbs
    if not chain:
        return Outcome.unknown("no relation found in the question")
    if len(chain) > 4:
        return Outcome.unknown("relation chain too long")

    results = []
    first_error: Optional[ChainError] = None
    seen = set()
    for perm in [tuple(chain)] + list(itertools.permutations(chain)):
        if perm in seen:
            continue
        seen.add(perm)
        try:
            val = _follow(kb, card, list(perm))
        except ChainError as e:
            if first_error is None:
                first_error = e
            continue
        if _hint_ok(hint, val, kb):
            results.append((perm, val))
    if not results:
        if first_error is not None and kb.well_formed:
            return Outcome.infeasible("knowledge base cannot answer: %s" % first_error, chain=chain)
        if first_error is not None:
            return Outcome.unknown("chain failed in a knowledge base with an unusual format: %s" % first_error)
        return Outcome.unknown("chain resolved to a value of an unexpected kind")
    answers = {v for _p, v in results}
    if len(answers) > 1 and results[0][0] != tuple(chain):
        return Outcome.unknown("relation order is ambiguous")
    perm, value = results[0]
    if hint == "year" and not re.fullmatch(r"\d{3,4}", value.strip()):
        years = re.findall(r"\b\d{4}\b", value)
        if len(years) != 1:
            return Outcome.unknown("cannot extract a year from %r" % value)
        value = years[0]
    return Outcome.solved(value, "%s -> %s" % (card.name, " -> ".join(perm)), chain=list(perm), entity=card.name)
