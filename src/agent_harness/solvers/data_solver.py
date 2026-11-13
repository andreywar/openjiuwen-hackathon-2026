"""Zero-token solver for the CSV (sales) questions.

The question is decomposed into *filters* (``product 'widget'``, ``months 5
through 6 inclusive``, ``region 'west'`` ...) and one *metric* (sum of a
column, row count, revenue, average, min/max, arg-max by group, distinct
count). Every word of the question has to be explained by one of the two;
otherwise the solver answers UNKNOWN and the LLM tier takes over.

Infeasibility that is detected for free: missing/empty file, a column that the
statement relies on but the file does not have, non-numeric values where
numbers are required, aggregates over an empty selection.
"""
from __future__ import annotations

import csv
import io
import math
import os
import re
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .. import wsfiles
from .base import Outcome, content_sentences


class Infeasible(Exception):
    pass


class Unknown(Exception):
    pass


# ----------------------------------------------------------------- loading
def find_csv(instruction: str, workspace: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (path, name mentioned in the instruction)."""
    named = re.findall(r"([\w./\\-]+\.csv)\b", instruction or "", flags=re.I)
    present = wsfiles.top_level_names(workspace, (".csv",))
    for n in named:
        p = wsfiles.safe_file(workspace, n)  # stays inside the workspace, links included
        if p is not None:
            return p, n
    if named:
        return None, named[0]
    if len(present) == 1:
        return wsfiles.safe_file(workspace, present[0]), None
    return None, None


def load_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8-sig", "replace")
    if not text.strip():
        raise Infeasible("data file is empty")
    first = text.splitlines()[0]
    delim = max([",", ";", "\t", "|"], key=first.count)
    if first.count(delim) == 0:
        delim = ","
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise Infeasible("data file is empty")
    header = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        out.append({h: r[i].strip() for i, h in enumerate(header)})
    return header, out


def _num(v: str) -> Optional[Fraction]:
    v = (v or "").strip().replace(",", "") if v else ""
    if not re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)", v):
        return None
    return Fraction(v)


# ----------------------------------------------------------------- parsing
_PLURAL = {"units": "unit", "months": "month", "prices": "price", "regions": "region",
           "products": "product", "orders": "order"}
_CMP = {
    "greater than": ">", "more than": ">", "above": ">", "over": ">", "exceeding": ">", "higher than": ">",
    "at least": ">=", "no less than": ">=", ">=": ">=", ">": ">",
    "less than": "<", "below": "<", "under": "<", "lower than": "<", "fewer than": "<",
    "at most": "<=", "no more than": "<=", "<=": "<=", "<": "<",
    "equal to": "==", "exactly": "==", "=": "==",
}
_PLACEHOLDERS = {"", "n/a", "na", "nan", "null", "none", "-", "--", "?", "unknown", "missing", "tbd", "error"}
_ROW_NOUNS = r"(?:orders|rows|records|sales|transactions|entries|lines|line items|order ids)"
_QUOTE_O = "['\"‘“`]"
_QUOTE_C = "['\"’”`]"


class Table:
    def __init__(self, header: List[str], rows: List[Dict[str, str]], claimed: List[str]):
        self.header = header
        self.rows = rows
        self.claimed = claimed
        self.by_lower = {h.lower(): h for h in header}
        self.numeric = {}
        for h in header:
            vals = [r[h] for r in rows if r[h] != ""]
            self.numeric[h] = bool(vals) and all(_num(v) is not None for v in vals)

    def col(self, word: str) -> Optional[str]:
        """Resolve a word of the question to an actual column name."""
        w = word.lower().strip()
        for cand in (w, _PLURAL.get(w, w), w.rstrip("s"), w + "s", w.replace(" ", "_")):
            if cand in self.by_lower:
                return self.by_lower[cand]
        return None

    def claimed_only(self, word: str) -> bool:
        """Column promised by the statement but absent from the file."""
        w = word.lower().strip()
        cands = {w, _PLURAL.get(w, w), w.rstrip("s"), w + "s"}
        return any(c.lower() in cands for c in self.claimed) and self.col(word) is None

    def col_alt(self) -> str:
        names = set()
        for h in self.header + self.claimed:
            h = h.lower()
            names.update({h, h + "s", h.rstrip("s"), h.replace("_", " ")})
        names.discard("")
        return "(?:" + "|".join(sorted((re.escape(n) for n in names), key=len, reverse=True)) + ")"


def _claimed_columns(instruction: str) -> List[str]:
    m = re.search(r"\bcolumns?\b[:\s]+([A-Za-z0-9_]+(?:\s*,\s*(?:and\s+)?[A-Za-z0-9_]+)+)", instruction or "")
    if not m:
        return []
    return [c.strip() for c in re.split(r"\s*,\s*(?:and\s+)?", m.group(1)) if c.strip()]


def _question(instruction: str) -> str:
    sents = content_sentences(instruction)
    qs = [s for s in sents if not re.search(r"\bhas (?:the )?columns?\b|\bcontains? (?:the )?columns?\b", s, re.I)]
    if not qs:
        raise Unknown("no question found")
    if len(qs) > 1:
        # keep statement sentences only if exactly one of them is a question
        marked = [s for s in qs if s.rstrip().endswith("?")]
        rest = [s for s in qs if s not in marked]
        if len(marked) != 1 or any(not re.search(r"round|decimal|integer|whole number", s, re.I) for s in rest):
            raise Unknown("several statement sentences")
        return marked[0] + " " + " ".join(rest)
    return qs[0]


class Query:
    def __init__(self) -> None:
        self.filters: List[Tuple[str, str, object]] = []  # (column, op, value)
        self.metric: Optional[Tuple] = None
        self.round_digits: Optional[int] = None
        self.round_mode: Optional[str] = None


def _need_col(t: Table, word: str) -> str:
    c = t.col(word)
    if c is None:
        if t.claimed_only(word):
            raise Infeasible("column %r is described in the task but absent from the file" % word)
        raise Unknown("unknown column %r" % word)
    return c


def _extract_filters(q: str, t: Table, query: Query) -> str:
    COL = t.col_alt()
    prep = r"(?:\b(?:of|for|in|from|by|during|with|where|within|across|at|on)\s+)?(?:the\s+)?"
    num = r"(-?\d+(?:\.\d+)?)"

    def cut(m: re.Match) -> str:
        return " "

    # "between months 5 and 6 (inclusive)"
    between_rx = re.compile(
        prep + r"between\s+(?:the\s+)?(" + COL + r")\s+" + num + r"\s+and\s+(?:(?:" + COL + r")\s+)?" + num +
        r"(?:\s*,?\s*\(?(inclusive|exclusive|both inclusive)\)?)?", re.I)
    # explicit lists: "months 3 and 7", "months 2, 5 or 9"
    list_rx = re.compile(
        prep + r"(" + COL + r")\s+(-?\d+(?:\s*,\s*-?\d+)*\s*,?\s*(?:and|or)\s+-?\d+)\b(?!\s*(?:inclusive|\(inclusive))",
        re.I)
    # numeric range: "months 5 through 6 inclusive", "from month 5 to month 6"
    range_rx = re.compile(
        prep + r"(?:from\s+)?(?:the\s+)?(" + COL + r")\s+" + num +
        r"\s*(?:through|thru|to|until|till|up to|-|–|—)\s*(?:(?:" + COL + r")\s+)?" + num +
        r"(?:\s*,?\s*\(?(inclusive|exclusive|both inclusive)\)?)?", re.I)
    while True:
        m = list_rx.search(q)
        if not m or between_rx.search(q):
            break
        col = _need_col(t, m.group(1))
        values = frozenset(Fraction(x) for x in re.findall(r"-?\d+", m.group(2)))
        query.filters.append((col, "in", values))
        q = q[:m.start()] + " " + q[m.end():]
    while True:
        m = between_rx.search(q) or range_rx.search(q)
        if not m:
            break
        col = _need_col(t, m.group(1))
        lo, hi = Fraction(m.group(2)), Fraction(m.group(3))
        if m.group(4) and m.group(4).lower() == "exclusive":
            query.filters.append((col, ">", lo))
            query.filters.append((col, "<", hi))
        else:
            query.filters.append((col, ">=", lo))
            query.filters.append((col, "<=", hi))
        q = q[:m.start()] + " " + q[m.end():]

    # numeric comparison: "price above 50", "units of at least 10"
    cmp_alt = "|".join(sorted((re.escape(k) for k in _CMP), key=len, reverse=True))
    cmp_rx = re.compile(
        prep + r"(?:a\s+|an\s+)?(" + COL + r")\s+(?:is\s+|was\s+|of\s+|that is\s+)?(" + cmp_alt + r")\s*" + num, re.I)
    while True:
        m = cmp_rx.search(q)
        if not m:
            break
        col = _need_col(t, m.group(1))
        query.filters.append((col, _CMP[m.group(2).lower()], Fraction(m.group(3))))
        q = q[:m.start()] + " " + q[m.end():]

    # quoted categorical value with its column: "product 'widget'", "the 'west' region"
    qv = _QUOTE_O + r"([^'\"‘’“”`]+)" + _QUOTE_C
    cat_rx = re.compile(prep + r"(" + COL + r")\s+(?:is\s+|=\s*|equal to\s+|named\s+|called\s+)?" + qv, re.I)
    cat_rev_rx = re.compile(prep + qv + r"\s+(" + COL + r")\b", re.I)
    while True:
        m = cat_rx.search(q)
        if m:
            col, val = _need_col(t, m.group(1)), m.group(2)
        else:
            m = cat_rev_rx.search(q)
            if not m:
                break
            col, val = _need_col(t, m.group(2)), m.group(1)
        query.filters.append((col, "==", val.strip()))
        q = q[:m.start()] + " " + q[m.end():]

    # a quoted value next to a word that is not a column at all
    m = re.search(r"\b([A-Za-z_]+)\s+" + qv, q)
    if m and m.group(1).lower() not in ("of", "for", "in", "from", "by", "the", "named", "called", "is"):
        if t.claimed_only(m.group(1)):
            raise Infeasible("column %r is described in the task but absent from the file" % m.group(1))

    # remaining quoted value without a column: find the column by its values
    bare_rx = re.compile(prep + qv, re.I)
    while True:
        m = bare_rx.search(q)
        if not m:
            break
        val = m.group(1).strip()
        cols = [h for h in t.header if not t.numeric[h] and any(r[h].lower() == val.lower() for r in t.rows)]
        if len(cols) == 1:
            query.filters.append((cols[0], "==", val))
        elif not cols:
            cat = [h for h in t.header if not t.numeric[h]]
            if not cat:
                raise Unknown("quoted value %r matches nothing" % val)
            query.filters.append((cat[0], "==", val))  # value absent everywhere -> empty selection
        else:
            raise Unknown("quoted value %r is ambiguous" % val)
        q = q[:m.start()] + " " + q[m.end():]

    # numeric equality: "in month 7"
    eq_rx = re.compile(prep + r"(" + COL + r")\s+(?:number\s+|no\.?\s+|#\s*|=\s*|is\s+|equal to\s+)?" + num + r"\b", re.I)
    while True:
        m = eq_rx.search(q)
        if not m:
            break
        col = _need_col(t, m.group(1))
        query.filters.append((col, "==", Fraction(m.group(2))))
        q = q[:m.start()] + " " + q[m.end():]

    # unquoted categorical values: "of widget", "in the west region"
    for h in t.header:
        if t.numeric[h]:
            continue
        values = sorted({r[h] for r in t.rows if len(r[h]) >= 2}, key=len, reverse=True)
        hn = re.escape(h.lower())
        for v in values:
            rx = re.compile(prep + r"(?:" + hn + r"s?\s+)?\b" + re.escape(v) + r"s?\b(?:\s+" + hn + r"s?\b)?", re.I)
            m = rx.search(q)
            if m:
                others = [o for o in t.header if o != h and not t.numeric[o]
                          and any(r[o].lower() == v.lower() for r in t.rows)]
                if others:
                    raise Unknown("value %r occurs in several columns" % v)
                query.filters.append((h, "==", v))
                q = q[:m.start()] + " " + q[m.end():]
    return q


def _extract_rounding(q: str, query: Query) -> str:
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4}
    m = re.search(r",?\s*\(?\b(?:rounded|round(?: it| the (?:answer|result))?) to (\d+|zero|one|two|three|four) "
                  r"decimal (?:places?|digits?)\)?\.?", q, re.I)
    if m:
        d = m.group(1).lower()
        query.round_digits = words[d] if d in words else int(d)
        q = q[:m.start()] + " " + q[m.end():]
    m = re.search(r",?\s*\(?\b(?:rounded|round(?: it| the (?:answer|result))?) (down|up)(?: to (?:the nearest |an? )?"
                  r"(?:integer|whole number))?\)?\.?", q, re.I)
    if m:
        query.round_mode = m.group(1).lower()
        q = q[:m.start()] + " " + q[m.end():]
    m = re.search(r",?\s*\(?\b(?:rounded|round(?: it| the (?:answer|result))?) to (?:the nearest |an? )"
                  r"(?:integer|whole number)\)?\.?|\bas an integer\b", q, re.I)
    if m:
        query.round_digits = 0
        q = q[:m.start()] + " " + q[m.end():]
    return q


def _clean(q: str) -> str:
    q = q.lower()
    q = re.sub(r"\(\s*inclusive\s*\)|\binclusive\b", " ", q)
    q = re.sub(r"[?.!,;:()]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    # dangling prepositions left behind by removed filters
    prev = None
    while prev != q:
        prev = q
        q = re.sub(r"\s+(?:in|of|for|from|by|during|with|and|across|within|where|at|on)$", "", q)
    q = re.sub(r"\b(in|of|for|from|by|during|with) (?=(?:were|was|are|is|sold|ordered)\b)", "", q)
    return q.strip()


def _parse_metric(rem: str, t: Table, query: Query) -> None:
    COL = t.col_alt()
    sold = r"(?: (?:that )?(?:were|was|are|is|have been|has been|got|did we|did the company|did they))?" \
           r"(?: (?:sold|ordered|purchased|shipped|bought|sell|moved))?(?: (?:in total|overall|altogether|in all))?"
    is_ = r"(?:what (?:is|was|are|were) |find |compute |calculate |determine |give |report )?"
    revenue = r"(?:revenue|sales revenue|sales value|sales amount|turnover|income|sales volume in money)"
    formula = r"(?: (?:units ?(?:\*|x|×|times|multiplied by) ?price|price ?(?:\*|x|×|times|multiplied by) ?units))?"

    # revenue
    m = re.fullmatch(is_ + r"(?:the )?(?:total |overall |combined |sum of )?" + revenue + formula +
                     r"(?: generated| earned| made)?" + sold, rem)
    if m:
        query.metric = ("sum_product", _need_col(t, "units"), _need_col(t, "price"))
        return
    m = re.fullmatch(r"how much " + revenue + formula + r"(?: was| were| is| did)?(?: generated| earned| made)?" + sold, rem)
    if m:
        query.metric = ("sum_product", _need_col(t, "units"), _need_col(t, "price"))
        return
    # count rows
    m = re.fullmatch(r"how many " + _ROW_NOUNS + r"(?: are there| were there| exist| (?:were|are|was) "
                     r"(?:made|placed|recorded|listed|there))?" + sold, rem)
    if m is None:
        m = re.fullmatch(is_ + r"(?:the )?(?:total )?(?:number|count) of " + _ROW_NOUNS +
                         r"(?: (?:made|placed|recorded|listed))?", rem)
    if m:
        query.metric = ("count",)
        return
    # distinct
    m = re.fullmatch(r"how many (?:distinct|different|unique) (" + COL + r")(?: values)?(?: are there| were there|"
                     r" appear| exist| were sold| are listed)?" + sold, rem)
    if m is None:
        m = re.fullmatch(is_ + r"(?:the )?number of (?:distinct|different|unique) (" + COL + r")(?: values)?", rem)
    if m:
        query.metric = ("distinct", _need_col(t, m.group(1)))
        return
    # sum of a column: "how many units ... were sold", "total units"
    m = re.fullmatch(r"how many (" + COL + r")" + sold, rem)
    if m is None:
        m = re.fullmatch(is_ + r"(?:the )?(?:total|sum|combined|overall)(?: number| amount| quantity| sum)?(?: of)? ("
                         + COL + r")" + sold, rem)
    if m:
        col = _need_col(t, m.group(1))
        if not t.numeric.get(col) and t.rows:
            vals = [r[col] for r in t.rows]
            if not (any(v.lower() in _PLACEHOLDERS for v in vals) or any(_num(v) is not None for v in vals)):
                raise Unknown("sum over a non-numeric column")
            # Mixed column: bad cells matter only if they are in the selected rows,
            # which is checked when the aggregate is computed.
        query.metric = ("sum", col)
        return
    # average / min / max
    m = re.fullmatch(is_ + r"(?:the )?(average|mean|maximum|max|highest|largest|biggest|greatest|minimum|min|lowest|"
                     r"smallest|median) (?:value of |number of )?(?:the )?(" + COL + r")(?: per (?:order|row|sale))?"
                     + sold, rem)
    if m:
        kind = m.group(1)
        col = _need_col(t, m.group(2))
        if kind in ("average", "mean"):
            query.metric = ("avg", col)
        elif kind == "median":
            query.metric = ("median", col)
        elif kind in ("maximum", "max", "highest", "largest", "biggest", "greatest"):
            query.metric = ("max", col)
        else:
            query.metric = ("min", col)
        return
    # arg-max by group
    m = re.fullmatch(r"which (" + COL + r") (?:sold|had|has|recorded|generated|made|got|accounts? for|saw) the "
                     r"(most|highest|largest|greatest|biggest|fewest|least|lowest|smallest) "
                     r"(?:number of |total |amount of )?(units|revenue|orders|sales|" + COL + r")" + sold, rem)
    if m:
        group = _need_col(t, m.group(1))
        best = "max" if m.group(2) in ("most", "highest", "largest", "greatest", "biggest") else "min"
        what = m.group(3)
        if what == "revenue":
            agg = ("sum_product", _need_col(t, "units"), _need_col(t, "price"))
        elif what in ("orders", "sales"):
            agg = ("count",)
        else:
            agg = ("sum", _need_col(t, what))
        query.metric = ("arg", group, best, agg)
        return
    raise Unknown("unrecognised question: %r" % rem)


# --------------------------------------------------------------- execution
def _match(row: Dict[str, str], flt: Tuple[str, str, object], ci: bool) -> bool:
    col, op, val = flt
    cell = row.get(col, "")
    if op == "in":
        n = _num(cell)
        return n is not None and n in val
    if isinstance(val, Fraction):
        n = _num(cell)
        if n is None:
            return False
        return {"==": n == val, ">": n > val, ">=": n >= val, "<": n < val, "<=": n <= val}[op]
    if ci:
        return cell.lower() == str(val).lower()
    return cell == val


def _select(t: Table, query: Query) -> List[Dict[str, str]]:
    """Rows that satisfy every filter.

    A cell that does not allow to decide a filter (empty / placeholder / not a
    number where a number is compared) makes the row *undetermined*. Such a row
    matters only if all its other filters hold; then the selection itself is
    not known and the solver answers UNKNOWN instead of guessing either way.
    """
    modes = []
    for flt in query.filters:
        col, op, val = flt
        if isinstance(val, (Fraction, frozenset)):
            modes.append("num")
        else:  # categorical: exact match first, case-insensitive only if nothing matches exactly
            modes.append("exact" if any(_match(r, flt, False) for r in t.rows) else "ci")
    selected = []
    for r in t.rows:
        verdicts = []
        for flt, mode in zip(query.filters, modes):
            col, op, val = flt
            cell = r.get(col, "")
            if mode == "num":
                verdicts.append(None if _num(cell) is None else _match(r, flt, False))
            elif cell.lower() in _PLACEHOLDERS and str(val).lower() not in _PLACEHOLDERS:
                verdicts.append(None)
            else:
                verdicts.append(_match(r, flt, mode == "ci"))
        if any(v is False for v in verdicts):
            continue
        if any(v is None for v in verdicts):
            raise Unknown("a row cannot be classified: missing or malformed value in a filter column")
        selected.append(r)
    return selected


def _numbers(rows: List[Dict[str, str]], col: str) -> List[Fraction]:
    out = []
    for r in rows:
        n = _num(r.get(col, ""))
        if n is None:
            raise Infeasible("missing or non-numeric value in column %r" % col)
        out.append(n)
    return out


def _aggregate(rows: List[Dict[str, str]], metric: Tuple):
    kind = metric[0]
    if kind == "count":
        return Fraction(len(rows))
    if kind == "sum":
        return sum(_numbers(rows, metric[1]), Fraction(0))
    if kind == "sum_product":
        a, b = _numbers(rows, metric[1]), _numbers(rows, metric[2])
        return sum((x * y for x, y in zip(a, b)), Fraction(0))
    if kind == "distinct":
        return Fraction(len({r[metric[1]] for r in rows if r[metric[1]] != ""}))
    if not rows:
        raise Infeasible("no rows match the conditions of the question")
    vals = _numbers(rows, metric[1])
    if kind == "avg":
        return sum(vals, Fraction(0)) / len(vals)
    if kind == "max":
        return max(vals)
    if kind == "min":
        return min(vals)
    if kind == "median":
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    raise Unknown("unsupported metric %r" % (kind,))


def _format(value: Fraction, query: Query) -> str:
    if query.round_mode == "down":
        return str(math.floor(value))
    if query.round_mode == "up":
        return str(math.ceil(value))
    if value.denominator == 1:
        if query.round_digits:
            return str(round(float(value), query.round_digits))
        return str(value.numerator)
    if query.round_digits is not None:
        if query.round_digits == 0:
            return str(int(round(float(value))))
        return str(round(float(value), query.round_digits))
    # exact decimal if it terminates, otherwise the task did not say how to round
    d = value.denominator
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    if d == 1:
        text = ("%.10f" % float(value)).rstrip("0").rstrip(".")
        return text
    raise Unknown("non-terminating result and no rounding rule")


def solve(instruction: str, workspace: str) -> Outcome:
    try:
        path, named = find_csv(instruction, workspace)
        if path is None:
            if named:
                return Outcome.infeasible("data file %s is missing" % named)
            return Outcome.unknown("no CSV file identified")
        header, rows = load_csv(path)
        if not any(header):
            return Outcome.infeasible("CSV has no header")
        table = Table(header, rows, _claimed_columns(instruction))
        query = Query()
        q = _question(instruction)
        q = _extract_rounding(q, query)
        q = _extract_filters(q, table, query)
        rem = _clean(q)
        _parse_metric(rem, table, query)
        selected = _select(table, query)
        metric = query.metric
        if metric[0] == "arg":
            _, group, best, agg = metric
            groups: Dict[str, List[Dict[str, str]]] = {}
            for r in selected:
                groups.setdefault(r[group], []).append(r)
            if not groups:
                return Outcome.infeasible("no rows match the conditions of the question")
            scored = sorted(((_aggregate(rs, agg), g) for g, rs in groups.items()),
                            key=lambda x: x[0], reverse=(best == "max"))
            if len(scored) > 1 and scored[0][0] == scored[1][0]:
                return Outcome.unknown("tie between groups")
            answer = scored[0][1]
        else:
            answer = _format(_aggregate(selected, metric), query)
        return Outcome.solved(answer, "metric=%s filters=%d rows=%d/%d" % (
            metric[0], len(query.filters), len(selected), len(rows)),
            matched_rows=len(selected), total_rows=len(rows))
    except Infeasible as e:
        return Outcome.infeasible(str(e))
    except Unknown as e:
        return Outcome.unknown(str(e))
    except Exception as e:  # noqa: BLE001
        return Outcome.unknown("data solver error: %s: %s" % (type(e).__name__, e))
