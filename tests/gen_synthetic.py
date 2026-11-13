#!/usr/bin/env python3
"""Generator of a synthetic training set (development only, NOT in the image).

The organisers publish 8 feasible examples and say that 15-25% of the hidden
tasks are infeasible and that wordings may differ. This script builds a local
set that imitates both: for every task type it produces

* tasks in the published wording with random data,
* re-worded variants of the same questions and a few new question kinds,
* infeasible tasks (missing entity / column / file, division by zero,
  contradictory or broken tests, ...).

Each ``task.json`` carries, besides the public fields, development-only fields:

    expected, check   - reference answer (as in the public examples)
    feasible          - false for infeasible tasks
    expect            - what the harness should do:
                        "free"         solve with zero tokens
                        "llm"          may need the LLM tier
                        "free_reject"  refuse with zero tokens
                        "llm_reject"   infeasibility can only be judged by the LLM
    note              - what the case is about

    python tests/gen_synthetic.py --out tests/synthetic_tasks --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil

TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
MATH_TOOL = " Compute it with the bash tool (python3 -c)."
DATA_TOOL = " Use bash with python3 to compute it."
SEARCH_TOOL = " Use the file tools or bash (ls, grep, cat) to find it."
CODING_TEXT = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
               "(do not edit the tests) until all tests pass. Work only inside the current directory. When you are "
               "done, write ONLY the word DONE to the file answer.txt in the current directory (for example with "
               "bash: echo 42 > answer.txt).")


class Out:
    def __init__(self, root: str):
        self.root = root
        self.n = 0
        self.index = []

    def add(self, ttype: str, instruction: str, expected, check: str, files: dict = None, feasible: bool = True,
            expect: str = "free", note: str = "", reward: int = None, rng: random.Random = None):
        tid = "%s_%04d" % (ttype, self.n)
        self.n += 1
        d = os.path.join(self.root, tid)
        ws = os.path.join(d, "workspace")
        os.makedirs(ws, exist_ok=True)
        for rel, content in (files or {}).items():
            p = os.path.join(ws, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
        meta = {"task_id": tid, "task_type": ttype, "instruction": instruction,
                "reward": reward if reward is not None else (rng or random).randint(1, 5),
                "expected": str(expected), "check": check, "feasible": feasible, "expect": expect, "note": note}
        with open(os.path.join(d, "task.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        self.index.append(meta)


# ===================================================================== math
def gen_math(out: Out, rng: random.Random, n: int):
    def step_split(wording=0):
        k = rng.randint(2, 5)
        text = [
            "Then the stock is split equally into %d warehouses and it keeps one share (rounded down)." % k,
            "Next, the stock is divided evenly among %d warehouses and it keeps one share (rounded down)." % k,
            "After that the inventory is split equally between %d sites, and the warehouse keeps one share, rounded down." % k,
        ][wording]
        return text, (lambda s: s // k)

    def step_double(wording=0):
        text = ["Then the stock is doubled by merging with a sister warehouse.",
                "Then the stock is tripled by merging with two sister warehouses.",
                "After that, the inventory is doubled."][wording]
        mult = [2, 3, 2][wording]
        return text, (lambda s: s * mult)

    def step_ship(wording=0):
        x, d = rng.randint(5, 60), rng.randint(2, 6)
        text = ["It ships %d boxes per day for %d days." % (x, d),
                "Then it sends out %d boxes each day for %d days." % (x, d),
                "The warehouse sells %d boxes every day for %d days." % (x, d)][wording]
        return text, (lambda s: s - x * d)

    def step_recv(wording=0):
        k, m = rng.randint(2, 6), rng.randint(20, 120)
        text = ["It receives %d deliveries of %d boxes each." % (k, m),
                "Then it gets %d shipments of %d boxes each." % (k, m),
                "Next, it receives %d batches of %d boxes." % (k, m)][wording]
        return text, (lambda s: s + k * m)

    def step_simple(wording=0):
        x = rng.randint(10, 200)
        if wording == 0:
            return "It receives %d more boxes." % x, (lambda s: s + x)
        if wording == 1:
            return "Then it ships %d boxes." % x, (lambda s: s - x)
        return "After that, %d boxes are added." % x, (lambda s: s + x)

    makers = [step_split, step_double, step_ship, step_recv]
    question = "How many boxes does it have at the end?"
    # published wording
    for _ in range(n):
        start = rng.randint(800, 5000)
        steps = [rng.choice(makers)(0) for _ in range(rng.randint(2, 6))]
        s = start
        for _t, fn in steps:
            s = fn(s)
        ins = "A warehouse starts with %d boxes. %s %s" % (start, " ".join(t for t, _ in steps), question)
        out.add("math", ins + MATH_TOOL + TAIL, s, "exact_int", rng=rng, note="published wording")
    # re-worded
    openings = ["A warehouse starts with %d boxes.", "The warehouse begins with %d boxes.",
                "A warehouse initially holds %d boxes.", "Initially, a warehouse has %d boxes."]
    questions = [question, "How many boxes remain at the end?", "How many boxes are left in the warehouse?",
                 "What is the final number of boxes?", "How many boxes does the warehouse have now?"]
    for _ in range(n):
        start = rng.randint(800, 5000)
        steps = []
        for _i in range(rng.randint(2, 5)):
            mk = rng.choice(makers + [step_simple])
            steps.append(mk(rng.randint(0, 2)))
        s = start
        for _t, fn in steps:
            s = fn(s)
        ins = "%s %s %s" % (rng.choice(openings) % start, " ".join(t for t, _ in steps), rng.choice(questions))
        out.add("math", ins + MATH_TOOL + TAIL, s, "exact_int", rng=rng, note="re-worded")
    # unseen operation -> LLM tier is acceptable
    out.add("math", "A warehouse starts with 1200 boxes. A flood destroys 25 percent of the boxes. Then it receives "
            "3 deliveries of 40 boxes each. " + question + MATH_TOOL + TAIL, 1020, "exact_int",
            expect="llm", rng=rng, note="unseen operation (percent)")
    out.add("math", "A warehouse starts with 640 boxes. Each box is repacked into 2 smaller boxes. Then it ships "
            "100 boxes. " + question + MATH_TOOL + TAIL, 1180, "exact_int", expect="llm", rng=rng,
            note="unseen operation (repack)")
    # infeasible
    for _ in range(max(2, n // 4)):
        start = rng.randint(800, 5000)
        ins = ("A warehouse starts with %d boxes. Then the stock is doubled by merging with a sister warehouse. "
               "Then the stock is split equally into 0 warehouses and it keeps one share (rounded down). %s"
               % (start, question))
        out.add("math", ins + MATH_TOOL + TAIL, "", "exact_int", feasible=False, expect="free_reject", rng=rng,
                note="division by zero")
    out.add("math", "A warehouse starts with 900 boxes. It ships some boxes per day for 4 days. " + question
            + MATH_TOOL + TAIL, "", "exact_int", feasible=False, expect="free_reject", rng=rng,
            note="missing quantity")
    out.add("math", "A warehouse starts with an unknown number of boxes. It receives 2 deliveries of 50 boxes each. "
            + question + MATH_TOOL + TAIL, "", "exact_int", feasible=False, expect="free_reject", rng=rng,
            note="missing start value")
    out.add("math", "A warehouse starts with 900 boxes. It ships 30 boxes per day for 4 days. How many trucks does "
            "it own at the end?" + MATH_TOOL + TAIL, "", "exact_int", feasible=False, expect="llm_reject", rng=rng,
            note="question about something never mentioned")


# ===================================================================== data
REGIONS = ["north", "south", "east", "west"]
PRODUCTS = ["widget", "gadget", "gizmo", "sprocket", "doohickey"]


def make_sales(rng: random.Random, rows: int, products=None):
    products = products or PRODUCTS[:4]
    data = []
    for i in range(1, rows + 1):
        data.append({"order_id": i, "region": rng.choice(REGIONS), "product": rng.choice(products),
                     "month": rng.randint(1, 12), "units": rng.randint(1, 40), "price": rng.randint(5, 100)})
    return data


def csv_text(data, cols=("order_id", "region", "product", "month", "units", "price")):
    lines = [",".join(cols)]
    for r in data:
        lines.append(",".join(str(r[c]) for c in cols))
    return "\n".join(lines) + "\n"


def gen_data(out: Out, rng: random.Random, n: int):
    head = "The file sales.csv has columns order_id,region,product,month,units,price (%d rows). "

    def add(q, expected, data, expect="free", note="", feasible=True, files=None, check="exact_int", rows=None):
        ins = head % (rows if rows is not None else len(data)) + q + DATA_TOOL + TAIL
        out.add("data_analysis", ins, expected, check, files=files if files is not None else {"sales.csv": csv_text(data)},
                feasible=feasible, expect=expect, note=note, rng=rng)

    for _ in range(n):
        data = make_sales(rng, rng.choice([20, 50, 300]))
        p = rng.choice(PRODUCTS[:4])
        a = rng.randint(1, 10)
        b = a + rng.randint(0, 2)
        exp = sum(r["units"] for r in data if r["product"] == p and a <= r["month"] <= b)
        add("How many units of product '%s' were sold in months %d through %d inclusive?" % (p, a, b), exp, data,
            note="published wording")
    for _ in range(n):
        data = make_sales(rng, rng.choice([20, 60, 200]))
        p, reg = rng.choice(PRODUCTS[:4]), rng.choice(REGIONS)
        a = rng.randint(1, 9)
        b = a + rng.randint(1, 3)
        kind = rng.randint(0, 9)
        if kind == 0:
            add("How many units of product '%s' were sold in month %d?" % (p, a),
                sum(r["units"] for r in data if r["product"] == p and r["month"] == a), data, note="single month")
        elif kind == 1:
            add("How many units were sold in region '%s' in months %d through %d inclusive?" % (reg, a, b),
                sum(r["units"] for r in data if r["region"] == reg and a <= r["month"] <= b), data, note="region filter")
        elif kind == 2:
            add("How many units of product '%s' were sold in region '%s'?" % (p, reg),
                sum(r["units"] for r in data if r["product"] == p and r["region"] == reg), data, note="product+region")
        elif kind == 3:
            add("What is the total revenue (units * price) for product '%s' in months %d through %d inclusive?" % (p, a, b),
                sum(r["units"] * r["price"] for r in data if r["product"] == p and a <= r["month"] <= b), data,
                note="revenue")
        elif kind == 4:
            add("How many orders were placed in region '%s'?" % reg,
                sum(1 for r in data if r["region"] == reg), data, note="count rows")
        elif kind == 5:
            add("What is the total number of units sold for product '%s'?" % p,
                sum(r["units"] for r in data if r["product"] == p), data, note="total units wording")
        elif kind == 6:
            add("How many units of %s were sold in the %s region?" % (p, reg),
                sum(r["units"] for r in data if r["product"] == p and r["region"] == reg), data, note="unquoted values")
        elif kind == 7:
            add("What is the maximum price of product '%s'?" % p,
                max([r["price"] for r in data if r["product"] == p] or [0]), data, note="max")
        elif kind == 8:
            add("How many units of product '%s' were sold between months %d and %d inclusive?" % (p, a, b),
                sum(r["units"] for r in data if r["product"] == p and a <= r["month"] <= b), data,
                note="between ... and")
        else:
            tot = {}
            for r in data:
                tot[r["region"]] = tot.get(r["region"], 0) + r["units"]
            best = sorted(tot.items(), key=lambda x: -x[1])
            if len(best) > 1 and best[0][1] == best[1][1]:
                continue
            add("Which region sold the most units?", best[0][0], data, note="arg-max", check="exact_str")
    # needs LLM: a question shape the parser does not know
    data = make_sales(rng, 40)
    months = sorted({r["month"] for r in data if r["product"] == "widget"})
    add("In how many different months was product 'widget' sold at least once?", len(months), data, expect="llm",
        note="unseen question shape")
    # ---- infeasible
    data = make_sales(rng, 30)
    add("How many units of product 'widget' were sold in months 3 through 5 inclusive?", "", data, feasible=False,
        expect="free_reject", note="file missing", files={})
    cols = ("order_id", "region", "product", "month", "price")
    add("How many units of product 'widget' were sold in months 3 through 5 inclusive?", "", data, feasible=False,
        expect="free_reject", note="units column missing", files={"sales.csv": csv_text(data, cols)})
    cols = ("order_id", "region", "product", "units", "price")
    add("How many units of product 'widget' were sold in months 3 through 5 inclusive?", "", data, feasible=False,
        expect="free_reject", note="month column missing", files={"sales.csv": csv_text(data, cols)})
    add("How many units of product 'widget' were sold in warehouse 'A'?", "", data, feasible=False,
        expect="llm_reject", note="question about a column that never existed")
    add("How many units of product 'flux capacitor' were sold in months 3 through 5 inclusive?", "", data,
        feasible=False, expect="free_reject", note="product absent (harmless to answer 0)")
    broken = csv_text(data).replace(",%d,%d\n" % (data[3]["units"], data[3]["price"]), ",n/a,%d\n" % data[3]["price"])
    bad = [dict(r, units="n/a") for r in data]
    add("How many units of product '%s' were sold in months 1 through 12 inclusive?" % data[0]["product"], "", data,
        feasible=False, expect="free_reject", note="non-numeric units", files={"sales.csv": csv_text(bad)})
    add("How many units of product 'widget' were sold in months 3 through 5 inclusive?", "", [], feasible=False,
        expect="free_reject", note="empty file", files={"sales.csv": ""}, rows=20)
    del broken


# =================================================================== search
FIRST = ["Alice", "Boris", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira", "Maya", "Nikolai",
         "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya"]
LAST = ["Novak", "Smith", "Haddad", "Wang", "Kowalski", "Petrov", "Sato", "Garcia", "Muller", "Ivanova"]
COMPANIES = ["Arktika Labs", "Bluefin Systems", "Cobalt Works", "Dune Robotics", "Ember Bio", "Fjord Data",
             "Granite AI", "Helix Motors", "Iris Optics", "Juniper Foods"]
CITIES = ["Tbilisi", "Osaka", "Lisbon", "Nairobi", "Tallinn", "Lima", "Hanoi", "Oslo", "Porto", "Kazan"]


def make_kb(rng: random.Random):
    companies = {c: {"Headquarters": rng.choice(CITIES), "Founded": rng.randint(1950, 2020)} for c in COMPANIES}
    names = set()
    while len(names) < 26:
        names.add("%s %s" % (rng.choice(FIRST), rng.choice(LAST)))
    names = sorted(names)
    people = {}
    for nm in names:
        people[nm] = {"Born": rng.randint(1950, 2000), "Employer": rng.choice(COMPANIES),
                      "Mentor": rng.choice([x for x in names if x != nm])}
    return people, companies


def kb_files(people, companies, skip=(), drop=None):
    files = {}
    for nm, facts in people.items():
        if nm in skip:
            continue
        lines = ["# %s" % nm, ""] + ["%s: %s" % (k, v) for k, v in facts.items()
                                     if not (drop and drop == (nm, k))]
        files["kb/person_%s.md" % nm.lower().replace(" ", "_")] = "\n".join(lines) + "\n"
    for c, facts in companies.items():
        if c in skip:
            continue
        lines = ["# %s" % c, ""] + ["%s: %s" % (k, v) for k, v in facts.items() if not (drop and drop == (c, k))]
        files["kb/company_%s.md" % c.lower().replace(" ", "_")] = "\n".join(lines) + "\n"
    return files


def gen_search(out: Out, rng: random.Random, n: int):
    head = "The directory kb/ contains a small knowledge base of markdown files. "

    def add(q, expected, files, expect="free", note="", feasible=True):
        out.add("search", head + q + SEARCH_TOOL + TAIL, expected, "exact_str", files=files, feasible=feasible,
                expect=expect, note=note, rng=rng)

    for i in range(n * 2):
        people, companies = make_kb(rng)
        files = kb_files(people, companies)
        p = rng.choice(sorted(people))
        emp = people[p]["Employer"]
        men = people[p]["Mentor"]
        kind = i % 14
        if kind == 0:
            add("In which year was %s born?" % p, people[p]["Born"], files, note="published wording")
        elif kind == 1:
            add("In which city is the headquarters of %s's employer?" % p, companies[emp]["Headquarters"], files,
                note="published wording")
        elif kind == 2:
            add("Who is the mentor of %s?" % p, men, files, note="mentor")
        elif kind == 3:
            add("In which year was %s's employer founded?" % p, companies[emp]["Founded"], files,
                note="employer founded")
        elif kind == 4:
            add("In which year was %s's mentor born?" % p, people[men]["Born"], files, note="mentor born")
        elif kind == 5:
            add("Who is the employer of %s?" % p, emp, files, note="employer")
        elif kind == 6:
            add("In which city is the headquarters of the employer of %s's mentor?" % p,
                companies[people[men]["Employer"]]["Headquarters"], files, note="three hops")
        elif kind == 7:
            add("Where is %s headquartered?" % emp, companies[emp]["Headquarters"], files, note="company direct")
        elif kind == 8:
            add("What is the birth year of %s?" % p, people[p]["Born"], files, note="birth year wording")
        elif kind == 9:
            add("Which company does %s work for?" % p, emp, files, note="works for wording")
        elif kind == 10:
            add("In which year was %s founded?" % emp, companies[emp]["Founded"], files, note="company founded")
        elif kind == 11:
            add("Who is the mentor of %s's mentor?" % p, people[men]["Mentor"], files, note="mentor of mentor")
        elif kind == 12:
            add("In which city is %s's employer headquartered?" % p, companies[emp]["Headquarters"], files,
                note="headquartered wording")
        else:
            add("Who mentors the person who mentors %s?" % p, people[men]["Mentor"], files, expect="llm",
                note="verb wording, LLM acceptable")
    # ---- infeasible
    for _ in range(max(2, n // 3)):
        people, companies = make_kb(rng)
        files = kb_files(people, companies)
        add("In which year was Zoltan Brightwater born?", "", files, feasible=False, expect="free_reject",
            note="unknown person")
        p = rng.choice(sorted(people))
        emp = people[p]["Employer"]
        add("In which city is the headquarters of %s's employer?" % p, "", kb_files(people, companies, skip=(emp,)),
            feasible=False, expect="free_reject", note="employer card missing")
        add("In which year was %s born?" % p, "", kb_files(people, companies, drop=(p, "Born")), feasible=False,
            expect="free_reject", note="fact missing")
        add("In which city is the headquarters of %s's employer?" % p, "",
            kb_files(people, companies, drop=(emp, "Headquarters")), feasible=False, expect="free_reject",
            note="headquarters fact missing")
    people, companies = make_kb(rng)
    add("In which year was %s born?" % sorted(people)[0], "", {}, feasible=False, expect="free_reject",
        note="kb directory missing")
    p = sorted(people)[1]
    files = kb_files(people, companies)
    files["kb/person_%s_copy.md" % p.lower().replace(" ", "_")] = "# %s\n\nBorn: 1901\nEmployer: %s\n" % (
        p, people[p]["Employer"])
    add("In which year was %s born?" % p, "", files, feasible=False, expect="free_reject",
        note="two contradictory cards")


# =================================================================== coding
# name -> (correct source, [buggy sources], test body lines)
FUNCS = {
    "reverse_words": ("def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
                      ["def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n",
                       "def reverse_words(s):\n    return ' '.join(s.split())\n"],
                      ["self.assertEqual(m.reverse_words('c a b'), 'b a c')"]),
    "total": ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
              ["def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
               "def total(xs):\n    s = 1\n    for x in xs:\n        s += x\n    return s\n"],
              ["self.assertEqual(m.total([1, 2, 3]), 6)"]),
    "maximum": ("def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
                ["def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n"],
                ["self.assertEqual(m.maximum([3, 9, 2]), 9)"]),
    "minimum": ("def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
                ["def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
                 "def minimum(xs):\n    return max(xs)\n"],
                ["self.assertEqual(m.minimum([3, 9, 2]), 2)"]),
    "average": ("def average(xs):\n    return sum(xs) / len(xs)\n",
                ["def average(xs):\n    return sum(xs) / (len(xs) + 1)\n",
                 "def average(xs):\n    return sum(xs) * len(xs)\n",
                 "def average(xs):\n    return len(xs) / len(xs)\n"],
                ["self.assertEqual(m.average([2, 4, 6]), 4)"]),
    "first_n": ("def first_n(xs, n):\n    return xs[:n]\n",
                ["def first_n(xs, n):\n    return xs[n:]\n", "def first_n(xs, n):\n    return xs[:n + 1]\n",
                 "def first_n(xs, n):\n    return xs[:n - 1]\n"],
                ["self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])"]),
    "count_even": ("def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
                   ["def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
                    "def count_even(xs):\n    return sum(1 for x in xs if x % 3 == 0)\n"],
                   ["self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)"]),
    "is_palindrome": ("def is_palindrome(s):\n    return s == s[::-1]\n",
                      ["def is_palindrome(s):\n    return s != s[::-1]\n",
                       "def is_palindrome(s):\n    return s == s[::1]\n"],
                      ["self.assertTrue(m.is_palindrome('level'))", "self.assertFalse(m.is_palindrome('lever'))"]),
    "factorial": ("def factorial(n):\n    r = 1\n    for i in range(1, n + 1):\n        r *= i\n    return r\n",
                  ["def factorial(n):\n    r = 1\n    for i in range(1, n):\n        r *= i\n    return r\n",
                   "def factorial(n):\n    r = 0\n    for i in range(1, n + 1):\n        r *= i\n    return r\n",
                   "def factorial(n):\n    r = 1\n    for i in range(1, n + 1):\n        r += i\n    return r\n"],
                  ["self.assertEqual(m.factorial(5), 120)", "self.assertEqual(m.factorial(0), 1)"]),
    "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
              ["def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n",
               "def clamp(x, lo, hi):\n    return max(lo, max(x, hi))\n"],
              ["self.assertEqual(m.clamp(15, 0, 10), 10)", "self.assertEqual(m.clamp(-5, 0, 10), 0)",
               "self.assertEqual(m.clamp(5, 0, 10), 5)"]),
    "last_n": ("def last_n(xs, n):\n    return xs[-n:]\n",
               ["def last_n(xs, n):\n    return xs[:n]\n", "def last_n(xs, n):\n    return xs[n:]\n"],
               ["self.assertEqual(m.last_n([5, 6, 7, 8], 3), [6, 7, 8])"]),
    "count_vowels": ("def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')\n",
                     ["def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch not in 'aeiou')\n",
                      "def count_vowels(s):\n    return sum(1 for ch in s.upper() if ch in 'aeiou')\n"],
                     ["self.assertEqual(m.count_vowels('Education'), 5)"]),
    "is_prime": ("def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n"
                 "        if n % d == 0:\n            return False\n    return True\n",
                 ["def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5) + 1):\n"
                  "        if n % d == 0:\n            return True\n    return True\n",
                  "def is_prime(n):\n    if n < 2:\n        return False\n    for d in range(2, int(n ** 0.5)):\n"
                  "        if n % d == 0:\n            return False\n    return True\n"],
                 ["self.assertTrue(m.is_prime(13))", "self.assertFalse(m.is_prime(9))", "self.assertFalse(m.is_prime(1))"]),
    "sum_squares": ("def sum_squares(xs):\n    return sum(x * x for x in xs)\n",
                    ["def sum_squares(xs):\n    return sum(x + x for x in xs)\n",
                     "def sum_squares(xs):\n    return sum(x * 2 for x in xs)\n"],
                    ["self.assertEqual(m.sum_squares([1, 2, 3]), 14)"]),
    "unique": ("def unique(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n",
               ["def unique(xs):\n    out = []\n    for x in xs:\n        if x in out:\n            out.append(x)\n    return out\n"],
               ["self.assertEqual(m.unique([3, 1, 3, 2, 1]), [3, 1, 2])"]),
    "gcd": ("def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
            ["def gcd(a, b):\n    while b:\n        a, b = b, a // b\n    return a\n",
             "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return b\n"],
            ["self.assertEqual(m.gcd(12, 18), 6)", "self.assertEqual(m.gcd(7, 5), 1)"]),
    "fib": ("def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
            ["def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a - b\n    return a\n",
             "def fib(n):\n    a, b = 0, 1\n    for _ in range(n + 1):\n        a, b = b, a + b\n    return a\n"],
            ["self.assertEqual(m.fib(10), 55)", "self.assertEqual(m.fib(1), 1)"]),
    "capitalize_words": ("def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
                         ["def capitalize_words(s):\n    return ' '.join(w.upper() for w in s.split())\n",
                          "def capitalize_words(s):\n    return ''.join(w.capitalize() for w in s.split())\n"],
                         ["self.assertEqual(m.capitalize_words('hello big world'), 'Hello Big World')"]),
    "running_total": ("def running_total(xs):\n    out = []\n    s = 0\n    for x in xs:\n        s += x\n        out.append(s)\n    return out\n",
                      ["def running_total(xs):\n    out = []\n    s = 0\n    for x in xs:\n        s += x\n        out.append(x)\n    return out\n",
                       "def running_total(xs):\n    out = []\n    s = 0\n    for x in xs:\n        s += x\n        out.append(s)\n        return out\n"],
                      ["self.assertEqual(m.running_total([1, 2, 3]), [1, 3, 6])"]),
    "second_largest": ("def second_largest(xs):\n    return sorted(set(xs))[-2]\n",
                       ["def second_largest(xs):\n    return sorted(set(xs))[-1]\n",
                        "def second_largest(xs):\n    return sorted(set(xs))[1]\n"],
                       ["self.assertEqual(m.second_largest([4, 9, 1, 9, 7]), 7)"]),
}
# bugs that need more than one edit -> the LLM tier is expected
HARD = {
    "median": ("def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    if n % 2:\n        return s[n // 2]\n    return (s[n // 2 - 1] + s[n // 2]) / 2\n",
               "def median(xs):\n    n = len(xs)\n    return xs[n // 2]\n",
               ["self.assertEqual(m.median([5, 1, 3]), 3)", "self.assertEqual(m.median([4, 1, 3, 2]), 2.5)"]),
    "word_count": ("def word_count(s):\n    out = {}\n    for w in s.split():\n        out[w] = out.get(w, 0) + 1\n    return out\n",
                   "def word_count(s):\n    out = {}\n    for w in s.split():\n        out[w] = 1\n    return len(out)\n",
                   ["self.assertEqual(m.word_count('a b a'), {'a': 2, 'b': 1})"]),
}


def test_file(names, bodies, extra=""):
    lines = ["import unittest", "", "import mylib as m", "", "", "class T(unittest.TestCase):"]
    for nm in names:
        lines.append("    def test_%s(self):" % nm)
        for b in bodies[nm]:
            lines.append("        " + b)
        lines.append("")
    if extra:
        lines.append(extra)
    lines += ["", "if __name__ == '__main__':", "    unittest.main()", ""]
    return "\n".join(lines)


def gen_coding(out: Out, rng: random.Random, n: int):
    names_all = sorted(FUNCS)
    bodies = {k: v[2] for k, v in FUNCS.items()}
    bodies.update({k: v[2] for k, v in HARD.items()})

    def build(names, buggy):
        parts = []
        for nm in names:
            if nm in HARD:
                parts.append(HARD[nm][1] if nm in buggy else HARD[nm][0])
            else:
                parts.append(buggy[nm] if nm in buggy else FUNCS[nm][0])
        return "\n\n".join(parts)

    for _ in range(n):
        names = rng.sample(names_all, rng.randint(3, 6))
        k = rng.randint(1, min(3, len(names)))
        buggy = {nm: rng.choice(FUNCS[nm][1]) for nm in rng.sample(names, k)}
        out.add("coding", CODING_TEXT, "tests_pass", "unittest",
                files={"mylib.py": build(names, buggy), "test_mylib.py": test_file(names, bodies)},
                expect="free", rng=rng, note="single-edit bugs: " + ",".join(sorted(buggy)))
    # every single buggy variant once (coverage of the mutation operators)
    for nm in names_all:
        for i, bug in enumerate(FUNCS[nm][1]):
            others = [x for x in rng.sample(names_all, 3) if x != nm][:2]
            names = [nm] + others
            out.add("coding", CODING_TEXT, "tests_pass", "unittest",
                    files={"mylib.py": build(names, {nm: bug}), "test_mylib.py": test_file(names, bodies)},
                    expect="free", rng=rng, note="variant %s#%d" % (nm, i))
    for nm in sorted(HARD):
        names = [nm] + rng.sample(names_all, 2)
        out.add("coding", CODING_TEXT, "tests_pass", "unittest",
                files={"mylib.py": build(names, {nm: 1}), "test_mylib.py": test_file(names, bodies)},
                expect="llm", rng=rng, note="multi-edit bug: " + nm)
    # already green
    names = rng.sample(names_all, 4)
    out.add("coding", CODING_TEXT, "tests_pass", "unittest",
            files={"mylib.py": build(names, {}), "test_mylib.py": test_file(names, bodies)},
            expect="free", rng=rng, note="tests already pass")
    # ---- infeasible
    names = ["total", "maximum", "count_even"]
    buggy = {"total": FUNCS["total"][1][0]}
    contradict = dict(bodies)
    contradict["total"] = ["self.assertEqual(m.total([1, 2, 3]), 6)", "self.assertEqual(m.total([1, 2, 3]), 7)"]
    out.add("coding", CODING_TEXT, "", "unittest", feasible=False, expect="free_reject", rng=rng,
            files={"mylib.py": build(names, buggy), "test_mylib.py": test_file(names, contradict)},
            note="contradictory assertions in one test")
    extra = "    def test_total_again(self):\n        self.assertEqual(m.total([1, 2, 3]), 5)\n"
    out.add("coding", CODING_TEXT, "", "unittest", feasible=False, expect="free_reject", rng=rng,
            files={"mylib.py": build(names, buggy), "test_mylib.py": test_file(names, bodies, extra)},
            note="contradictory assertions in two tests")
    extra = "    def test_sanity(self):\n        self.assertEqual(1 + 1, 3)\n"
    out.add("coding", CODING_TEXT, "", "unittest", feasible=False, expect="free_reject", rng=rng,
            files={"mylib.py": build(names, buggy), "test_mylib.py": test_file(names, bodies, extra)},
            note="failing test independent of the library")
    out.add("coding", CODING_TEXT, "", "unittest", feasible=False, expect="free_reject", rng=rng,
            files={"mylib.py": build(names, buggy),
                   "test_mylib.py": test_file(names, bodies).replace("class T(unittest.TestCase):", "class T(unittest.TestCase:")},
            note="syntax error in the test file")
    out.add("coding", CODING_TEXT, "", "unittest", feasible=False, expect="free_reject", rng=rng,
            files={"mylib.py": build(names, buggy),
                   "test_mylib.py": test_file(names, bodies).replace("import mylib as m", "import mylib as m\nimport notinstalledlib")},
            note="test imports a missing module")
    out.add("coding", CODING_TEXT, "", "unittest", feasible=False, expect="free_reject", rng=rng,
            files={"mylib.py": build(names, buggy)}, note="test file missing")
    out.add("coding", CODING_TEXT, "", "unittest", feasible=False, expect="free_reject", rng=rng,
            files={"test_mylib.py": test_file(names, bodies)}, note="library file missing")
    impossible = dict(bodies)
    impossible["total"] = ["self.assertEqual(m.total([1, 2, 3]), 6)", "self.assertEqual(m.total([2, 1, 3]), 9)"]
    out.add("coding", CODING_TEXT, "", "unittest", feasible=False, expect="llm_reject", rng=rng,
            files={"mylib.py": build(names, buggy), "test_mylib.py": test_file(names, impossible)},
            note="semantically contradictory tests (needs judgement)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_tasks"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=12, help="base number of tasks per group")
    ap.add_argument("--types", default="math,data_analysis,search,coding")
    args = ap.parse_args()
    if os.path.isdir(args.out) and os.path.basename(os.path.normpath(args.out)).startswith("synthetic"):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    out = Out(args.out)
    types = args.types.split(",")
    if "math" in types:
        gen_math(out, rng, args.n)
    if "data_analysis" in types:
        gen_data(out, rng, args.n)
    if "search" in types:
        gen_search(out, rng, args.n)
    if "coding" in types:
        gen_coding(out, rng, args.n)
    by = {}
    for m in out.index:
        key = (m["task_type"], m["expect"])
        by[key] = by.get(key, 0) + 1
    print("generated %d tasks in %s" % (len(out.index), args.out))
    for k in sorted(by):
        print("  %-14s %-12s %d" % (k[0], k[1], by[k]))


if __name__ == "__main__":
    main()
