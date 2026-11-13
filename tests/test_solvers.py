"""Unit tests of the zero-token solvers (development only, not in the image).

    python -m unittest discover -s tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent_harness import answers  # noqa: E402
from agent_harness.budget import Ledger, weighted_cost  # noqa: E402
from agent_harness.solvers import coding_solver, data_solver, math_solver, search_solver  # noqa: E402
from agent_harness.solvers.base import INFEASIBLE, SOLVED, UNKNOWN  # noqa: E402
from agent_harness.tasks import discover_tasks, load_task  # noqa: E402

TAIL = (" Compute it with the bash tool (python3 -c). Work only inside the current directory. When you are done, "
        "write ONLY the final answer to the file answer.txt in the current directory (for example with bash: "
        "echo 42 > answer.txt).")


class MathTests(unittest.TestCase):
    def solve(self, text):
        return math_solver.solve(text + TAIL)

    def test_published_examples(self):
        o = self.solve("A warehouse starts with 1752 boxes. Then the stock is split equally into 2 warehouses and "
                       "it keeps one share (rounded down). Then the stock is doubled by merging with a sister "
                       "warehouse. Then the stock is split equally into 3 warehouses and it keeps one share "
                       "(rounded down). Then the stock is doubled by merging with a sister warehouse. Then the "
                       "stock is split equally into 3 warehouses and it keeps one share (rounded down). How many "
                       "boxes does it have at the end?")
        self.assertEqual((o.status, o.answer), (SOLVED, "389"))
        o = self.solve("A warehouse starts with 1669 boxes. It ships 36 boxes per day for 3 days. It receives 2 "
                       "deliveries of 87 boxes each. How many boxes does it have at the end?")
        self.assertEqual((o.status, o.answer), (SOLVED, "1735"))

    def test_division_by_zero_is_infeasible(self):
        o = self.solve("A warehouse starts with 100 boxes. Then the stock is split equally into 0 warehouses and it "
                       "keeps one share (rounded down). How many boxes does it have at the end?")
        self.assertEqual(o.status, INFEASIBLE)

    def test_missing_quantity_is_infeasible(self):
        o = self.solve("A warehouse starts with 100 boxes. It ships some boxes per day for 3 days. How many boxes "
                       "does it have at the end?")
        self.assertEqual(o.status, INFEASIBLE)

    def test_unknown_wordings_are_not_guessed(self):
        for text in (
            "A warehouse starts with 100 boxes. A fire destroys half of them. How many boxes does it have at the end?",
            "A warehouse starts with 100 boxes. It ships 5 boxes per day for 3 days. How many boxes did it ship?",
            "A warehouse starts with 100 boxes. Then the stock is split equally into 3 warehouses and it keeps two "
            "shares (rounded down). How many boxes does it have at the end?",
            "A warehouse starts with 100 boxes. It ships 5 boxes per day for 3 days, except on Sunday. How many "
            "boxes does it have at the end?",
        ):
            self.assertEqual(self.solve(text).status, UNKNOWN, text)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        rows = ["order_id,region,product,month,units,price",
                "1,west,widget,5,5,10", "2,north,widget,6,7,20", "3,north,gizmo,7,9,30",
                "4,west,widget,7,11,40", "5,south,gadget,3,2,50"]
        with open(os.path.join(self.dir, "sales.csv"), "w") as f:
            f.write("\n".join(rows) + "\n")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def ask(self, q):
        text = ("The file sales.csv has columns order_id,region,product,month,units,price (5 rows). " + q +
                " Use bash with python3 to compute it." + TAIL)
        return data_solver.solve(text, self.dir)

    def test_questions(self):
        cases = {
            "How many units of product 'widget' were sold in months 5 through 6 inclusive?": "12",
            "How many units of product 'widget' were sold in month 7?": "11",
            "How many units were sold in region 'north'?": "16",
            "What is the total revenue (units * price) for product 'widget'?": str(5 * 10 + 7 * 20 + 11 * 40),
            "How many orders were placed in region 'west'?": "2",
            "How many units of product 'widget' were sold in months 5 and 7?": "16",
            "How many units of product 'widget' were sold between months 5 and 7 inclusive?": "23",
            "Which product sold the most units?": "widget",
            "How many units of product 'nothing' were sold in months 1 through 12 inclusive?": "0",
        }
        for q, exp in cases.items():
            o = self.ask(q)
            self.assertEqual((o.status, o.answer), (SOLVED, exp), q + " -> " + o.reason)

    def test_not_guessed(self):
        for q in ("How many units of product 'widget' were sold outside months 5 through 6?",
                  "How many units of product 'widget' or 'gizmo' were sold?",
                  "Which region sold the most units?",  # north and west tie at 16
                  "How many units were sold in the first quarter?"):
            self.assertEqual(self.ask(q).status, UNKNOWN, q)

    def test_missing_column_and_file(self):
        with open(os.path.join(self.dir, "sales.csv"), "w") as f:
            f.write("order_id,region,product,month,price\n1,west,widget,5,10\n")
        o = self.ask("How many units of product 'widget' were sold in months 5 through 6 inclusive?")
        self.assertEqual(o.status, INFEASIBLE)
        os.remove(os.path.join(self.dir, "sales.csv"))
        o = self.ask("How many units of product 'widget' were sold in months 5 through 6 inclusive?")
        self.assertEqual(o.status, INFEASIBLE)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        kb = os.path.join(self.dir, "kb")
        os.makedirs(kb)
        cards = {
            "person_ann_lee.md": "# Ann Lee\n\nBorn: 1980\nEmployer: Fjord Data\nMentor: Bob Roe\n",
            "person_bob_roe.md": "# Bob Roe\n\nBorn: 1961\nEmployer: Dune Robotics\nMentor: Ann Lee\n",
            "person_cy_orphan.md": "# Cy Orphan\n\nBorn: 1970\nEmployer: Ghost Corp\nMentor: Bob Roe\n",
            "company_fjord_data.md": "# Fjord Data\n\nHeadquarters: Osaka\nFounded: 2001\n",
            "company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Lima\nFounded: 1990\n",
        }
        for name, text in cards.items():
            with open(os.path.join(kb, name), "w") as f:
                f.write(text)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def ask(self, q):
        text = ("The directory kb/ contains a small knowledge base of markdown files. " + q +
                " Use the file tools or bash (ls, grep, cat) to find it." + TAIL)
        return search_solver.solve(text, self.dir)

    def test_chains(self):
        cases = {
            "In which year was Ann Lee born?": "1980",
            "In which city is the headquarters of Ann Lee's employer?": "Osaka",
            "Who is the mentor of Ann Lee?": "Bob Roe",
            "In which year was Ann Lee's mentor born?": "1961",
            "In which city is the headquarters of the employer of Ann Lee's mentor?": "Lima",
            "In which year was Ann Lee's employer founded?": "2001",
            "Which company does Ann Lee work for?": "Fjord Data",
            "Who is the employer of Ann Lee?": "Fjord Data",
            "Where is Fjord Data headquartered?": "Osaka",
        }
        for q, exp in cases.items():
            o = self.ask(q)
            self.assertEqual((o.status, o.answer), (SOLVED, exp), q + " -> " + o.reason)

    def test_infeasible(self):
        self.assertEqual(self.ask("In which year was Zed Nobody born?").status, INFEASIBLE)
        self.assertEqual(self.ask("In which city is the headquarters of Cy Orphan's employer?").status, INFEASIBLE)

    def test_not_guessed(self):
        for q in ("In which year was Ann Lee born, and who is her mentor?",
                  "What is the salary of Ann Lee?",
                  "Who was born in 1980?"):
            self.assertEqual(self.ask(q).status, UNKNOWN, q)


class CodingTests(unittest.TestCase):
    def test_contradictory_tests_detected(self):
        src = ("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n"
               "    def test_a(self):\n        self.assertEqual(m.total([1, 2]), 3)\n"
               "    def test_b(self):\n        self.assertEqual(m.total([1, 2]), 4)\n")
        # A contradiction is proven only when the library function is evidently stateless.
        pure = "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n"
        info = coding_solver.analyze_tests(src, "mylib", pure)
        self.assertTrue(info.contradictions)
        self.assertEqual(info.refs["T.test_a"], {"total"})
        # Without the library source nothing can be proven: it is only a suspicion.
        info = coding_solver.analyze_tests(src, "mylib")
        self.assertFalse(info.contradictions)
        self.assertTrue(info.suspicions)

    def test_merge_patch(self):
        source = "import math\n\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n"
        merged = coding_solver.merge_patch(source, "def b():\n    return 3\n")
        self.assertIn("return 3", merged)
        self.assertIn("def a():\n    return 1", merged)
        self.assertNotIn("return 2", merged)

    def test_parse_unittest_output(self):
        out = ("test_a (__main__.T.test_a) ... ok\ntest_b (__main__.T.test_b) ... FAIL\n\n" + "=" * 70 +
               "\nFAIL: test_b (__main__.T.test_b)\n" + "-" * 70 +
               "\nTraceback (most recent call last):\n  File \"t.py\", line 5, in test_b\nAssertionError: 1 != 2\n\n" +
               "-" * 70 + "\nRan 2 tests in 0.001s\n\nFAILED (failures=1)\n")
        statuses, messages = coding_solver.parse_unittest_output(out)
        self.assertEqual(statuses, {"T.test_a": "ok", "T.test_b": "FAIL"})
        self.assertIn("1 != 2", messages["T.test_b"])


class InfraTests(unittest.TestCase):
    def test_cost_formula(self):
        self.assertAlmostEqual(weighted_cost(100, 50, 10), 60.0)   # reasoning already inside output
        # reported separately (or semantics unknown): added, whatever the magnitudes are
        self.assertAlmostEqual(weighted_cost(100, 5, 40, reasoning_separate=True), 55.0)
        self.assertAlmostEqual(weighted_cost(100, 100, 20, reasoning_separate=True), 130.0)
        led = Ledger(total=1000, soft_fraction=0.5)
        self.assertTrue(led.try_reserve(400))
        self.assertFalse(led.try_reserve(200))

    def test_loader_ignores_grading_fields(self):
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, "t1", "workspace"))
            with open(os.path.join(d, "t1", "task.json"), "w") as f:
                json.dump({"task_id": "t1", "task_type": "math", "instruction": "x", "reward": 3,
                           "expected": "42", "check": "exact_int"}, f)
            task = load_task(os.path.join(d, "t1"))
            self.assertEqual(task.reward, 3.0)
            self.assertFalse(hasattr(task, "expected"))
            self.assertFalse(hasattr(task, "check"))
            self.assertNotIn("42", repr(vars(task)))
            self.assertEqual(len(discover_tasks(d)), 1)
            self.assertTrue(answers.write_answer(task, "7"))
            with open(os.path.join(d, "t1", "workspace", "answer.txt")) as f:
                self.assertEqual(f.read(), "7\n")
            with open(os.path.join(d, "t1", "answer.txt")) as f:
                self.assertEqual(f.read(), "7\n")
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
