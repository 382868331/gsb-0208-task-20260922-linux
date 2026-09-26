"""可选编辑距离模型测试：短串独立枚举、多路径、歧义和输入置换。"""

from __future__ import annotations

import itertools
import math
import random
import unittest

from umicorrect import (
    EditDistanceOptions,
    Read,
    analyze_reads,
    cluster_umis_with_edit_distance,
    shortest_edit_evidence,
)


def brute_force_shortest_paths(root, observed, qualities):
    """独立枚举网格编辑路径，只用于长度不超过 3 的短串。"""
    n = len(observed)
    found = []

    def terms(q):
        err = 10.0 ** (-q / 10.0)
        return {
            "M": math.log10(1.0 - err),
            "S": math.log10(err / 9.0),
            "I": math.log10(err / 12.0),
            "D": math.log10(err / 3.0),
        }

    def walk(i, j, built, edits, log_probability, transcript):
        if i == len(root) and j == n:
            if "".join(built) == observed:
                found.append((edits, log_probability, tuple(transcript)))
            return
        if i < len(root) and j < n:
            op = "M" if root[i] == observed[j] else "S"
            q = qualities[j]
            walk(
                i + 1,
                j + 1,
                built + [observed[j]],
                edits + (op == "S"),
                log_probability + terms(q)[op],
                transcript + [op],
            )
        if i < len(root):
            q = qualities[min(j, n - 1)]
            walk(i + 1, j, built, edits + 1, log_probability + terms(q)["D"], transcript + ["D"])
        if j < n:
            q = qualities[j]
            walk(
                i,
                j + 1,
                built + [observed[j]],
                edits + 1,
                log_probability + terms(q)["I"],
                transcript + ["I"],
            )

    walk(0, 0, [], 0, 0.0, [])
    minimum = min(item[0] for item in found)
    shortest = [item for item in found if item[0] == minimum]
    total = float("-inf")
    for _, value, _ in shortest:
        total = value if total == float("-inf") else math.log10(10.0**total + 10.0**value)
    return minimum, total, len(shortest)


class IndependentPathEnumerationTests(unittest.TestCase):
    def test_all_short_strings_against_brute_force(self) -> None:
        quality_patterns = [
            (20,),
            (20, 25),
            (30, 15, 25),
            (7, 33, 18),
        ]
        strings = [
            "".join(chars)
            for length in range(1, 4)
            for chars in itertools.product("ACGT", repeat=length)
        ]
        checked = 0
        for root in strings:
            for observed in strings:
                if abs(len(root) - len(observed)) > 1:
                    continue
                qs = quality_patterns[len(observed) - 1]
                got = shortest_edit_evidence(root, observed, qs)
                want = brute_force_shortest_paths(root, observed, qs)
                self.assertEqual(got[0], want[0], (root, observed))
                self.assertEqual(got[2], want[2], (root, observed, got, want))
                self.assertAlmostEqual(got[1], want[1], places=12, msg=(root, observed))
                checked += 1
        self.assertGreater(checked, 1000)


class HomopolymerAndClusteringTests(unittest.TestCase):
    def test_homopolymer_deletion_counts_three_equal_paths(self) -> None:
        distance, log_probability, paths = shortest_edit_evidence("AAA", "AA", (20, 20))
        self.assertEqual((distance, paths), (1, 3))
        err = 10.0 ** (-20 / 10.0)
        expected = math.log10(3.0 * (1.0 - err) ** 2 * err / 3.0)
        self.assertAlmostEqual(log_probability, expected, places=12)

    def test_homopolymer_insertion_counts_four_equal_paths(self) -> None:
        distance, log_probability, paths = shortest_edit_evidence("AAA", "AAAA", (20, 20, 20, 20))
        self.assertEqual((distance, paths), (1, 4))
        err = 10.0 ** (-20 / 10.0)
        expected = math.log10(4.0 * (1.0 - err) ** 3 * err / 12.0)
        self.assertAlmostEqual(log_probability, expected, places=12)

    def test_length_difference_one_merges_when_evidence_clear(self) -> None:
        counts = {"AAAA": 10, "AAA": 1}
        qualities = {"AAA": [(20, 20, 20)]}
        result = cluster_umis_with_edit_distance(
            counts,
            qualities,
            edit_options=EditDistanceOptions(min_log10_gap=1.0),
        )
        self.assertEqual([set(c.members) for c in result.clusters], [{"AAAA", "AAA"}])
        assignment = result.assignments[-1]
        self.assertEqual(assignment.state, "merged")
        self.assertEqual(assignment.assigned_root, "AAAA")

    def test_two_high_abundance_tied_candidates_remain_unmerged(self) -> None:
        counts = {"AAAA": 10, "TAAA": 10, "CAAA": 1}
        qualities = {"CAAA": [(20, 20, 20, 20)]}
        result = cluster_umis_with_edit_distance(
            counts,
            qualities,
            edit_options=EditDistanceOptions(min_log10_gap=1.0),
        )
        self.assertEqual(len(result.clusters), 3)
        assignment = next(a for a in result.assignments if a.umi == "CAAA")
        self.assertEqual(assignment.state, "ambiguous")
        self.assertIsNone(assignment.assigned_root)
        self.assertEqual(len(assignment.candidates), 2)
        self.assertEqual(assignment.best_gap, 0.0)


class PipelineAndPermutationTests(unittest.TestCase):
    @staticmethod
    def _reads():
        reads = [
            Read("AAAA", "AAAA", (30, 30, 30, 30), (30, 30, 30, 30))
            for _ in range(10)
        ]
        reads += [
            Read("AAA", "AAAA", (28, 28, 28, 28), (30, 30, 30)),
            Read("CAAA", "CCCC", (20, 20, 20, 20), (20, 20, 20, 20)),
        ]
        reads += [
            Read("TAAA", "TTTT", (30, 30, 30, 30), (30, 30, 30, 30))
            for _ in range(10)
        ]
        return reads

    def test_pipeline_returns_merge_and_ambiguous_status(self) -> None:
        result = analyze_reads(
            self._reads(),
            edit_options=EditDistanceOptions(min_log10_gap=0.5),
        )
        self.assertEqual(result.molecules_after, 3)
        status = {a.umi: a.state for a in result.edit_assignments}
        self.assertEqual(status["AAA"], "merged")
        self.assertEqual(status["CAAA"], "ambiguous")

    def test_input_permutation_does_not_change_assignment(self) -> None:
        reads = self._reads()
        rng = random.Random(20260926)
        reference = analyze_reads(reads, edit_options=EditDistanceOptions())
        for _ in range(20):
            shuffled = reads[:]
            rng.shuffle(shuffled)
            got = analyze_reads(shuffled, edit_options=EditDistanceOptions())
            self.assertEqual(
                sorted((m.root_umi, m.members) for m in got.molecules),
                sorted((m.root_umi, m.members) for m in reference.molecules),
            )
            self.assertEqual(
                [(a.umi, a.state, a.assigned_root, a.best_gap) for a in got.edit_assignments],
                [(a.umi, a.state, a.assigned_root, a.best_gap) for a in reference.edit_assignments],
            )


if __name__ == "__main__":
    unittest.main()
