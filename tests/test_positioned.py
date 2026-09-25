"""带显式参考起点的读段模式测试。

覆盖：两端重叠、中间空洞、负起点拒绝、同 UMI 不同起点、完全无重叠、
逐坐标手算对数似然、输入乱序不改根分组与共识、与等长入口兼容。
"""

from __future__ import annotations

import math
import random
import unittest

from reference import reference_positioned_consensus

from umicorrect import (
    InvalidArgumentError,
    InvalidReadError,
    LimitExceededError,
    PositionedRead,
    Read,
    analyze_positioned_reads,
    analyze_reads,
    positioned_consensus,
)

LOG10 = math.log10


def pr(umi, start, seq, q):
    return PositionedRead(umi, start, seq, tuple([q] * len(seq)))


class OverlapTests(unittest.TestCase):
    def test_two_end_overlap_votes_at_coordinates(self) -> None:
        # 长读段覆盖 [0,6)，两条短读段分别搭在两端：
        # 左短读 [0,3) 呼 AAA@30，右短读 [3,6) 呼 TTT@30。
        # 短读段只在自己的坐标投票，不得把长读段位置顶错位。
        reads = [
            pr("AAAA", 0, "AAACCC", 20),   # 长读段：坐标 0..5
            pr("AAAA", 0, "AAA", 30),       # 左短读：坐标 0..2
            pr("AAAA", 3, "TTT", 30),       # 右短读：坐标 3..5
        ]
        result = analyze_positioned_reads(reads)
        (mol,) = result.molecules
        self.assertEqual(mol.ref_start, 0)
        self.assertEqual(mol.consensus_seq, "AAATTT")
        self.assertEqual(mol.coverage, (2, 2, 2, 2, 2, 2))
        # 坐标 3：长读段呼 C@20，右短读呼 T@30。
        # L(T)/L(C) = (0.999*(0.01/3)) / ((0.001/3)*0.99) = 999/99*... 手算：
        # ll(T) = log10(0.999) + log10(0.01/3)；ll(C) = log10(0.001/3)+log10(0.99)
        e20, e30 = 0.01, 0.001
        want_gap = LOG10(((1 - e30) * (e20 / 3)) / ((e30 / 3) * (1 - e20)))
        self.assertAlmostEqual(mol.confidence_gap[3], want_gap, places=12)
        # 坐标 0：两读一致呼 A，gap = log10( (0.99*0.999) / ((0.01/3)*(0.001/3)) )
        want_gap0 = LOG10(((1 - e20) * (1 - e30)) / ((e20 / 3) * (e30 / 3)))
        self.assertAlmostEqual(mol.confidence_gap[0], want_gap0, places=12)

    def test_short_reads_do_not_shift_long_read_positions(self) -> None:
        # 三条短读段都叠在长读段左端；长读段右端坐标不受短读段数量影响。
        reads = [pr("AAAA", 0, "AAAAGGGG", 25)]
        reads += [pr("AAAA", 0, "AAAA", 30) for _ in range(3)]
        result = analyze_positioned_reads(reads)
        (mol,) = result.molecules
        self.assertEqual(mol.consensus_seq, "AAAAGGGG")
        self.assertEqual(mol.coverage, (4, 4, 4, 4, 1, 1, 1, 1))


class HoleTests(unittest.TestCase):
    def test_middle_hole_is_N_with_missing_gap(self) -> None:
        # 两段覆盖 [0,3) 与 [5,8)，坐标 3、4 为空洞。
        reads = [
            pr("AAAA", 0, "ACG", 30),
            pr("AAAA", 5, "TGC", 30),
        ]
        result = analyze_positioned_reads(reads)
        (mol,) = result.molecules
        self.assertEqual(mol.consensus_seq, "ACGNNTGC")
        self.assertEqual(mol.coverage, (1, 1, 1, 0, 0, 1, 1, 1))
        # 空洞：信心差缺失（None），不是 0.0，也不是低质量观测
        self.assertIsNone(mol.confidence_gap[3])
        self.assertIsNone(mol.confidence_gap[4])
        # 覆盖位置信心差是有限正数
        self.assertGreater(mol.confidence_gap[0], 0.0)

    def test_completely_non_overlapping_same_umi_one_cluster(self) -> None:
        # 同 UMI 两段完全不重叠：仍属同簇，输出跨 min..max，中间空洞 N。
        reads = [
            pr("AAAA", 0, "AA", 30),
            pr("AAAA", 10, "TT", 30),
        ]
        result = analyze_positioned_reads(reads)
        self.assertEqual(result.molecules_after, 1)
        (mol,) = result.molecules
        self.assertEqual(mol.ref_start, 0)
        self.assertEqual(len(mol.consensus_seq), 12)
        self.assertEqual(mol.consensus_seq, "AA" + "N" * 8 + "TT")
        self.assertEqual(mol.coverage, (1, 1) + (0,) * 8 + (1, 1))
        self.assertTrue(all(g is None for g in mol.confidence_gap[2:10]))


class StartValidationTests(unittest.TestCase):
    def test_negative_start_rejected_at_construction(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "start"):
            PositionedRead("AA", -1, "AC", (20, 20))

    def test_bool_and_float_start_rejected(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            PositionedRead("AA", True, "AC", (20, 20))
        with self.assertRaisesRegex(InvalidArgumentError, "整数"):
            PositionedRead("AA", 1.5, "AC", (20, 20))
        with self.assertRaisesRegex(InvalidArgumentError, "NaN/Infinity"):
            PositionedRead("AA", float("nan"), "AC", (20, 20))

    def test_batch_locates_bad_start(self) -> None:
        obj = object.__new__(PositionedRead)
        object.__setattr__(obj, "umi", "AA")
        object.__setattr__(obj, "start", -3)
        object.__setattr__(obj, "sequence", "AC")
        object.__setattr__(obj, "qualities", (20, 20))
        with self.assertRaisesRegex(InvalidReadError, r"reads\[0\].*start"):
            analyze_positioned_reads([obj])

    def test_start_limit_exceeded(self) -> None:
        with self.assertRaises(LimitExceededError):
            PositionedRead("AA", 10_000_001, "AC", (20, 20))

    def test_empty_batch_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "空批次"):
            analyze_positioned_reads([])


class SameUmiDifferentStartTests(unittest.TestCase):
    def test_same_umi_different_starts_merge_into_one_molecule(self) -> None:
        # 同 UMI 两条读段起点不同、部分重叠：一个分子，坐标汇聚。
        reads = [
            pr("AAAA", 0, "AACG", 30),    # 坐标 0..3
            pr("AAAA", 2, "CGTT", 30),    # 坐标 2..5
        ]
        result = analyze_positioned_reads(reads)
        self.assertEqual(result.nodes_before, 1)
        self.assertEqual(result.molecules_after, 1)
        (mol,) = result.molecules
        self.assertEqual(mol.ref_start, 0)
        self.assertEqual(mol.consensus_seq, "AACGTT")
        self.assertEqual(mol.coverage, (1, 1, 2, 2, 1, 1))

    def test_umi_absorption_still_applies_with_positions(self) -> None:
        # 大根 AAAA(5) 吸收 AAAT(2)（Hamming=1, 5>=3）；两者起点不同。
        reads = [pr("AAAA", 0, "AAAA", 30) for _ in range(5)]
        reads += [pr("AAAT", 4, "CCCC", 30) for _ in range(2)]
        result = analyze_positioned_reads(reads)
        self.assertEqual(result.nodes_before, 2)
        self.assertEqual(result.molecules_after, 1)
        (mol,) = result.molecules
        self.assertEqual(mol.root_umi, "AAAA")
        self.assertEqual(mol.members, ("AAAA", "AAAT"))
        self.assertEqual(mol.consensus_seq, "AAAACCCC")
        self.assertEqual(mol.coverage, (5, 5, 5, 5, 2, 2, 2, 2))


class HandComputedLikelihoodTests(unittest.TestCase):
    def test_per_coordinate_hand_computed(self) -> None:
        # 坐标 5：read0 呼 A@Q20，read1 呼 C@Q30。
        # L(C)/L(A) = ((0.01/3)*0.999)/(0.99*(0.001/3)) = 111/11 -> C 胜
        reads = [
            pr("AA", 3, "A", 20),   # 坐标 3
            pr("AA", 5, "A", 20),   # 坐标 5
            pr("AA", 5, "C", 30),   # 坐标 5
        ]
        ref_start, seq, gaps, cov = positioned_consensus(reads, (0, 1, 2))
        self.assertEqual(ref_start, 3)
        self.assertEqual(seq, "ANC")   # 坐标 4 是空洞
        self.assertEqual(cov, (1, 0, 2))
        self.assertAlmostEqual(gaps[0], LOG10(297), places=12)
        self.assertIsNone(gaps[1])
        self.assertAlmostEqual(gaps[2], LOG10(111 / 11), places=12)

    def test_q0_and_tie_rules_preserved(self) -> None:
        # Q=0：呼中碱基似然 -inf，其余三者并列 -> N, gap 0.0（非 None）。
        reads = [pr("AA", 7, "A", 0)]
        ref_start, seq, gaps, cov = positioned_consensus(reads, (0,))
        self.assertEqual((ref_start, seq, cov), (7, "N", (1,)))
        self.assertEqual(gaps[0], 0.0)
        # 等质量 A/C 并列 -> N, gap 0.0
        reads2 = [pr("AA", 0, "A", 20), pr("AA", 0, "C", 20)]
        _, seq2, gaps2, _ = positioned_consensus(reads2, (0, 1))
        self.assertEqual(seq2, "N")
        self.assertEqual(gaps2[0], 0.0)


class DeterminismTests(unittest.TestCase):
    def test_shuffle_keeps_roots_and_consensus(self) -> None:
        reads = (
            [pr("AAAA", 0, "AAAA", 30) for _ in range(6)]
            + [pr("AAAA", 2, "AG", 25)]
            + [pr("AAAT", 1, "TT", 20) for _ in range(2)]
            + [pr("GGGG", 5, "GGG", 30)]
        )
        base = analyze_positioned_reads(reads)
        rng = random.Random(99)
        for _ in range(10):
            shuffled = list(reads)
            rng.shuffle(shuffled)
            got = analyze_positioned_reads(shuffled)
            self.assertEqual(
                [(m.root_umi, m.members) for m in got.molecules],
                [(m.root_umi, m.members) for m in base.molecules],
            )
            for m1, m2 in zip(base.molecules, got.molecules):
                self.assertEqual(m1.ref_start, m2.ref_start)
                self.assertEqual(m1.consensus_seq, m2.consensus_seq)
                self.assertEqual(m1.coverage, m2.coverage)
                for g1, g2 in zip(m1.confidence_gap, m2.confidence_gap):
                    if g1 is None:
                        self.assertIsNone(g2)
                    else:
                        self.assertAlmostEqual(g1, g2, places=12)

    def test_random_batches_against_independent_reference(self) -> None:
        rng = random.Random(20260925)
        for _ in range(30):
            n = rng.randint(1, 6)
            umis = list(dict.fromkeys(
                "".join(rng.choice("ACGT") for _ in range(3)) for _ in range(n)
            ))
            reads = []
            for u in umis:
                for _ in range(rng.randint(1, 4)):
                    start = rng.randint(0, 4)
                    ln = rng.randint(1, 4)
                    seq = "".join(rng.choice("ACGT") for _ in range(ln))
                    reads.append(pr(u, start, seq, rng.randint(0, 40)))
            result = analyze_positioned_reads(reads)
            for mol in result.molecules:
                rs, seq, gaps, cov = reference_positioned_consensus(
                    reads, list(mol.read_indices)
                )
                self.assertEqual(mol.ref_start, rs)
                self.assertEqual(mol.consensus_seq, seq)
                self.assertEqual(list(mol.coverage), cov)
                for got, want in zip(mol.confidence_gap, gaps):
                    if want is None:
                        self.assertIsNone(got)
                    else:
                        self.assertAlmostEqual(got, want, places=12)


class CompatibilityTests(unittest.TestCase):
    def test_start_zero_equal_length_matches_analyze_reads(self) -> None:
        # 全部 start=0 且等长：定位模式共识/信心差与等长入口完全一致。
        specs = (
            [("AAAA", "ACGT", (30, 25, 20, 15))] * 5
            + [("AAAA", "ACGA", (10, 10, 10, 10))]
            + [("AAAT", "ACGT", (20, 20, 20, 20))] * 2
        )
        eq_reads = [Read(u, s, q) for u, s, q in specs]
        pos_reads = [PositionedRead(u, 0, s, q) for u, s, q in specs]
        eq = analyze_reads(eq_reads)
        pos = analyze_positioned_reads(pos_reads)
        self.assertEqual(eq.nodes_before, pos.nodes_before)
        self.assertEqual(eq.molecules_after, pos.molecules_after)
        for me, mp in zip(eq.molecules, pos.molecules):
            self.assertEqual(me.root_umi, mp.root_umi)
            self.assertEqual(me.members, mp.members)
            self.assertEqual(me.consensus_seq, mp.consensus_seq)
            self.assertEqual(me.confidence_gap, mp.confidence_gap)
            self.assertEqual(mp.ref_start, 0)
            self.assertTrue(all(c == len(mp.read_indices) for c in mp.coverage))


if __name__ == "__main__":
    unittest.main()
