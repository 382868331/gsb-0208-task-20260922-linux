"""锚定（显式参考起点）读段模式测试。

覆盖：两端重叠、中间空洞、负起点拒绝、同 UMI 不同起点、完全无重叠、
逐坐标手算对数似然、输入顺序打乱不改根分组与共识、等长入口兼容性。
手算期望值由本文件直接用 math.log10 按定义计算，不调用被测核心。
"""

from __future__ import annotations

import math
import random
import unittest

from umicorrect import (
    AnchoredRead,
    InvalidArgumentError,
    InvalidReadError,
    Read,
    analyze_anchored_reads,
    analyze_reads,
)


def log10_correct(q: int) -> float:
    return math.log10(1.0 - 10.0 ** (-q / 10.0))


def log10_wrong(q: int) -> float:
    return math.log10(10.0 ** (-q / 10.0) / 3.0)


def hand_ll(votes):
    """votes: [(base, q), ...] -> dict base -> 累计 log10 似然（按定义手算）。"""
    ll = {b: 0.0 for b in "ACGT"}
    for base, q in votes:
        for b in "ACGT":
            ll[b] += log10_correct(q) if b == base else log10_wrong(q)
    return ll


def hand_gap(ll):
    vals = sorted(ll.values(), reverse=True)
    return vals[0] - vals[1]


class TestAnchoredBasics(unittest.TestCase):
    def test_negative_start_rejected_at_construction(self):
        with self.assertRaises(InvalidArgumentError):
            AnchoredRead(umi="AAA", start=-1, sequence="AC", qualities=(30, 30))

    def test_start_bool_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            AnchoredRead(umi="AAA", start=True, sequence="AC", qualities=(30, 30))

    def test_start_float_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            AnchoredRead(umi="AAA", start=1.5, sequence="AC", qualities=(30, 30))

    def test_two_end_overlap(self):
        # 两条读段分别覆盖参考两端，中间重叠 2 个坐标
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="ACGT", qualities=(30, 30, 30, 30)),
            AnchoredRead(umi="AAA", start=2, sequence="GTAC", qualities=(30, 30, 30, 30)),
        ]
        result = analyze_anchored_reads(reads)
        self.assertEqual(result.molecules_after, 1)
        mol = result.molecules[0]
        self.assertEqual((mol.ref_start, mol.ref_end), (0, 6))
        self.assertEqual(mol.consensus_seq, "ACGTAC")
        self.assertEqual(mol.coverage, (1, 1, 2, 2, 1, 1))

    def test_middle_gap_is_N_with_missing_gap(self):
        # 同一 UMI 两条读段，中间坐标 3..4 无任何覆盖
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="ACG", qualities=(30, 30, 30)),
            AnchoredRead(umi="AAA", start=5, sequence="TA", qualities=(30, 30)),
        ]
        mol = analyze_anchored_reads(reads).molecules[0]
        self.assertEqual((mol.ref_start, mol.ref_end), (0, 7))
        self.assertEqual(mol.consensus_seq, "ACGNNTA")
        self.assertEqual(mol.coverage, (1, 1, 1, 0, 0, 1, 1))
        # 空洞置信差为缺失（None），不是 0.0，也不是低质量观测
        self.assertIsNone(mol.confidence_gap[3])
        self.assertIsNone(mol.confidence_gap[4])
        for pos in (0, 1, 2, 5, 6):
            self.assertIsNotNone(mol.confidence_gap[pos])
            self.assertGreater(mol.confidence_gap[pos], 0.0)

    def test_same_umi_different_starts_one_cluster(self):
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="AC", qualities=(30, 30)),
            AnchoredRead(umi="AAA", start=4, sequence="GT", qualities=(30, 30)),
        ]
        result = analyze_anchored_reads(reads)
        self.assertEqual(result.nodes_before, 1)
        self.assertEqual(result.molecules_after, 1)
        mol = result.molecules[0]
        self.assertEqual((mol.ref_start, mol.ref_end), (0, 6))
        self.assertEqual(mol.consensus_seq, "ACNNGT")

    def test_complete_non_overlap_same_umi(self):
        # 同 UMI 完全无重叠：仍属一簇，中间整段为空洞
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="AA", qualities=(30, 30)),
            AnchoredRead(umi="AAA", start=10, sequence="TT", qualities=(30, 30)),
        ]
        mol = analyze_anchored_reads(reads).molecules[0]
        self.assertEqual((mol.ref_start, mol.ref_end), (0, 12))
        self.assertEqual(mol.consensus_seq, "AA" + "N" * 8 + "TT")
        self.assertEqual(mol.coverage, (1, 1) + (0,) * 8 + (1, 1))

    def test_complete_non_overlap_different_umi(self):
        # 不同 UMI（计数相等、Hamming>1）：各自成分子，各自区间独立
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="AC", qualities=(30, 30)),
            AnchoredRead(umi="CCC", start=50, sequence="GT", qualities=(30, 30)),
        ]
        result = analyze_anchored_reads(reads)
        self.assertEqual(result.molecules_after, 2)
        by_root = {m.root_umi: m for m in result.molecules}
        self.assertEqual((by_root["AAA"].ref_start, by_root["AAA"].ref_end), (0, 2))
        self.assertEqual((by_root["CCC"].ref_start, by_root["CCC"].ref_end), (50, 52))
        self.assertEqual(by_root["AAA"].consensus_seq, "AC")
        self.assertEqual(by_root["CCC"].consensus_seq, "GT")


class TestAnchoredHandLikelihood(unittest.TestCase):
    def test_per_coordinate_hand_computed(self):
        # 逐坐标手算：坐标 1 被两条读段覆盖，坐标 0、2 各一条
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="AC", qualities=(20, 20)),
            AnchoredRead(umi="AAA", start=1, sequence="CT", qualities=(30, 30)),
        ]
        mol = analyze_anchored_reads(reads).molecules[0]
        self.assertEqual((mol.ref_start, mol.ref_end), (0, 3))
        self.assertEqual(mol.consensus_seq, "ACT")
        self.assertEqual(mol.coverage, (1, 2, 1))

        # 坐标 0：仅读段0 的 'A'@Q20
        ll0 = hand_ll([("A", 20)])
        self.assertAlmostEqual(mol.confidence_gap[0], hand_gap(ll0), places=12)
        # 坐标 1：读段0 'C'@Q20 + 读段1 'C'@Q30
        ll1 = hand_ll([("C", 20), ("C", 30)])
        self.assertAlmostEqual(mol.confidence_gap[1], hand_gap(ll1), places=12)
        # 坐标 2：仅读段1 的 'T'@Q30
        ll2 = hand_ll([("T", 30)])
        self.assertAlmostEqual(mol.confidence_gap[2], hand_gap(ll2), places=12)

    def test_short_reads_do_not_misalign_long_read(self):
        # 长读段 ACGT@0，两条短读段分别从坐标 2、3 起；
        # 坐标对齐必须按参考坐标，短读段不得把长读段碱基挤错位。
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="ACGT", qualities=(30,) * 4),
            AnchoredRead(umi="AAA", start=2, sequence="GC", qualities=(30, 30)),
            AnchoredRead(umi="AAA", start=3, sequence="TA", qualities=(30, 30)),
        ]
        mol = analyze_anchored_reads(reads).molecules[0]
        self.assertEqual((mol.ref_start, mol.ref_end), (0, 5))
        self.assertEqual(mol.coverage, (1, 1, 2, 3, 1))
        # 坐标 2：长读段 'G'@30 + 短读段 'G'@30 -> G
        # 坐标 3：长读段 'T'@30 + 短读段1 'C'@30 + 短读段2 'T'@30 -> T
        # 坐标 4：仅短读段2 'A'@30 -> A
        self.assertEqual(mol.consensus_seq, "ACGTA")
        ll3 = hand_ll([("T", 30), ("C", 30), ("T", 30)])
        self.assertAlmostEqual(mol.confidence_gap[3], hand_gap(ll3), places=12)

    def test_q0_and_tie_rules(self):
        # Q=0：呼中碱基 ll=-inf，三个未呼中碱基精确并列 -> N，gap=0.0
        reads = [AnchoredRead(umi="AAA", start=0, sequence="A", qualities=(0,))]
        mol = analyze_anchored_reads(reads).molecules[0]
        self.assertEqual(mol.consensus_seq, "N")
        self.assertEqual(mol.confidence_gap, (0.0,))
        self.assertEqual(mol.coverage, (1,))

        # 两条等质量对立读段：A 与 C 精确并列 -> N，gap=0.0
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="A", qualities=(25,)),
            AnchoredRead(umi="AAA", start=0, sequence="C", qualities=(25,)),
        ]
        mol = analyze_anchored_reads(reads).molecules[0]
        self.assertEqual(mol.consensus_seq, "N")
        self.assertEqual(mol.confidence_gap, (0.0,))

    def test_low_quality_majority_overturned(self):
        # 坐标 0：两条 Q7 的 'C' 被一条 Q30 的 'A' 推翻（手算验证）
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="C", qualities=(7,)),
            AnchoredRead(umi="AAA", start=0, sequence="C", qualities=(7,)),
            AnchoredRead(umi="AAA", start=0, sequence="A", qualities=(30,)),
        ]
        mol = analyze_anchored_reads(reads).molecules[0]
        ll = hand_ll([("C", 7), ("C", 7), ("A", 30)])
        self.assertGreater(ll["A"], ll["C"])  # 手算先确认前提
        self.assertEqual(mol.consensus_seq, "A")
        self.assertAlmostEqual(mol.confidence_gap[0], hand_gap(ll), places=12)


class TestAnchoredClustering(unittest.TestCase):
    def test_umi_absorption_preserved(self):
        # 大根 AAA(4) 吸收 Hamming=1 的 AAT(1)：4 >= 2*1-1
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="AC", qualities=(30, 30)),
            AnchoredRead(umi="AAA", start=0, sequence="AC", qualities=(30, 30)),
            AnchoredRead(umi="AAA", start=0, sequence="AC", qualities=(30, 30)),
            AnchoredRead(umi="AAA", start=0, sequence="AC", qualities=(30, 30)),
            AnchoredRead(umi="AAT", start=2, sequence="GT", qualities=(30, 30)),
        ]
        result = analyze_anchored_reads(reads)
        self.assertEqual(result.nodes_before, 2)
        self.assertEqual(result.molecules_after, 1)
        mol = result.molecules[0]
        self.assertEqual(mol.root_umi, "AAA")
        self.assertEqual(mol.members, ("AAA", "AAT"))
        self.assertEqual((mol.ref_start, mol.ref_end), (0, 4))
        self.assertEqual(mol.consensus_seq, "ACGT")

    def test_shuffle_preserves_roots_and_consensus(self):
        base_reads = [
            AnchoredRead(umi="AAA", start=0, sequence="ACGT", qualities=(30, 20, 30, 20)),
            AnchoredRead(umi="AAA", start=2, sequence="GTAC", qualities=(25, 25, 25, 25)),
            AnchoredRead(umi="AAT", start=1, sequence="CG", qualities=(30, 30)),
            AnchoredRead(umi="CCC", start=10, sequence="AAAA", qualities=(30,) * 4),
            AnchoredRead(umi="CCC", start=12, sequence="AT", qualities=(20, 20)),
        ]
        ref = analyze_anchored_reads(base_reads)
        rng = random.Random(20260925)
        for _ in range(20):
            shuffled = list(base_reads)
            rng.shuffle(shuffled)
            got = analyze_anchored_reads(shuffled)
            self.assertEqual(
                [(m.root_umi, m.members) for m in got.molecules],
                [(m.root_umi, m.members) for m in ref.molecules],
            )
            for gm, rm in zip(got.molecules, ref.molecules):
                self.assertEqual(gm.consensus_seq, rm.consensus_seq)
                self.assertEqual(gm.coverage, rm.coverage)
                self.assertEqual(
                    (gm.ref_start, gm.ref_end), (rm.ref_start, rm.ref_end)
                )
                for g_gap, r_gap in zip(gm.confidence_gap, rm.confidence_gap):
                    if g_gap is None or r_gap is None:
                        self.assertIs(g_gap, r_gap)
                    else:
                        self.assertAlmostEqual(g_gap, r_gap, places=12)

    def test_equal_length_entry_compatible(self):
        # 全部 start=0 且等长时，锚定共识与等长入口结果一致
        plain = [
            Read(umi="AAAA", sequence="ACGT", qualities=(30, 20, 30, 20)),
            Read(umi="AAAA", sequence="ACGA", qualities=(25, 25, 25, 25)),
            Read(umi="AAAT", sequence="ACGT", qualities=(30, 30, 30, 30)),
        ]
        anchored = [
            AnchoredRead(umi=r.umi, start=0, sequence=r.sequence, qualities=r.qualities)
            for r in plain
        ]
        ref = analyze_reads(plain)
        got = analyze_anchored_reads(anchored)
        self.assertEqual(ref.molecules_after, got.molecules_after)
        for rm, gm in zip(ref.molecules, got.molecules):
            self.assertEqual(rm.root_umi, gm.root_umi)
            self.assertEqual(rm.members, gm.members)
            self.assertEqual(rm.consensus_seq, gm.consensus_seq)
            self.assertEqual((gm.ref_start, gm.ref_end), (0, 4))
            for r_gap, g_gap in zip(rm.confidence_gap, gm.confidence_gap):
                self.assertAlmostEqual(r_gap, g_gap, places=12)


class TestAnchoredValidation(unittest.TestCase):
    def test_empty_batch_rejected(self):
        with self.assertRaises(InvalidArgumentError):
            analyze_anchored_reads([])

    def test_batch_located_error(self):
        # 非法碱基在构造期即被拒绝
        with self.assertRaises(InvalidArgumentError):
            AnchoredRead(umi="AAA", start=0, sequence="AN", qualities=(30, 30))
        # 批级错误带 reads[i] 定位前缀
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="AC", qualities=(30, 30)),
            AnchoredRead(umi="AAAA", start=0, sequence="AC", qualities=(30, 30)),
        ]
        with self.assertRaises(InvalidReadError) as ctx:
            analyze_anchored_reads(reads)
        self.assertIn("reads[1]", str(ctx.exception))

    def test_wrong_type_rejected(self):
        with self.assertRaises(InvalidReadError):
            analyze_anchored_reads([Read(umi="AAA", sequence="ACG", qualities=(30, 30, 30))])

    def test_variable_lengths_accepted(self):
        reads = [
            AnchoredRead(umi="AAA", start=0, sequence="ACGTA", qualities=(30,) * 5),
            AnchoredRead(umi="AAA", start=1, sequence="CG", qualities=(30, 30)),
        ]
        mol = analyze_anchored_reads(reads).molecules[0]
        self.assertEqual(mol.consensus_seq, "ACGTA")
        self.assertEqual(mol.coverage, (1, 2, 2, 1, 1))


if __name__ == "__main__":
    unittest.main()
