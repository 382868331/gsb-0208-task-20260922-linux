"""逐碱基对数似然共识测试：手算常量 + 独立参考对拍。"""

from __future__ import annotations

import math
import unittest

from umi_consensus import consensus

from reference import ref_consensus_scores

# 手算常量（Q = -10 log10 p，等概率三向错配）：
# Q=10: p=10^-1 = 0.1；     match = log10(0.9)      = -0.04575749056067512
#                            mismatch = log10(0.1/3) = -1.4771212547196626
LOG_MATCH_10 = math.log10(0.9)
LOG_MISMATCH_10 = math.log10(0.1 / 3.0)
# 手算核对值（直接写出十进制结果，不靠被测实现）：
HAND_LOG10_09 = -0.04575749056067512
HAND_LOG10_81 = -0.09151498112135023  # log10(0.9 * 0.9)
HAND_LOG10_00333 = -1.4771212547196626  # log10(1/30)


class HandComputedLikelihoodTests(unittest.TestCase):
    def test_constants_match_hand_arithmetic(self):
        self.assertAlmostEqual(LOG_MATCH_10, HAND_LOG10_09, places=15)
        self.assertAlmostEqual(LOG_MISMATCH_10, HAND_LOG10_00333, places=15)

    def test_two_identical_high_quality_reads(self):
        seq, conf, scores = consensus([("A", (10,)), ("A", (10,))])
        self.assertEqual(seq, "A")
        table = scores[0]
        # 两个 Q10 A：LL(A) = 2*log10(0.9)，LL(C/G/T) = 2*log10(1/30)
        self.assertAlmostEqual(table["A"], HAND_LOG10_81, places=14)
        self.assertAlmostEqual(table["C"], 2 * HAND_LOG10_00333, places=14)
        self.assertAlmostEqual(table["G"], table["C"], places=15)
        self.assertAlmostEqual(table["T"], table["C"], places=15)
        # 信心差 = 最高 - 次高
        self.assertAlmostEqual(
            conf[0], HAND_LOG10_81 - 2 * HAND_LOG10_00333, places=14
        )
        self.assertGreater(conf[0], 0.0)

    def test_tie_yields_N_and_zero_confidence(self):
        # 一个 Q10 A 与一个 Q10 C：LL(A)=LL(C)=match+mismatch，严格并列。
        seq, conf, scores = consensus([("A", (10,)), ("C", (10,))])
        self.assertEqual(seq, "N")
        self.assertEqual(conf[0], 0.0)
        self.assertAlmostEqual(scores[0]["A"], scores[0]["C"], places=15)
        self.assertLess(scores[0]["G"], scores[0]["A"])

    def test_multi_position_consensus(self):
        reads = [("AC", (20, 10)), ("AG", (20, 30)), ("AT", (5, 30))]
        seq, conf, scores = consensus(reads)
        self.assertEqual(len(seq), 2)
        self.assertEqual(seq[0], "A")
        # 位置 1：C@10, G@30, T@30；G/T 质量谱完全相同 → 严格并列 → N
        self.assertEqual(seq[1], "N")
        self.assertEqual(conf[1], 0.0)


class LowQualityMajorityTests(unittest.TestCase):
    def test_low_quality_majority_overruled(self):
        # 多数派：2 条 Q2 的 A（p≈0.631，比错配好不了多少）；
        # 少数派：1 条 Q40 的 C（p=1e-4）。高置信少数应胜出。
        reads = [("A", (2,)), ("A", (2,)), ("C", (40,))]
        seq, conf, _ = consensus(reads)
        self.assertEqual(seq, "C")
        self.assertGreater(conf[0], 0.0)

    def test_low_quality_majority_holds_when_evidence_close(self):
        # 2 条 Q20 的 A vs 1 条 Q20 的 C：多数派胜。
        seq, _, _ = consensus([("A", (20,)), ("A", (20,)), ("C", (20,))])
        self.assertEqual(seq, "A")


class QZeroTests(unittest.TestCase):
    def test_q0_single_read_is_three_way_tie(self):
        # Q0：p=1，读到的碱基必错；其余三个碱基等可能 → N。
        seq, conf, scores = consensus([("A", (0,))])
        self.assertEqual(seq, "N")
        self.assertEqual(conf[0], 0.0)
        self.assertEqual(scores[0]["A"], float("-inf"))
        self.assertGreater(scores[0]["C"], float("-inf"))

    def test_q0_three_calls_leave_unique_base(self):
        # A/C/G 各一条 Q0：这三个真值似然为 0（-inf），只剩 T 有限。
        seq, conf, scores = consensus([("A", (0,)), ("C", (0,)), ("G", (0,))])
        self.assertEqual(seq, "T")
        self.assertTrue(math.isinf(conf[0]))
        self.assertGreater(conf[0], 0.0)
        self.assertEqual(scores[0]["T"], 3 * math.log10(1.0 / 3.0))


class ReferenceCrossCheckTests(unittest.TestCase):
    """与逐条累加的独立参考在多组数据上对拍分数与判定。"""

    def _winners(self, table):
        best = max(table.values())
        return frozenset(b for b, v in table.items() if v == best)

    def test_cross_check_mixed(self):
        cases = [
            [("A", (10,)), ("A", (10,))],
            [("A", (10,)), ("C", (10,))],
            [("A", (2,)), ("A", (2,)), ("C", (40,))],
            [("ACGT", (3, 11, 25, 40)), ("ACGA", (8, 11, 25, 40)),
             ("TCGT", (30, 2, 25, 0))],
            [("GG", (q, q % 41)) for q in range(0, 41)],
        ]
        for reads in cases:
            seq, conf, scores = consensus(reads)
            ref_scores = ref_consensus_scores(
                [(s, tuple(qs)) for s, qs in reads]
            )
            self.assertEqual(len(scores), len(ref_scores))
            for pos, (got, ref) in enumerate(zip(scores, ref_scores)):
                for base in "ACGT":
                    if math.isinf(ref[base]):
                        self.assertEqual(got[base], ref[base], msg=(pos, base))
                    else:
                        self.assertAlmostEqual(
                            got[base], ref[base], places=12, msg=(pos, base)
                        )
                got_winners = self._winners(got)
                ref_winners = self._winners(ref)
                self.assertEqual(got_winners, ref_winners)
                expected_call = (
                    next(iter(got_winners)) if len(got_winners) == 1 else "N"
                )
                self.assertEqual(seq[pos], expected_call)

    def test_order_independence(self):
        # 分数只取决于 (碱基,质量) 计数 → 打乱读段顺序结果必须逐位相同。
        import random

        reads = [("AC", (7, 30)), ("AC", (19, 5)), ("GC", (12, 30)),
                 ("AT", (40, 11)), ("AC", (1, 9))]
        rng = random.Random(1234)  # 确定性洗牌，仅用于排列输入
        shuffled = reads[:]
        rng.shuffle(shuffled)
        seq1, conf1, scores1 = consensus(reads)
        seq2, conf2, scores2 = consensus(shuffled)
        self.assertEqual(seq1, seq2)
        self.assertEqual(conf1, conf2)
        self.assertEqual(scores1, scores2)

    def test_confidence_is_best_minus_second(self):
        reads = [("AC", (20, 10)), ("AG", (20, 30)), ("AT", (5, 30))]
        _, conf, scores = consensus(reads)
        for pos, table in enumerate(scores):
            ordered = sorted(table.values(), reverse=True)
            if ordered[0] == ordered[1]:
                self.assertEqual(conf[pos], 0.0)
            elif math.isinf(ordered[1]):
                self.assertTrue(math.isinf(conf[pos]))
            else:
                self.assertAlmostEqual(conf[pos], ordered[0] - ordered[1], places=15)


if __name__ == "__main__":
    unittest.main()
