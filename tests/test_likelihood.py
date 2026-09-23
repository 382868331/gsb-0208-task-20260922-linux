"""手算质量对数似然、并列 N、低质量多数等共识测试。"""

from __future__ import annotations

import math
import unittest

from umicorrect import (  # noqa: E402
    InvalidArgumentError,
    Read,
    consensus,
)

LOG10 = math.log10


class HandComputedLikelihoodTests(unittest.TestCase):
    """期望值全部由题面公式手工化简得到，不调用被测实现求期望。"""

    def test_single_read_q20_gap_is_log10_297(self) -> None:
        # e = 10^(-20/10) = 0.01
        # 共识=A；gap = log10((1-e)/(e/3)) = log10(0.99*3/0.01) = log10(297)
        reads = [Read("A", "A", (20,))]
        seq, gaps = consensus(reads, 1, read_indices=(0,))
        self.assertEqual(seq, "A")
        self.assertAlmostEqual(gaps[0], LOG10(297), places=12)
        # 手算十进制值：log10(297) = 2.472756...
        self.assertLess(abs(gaps[0] - 2.472756449317212), 1e-9)

    def test_two_reads_higher_q_wins_despite_tied_vote(self) -> None:
        # read1 呼 A @Q20(e=0.01)，read2 呼 C @Q30(e=0.001)。
        # L(C)/L(A) = ((0.01/3)*0.999) / (0.99*(0.001/3))
        #           = 0.00999/0.00099 = 999/99 = 111/11 > 1  -> C 胜
        # gap = log10(111/11) ≈ 1.0039
        reads = [
            Read("A", "A", (20,)),
            Read("A", "C", (30,)),
        ]
        seq, gaps = consensus(reads, 1, read_indices=(0, 1))
        self.assertEqual(seq, "C")
        self.assertAlmostEqual(gaps[0], LOG10(111 / 11), places=12)
        self.assertGreater(gaps[0], 1.0)
        self.assertLess(gaps[0], 1.01)

    def test_exact_tie_outputs_N_gap_zero(self) -> None:
        # A@Q20 与 C@Q20：ll(A)=logc+logw，ll(C)=logw+logc，精确相等。
        reads = [
            Read("A", "A", (20,)),
            Read("A", "C", (20,)),
        ]
        seq, gaps = consensus(reads, 1, read_indices=(0, 1))
        self.assertEqual(seq, "N")
        self.assertEqual(gaps[0], 0.0)

    def test_q0_correct_base_impossible_three_way_tie(self) -> None:
        # Q=0: e=1，呼中的碱基概率为 0（log10 0 = -inf），其余三者并列
        # log10(1/3) -> 输出 N，gap 0。
        reads = [Read("A", "A", (0,))]
        seq, gaps = consensus(reads, 1, read_indices=(0,))
        self.assertEqual(seq, "N")
        self.assertEqual(gaps[0], 0.0)

    def test_low_quality_majority_overridden_by_one_high_q_read(self) -> None:
        # 两条 Q=3 呼 A（e≈0.501，正确概率 0.499，接近抛硬币），
        # 一条 Q=40 呼 C（e=1e-4）。手算：
        # L(A) = 0.4988^2 * (1e-4/3) ≈ 8.3e-6
        # L(C) = (0.5012/3)^2 * 0.9999 ≈ 2.79e-2
        # 低质量多数必须被高质量单读推翻，共识为 C。
        reads = [
            Read("A", "A", (3,)),
            Read("A", "A", (3,)),
            Read("A", "C", (40,)),
        ]
        seq, gaps = consensus(reads, 1, read_indices=(0, 1, 2))
        self.assertEqual(seq, "C")
        e3 = 10.0 ** (-3 / 10)
        e40 = 10.0 ** (-40 / 10)
        lc = (1 - e3) ** 2 * (e40 / 3)
        la = (e3 / 3) ** 2 * (1 - e40)
        self.assertAlmostEqual(gaps[0], LOG10(la / lc), places=10)
        self.assertGreater(gaps[0], 3.0)

    def test_multi_position_independence(self) -> None:
        # 位置0：两读一致呼 C@30；位置1：A@10 与 G@10 并列 -> N。
        reads = [
            Read("AA", "CA", (30, 10)),
            Read("AA", "CG", (30, 10)),
        ]
        seq, gaps = consensus(reads, 2, read_indices=(0, 1))
        self.assertEqual(seq, "CN")
        self.assertEqual(gaps[1], 0.0)
        # 位置0：L(C)/L(A) = ((0.999)^2)/((0.001/3)^2) = (2997)^2
        self.assertAlmostEqual(gaps[0], LOG10(2997**2), places=10)


class ConsensusValidationTests(unittest.TestCase):
    def test_empty_cluster_rejected(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            consensus([Read("A", "A", (20,))], 1, read_indices=())

    def test_bool_quality_rejected_at_construction(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            Read("A", "A", (True,))
        with self.assertRaises(InvalidArgumentError):
            Read("A", "A", (False,))

    def test_nan_inf_quality_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "NaN/Infinity"):
            Read("A", "A", (float("nan"),))
        with self.assertRaisesRegex(InvalidArgumentError, "NaN/Infinity"):
            Read("A", "A", (float("inf"),))

    def test_quality_out_of_range_rejected_with_position(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, r"qualities\[1\]"):
            Read("AC", "AC", (20, 41))
        with self.assertRaisesRegex(InvalidArgumentError, r"qualities\[0\]"):
            Read("A", "A", (-1,))

    def test_read_length_nan_rejected(self) -> None:
        reads = [Read("A", "A", (20,))]
        with self.assertRaisesRegex(InvalidArgumentError, "NaN/Infinity"):
            consensus(reads, float("nan"), read_indices=(0,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(InvalidArgumentError, "整数"):
            consensus(reads, 1.0, read_indices=(0,))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
