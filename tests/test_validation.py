"""输入校验与错误语义测试。"""

from __future__ import annotations

import math
import unittest

from umi_consensus import (
    LIMITS,
    ValidationError,
    cluster_umis,
    collapse,
    consensus,
)


class ValidationTests(unittest.TestCase):
    def test_empty_reads_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            consensus([])
        self.assertIn("reads", ctx.exception.field or "")

    def test_empty_records_batch_is_empty_result(self):
        # 空批次允许（与“空读段组无法共识”不同），返回空元组。
        self.assertEqual(collapse(()), ())
        self.assertEqual(cluster_umis({}), ())

    def test_nan_infinity_and_floats_rejected(self):
        for bad in (math.nan, math.inf, -math.inf, 30.5, 0.0):
            with self.assertRaises(ValidationError):
                consensus([("A", (bad,))])
            with self.assertRaises(ValidationError):
                cluster_umis({"A": bad})

    def test_bool_rejected_as_integer(self):
        with self.assertRaises(ValidationError):
            consensus([("A", (True,))])
        with self.assertRaises(ValidationError):
            cluster_umis({"A": True})
        with self.assertRaises(ValidationError):
            collapse([("AA", "A", (False,))])

    def test_bad_bases_and_lowercase_rejected(self):
        for bad_seq in ("N", "a", "ACGTN", ".", " A"):
            with self.assertRaises(ValidationError):
                consensus([(bad_seq, (10,) * len(bad_seq))])
        with self.assertRaises(ValidationError):
            collapse([("AN", "A", (10,))])

    def test_quality_range(self):
        for q in (-1, 41, 100):
            with self.assertRaises(ValidationError):
                consensus([("AC", (10, q))])
        # 边界本身合法
        seq, conf, _ = consensus([("AC", (0, 40))])
        self.assertEqual(seq, "NC")  # Q0 单读段：三个错配碱基并列 → N
        self.assertEqual(conf[0], 0.0)
        # 三个 Q0 读段分别排除 A/C/G 后，T 是唯一有限似然 → +inf 信心差
        seq, conf, _ = consensus([("A", (0,)), ("C", (0,)), ("G", (0,))])
        self.assertEqual(seq, "T")
        self.assertTrue(math.isinf(conf[0]))

    def test_length_mismatch_is_locatable(self):
        with self.assertRaises(ValidationError) as ctx:
            consensus([("AC", (10,))])
        self.assertEqual(ctx.exception.field, "qualities")

        with self.assertRaises(ValidationError) as ctx:
            collapse([
                ("AA", "AC", (10, 10)),
                ("AAA", "AC", (10, 10)),
            ])
        self.assertEqual(ctx.exception.index, 1)
        self.assertEqual(ctx.exception.field, "umi")

        with self.assertRaises(ValidationError) as ctx:
            collapse([
                ("AA", "AC", (10, 10)),
                ("AA", "ACG", (10, 10, 10)),
            ])
        self.assertEqual(ctx.exception.index, 1)

    def test_counts_must_be_positive_and_mapping(self):
        for bad in ({}, ):
            self.assertEqual(cluster_umis(bad), ())
        with self.assertRaises(ValidationError):
            cluster_umis({"A": 0})
        with self.assertRaises(ValidationError):
            cluster_umis({"A": -3})
        with self.assertRaises(ValidationError):
            cluster_umis([("A", 1)])  # 不是 Mapping
        with self.assertRaises(ValidationError) as ctx:
            cluster_umis({"": 1})
        self.assertIn("umi", ctx.exception.field)

    def test_bad_record_shape(self):
        with self.assertRaises(ValidationError):
            collapse([("AA", "AC")])  # 二元组不是合法记录
        with self.assertRaises(ValidationError):
            collapse(["AAC"])  # 字符串不是记录

    def test_not_iterable(self):
        with self.assertRaises(ValidationError):
            collapse(123)
        with self.assertRaises(ValidationError):
            consensus(None)

    def test_length_limits(self):
        long_umi = "A" * (LIMITS["umi_length_max"] + 1)
        with self.assertRaises(ValidationError):
            cluster_umis({long_umi: 1})
        with self.assertRaises(ValidationError):
            collapse([(long_umi, "A", (10,))])
        long_read = "A" * (LIMITS["read_length_max"] + 1)
        with self.assertRaises(ValidationError):
            consensus([(long_read, (10,) * len(long_read))])

    def test_input_not_mutated_on_failure(self):
        counts = {"AAA": 3, "bad": 2}
        before = dict(counts)
        with self.assertRaises(ValidationError):
            cluster_umis(counts)
        self.assertEqual(counts, before)

    def test_generator_consumed_only_after_full_validation_path(self):
        # 第三条非法：整体失败，不产生结果（纯函数语义）。
        def gen():
            yield ("AA", "AC", (10, 10))
            yield ("AA", "AC", (10, 10))
            yield ("AA", "N", (10,))

        with self.assertRaises(ValidationError) as ctx:
            collapse(gen())
        self.assertEqual(ctx.exception.index, 2)


if __name__ == "__main__":
    unittest.main()
