"""analyze_reads 端到端测试：独立参考共识、空批次拒绝、原子失败、确定性。"""

from __future__ import annotations

import math
import random
import unittest

from reference import reference_clusters, reference_consensus

from umicorrect import (
    InvalidArgumentError,
    InvalidReadError,
    LimitExceededError,
    Read,
    SearchBudgetExhaustedError,
    analyze_reads,
    cluster_umis,
)


def build_batch(specs):
    """specs: list of (umi, base_char, q)。"""
    reads = []
    for umi, base, q in specs:
        seq = base * len(umi)
        reads.append(Read(umi, seq, tuple([q] * len(umi))))
    return reads


class EndToEndTests(unittest.TestCase):
    def test_merge_reduces_molecule_count_and_consensus_matches_reference(self) -> None:
        # 大根 AAAA 30 条呼 AAAA@30；错误 UMI AAAT 2 条在位置 2 呼 G@10。
        specs = [("AAAA", "A", 30)] * 30
        reads = build_batch(specs)
        reads += [Read("AAAT", "AAGA", (10, 10, 10, 10)) for _ in range(2)]
        result = analyze_reads(reads)

        self.assertEqual(result.nodes_before, 2)
        self.assertEqual(result.molecules_after, 1)
        self.assertEqual(result.merged_count, 1)
        (mol,) = result.molecules
        self.assertEqual(mol.root_umi, "AAAA")
        self.assertEqual(mol.members, ("AAAA", "AAAT"))
        self.assertEqual(len(mol.read_indices), 32)
        self.assertEqual(mol.consensus_seq, "AAAA")
        # 与独立参考共识逐项一致
        ref_seq, ref_gaps = reference_consensus(reads, 4, list(mol.read_indices))
        self.assertEqual(mol.consensus_seq, ref_seq)
        for got, want in zip(mol.confidence_gap, ref_gaps):
            self.assertTrue(math.isfinite(got))
            self.assertAlmostEqual(got, want, places=12)
        # 错误位置（2）信心差应明显小于完全一致的位置
        self.assertLess(mol.confidence_gap[2], mol.confidence_gap[0])

    def test_bridge_scenario_two_molecules_survive(self) -> None:
        # 两个大根各呼不同碱基，桥只归一侧：分子数仍为 2。
        specs = (
            [("AAAA", "A", 25)] * 25
            + [("AATT", "C", 25)] * 25
            + [("AAAT", "A", 8)]
        )
        reads = build_batch(specs)
        result = analyze_reads(reads)
        self.assertEqual(result.nodes_before, 3)
        self.assertEqual(result.molecules_after, 2)
        roots = {m.root_umi for m in result.molecules}
        self.assertEqual(roots, {"AAAA", "AATT"})

    def test_random_batches_against_independent_reference(self) -> None:
        rng = random.Random(424242)
        for _ in range(40):
            length = rng.randint(1, 5)
            n_umi = rng.randint(1, 8)
            umis = ["".join(rng.choice("ACGT") for _ in range(length)) for _ in range(n_umi)]
            umis = list(dict.fromkeys(umis))
            specs = []
            for u in umis:
                for _ in range(rng.randint(1, 6)):
                    specs.append((u, rng.choice("ACGT"), rng.randint(0, 40)))
            reads = build_batch(specs)
            result = analyze_reads(reads)

            counts = {}
            for r in reads:
                counts[r.umi] = counts.get(r.umi, 0) + 1
            want_groups = reference_clusters(counts)
            self.assertEqual(
                sorted(sorted(g) for g in want_groups),
                sorted(list(m.members) for m in result.molecules),
            )
            groups_by_root = {
                min(g, key=lambda u: (-counts[u], u)): g for g in want_groups
            }
            for mol in result.molecules:
                idxs = [
                    i for i, r in enumerate(reads) if r.umi in groups_by_root[mol.root_umi]
                ]
                ref_seq, ref_gaps = reference_consensus(reads, length, idxs)
                self.assertEqual(mol.consensus_seq, ref_seq)
                for got, want in zip(mol.confidence_gap, ref_gaps):
                    self.assertAlmostEqual(got, want, places=12)

    def test_deterministic_across_runs_and_input_shuffles(self) -> None:
        specs = (
            [("AAAA", "A", 30)] * 30
            + [("AAAT", "A", 15)] * 3
            + [("AATT", "G", 30)] * 10
        )
        reads = build_batch(specs)
        r1 = analyze_reads(reads)
        r2 = analyze_reads(list(reversed(reads)))
        self.assertEqual(
            [(m.root_umi, m.members, m.consensus_seq) for m in r1.molecules],
            [(m.root_umi, m.members, m.consensus_seq) for m in r2.molecules],
        )
        self.assertEqual(
            (r1.nodes_before, r1.molecules_after),
            (r2.nodes_before, r2.molecules_after),
        )
        # 输入重排只改变有序求和的累加次序，gap 允许 ulp 级差（README 口径）
        for m1, m2 in zip(r1.molecules, r2.molecules):
            for g1, g2 in zip(m1.confidence_gap, m2.confidence_gap):
                self.assertAlmostEqual(g1, g2, places=12)


class BatchValidationTests(unittest.TestCase):
    def test_empty_batch_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "空批次"):
            analyze_reads([])

    def test_empty_umi_and_empty_sequence_rejected(self) -> None:
        # 字段级非法在 Read 构造期即拒绝（可定位到字段），不等到批次调用。
        with self.assertRaisesRegex(InvalidArgumentError, "umi 为空"):
            Read("", "A", (20,))
        with self.assertRaisesRegex(InvalidArgumentError, "sequence 为空"):
            Read("A", "", ())

    def test_length_mismatch_locatable(self) -> None:
        # 跨读段固定长度不一致是批级错误，定位到 reads[2]。
        with self.assertRaisesRegex(InvalidReadError, r"reads\[2\]"):
            analyze_reads(
                [
                    Read("AA", "AA", (20, 20)),
                    Read("AA", "AA", (20, 20)),
                    Read("AAA", "AAA", (20, 20, 20)),
                ]
            )

    def test_illegal_base_locatable(self) -> None:
        # 构造期：字段路径可定位
        with self.assertRaisesRegex(InvalidArgumentError, r"umi\[1\]"):
            Read("AN", "AA", (20, 20))

    def test_batch_path_adds_reads_prefix(self) -> None:
        # 绕过构造期校验（直接安放字段），验证批级路径加 reads[0] 前缀
        obj = object.__new__(Read)
        object.__setattr__(obj, "umi", "AA")
        object.__setattr__(obj, "sequence", "NA")
        object.__setattr__(obj, "qualities", (20, 20))
        with self.assertRaisesRegex(InvalidReadError, r"reads\[0\].*sequence\[0\]"):
            analyze_reads([obj])

    def test_not_a_read_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidReadError, r"reads\[0\]"):
            analyze_reads([("AA", "AA", (20, 20))])

    def test_non_list_batch_rejected(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            analyze_reads((r for r in []))

    def test_batch_failure_is_atomic_no_partial_state(self) -> None:
        # analyze_reads 无状态；字段级非法在构造期、批级非法在校验期，
        # 均先于任何结果构建。这里用批级非法（固定长度不一致）验证：
        good = [Read("AA", "AA", (20, 20)), Read("AT", "AT", (10, 10))]
        bad = good + [Read("AAA", "AAA", (20, 20, 20))]
        with self.assertRaises(InvalidReadError):
            analyze_reads(bad)
        # 失败调用不留部分结果；修正批次后正常计算
        result_after = analyze_reads(good)
        self.assertEqual(result_after.molecules_after, 1)

    def test_limit_exceeded_separate_exception_type(self) -> None:
        # UMI 长度上限 32：构造 33 长 UMI
        with self.assertRaises(LimitExceededError):
            Read("A" * 33, "A" * 33, tuple([20] * 33))

    def test_budget_exhausted_propagates_and_is_not_value_error(self) -> None:
        specs = [("AAA", "A", 20)] * 5 + [("AAC", "C", 10)] * 3 + [("AAG", "G", 10)]
        reads = build_batch(specs)
        with self.assertRaises(SearchBudgetExhaustedError):
            analyze_reads(reads, probe_budget=1)
        # 放大预算后同一输入成功（预算耗尽≠无解）
        ok = analyze_reads(reads, probe_budget=10)
        self.assertEqual(ok.molecules_after, 1)


class RoundTripSupplementTests(unittest.TestCase):
    """补充性质（非期望值来源）：分区覆盖全部读段且互不相交。"""

    def test_partition_is_complete_and_disjoint(self) -> None:
        rng = random.Random(7)
        for _ in range(20):
            umis = ["".join(rng.choice("ACGT") for _ in range(3)) for _ in range(10)]
            umis = list(dict.fromkeys(umis))
            counts = {u: rng.randint(1, 8) for u in umis}
            clusters = cluster_umis(counts)
            seen: list[str] = []
            for cl in clusters:
                seen.extend(cl.members)
            self.assertEqual(sorted(seen), sorted(counts))


if __name__ == "__main__":
    unittest.main()
