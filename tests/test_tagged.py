"""双端 UMI + 坐标 + 链方向管线测试。

期望值来源（均不调用被测核心生成期望）：
* 规范化、成对、去重等场景为按题面规则手算的固定用例；
* :func:`independent_normalize` —— 测试内独立书写的反向互补规范化
  （不导入 umicorrect）；
* :mod:`reference` 中的穷举聚类/Floyd 传递闭包与逐读段似然参考。
"""

from __future__ import annotations

import math
import random
import unittest

from reference import reference_clusters, reference_consensus

from umicorrect import (
    DuplicateRecordError,
    InvalidArgumentError,
    InvalidReadError,
    LimitExceededError,
    SearchBudgetExhaustedError,
    TaggedRead,
    analyze_tagged_reads,
)

LOG10 = math.log10
_COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def rc(seq: str) -> str:
    """测试内独立实现的反向互补（不使用被测的 reverse_complement）。"""
    return "".join(_COMP[ch] for ch in reversed(seq))


def independent_normalize(rec: TaggedRead):
    """独立规范化：返回 (n1, n2, nseq, nqual)，反链取反向互补并反转质量。"""
    if rec.strand == "-":
        return (
            rc(rec.umi1),
            rc(rec.umi2),
            rc(rec.sequence),
            tuple(reversed(rec.qualities)),
        )
    return rec.umi1, rec.umi2, rec.sequence, tuple(rec.qualities)


def make(umi1, umi2, coord, strand, seq, q, rid):
    """快捷构造：q 为 int（全位置同质量）或质量序列。"""
    qual = tuple(q) if isinstance(q, (list, tuple)) else tuple([q] * len(seq))
    return TaggedRead(umi1, umi2, coord, strand, seq, qual, rid)


class NormalizationAndStrandIdentityTests(unittest.TestCase):
    def test_reverse_complement_hand_computed(self) -> None:
        from umicorrect import reverse_complement
        self.assertEqual(reverse_complement("A"), "T")
        self.assertEqual(reverse_complement("ACGT"), "ACGT")  # 自互补排列
        self.assertEqual(reverse_complement("AACG"), "CGTT")
        self.assertEqual(reverse_complement(reverse_complement("AACGT")), "AACGT")

    def test_minus_strand_umis_normalized_to_plus_space(self) -> None:
        # '+' 记录键 AC|GT；'-' 记录原始 UMI GT|AC（恰为两端各自 rc），
        # 规范化后键同为 AC|GT，但链身份不同 -> 两个分子，不被合并。
        recs = [
            make("AC", "GT", 100, "+", "ACGT", 30, "p1"),
            make("GT", "AC", 100, "-", "ACGT", 25, "m1"),
        ]
        result = analyze_tagged_reads(recs)
        self.assertEqual((result.nodes_before, result.molecules_after), (2, 2))
        by_strand = {m.strand: m for m in result.molecules}
        self.assertEqual(set(by_strand), {"+", "-"})
        # 两个分子的规范化根键相同（都在正链空间显示 AC|GT）
        self.assertEqual(by_strand["+"].root_umi, "AC|GT")
        self.assertEqual(by_strand["-"].root_umi, "AC|GT")
        # 但 read_indices / record_ids 完全不相交，链身份保留
        self.assertEqual(by_strand["+"].record_ids, ("p1",))
        self.assertEqual(by_strand["-"].record_ids, ("m1",))
        self.assertEqual(
            set(by_strand["+"].read_indices) & set(by_strand["-"].read_indices),
            set(),
        )

    def test_reverse_complement_equivalent_reads_same_strand_still_distinct(self) -> None:
        # 同坐标同正链：键 AC|GT 与“整体反向互补键”GT|AC（自互补发夹型关系），
        # 四个位置全不同（Hamming=4），是两个不同分子，不得因 rc 等价而合并。
        recs = [
            make("AC", "GT", 7, "+", "ACGT", 30, "a"),
            make("GT", "AC", 7, "+", "ACGT", 30, "b"),
        ]
        result = analyze_tagged_reads(recs)
        self.assertEqual(result.molecules_after, 2)
        self.assertEqual({m.root_umi for m in result.molecules}, {"AC|GT", "GT|AC"})

    def test_minus_strand_consensus_reported_in_plus_space_with_reversed_qualities(self) -> None:
        # 手算：'-' 原始 seq=ACG, qual=(7,30,30)。
        # 规范化 seq=rc(ACG)=CGT，规范化 qual=(30,30,7)：低质量落到位置 2。
        recs = [make("AAA", "TTT", 3, "-", "ACG", (7, 30, 30), "x")]
        result = analyze_tagged_reads(recs)
        (mol,) = result.molecules
        self.assertEqual(mol.strand, "-")
        self.assertEqual(mol.consensus_seq, "CGT")
        e7 = 10.0 ** (-7 / 10)
        want_gap2 = LOG10(3.0 * (1.0 - e7) / e7)
        self.assertAlmostEqual(mol.confidence_gap[2], want_gap2, places=10)
        # 位置 0/1 是 Q30，位置 2 是随反转搬来的 Q7 -> gap 显著更小
        self.assertLess(mol.confidence_gap[2], mol.confidence_gap[0])
        self.assertAlmostEqual(mol.confidence_gap[0], LOG10(2997.0), places=10)


class PairingRuleTests(unittest.TestCase):
    def test_one_end_single_difference_absorbed_under_threshold(self) -> None:
        # AC|GT(5) -> AT|GT(2)：umi1 端单碱基差，5 >= 2*2-1=3，吸收。
        recs = [make("AC", "GT", 1, "+", "ACGT", 30, f"a{k}") for k in range(5)]
        recs += [make("AT", "GT", 1, "+", "ACGT", 30, f"b{k}") for k in range(2)]
        result = analyze_tagged_reads(recs)
        self.assertEqual((result.nodes_before, result.molecules_after), (2, 1))
        (mol,) = result.molecules
        self.assertEqual(mol.root_umi, "AC|GT")
        self.assertEqual(mol.members, ("AC|GT", "AT|GT"))
        self.assertEqual(len(mol.read_indices), 7)

    def test_two_end_difference_not_absorbed(self) -> None:
        # AC|GT 与 AT|GA 两端各差一个碱基（组合键 Hamming=2），永不成对。
        recs = [
            make("AC", "GT", 1, "+", "ACGT", 30, "a") for _ in range(20)
        ] + [make("AT", "GA", 1, "+", "ACGT", 30, "b")]
        result = analyze_tagged_reads(recs)
        self.assertEqual(result.molecules_after, 2)

    def test_threshold_boundary_on_dual_keys(self) -> None:
        # s=3 -> 阈值 2s-1=5：c=5 吸收，c=4 不吸收（单端单碱基差）。
        yes = [make("AC", "GT", 1, "+", "AC", 30, f"y{k}") for k in range(5)]
        yes += [make("AT", "GT", 1, "+", "AC", 30, f"ys{k}") for k in range(3)]
        self.assertEqual(analyze_tagged_reads(yes).molecules_after, 1)

        no = [make("AC", "GT", 1, "+", "AC", 30, f"n{k}") for k in range(4)]
        no += [make("AT", "GT", 1, "+", "AC", 30, f"ns{k}") for k in range(3)]
        self.assertEqual(analyze_tagged_reads(no).molecules_after, 2)

    def test_different_coordinates_never_pair(self) -> None:
        recs = [
            make("AC", "GT", 100, "+", "AC", 30, "a") for _ in range(20)
        ] + [make("AC", "GT", 200, "+", "AC", 30, "b") for _ in range(2)]
        result = analyze_tagged_reads(recs)
        # 规范化键完全相同，但坐标不同 -> 两个分子；计数不跨坐标累加。
        self.assertEqual(result.nodes_before, 2)
        self.assertEqual(result.molecules_after, 2)
        coords = sorted(m.coordinate for m in result.molecules)
        self.assertEqual(coords, [100, 200])

    def test_opposite_strands_never_pair_even_with_abundance_gap(self) -> None:
        # '+' AC|GT 20 条，'-' 规范化后也是 AC|GT 的 1 条：跨链无有向边，
        # 反链分子不被正链大根吸收。
        recs = [make("AC", "GT", 9, "+", "AC", 30, f"p{k}") for k in range(20)]
        recs.append(make("GT", "AC", 9, "-", "AC", 30, "m0"))
        result = analyze_tagged_reads(recs)
        self.assertEqual(result.molecules_after, 2)
        minus = [m for m in result.molecules if m.strand == "-"]
        self.assertEqual(len(minus), 1)
        self.assertEqual(minus[0].members, ("AC|GT",))
        self.assertEqual(minus[0].read_indices, (20,))

    def test_minus_strand_cluster_pairs_and_consensus_matches_reference(self) -> None:
        # 全反链：大根原始 UMI GT|AC（规范化 AC|GT）10 条；错误 UMI GT|AT
        # （规范化 AC|AT，与根在 umi2 端差 1）4 条。必须在 '-' 分区内成对，
        # 不与任何正链记录混合；共识与独立参考（独立规范化 + 逐读段似然）一致。
        recs = [make("GT", "AC", 5, "-", "ACGT", 30, f"m-{k}") for k in range(10)]
        recs += [make("GT", "AT", 5, "-", "ACGT", (10, 30, 30, 30), f"e-{k}")
                 for k in range(4)]
        result = analyze_tagged_reads(recs)
        self.assertEqual((result.nodes_before, result.molecules_after), (2, 1))
        (mol,) = result.molecules
        self.assertEqual(mol.strand, "-")
        self.assertEqual(mol.members, ("AC|AT", "AC|GT"))
        self.assertEqual(len(mol.read_indices), 14)

        # 独立期望：测试内规范化构造视图，调用独立似然参考
        view, idxs = [], []
        for i, r in enumerate(recs):
            _, _, nseq, nqual = independent_normalize(r)
            view.append(type("V", (), {"sequence": nseq, "qualities": nqual})())
            idxs.append(i)
        exp_seq, exp_gaps = reference_consensus(view, 4, idxs)
        self.assertEqual(mol.consensus_seq, exp_seq)
        for got, want in zip(mol.confidence_gap, exp_gaps):
            self.assertAlmostEqual(got, want, places=12)

    def test_pairing_matches_independent_reference(self) -> None:
        # 手算规范化后直接用独立参考的 Floyd 闭包核对成员划分。
        recs = [
            make("AC", "GT", 1, "+", "AC", 30, f"r1-{k}") for k in range(10)
        ]
        recs += [make("AT", "GT", 1, "+", "AC", 30, f"r2-{k}") for k in range(4)]
        recs += [make("AT", "GA", 1, "+", "AC", 30, f"r3-{k}") for k in range(2)]
        recs += [make("GG", "GG", 1, "+", "GG", 5, "r4-0")]
        result = analyze_tagged_reads(recs)

        norm_counts: dict[str, int] = {}
        for r in recs:
            n1, n2, _, _ = independent_normalize(r)
            key = n1 + "|" + n2
            norm_counts[key] = norm_counts.get(key, 0) + 1
        want_groups = reference_clusters(norm_counts)
        got_groups = sorted(sorted(m.members) for m in result.molecules)
        self.assertEqual(got_groups, sorted(sorted(g) for g in want_groups))


class DuplicateByIdTests(unittest.TestCase):
    def test_identical_resubmission_collapsed_keeping_first_index(self) -> None:
        recs = [
            make("AC", "GT", 1, "+", "AC", 30, "X"),
            make("AC", "GT", 1, "+", "AC", 30, "X"),  # 完全一致的重复提交
            make("AT", "GT", 1, "+", "AC", 30, "Y"),
        ]
        result = analyze_tagged_reads(recs)
        self.assertEqual(result.duplicates_removed, 1)
        # 计数按去重后记录：AC|GT=1（而非 2），AT|GT=1；两 count=1 且
        # Hamming=1 -> 1>=1 成立，合成一簇；成员下标只保留首次出现的 0。
        self.assertEqual(result.molecules_after, 1)
        (mol,) = result.molecules
        self.assertEqual(mol.read_indices, (0, 2))
        self.assertEqual(mol.record_ids, ("X", "Y"))

    def test_conflicting_duplicate_raises_with_locations_and_field(self) -> None:
        recs = [
            make("AC", "GT", 1, "+", "AC", 30, "X"),
            make("AC", "GG", 1, "+", "AC", 30, "X"),  # umi2 冲突
        ]
        with self.assertRaises(DuplicateRecordError) as ctx:
            analyze_tagged_reads(recs)
        msg = str(ctx.exception)
        self.assertIn("records[1]", msg)
        self.assertIn("records[0]", msg)
        self.assertIn("'X'", msg)
        self.assertIn("umi2", msg)

    def test_conflict_in_each_tracked_field_rejected(self) -> None:
        base = dict(umi1="AC", umi2="GT", coordinate=1, strand="+",
                    seq="AC", q=30, rid="Z")

        def build(**over):
            kw = dict(base)
            kw.update(over)
            return make(kw["umi1"], kw["umi2"], kw["coordinate"], kw["strand"],
                        kw["seq"], kw["q"], kw["rid"])

        variants = [
            dict(umi1="AT"),
            dict(coordinate=2),
            dict(strand="-"),
            dict(seq="AG"),
            dict(q=20),
        ]
        for over in variants:
            with self.subTest(over=over):
                recs = [build(), build(**over)]
                with self.assertRaises(DuplicateRecordError):
                    analyze_tagged_reads(recs)

    def test_duplicate_detection_is_atomic_and_input_unchanged(self) -> None:
        recs = [
            make("AC", "GT", 1, "+", "AC", 30, "X"),
            make("AC", "GT", 1, "+", "AC", 30, "X"),
            make("AT", "GT", 1, "+", "AC", 30, "X"),  # 同 ID 冲突
        ]
        before = len(recs)
        with self.assertRaises(DuplicateRecordError):
            analyze_tagged_reads(recs)
        self.assertEqual(len(recs), before)  # 输入未被修改
        # 修正为不同 ID 后同批可正常分析（失败不留半更新状态）
        recs[2] = make("AT", "GT", 1, "+", "AC", 30, "W")
        ok = analyze_tagged_reads(recs)
        self.assertEqual(ok.duplicates_removed, 1)
        self.assertEqual(ok.molecules_after, 1)


class TaggedValidationTests(unittest.TestCase):
    def test_empty_batch_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "空批次"):
            analyze_tagged_reads([])

    def test_not_a_tagged_read_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidReadError, r"records\[0\]"):
            analyze_tagged_reads([("AC", "GT", 1, "+", "AC", (30, 30), "z")])

    def test_non_list_batch_rejected(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            analyze_tagged_reads((r for r in ()))

    def test_unequal_dual_umi_lengths_rejected_at_construction(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "双端 UMI 长度"):
            TaggedRead("AC", "GTT", 1, "+", "AC", (30, 30), "z")

    def test_illegal_base_and_empty_id_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, r"umi1\[1\]"):
            TaggedRead("AN", "GT", 1, "+", "AC", (30, 30), "z")
        with self.assertRaisesRegex(InvalidArgumentError, "record_id"):
            TaggedRead("AC", "GT", 1, "+", "AC", (30, 30), "")

    def test_bad_strand_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "strand"):
            TaggedRead("AC", "GT", 1, ".", "AC", (30, 30), "z")

    def test_coordinate_validation(self) -> None:
        TaggedRead("AC", "GT", 0, "+", "AC", (30, 30), "z")  # 边界 0 合法
        with self.assertRaisesRegex(InvalidArgumentError, "coordinate"):
            TaggedRead("AC", "GT", -1, "+", "AC", (30, 30), "z")
        with self.assertRaisesRegex(InvalidArgumentError, "coordinate"):
            TaggedRead("AC", "GT", 2_147_483_648, "+", "AC", (30, 30), "z")
        with self.assertRaisesRegex(InvalidArgumentError, "bool"):
            TaggedRead("AC", "GT", True, "+", "AC", (30, 30), "z")

    def test_quality_and_cross_record_length(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, r"qualities\[0\]"):
            TaggedRead("AC", "GT", 1, "+", "AC", (41, 30), "z")
        good = TaggedRead("AC", "GT", 1, "+", "AC", (30, 30), "z")
        bad = TaggedRead("ACG", "GTA", 1, "+", "ACG", (30, 30, 30), "y")
        with self.assertRaisesRegex(InvalidReadError, r"records\[1\]"):
            analyze_tagged_reads([good, bad])

    def test_umi_length_limit_separate_exception(self) -> None:
        long_u = "A" * 33
        with self.assertRaises(LimitExceededError):
            TaggedRead(long_u, long_u, 1, "+", "AC", (30, 30), "z")

    def test_budget_validation(self) -> None:
        recs = [
            make("AC", "GT", 1, "+", "AC", 30, "a"),
            make("AT", "GT", 1, "+", "AC", 30, "b"),
        ]
        with self.assertRaisesRegex(InvalidArgumentError, "整数"):
            analyze_tagged_reads(recs, probe_budget=2.0)
        with self.assertRaises(InvalidArgumentError):
            analyze_tagged_reads(recs, probe_budget=True)
        with self.assertRaisesRegex(InvalidArgumentError, "NaN/Infinity"):
            analyze_tagged_reads(recs, probe_budget=float("nan"))


class BudgetAcrossPartitionsTests(unittest.TestCase):
    def test_probe_budget_shared_and_exhaustion_recoverable(self) -> None:
        # 两个分区，每个 2 节点需 1 次探测；预算 1 在第二个分区耗尽。
        recs = [
            make("AC", "GT", 1, "+", "AC", 30, "a1"),
            make("AT", "GT", 1, "+", "AC", 30, "a2"),
            make("AC", "GT", 2, "+", "AC", 30, "b1"),
            make("AT", "GT", 2, "+", "AC", 30, "b2"),
        ]
        with self.assertRaises(SearchBudgetExhaustedError) as ctx:
            analyze_tagged_reads(recs, probe_budget=1)
        self.assertIsInstance(ctx.exception, RuntimeError)
        self.assertNotIsInstance(ctx.exception, ValueError)
        self.assertEqual(ctx.exception.budget, 1)
        self.assertGreaterEqual(ctx.exception.probes_used, 2)
        # 预算耗尽不返回截断结果；放大预算后同一输入成功。
        ok = analyze_tagged_reads(recs, probe_budget=10)
        self.assertEqual(ok.molecules_after, 2)


class DeterminismAndRandomReferenceTests(unittest.TestCase):
    def test_shuffle_input_gives_identical_molecules(self) -> None:
        rng = random.Random(123)
        recs = [
            make("AC", "GT", 1, "+", "ACGT", 30, f"a{k}") for k in range(8)
        ]
        recs += [make("AT", "GT", 1, "+", "ACGT", 12, f"b{k}") for k in range(3)]
        recs += [make("GG", "CC", 1, "-", "ACGT", 25, f"c{k}") for k in range(4)]
        r1 = analyze_tagged_reads(recs)
        shuffled = list(recs)
        rng.shuffle(shuffled)
        r2 = analyze_tagged_reads(shuffled)
        key = lambda mols: [  # noqa: E731
            (m.coordinate, m.strand, m.root_umi, m.members,
             tuple(sorted(m.record_ids)), m.consensus_seq)
            for m in mols
        ]
        self.assertEqual(key(r1.molecules), key(r2.molecules))

    def test_random_batches_against_independent_reference(self) -> None:
        rng = random.Random(20260924)
        for trial in range(40):
            ulen = rng.randint(1, 3)
            rlen = rng.randint(1, 4)
            coords = [rng.randint(0, 3) for _ in range(2)]

            def rand_umi():
                return "".join(rng.choice("ACGT") for _ in range(ulen))

            recs = []
            rid = 0
            n_entities = rng.randint(1, 7)
            for _ in range(n_entities):
                u1, u2 = rand_umi(), rand_umi()
                coord = rng.choice(coords)
                strand = rng.choice("+-")
                reps = rng.randint(1, 5)
                for _ in range(reps):
                    seq = "".join(rng.choice("ACGT") for _ in range(rlen))
                    qual = tuple(rng.randint(0, 40) for _ in range(rlen))
                    recs.append(TaggedRead(u1, u2, coord, strand, seq, qual, f"r{rid}"))
                    rid += 1

            result = analyze_tagged_reads(recs)

            # ---- 独立期望：手工规范化 -> (coord,strand) 分组 -> 独立参考 ----
            partitions: dict[tuple[int, str], list] = {}
            for r in recs:
                n1, n2, nseq, nqual = independent_normalize(r)
                partitions.setdefault((r.coordinate, r.strand), []).append(
                    (r.record_id, n1 + "|" + n2, nseq, nqual)
                )

            total_nodes = 0
            expect_mols = []
            for (coord, strand) in sorted(partitions):
                entries = partitions[(coord, strand)]
                counts: dict[str, int] = {}
                for _, key, _, _ in entries:
                    counts[key] = counts.get(key, 0) + 1
                total_nodes += len(counts)
                groups = reference_clusters(counts)
                for group in groups:
                    root = min(group, key=lambda u: (-counts[u], u))
                    idxs = [i for i, e in enumerate(entries) if e[1] in set(group)]
                    view = [type("V", (), {"sequence": e[2], "qualities": e[3]})()
                            for e in entries]
                    exp_seq, exp_gaps = reference_consensus(view, rlen, idxs)
                    expect_mols.append((coord, strand, root,
                                        tuple(sorted(group)), exp_seq, exp_gaps,
                                        sorted(e[0] for i, e in enumerate(entries)
                                               if i in idxs)))

            self.assertEqual(result.nodes_before, total_nodes)
            self.assertEqual(result.molecules_after, len(expect_mols))
            got = sorted(
                (m.coordinate, m.strand, m.root_umi, m.members, m.consensus_seq)
                for m in result.molecules
            )
            want = sorted((e[0], e[1], e[2], e[3], e[4]) for e in expect_mols)
            self.assertEqual(got, want)
            # gap 与独立参考逐项核对
            gap_want = {
                (e[0], e[1], e[2]): e[5] for e in expect_mols
            }
            for m in result.molecules:
                for g, w in zip(m.confidence_gap,
                                gap_want[(m.coordinate, m.strand, m.root_umi)]):
                    self.assertAlmostEqual(g, w, places=12)


if __name__ == "__main__":
    unittest.main()
