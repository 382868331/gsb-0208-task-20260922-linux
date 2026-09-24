"""双端 UMI / 坐标 / 链规范化与显式成对的测试。

期望值来源（不调用被测实现生成期望）：
* 反向互补、规范化键、成对归属均为手算固定用例，并用测试内**独立书写**
  的 :func:`indep_rc` / :func:`indep_key` 复核（不导入
  normalize_record / reverse_complement）；
* 共识期望值复用 tests/reference.py 的独立似然参考
  :func:`reference_consensus`，输入序列由测试内独立规范化得到；
* 另在一个小规模确定性随机实例上比较“整批划分”。
旧测试（test_likelihood / test_clustering / test_pipeline）保持不变。
"""

from __future__ import annotations

import math
import random
import unittest

from reference import reference_consensus

from umicorrect import (
    InvalidArgumentError,
    InvalidReadError,
    LimitExceededError,
    PairedRecord,
    RecordConflictError,
    UMICorrectError,
    analyze_paired_records,
    group_paired_records,
    normalize_record,
    reverse_complement,
)


# ---------------------------------------------------------------------------
# 测试内独立参考：与 umicorrect 实现无关地手写反向互补与匹配键
# ---------------------------------------------------------------------------
_COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def indep_rc(s: str) -> str:
    return "".join(_COMP[ch] for ch in reversed(s))


def indep_normalized(rec: PairedRecord) -> tuple:
    """独立计算规范化键 (contig, lo, hi, strand, ul, ur)。"""
    lo, hi = (rec.start, rec.end) if rec.start < rec.end else (rec.end, rec.start)
    if rec.strand == "+":
        ul, ur = rec.umi1, rec.umi2
    else:
        ul, ur = indep_rc(rec.umi2), indep_rc(rec.umi1)
    return rec.contig, lo, hi, rec.strand, ul, ur


def indep_norm_seq(rec: PairedRecord) -> tuple[str, tuple[int, ...]]:
    """独立规范化序列/质量到参考方向。"""
    if rec.strand == "+":
        return rec.sequence, tuple(rec.qualities)
    return indep_rc(rec.sequence), tuple(reversed(rec.qualities))


def mk(rid, u1, u2, contig="chr1", start=100, end=104, strand="+",
       seq="ACGT", qual=(30, 30, 30, 30)):
    return PairedRecord(rid, u1, u2, contig, start, end, strand, seq, qual)


def groups_by_key(result) -> dict[tuple, tuple]:
    return {m.key: m.record_ids for m in result.molecules}


class ReverseComplementHandTests(unittest.TestCase):
    def test_hand_computed_rc(self) -> None:
        self.assertEqual(reverse_complement("AAA"), "TTT")
        self.assertEqual(reverse_complement("ACGT"), "ACGT")      # 回文
        self.assertEqual(reverse_complement("AAC"), "GTT")
        self.assertEqual(reverse_complement("AATTCC"), "GGAATT")

    def test_rc_involution(self) -> None:
        # 小规模枚举全部 4^1..4^4 串验证 rc(rc(s)) == s、长度不变
        alphabet = "ACGT"
        for length in range(1, 5):
            def gen(prefix, n):
                if n == 0:
                    yield prefix
                    return
                for ch in alphabet:
                    yield from gen(prefix + ch, n - 1)
            for s in gen("", length):
                self.assertEqual(reverse_complement(reverse_complement(s)), s)

    def test_rc_rejects_bad_input(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            reverse_complement("")
        with self.assertRaises(InvalidArgumentError):
            reverse_complement("AN")
        with self.assertRaises(InvalidArgumentError):
            reverse_complement("ac")
        with self.assertRaises(InvalidArgumentError):
            reverse_complement(123)


class NormalizationHandTests(unittest.TestCase):
    def test_plus_strand_keeps_order(self) -> None:
        nr = normalize_record(mk("r1", "AAA", "AAC"))
        self.assertEqual((nr.umi_left, nr.umi_right), ("AAA", "AAC"))
        self.assertEqual(nr.strand, "+")
        self.assertEqual(nr.sequence, "ACGT")
        self.assertEqual(nr.contig, "chr1")
        self.assertEqual((nr.start, nr.end), (100, 104))

    def test_minus_strand_swaps_and_rc_each_umi(self) -> None:
        # 手算：raw u1=GTT=rc(AAC), u2=TTT=rc(AAA)
        # 规范化后必须与对应 + 链记录 (AAA, AAC) 的方向化 UMI 相同。
        nr = normalize_record(mk("r2", "GTT", "TTT", strand="-"))
        self.assertEqual((nr.umi_left, nr.umi_right), ("AAA", "AAC"))
        # 链身份原样保留，方向化不改变链
        self.assertEqual(nr.strand, "-")

    def test_minus_strand_sequence_and_qualities_are_reference_oriented(self) -> None:
        rec = mk("r3", "GTT", "TTT", strand="-", seq="AACC",
                 qual=(10, 20, 30, 40))
        nr = normalize_record(rec)
        # rc("AACC") = "GGTT"；质量随位置反转到对应碱基
        self.assertEqual(nr.sequence, "GGTT")
        self.assertEqual(nr.qualities, (40, 30, 20, 10))

    def test_coordinates_sorted_when_given_in_reverse(self) -> None:
        nr = normalize_record(mk("r4", "AAA", "AAC", start=200, end=150))
        self.assertEqual((nr.start, nr.end), (150, 200))


class PairingHandTests(unittest.TestCase):
    def test_duplicate_observations_pair_into_one_molecule(self) -> None:
        records = [
            mk("r1", "AAA", "AAC", seq="ACGT", qual=(30, 30, 30, 30)),
            mk("r2", "AAA", "AAC", seq="ACGA", qual=(30, 30, 30, 7)),
        ]
        result = analyze_paired_records(records)
        self.assertEqual(result.records_in, 2)
        self.assertEqual(result.molecules_after, 1)
        self.assertEqual(result.duplicate_count, 1)
        (mol,) = result.molecules
        self.assertEqual(mol.record_ids, ("r1", "r2"))
        self.assertEqual(mol.record_indices, (0, 1))
        self.assertEqual((mol.umi_left, mol.umi_right), ("AAA", "AAC"))
        self.assertEqual(mol.strand, "+")
        self.assertEqual((mol.start, mol.end), (100, 104))
        # 共识与独立参考一致（低质量错误读不影响前 3 位）
        ref_seq, ref_gaps = reference_consensus(
            [type("R", (), {"sequence": s, "qualities": q})()
             for s, q in [("ACGT", (30, 30, 30, 30)),
                          ("ACGA", (30, 30, 30, 7))]],
            4, [0, 1],
        )
        self.assertEqual(mol.consensus_seq, ref_seq)
        self.assertEqual(mol.consensus_seq, "ACGT")
        for got, want in zip(mol.confidence_gap, ref_gaps):
            self.assertAlmostEqual(got, want, places=12)

    def test_rc_equal_but_opposite_strand_are_two_molecules(self) -> None:
        # 核心场景：一条 + 记录与一条 - 记录，方向化后双 UMI 字面相同、
        # 坐标相同，但链不同——它们是两个不同分子，绝不跨链合并。
        # 方向化 UMI 均为 (AAA, AAC)：
        #   + raw (AAA, AAC)；- raw (rc(AAC), rc(AAA)) = (GTT, TTT)
        records = [
            mk("plus", "AAA", "AAC", seq="ACGT", strand="+"),
            mk("minus", "GTT", "TTT", seq=indep_rc("ACGT"), strand="-"),
        ]
        result = analyze_paired_records(records)
        self.assertEqual(result.molecules_after, 2)
        keys = [m.key for m in result.molecules]
        self.assertEqual({k for k in keys}, {
            ("chr1", 100, 104, "+", "AAA", "AAC"),
            ("chr1", 100, 104, "-", "AAA", "AAC"),
        })
        # 两条记录分属不同分子，各只有 1 条
        for mol in result.molecules:
            self.assertEqual(len(mol.record_ids), 1)
        # - 链分子的共识在参考方向给出
        minus_mol = next(m for m in result.molecules if m.strand == "-")
        self.assertEqual(minus_mol.consensus_seq, "ACGT")

    def test_palindromic_umi_pair_rc_equal_still_separated_by_strand(self) -> None:
        # 更强的边界：双 UMI 本身满足 rc(u2)=u1、rc(u1)=u2（AA/TT），
        # 原始串与方向化串都相同，唯一区别是链——仍是两个分子。
        records = [
            mk("p", "AA", "TT", seq="ACGT", strand="+"),
            mk("m", "AA", "TT", seq="ACGT", strand="-"),
        ]
        result = analyze_paired_records(records)
        self.assertEqual(result.molecules_after, 2)
        self.assertEqual(result.duplicate_count, 0)

    def test_same_strand_duplicates_pair_minus_strand(self) -> None:
        # 两条 - 链重复观测（raw UMI 相同）应成对，且共识在参考方向。
        records = [
            mk("m1", "GTT", "TTT", strand="-", seq="AACC", qual=(30, 30, 30, 30)),
            mk("m2", "GTT", "TTT", strand="-", seq="AACT", qual=(30, 30, 30, 10)),
        ]
        result = analyze_paired_records(records)
        self.assertEqual(result.molecules_after, 1)
        mol = result.molecules[0]
        self.assertEqual(mol.strand, "-")
        self.assertEqual((mol.umi_left, mol.umi_right), ("AAA", "AAC"))
        # 独立参考：先在测试内方向化再喂给独立似然实现
        norm = [indep_norm_seq(r) for r in records]
        ref = [type("R", (), {"sequence": s, "qualities": q})()
               for s, q in norm]
        ref_seq, ref_gaps = reference_consensus(ref, 4, [0, 1])
        self.assertEqual(mol.consensus_seq, ref_seq)
        self.assertEqual(mol.consensus_seq, "GGTT")
        for got, want in zip(mol.confidence_gap, ref_gaps):
            self.assertAlmostEqual(got, want, places=12)

    def test_coordinate_reversal_still_pairs(self) -> None:
        # 同一区间一条按 (100,104) 给、一条反向按 (104,100) 给，同链同 UMI
        # -> 规范化后同键成对。
        records = [
            mk("a", "AAA", "AAC", start=100, end=104),
            mk("b", "AAA", "AAC", start=104, end=100),
        ]
        result = analyze_paired_records(records)
        self.assertEqual(result.molecules_after, 1)
        self.assertEqual(result.molecules[0].key[:3], ("chr1", 100, 104))

    def test_exact_match_only_nearby_umis_do_not_collapse(self) -> None:
        # 显式匹配规则：UMI 差一个碱基、坐标差 1、contig 不同都不成对
        records = [mk("base", "AAA", "AAC")]
        variants = [
            ("u", "AAT", "AAC", "chr1", 100, 104, "+"),  # UMI Hamming=1
            ("c", "AAA", "AAC", "chr2", 100, 104, "+"),  # contig 不同
            ("p", "AAA", "AAC", "chr1", 101, 104, "+"),  # 坐标差 1
        ]
        for rid, u1, u2, ct, s, e, st in variants:
            records.append(mk(rid, u1, u2, contig=ct, start=s, end=e,
                              strand=st, seq="ACGT"))
        result = analyze_paired_records(records)
        self.assertEqual(result.molecules_after, 4)

    def test_molecule_and_record_ordering_is_deterministic(self) -> None:
        records = [
            mk("z", "TTT", "TTT", contig="chr2"),
            mk("a", "AAA", "AAA", contig="chr1"),
            mk("a2", "AAA", "AAA", contig="chr1", seq="ACGA",
               qual=(30, 30, 30, 5)),
        ]
        r1 = analyze_paired_records(records)
        r2 = analyze_paired_records(list(reversed(records)))
        self.assertEqual(
            [m.key for m in r1.molecules],
            [m.key for m in r2.molecules],
        )
        # 组内按下标升序：重排后 chr1 组内顺序随输入位置变化但按键稳定
        chr1_r1 = next(m for m in r1.molecules if m.contig == "chr1")
        chr1_r2 = next(m for m in r2.molecules if m.contig == "chr1")
        self.assertEqual(chr1_r1.record_ids, ("a", "a2"))
        self.assertEqual(chr1_r2.record_ids, ("a2", "a"))
        # 共识碱基不受求和次序影响；gap 仅允许 ulp 级差
        self.assertEqual(chr1_r1.consensus_seq, chr1_r2.consensus_seq)
        for g1, g2 in zip(chr1_r1.confidence_gap, chr1_r2.confidence_gap):
            self.assertAlmostEqual(g1, g2, places=12)

    def test_grouping_only_entry_leaves_consensus_empty(self) -> None:
        mols = group_paired_records([mk("r1", "AAA", "AAC"), mk("r2", "AAA", "AAC")])
        self.assertEqual(len(mols), 1)
        self.assertEqual(mols[0].consensus_seq, "")
        self.assertEqual(mols[0].confidence_gap, ())


class PairingConsensusBehaviorTests(unittest.TestCase):
    def test_exact_tie_in_group_outputs_N(self) -> None:
        # 同分子两条记录，位置 0 等质量呼 A/C -> 精确并列报 N（旧共识口径）
        records = [
            mk("r1", "AAA", "AAC", seq="ACGT", qual=(20, 30, 30, 30)),
            mk("r2", "AAA", "AAC", seq="CCGT", qual=(20, 30, 30, 30)),
        ]
        mol = analyze_paired_records(records).molecules[0]
        self.assertEqual(mol.consensus_seq, "NCGT")
        self.assertEqual(mol.confidence_gap[0], 0.0)

    def test_high_quality_single_read_overrides_low_quality_duplicates(self) -> None:
        records = [
            mk("r1", "AAA", "AAC", seq="ACGT", qual=(3, 30, 30, 30)),
            mk("r2", "AAA", "AAC", seq="ACGT", qual=(3, 30, 30, 30)),
            mk("r3", "AAA", "AAC", seq="CCGT", qual=(40, 30, 30, 30)),
        ]
        mol = analyze_paired_records(records).molecules[0]
        self.assertEqual(mol.consensus_seq[0], "C")
        self.assertGreater(mol.confidence_gap[0], 3.0)


class SmallRandomPartitionTests(unittest.TestCase):
    def test_random_batch_partition_matches_independent_key(self) -> None:
        rng = random.Random(20260924)
        umis = ["".join(rng.choice("ACGT") for _ in range(3)) for _ in range(8)]
        contigs = ["chr1", "chr2"]
        records = []
        rid_counter = 0
        for _ in range(60):
            u1 = rng.choice(umis)
            u2 = rng.choice(umis)
            # 约一半记录复用“热门键”以制造分子级重复
            if records and rng.random() < 0.5:
                base = rng.choice(records)
                u1, u2 = base.umi1, base.umi2
                ct, s, e, st = base.contig, base.start, base.end, base.strand
            else:
                ct = rng.choice(contigs)
                s = rng.randint(0, 50)
                e = s + rng.randint(1, 6)
                st = rng.choice("+-")
            seq = "".join(rng.choice("ACGT") for _ in range(4))
            qual = tuple(rng.randint(0, 40) for _ in range(4))
            rid_counter += 1
            records.append(PairedRecord(
                f"id{rid_counter}", u1, u2, ct, s, e, st, seq, qual,
            ))

        result = analyze_paired_records(records)

        # 独立期望划分：测试内自己算键
        expect: dict[tuple, list[str]] = {}
        for rec in records:
            expect.setdefault(indep_normalized(rec), []).append(rec.record_id)
        self.assertEqual(
            {m.key: list(m.record_ids) for m in result.molecules},
            {k: v for k, v in sorted(expect.items())},
        )

        # 每组共识对独立似然参考
        norm_records = []
        key_of = []
        for i, rec in enumerate(records):
            s2, q2 = indep_norm_seq(rec)
            norm_records.append(
                type("R", (), {"sequence": s2, "qualities": q2})()
            )
            key_of.append(indep_normalized(rec))
        for mol in result.molecules:
            idxs = [i for i, k in enumerate(key_of) if k == mol.key]
            ref_seq, ref_gaps = reference_consensus(norm_records, 4, idxs)
            self.assertEqual(mol.consensus_seq, ref_seq)
            for got, want in zip(mol.confidence_gap, ref_gaps):
                # gap 恒为有限值（并列位置为 0.0）
                self.assertTrue(math.isfinite(got))
                self.assertAlmostEqual(got, want, places=12)


class RecordConflictTests(unittest.TestCase):
    def test_duplicate_record_id_rejected_with_locations(self) -> None:
        records = [
            mk("same-id", "AAA", "AAC"),
            mk("other", "TTT", "GGG"),
            mk("same-id", "AAA", "AAC"),  # 与 records[0] 同 ID
        ]
        with self.assertRaisesRegex(RecordConflictError, r"records\[2\].*records\[0\]"):
            analyze_paired_records(records)
        with self.assertRaises(RecordConflictError):
            group_paired_records(records)

    def test_conflict_is_invalid_argument_and_caught_by_base(self) -> None:
        records = [mk("x", "AAA", "AAC"), mk("x", "AAA", "AAC")]
        with self.assertRaises(InvalidArgumentError):
            analyze_paired_records(records)
        with self.assertRaises(UMICorrectError):
            analyze_paired_records(records)

    def test_failure_is_atomic_no_partial_state(self) -> None:
        # 同 ID 即使内容完全一致也拒绝（库无法判断重发还是误用同号）；
        # 失败后修正 ID 的同批数据可正常成对。
        dup = [mk("r1", "AAA", "AAC"), mk("r1", "AAA", "AAC")]
        with self.assertRaises(RecordConflictError):
            analyze_paired_records(dup)
        fixed = [mk("r1", "AAA", "AAC"), mk("r2", "AAA", "AAC")]
        result = analyze_paired_records(fixed)
        self.assertEqual(result.molecules_after, 1)
        self.assertEqual(result.molecules[0].record_ids, ("r1", "r2"))

    def test_same_key_length_mismatch_not_silently_merged(self) -> None:
        # 匹配键相同但规范化后序列长度不一致：不能截断、不能静默合并。
        records = [
            mk("r1", "AAA", "AAC", seq="ACGT", qual=(30, 30, 30, 30)),
            mk("r2", "AAA", "AAC", seq="ACG", qual=(30, 30, 30)),
        ]
        with self.assertRaisesRegex(InvalidArgumentError, "长度"):
            analyze_paired_records(records)
        with self.assertRaises(InvalidArgumentError):
            group_paired_records(records)


class PairedValidationTests(unittest.TestCase):
    def test_empty_batch_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "空批次"):
            analyze_paired_records([])

    def test_not_a_record_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidReadError, r"records\[0\]"):
            analyze_paired_records([("x", "AAA", "AAC")])

    def test_batch_path_adds_records_prefix_even_if_constructor_bypassed(self) -> None:
        # 与单端 test_batch_path_adds_reads_prefix 对称：绕过构造器安放
        # 非法字段时，批级复检仍拒绝并带 records[0] 定位前缀。
        obj = object.__new__(PairedRecord)
        object.__setattr__(obj, "record_id", "r")
        object.__setattr__(obj, "umi1", "AAA")
        object.__setattr__(obj, "umi2", "AAC")
        object.__setattr__(obj, "contig", "c")
        object.__setattr__(obj, "start", 0)
        object.__setattr__(obj, "end", 4)
        object.__setattr__(obj, "strand", "+")
        object.__setattr__(obj, "sequence", "NCGT")
        object.__setattr__(obj, "qualities", (30, 30, 30, 30))
        with self.assertRaisesRegex(InvalidReadError, r"records\[0\].*sequence\[0\]"):
            analyze_paired_records([obj])

    def test_non_list_batch_rejected(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            analyze_paired_records(iter([mk("r1", "AAA", "AAC")]))

    def test_record_id_rules(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "record_id"):
            PairedRecord("", "AAA", "AAC", "c", 0, 4, "+", "ACGT", (30,) * 4)
        with self.assertRaisesRegex(InvalidArgumentError, "record_id"):
            PairedRecord(7, "AAA", "AAC", "c", 0, 4, "+", "ACGT", (30,) * 4)

    def test_umi_rules_with_positions(self) -> None:
        kw = dict(contig="c", start=0, end=4, strand="+",
                  sequence="ACGT", qualities=(30, 30, 30, 30))
        with self.assertRaisesRegex(InvalidArgumentError, r"umi1\[1\]"):
            PairedRecord("r", "ANA", "AAC", **kw)
        with self.assertRaisesRegex(InvalidArgumentError, "umi2 为空"):
            PairedRecord("r", "AAA", "", **kw)
        with self.assertRaises(LimitExceededError):
            PairedRecord("r", "A" * 33, "AAC", **kw)

    def test_coordinate_rules(self) -> None:
        kw = dict(umi1="AAA", umi2="AAC", contig="c",
                  sequence="ACGT", qualities=(30, 30, 30, 30))
        with self.assertRaisesRegex(InvalidArgumentError, "非负"):
            PairedRecord("r", start=-1, end=4, strand="+", **kw)
        with self.assertRaisesRegex(InvalidArgumentError, "空区间"):
            PairedRecord("r", start=5, end=5, strand="+", **kw)
        with self.assertRaisesRegex(InvalidArgumentError, "整数"):
            PairedRecord("r", start=1.0, end=4, strand="+", **kw)
        with self.assertRaisesRegex(InvalidArgumentError, "bool"):
            PairedRecord("r", start=True, end=4, strand="+", **kw)
        with self.assertRaisesRegex(InvalidArgumentError, "NaN/Infinity"):
            PairedRecord("r", start=float("nan"), end=4, strand="+", **kw)

    def test_strand_rule(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "strand"):
            PairedRecord("r", "AAA", "AAC", "c", 0, 4, "1",
                         "ACGT", (30, 30, 30, 30))

    def test_sequence_quality_rules(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, r"sequence\[2\]"):
            PairedRecord("r", "AAA", "AAC", "c", 0, 4, "+",
                         "ACNT", (30, 30, 30, 30))
        with self.assertRaisesRegex(InvalidArgumentError, "qualities 长度"):
            PairedRecord("r", "AAA", "AAC", "c", 0, 4, "+",
                         "ACGT", (30, 30))
        with self.assertRaisesRegex(InvalidArgumentError, r"qualities\[0\]"):
            PairedRecord("r", "AAA", "AAC", "c", 0, 4, "+",
                         "ACGT", (41, 30, 30, 30))
        with self.assertRaisesRegex(InvalidArgumentError, "bool"):
            PairedRecord("r", "AAA", "AAC", "c", 0, 4, "+",
                         "ACGT", (True, 30, 30, 30))

    def test_dual_umis_need_not_equal_sequence_length(self) -> None:
        # 双 UMI 是接头结构，与插入序列长度解耦，且两端可不等长。
        rec = PairedRecord("r", "A", "AAC", "c", 0, 9, "+",
                           "ACGTACGT", tuple([30] * 8))
        nr = normalize_record(rec)
        self.assertEqual((nr.umi_left, nr.umi_right), ("A", "AAC"))
        self.assertEqual(len(nr.sequence), 8)


if __name__ == "__main__":
    unittest.main()
