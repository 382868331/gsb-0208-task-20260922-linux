"""编辑距离吸收模型测试（本轮新增可选行为）。

期望值来源：
* 手算路径概率（同聚物等价对齐计数、替换/插入/删除因子）；
* tests.reference.reference_edit_path_probability —— 递归枚举全部
  对齐路径的独立参考（不导入被测核心）；
* tests.reference.reference_clusters_edit —— 全对枚举候选 +
  Floyd-Warshall 归属的独立聚类参考；
* 固定场景：长度差 1 吸收、两个高丰度候选并列保留未归并、歧义阈值两侧、
  输入置换不变性、默认模式回归（原 Hamming 行为不变）。
"""

from __future__ import annotations

import itertools
import math
import random
import unittest

from reference import (
    reference_clusters_edit,
    reference_edit_path_probability,
)

from umicorrect import (
    InvalidArgumentError,
    InvalidReadError,
    Read,
    analyze_reads,
    cluster_umis,
    edit_path_probability,
)

E3 = 0.001  # Q30 的错误率


def uniform_profile(umi, e=E3):
    return (e,) * len(umi)


def profiles_for(umis, e=E3):
    return {u: uniform_profile(u, e) for u in umis}


def partition_of(clusters):
    return {cl.root: frozenset(cl.members) for cl in clusters}


class EditPathProbabilityTests(unittest.TestCase):
    """短串独立枚举核对：核心 DP 的概率和必须等于逐条路径枚举之和。"""

    def test_homopolymer_deletion_alignments_all_counted(self) -> None:
        # AAAA -> AAA：删除 4 个位置中任意一个 A 都是独立等价对齐，
        # 概率和 = 4 * e_del（e_del = mean(profile) = 1e-3）。
        edits, prob = edit_path_probability("AAAA", "AAA", uniform_profile("AAA"))
        self.assertEqual(edits, 1)
        self.assertAlmostEqual(prob, 4 * E3, places=15)

    def test_homopolymer_insertion_alignments_all_counted(self) -> None:
        # AAA -> AAAA：在 4 个间隙插入 A 的 4 种等价对齐，
        # 概率和 = 4 * (e/3)。
        edits, prob = edit_path_probability("AAA", "AAAA", uniform_profile("AAAA"))
        self.assertEqual(edits, 1)
        self.assertAlmostEqual(prob, 4 * E3 / 3.0, places=15)

    def test_single_substitution(self) -> None:
        edits, prob = edit_path_probability("AAA", "AAT", uniform_profile("AAT"))
        self.assertEqual((edits, prob), (1, E3 / 3.0))

    def test_two_substitutions(self) -> None:
        edits, prob = edit_path_probability("AAAA", "AATT", uniform_profile("AATT"))
        self.assertEqual(edits, 2)
        self.assertAlmostEqual(prob, (E3 / 3.0) ** 2, places=20)

    def test_indel_plus_substitution(self) -> None:
        # AAA -> AT：删除一个 A（3 种等价对齐）+ 替换一个 A->T，
        # 概率和 = 3 * e_del * (e/3)。
        edits, prob = edit_path_probability("AAA", "AT", uniform_profile("AT"))
        self.assertEqual(edits, 2)
        self.assertAlmostEqual(prob, 3 * E3 * (E3 / 3.0), places=20)

    def test_two_indels_not_in_model(self) -> None:
        # 需要两次删除：模型不允许（indel 合计至多 1 次）。
        self.assertIsNone(edit_path_probability("ACGT", "AG", uniform_profile("AG")))
        # 编辑数 3（3 次替换）也超出总编辑数 2。
        self.assertIsNone(edit_path_probability("AAA", "TTT", uniform_profile("TTT")))

    def test_position_dependent_profile_used(self) -> None:
        # 替换发生在观测位置 j，概率因子为该位置的 e_j/3。
        profile = (0.01, 0.02, 0.04)
        edits, prob = edit_path_probability("AAA", "AAT", profile)
        self.assertEqual(edits, 1)
        self.assertAlmostEqual(prob, 0.04 / 3.0, places=15)

    def test_against_independent_enumeration_short_strings(self) -> None:
        # 长度 1..4 的全部串中固定子抽样 4000 对，与递归枚举参考逐一比对
        # （最短编辑数与概率和）。
        universe = [
            "".join(p)
            for n in (1, 2, 3, 4)
            for p in itertools.product("ACGT", repeat=n)
        ]
        rng = random.Random(20260926)
        pairs = [(u, v) for u in universe for v in universe]
        rng.shuffle(pairs)
        for u, v in pairs[:4000]:  # 固定子抽样，保持测试秒级完成
            profile = tuple(0.001 + 0.0005 * (j % 3) for j in range(len(v)))
            want = reference_edit_path_probability(u, v, profile)
            got = edit_path_probability(u, v, profile)
            if want is None:
                self.assertIsNone(got, msg=f"{u}->{v}")
            else:
                self.assertIsNotNone(got, msg=f"{u}->{v}")
                self.assertEqual(got[0], want[0], msg=f"{u}->{v}")
                self.assertAlmostEqual(got[1], want[1], places=12, msg=f"{u}->{v}")


class EditClusteringTests(unittest.TestCase):
    def test_length_diff_one_absorbed_in_edit_mode_only(self) -> None:
        counts = {"AAAA": 100, "AAA": 10}
        got = cluster_umis(counts, error_model="edit")
        self.assertEqual(
            partition_of(got), {"AAAA": frozenset({"AAAA", "AAA"})}
        )
        # 默认模式回归：hamming 模式拒绝变长 UMI（原契约不变）。
        with self.assertRaises(InvalidArgumentError):
            cluster_umis(counts)

    def test_default_mode_unchanged_for_equal_length(self) -> None:
        # 等长场景下 edit 不是默认：Hamming=2 的对在默认模式不合并。
        counts = {"AAAA": 100, "AATT": 10}
        self.assertEqual(
            partition_of(cluster_umis(counts)),
            {"AAAA": frozenset({"AAAA"}), "AATT": frozenset({"AATT"})},
        )
        self.assertEqual(
            partition_of(cluster_umis(counts, error_model="edit")),
            {"AAAA": frozenset({"AAAA", "AATT"})},
        )

    def test_two_tied_high_abundance_candidates_stay_unmerged(self) -> None:
        # AATT(5) 的两个候选父节点 AAAA(50) 与 TTTT(50) 都是 2 次替换，
        # 路径概率完全相等 -> 差距 0 <= 任意阈值 -> 保留未归并。
        counts = {"AAAA": 50, "TTTT": 50, "AATT": 5}
        got = cluster_umis(counts, error_model="edit")
        self.assertEqual(
            partition_of(got),
            {
                "AAAA": frozenset({"AAAA"}),
                "TTTT": frozenset({"TTTT"}),
                "AATT": frozenset({"AATT"}),
            },
        )

    def test_ambiguity_margin_threshold_both_sides(self) -> None:
        # AAAA(5) 的候选：AAAT(50) 1 次替换（得分 log10(e/3)），
        # AATT(40) 2 次替换（得分 2*log10(e/3)），差距 = -log10(e/3)。
        counts = {"AAAT": 50, "AATT": 40, "AAAA": 5}
        profiles = profiles_for(counts)
        gap = -math.log10(E3 / 3.0)
        # 阈值大于差距 -> 歧义，AAAA 保留未归并
        got = cluster_umis(
            counts,
            error_model="edit",
            umi_error_profiles=profiles,
            ambiguity_margin=gap + 0.5,
        )
        self.assertEqual(
            partition_of(got),
            {
                "AAAT": frozenset({"AAAT"}),
                "AATT": frozenset({"AATT"}),
                "AAAA": frozenset({"AAAA"}),
            },
        )
        # 阈值小于差距 -> 归最佳候选 AAAT
        got = cluster_umis(
            counts,
            error_model="edit",
            umi_error_profiles=profiles,
            ambiguity_margin=gap - 0.5,
        )
        self.assertEqual(
            partition_of(got),
            {
                "AAAT": frozenset({"AAAT", "AAAA"}),
                "AATT": frozenset({"AATT"}),
            },
        )

    def test_quality_profile_decides_winner_not_traversal_order(self) -> None:
        # 两个候选同为 1 次替换；观测位置质量不同使概率不同，
        # 归属由质量决定，与字典序/遍历顺序无关。
        # v = "ATA"：候选 "AAA"（位置 1 替换，e_1/3）与 "TTA"（位置 0 替换，
        # e_0/3）。令 e_0 远大于 e_1（位置 0 质量极低，该错误更可能发生）
        # -> "TTA" 胜出，尽管字典序 "AAA" 在前。
        counts = {"AAA": 50, "TTA": 50, "ATA": 5}
        profiles = profiles_for(counts)
        profiles["ATA"] = (0.01, 1e-6, 0.01)
        got = cluster_umis(
            counts, error_model="edit", umi_error_profiles=profiles
        )
        self.assertEqual(
            partition_of(got),
            {
                "AAA": frozenset({"AAA"}),
                "TTA": frozenset({"TTA", "ATA"}),
            },
        )

    def test_input_permutation_invariance(self) -> None:
        counts = {
            "AAAA": 80, "AATT": 60, "AAA": 7, "AAAT": 9,
            "ATAT": 4, "TTTT": 30, "AAGT": 2,
        }
        profiles = profiles_for(counts)
        want = partition_of(
            cluster_umis(
                counts, error_model="edit", umi_error_profiles=profiles
            )
        )
        keys = list(counts)
        rng = random.Random(7)
        for _ in range(20):
            rng.shuffle(keys)
            shuffled = {k: counts[k] for k in keys}
            got = partition_of(
                cluster_umis(
                    shuffled, error_model="edit", umi_error_profiles=profiles
                )
            )
            self.assertEqual(got, want)

    def test_against_independent_clustering_reference(self) -> None:
        rng = random.Random(20260927)
        universe = [
            "".join(p) for p in itertools.product("ACGT", repeat=4)
        ] + ["".join(p) for p in itertools.product("ACGT", repeat=3)]
        for _ in range(60):
            k = rng.randint(2, 10)
            umis = rng.sample(universe, k)
            counts = {u: rng.randint(1, 40) for u in umis}
            profiles = {
                u: tuple(rng.choice((0.0005, 0.001, 0.005)) for _ in u)
                for u in umis
            }
            margin = rng.choice((0.0, 0.5, 2.0))
            got = partition_of(
                cluster_umis(
                    counts,
                    error_model="edit",
                    umi_error_profiles=profiles,
                    ambiguity_margin=margin,
                )
            )
            groups = reference_clusters_edit(counts, profiles, margin)
            want = {}
            for group in groups:
                root = min(group, key=lambda u: (-counts[u], u))
                want[root] = frozenset(group)
            self.assertEqual(got, want, msg=f"counts={counts} margin={margin}")


class EditPipelineTests(unittest.TestCase):
    def _batch(self):
        reads = []
        for _ in range(20):
            reads.append(Read("AAAA", "ACGT", (30, 30, 30, 30)))
        for _ in range(2):
            reads.append(Read("AAA", "ACG", (25, 25, 25)))  # 缺失 1 碱基
        reads.append(Read("AATT", "AGGT", (28, 28, 28, 28)))  # 2 次替换
        return reads

    def test_analyze_reads_edit_mode_merges_indel(self) -> None:
        result = analyze_reads(self._batch(), error_model="edit")
        self.assertEqual(result.nodes_before, 3)
        self.assertEqual(result.molecules_after, 1)
        mol = result.molecules[0]
        self.assertEqual(mol.root_umi, "AAAA")
        self.assertEqual(mol.members, ("AAA", "AAAA", "AATT"))
        # 共识在根坐标系：高质量根读段主导
        self.assertEqual(mol.consensus_seq, "ACGT")

    def test_analyze_reads_default_mode_rejects_variable_length(self) -> None:
        # 默认模式回归：变长 UMI 批次仍被拒绝（原契约）。
        with self.assertRaises(InvalidReadError):
            analyze_reads(self._batch())

    def test_analyze_reads_permutation_invariance(self) -> None:
        reads = self._batch()
        want = analyze_reads(reads, error_model="edit")
        rng = random.Random(11)
        for _ in range(10):
            perm = list(reads)
            rng.shuffle(perm)
            got = analyze_reads(perm, error_model="edit")
            self.assertEqual(
                [(m.root_umi, m.members) for m in got.molecules],
                [(m.root_umi, m.members) for m in want.molecules],
            )
            self.assertEqual(
                [m.consensus_seq for m in got.molecules],
                [m.consensus_seq for m in want.molecules],
            )

    def test_invalid_error_model_rejected(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            cluster_umis({"AAA": 1}, error_model="levenshtein")
        with self.assertRaises(InvalidArgumentError):
            analyze_reads(self._batch(), error_model="")
        with self.assertRaises(InvalidArgumentError):
            cluster_umis({"AAA": 1}, error_model="edit", ambiguity_margin=-1.0)
        with self.assertRaises(InvalidArgumentError):
            cluster_umis(
                {"AAA": 1},
                error_model="edit",
                umi_error_profiles={"AAA": (0.5,)},  # 长度不符
            )


if __name__ == "__main__":
    unittest.main()
