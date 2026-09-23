"""UMI 吸收聚类测试。

期望值来源：
* 丰度桥、相等根、传递链等场景为按题面规则手算的固定用例；
* :func:`tests.reference.reference_clusters` —— 穷举边 + Floyd-Warshall
  传递闭包的独立参考（不导入被测核心）；
* 小规模枚举：固定 5 个 UMI、计数取值 {1,2,5} 的全部 3^5=243 种赋值，
  以及确定性随机种子下的若干随机实例。
"""

from __future__ import annotations

import random
import unittest

from reference import reference_clusters

from umicorrect import (
    InvalidArgumentError,
    SearchBudgetExhaustedError,
    cluster_umis,
)


def partition_of(clusters):
    """把核心输出转成 {root: frozenset(members)} 便于与参考比较。"""
    return {cl.root: frozenset(cl.members) for cl in clusters}


def expected_partition(umi_counts):
    groups = reference_clusters(umi_counts)
    out = {}
    for group in groups:
        root = min(group, key=lambda u: (-umi_counts[u], u))
        out[root] = frozenset(group)
    return out


class ExhaustiveSmallGraphTests(unittest.TestCase):
    # 5 个 UMI：前 4 个构成超立方体风格邻接，最后一个孤立。
    UMIS = ("AAA", "AAC", "ACC", "CCC", "GGG")

    def test_all_3_5_count_assignments(self) -> None:
        # 3^5 = 243 种计数赋值，逐一枚举（真正的小有向图穷举）。
        for code in range(3**5):
            counts = {}
            x = code
            for u in self.UMIS:
                counts[u] = (1, 2, 5)[x % 3]
                x //= 3
            got = partition_of(cluster_umis(counts))
            want = expected_partition(counts)
            self.assertEqual(got, want, msg=f"counts={counts}")

    def test_random_length4_instances_against_reference(self) -> None:
        rng = random.Random(20260923)
        universe = ["".join(b) for b in _product("ACGT", repeat=4)]
        for _ in range(300):
            k = rng.randint(1, 14)
            umis = rng.sample(universe, k)
            counts = {u: rng.randint(1, 12) for u in umis}
            got = partition_of(cluster_umis(counts))
            want = expected_partition(counts)
            self.assertEqual(got, want, msg=f"counts={counts}")


def _product(alphabet, repeat):
    if repeat == 0:
        yield ""
        return
    for prefix in _product(alphabet, repeat - 1):
        for ch in alphabet:
            yield prefix + ch


class AbundanceBridgeTests(unittest.TestCase):
    """低丰度桥不得合并两个大根。"""

    def test_bridge_between_two_big_roots(self) -> None:
        # AAAA(10) 与 AATT(10) Hamming=2；AAAT(1) 与两者各 Hamming=1。
        counts = {"AAAA": 10, "AATT": 10, "AAAT": 1}
        got = cluster_umis(counts)
        parts = partition_of(got)
        self.assertEqual(
            parts,
            {
                "AAAA": frozenset({"AAAA", "AAAT"}),  # 字典序 AAAA 先处理
                "AATT": frozenset({"AATT"}),
            },
        )
        # 合并后仍是 2 个分子，桥没有连通两大根
        self.assertEqual(len(got), 2)

    def test_bridge_with_chained_errors_still_no_root_merge(self) -> None:
        # 桥再挂一个下游错误 UMI，两大根依旧独立。
        # AAAA(20) -- AAAT(2) -- AAGT(1) ；对侧大根 AATT(20)。
        counts = {"AAAA": 20, "AATT": 20, "AAAT": 2, "AAGT": 1}
        # 边检查：AAAA->AAAT (20>=3)，AAAT->AAGT (2>=1)；
        # AATT->AAAT (20>=3) 但 AAAT 已被首个最高根 AAAA 拿走。
        got = cluster_umis(counts)
        parts = partition_of(got)
        self.assertEqual(
            parts,
            {
                "AAAA": frozenset({"AAAA", "AAAT", "AAGT"}),
                "AATT": frozenset({"AATT"}),
            },
        )


class EqualRootTests(unittest.TestCase):
    def test_equal_big_counts_cannot_absorb_both_survive(self) -> None:
        # 两个 count=2 的 Hamming-1 UMI：2 >= 2*2-1=3 不成立，各自成根。
        counts = {"AAA": 2, "AAC": 2}
        got = cluster_umis(counts)
        self.assertEqual(
            partition_of(got),
            {"AAA": frozenset({"AAA"}), "AAC": frozenset({"AAC"})},
        )

    def test_equal_singletons_merge_lexicographically_smaller_is_root(self) -> None:
        # 边界：count 同为 1 时 1 >= 2*1-1 = 1 成立；字典序小者为根。
        counts = {"TTA": 1, "ATA": 1}  # Hamming=1
        got = cluster_umis(counts)
        self.assertEqual(
            partition_of(got), {"ATA": frozenset({"ATA", "TTA"})}
        )

    def test_threshold_boundary_inclusion(self) -> None:
        # 大 count c、小 count s：吸收条件 c >= 2s-1 的恰好边界两侧。
        # s=3 -> 阈值 5：c=5 可吸收，c=4 不可。
        yes = cluster_umis({"AAA": 5, "TAA": 3})
        self.assertEqual(partition_of(yes), {"AAA": frozenset({"AAA", "TAA"})})
        no = cluster_umis({"AAA": 4, "TAA": 3})
        self.assertEqual(
            partition_of(no), {"AAA": frozenset({"AAA"}), "TAA": frozenset({"TAA"})}
        )


class TransitiveChainTests(unittest.TestCase):
    def test_chain_is_reachable_transitively(self) -> None:
        # AAAA(10)->AAAT(5)->AAGT(2)；AAAA 与 AAGT Hamming=2，
        # 仅靠传递可达归入同簇。
        counts = {"AAAA": 10, "AAAT": 5, "AAGT": 2}
        got = cluster_umis(counts)
        self.assertEqual(
            partition_of(got),
            {"AAAA": frozenset({"AAAA", "AAAT", "AAGT"})},
        )

    def test_members_sorted_and_cluster_order(self) -> None:
        counts = {"AAAA": 10, "AAAT": 5}
        (cl,) = cluster_umis(counts)
        self.assertEqual(cl.root, "AAAA")
        self.assertEqual(cl.members, ("AAAA", "AAAT"))
        self.assertEqual((cl.root_count, cl.total_count), (10, 15))


class BudgetAndErrorTests(unittest.TestCase):
    def test_budget_exhausted_is_distinct_from_impossible(self) -> None:
        counts = {"AAA": 5, "AAC": 3, "AAG": 1}
        # 3 个节点共 3 对探测；预算 2 必须在第 3 对时耗尽。
        with self.assertRaises(SearchBudgetExhaustedError) as ctx:
            cluster_umis(counts, probe_budget=2)
        self.assertEqual(ctx.exception.budget, 2)
        self.assertGreaterEqual(ctx.exception.probes_used, 2)
        # 预算耗尽是 RuntimeError（结果未知），不是 ValueError（证明非法）。
        self.assertIsInstance(ctx.exception, RuntimeError)
        self.assertNotIsInstance(ctx.exception, ValueError)

    def test_adequate_budget_always_yields_partition(self) -> None:
        # 合法输入不存在“无解”：预算充足时必然得到划分，最差为每节点独立。
        counts = {"AAA": 2, "AAC": 2, "AAG": 2}
        got = cluster_umis(counts, probe_budget=3)
        self.assertEqual(len(got), 3)

    def test_bool_budget_rejected(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            cluster_umis({"AA": 1}, probe_budget=True)

    def test_empty_counts_returns_empty_not_error(self) -> None:
        # cluster_umis 层面空表返回 ()（空批次拒绝由 analyze_reads 负责）。
        self.assertEqual(cluster_umis({}), ())

    def test_nonpositive_count_rejected(self) -> None:
        with self.assertRaises(InvalidArgumentError):
            cluster_umis({"AA": 0})
        with self.assertRaises(InvalidArgumentError):
            cluster_umis({"AA": True})

    def test_nan_infinity_budget_rejected(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "NaN/Infinity"):
            cluster_umis({"AA": 1}, probe_budget=float("nan"))
        with self.assertRaisesRegex(InvalidArgumentError, "NaN/Infinity"):
            cluster_umis({"AA": 1}, probe_budget=float("inf"))

    def test_float_budget_rejected_even_if_integral_valued(self) -> None:
        with self.assertRaisesRegex(InvalidArgumentError, "整数"):
            cluster_umis({"AA": 1}, probe_budget=2.0)


if __name__ == "__main__":
    unittest.main()
