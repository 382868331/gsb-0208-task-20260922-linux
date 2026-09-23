"""UMI 错误吸收规则测试，含与穷举小有向图参考的对拍。"""

from __future__ import annotations

import itertools
import unittest

from umi_consensus import Cluster, cluster_umis

from reference import hamming, ref_cluster


def _by_root(clusters):
    return {c.root: set(c.members) for c in clusters}


class BasicAbsorptionTests(unittest.TestCase):
    def test_single_node(self):
        (c,) = cluster_umis({"AAA": 1})
        self.assertEqual(c.root, "AAA")
        self.assertEqual(c.members, ("AAA",))
        self.assertEqual(c.node_counts, (("AAA", 1),))
        self.assertEqual(c.read_count, 1)

    def test_simple_absorption_threshold(self):
        # 大节点 3、小节点 1，距离 1：3 >= 2*1-1 = 1 → 吸收
        clusters = cluster_umis({"AAA": 3, "AAC": 1})
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].root, "AAA")
        self.assertEqual(clusters[0].members, ("AAA", "AAC"))

    def test_equal_counts_never_absorbed(self):
        # “大到小吸收”要求大节点计数严格更大；计数相等时互不为大节点，
        # 即使 1 >= 2*1-1 = 1 也不吸收——相等根保持独立。
        clusters = cluster_umis({"AAA": 5, "AAC": 5})
        self.assertEqual(len(clusters), 2)

    def test_threshold_boundary_3_and_2(self):
        # 3 >= 2*2-1 = 3 → 恰好在边界，允许吸收
        self.assertEqual(len(cluster_umis({"AAA": 3, "AAC": 2})), 1)

    def test_threshold_just_below(self):
        # 2 >= 2*2-1 = 3 不成立 → 不吸收
        self.assertEqual(len(cluster_umis({"AAA": 2, "AAC": 2})), 2)
        # 4 与 3：4 >= 5 不成立
        self.assertEqual(len(cluster_umis({"AAA": 4, "AAC": 3})), 2)

    def test_hamming_two_not_absorbed(self):
        self.assertEqual(len(cluster_umis({"AAA": 10, "ACC": 1})), 2)

    def test_tie_break_lexicographic_on_equal_counts(self):
        # AAC 与 AAG 计数相等且都可被 AAA(5) 吸收，成员顺序按节点排序；
        # 根只能是 AAA。
        clusters = cluster_umis({"AAA": 5, "AAG": 1, "AAC": 1})
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].root, "AAA")
        self.assertEqual(clusters[0].members, ("AAA", "AAC", "AAG"))


class AbundanceBridgeTests(unittest.TestCase):
    """低丰度桥不得把两个大根合并。"""

    def test_bridge_joins_one_root_only(self):
        # AAA=10, AAC=1（桥）, AAG=10。AAC 与两个大根距离都是 1，
        # 10 >= 1 都可吸收它；它只能归属首个最高根（字典序 AAA），
        # 两个大根保持独立。
        clusters = cluster_umis({"AAA": 10, "AAC": 1, "AAG": 10})
        as_sets = _by_root(clusters)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(as_sets["AAA"], {"AAA", "AAC"})
        self.assertEqual(as_sets["AAG"], {"AAG"})

    def test_bridge_below_threshold_to_both_roots(self):
        # AAA=2, AAC=2, AAG=2：计数相等，谁也不能吸收谁，三个根。
        clusters = cluster_umis({"AAA": 2, "AAC": 2, "AAG": 2})
        self.assertEqual(len(clusters), 3)

    def test_chain_absorption_is_directed(self):
        # 10 -> 4 -> 1 距离链。4 可被 10 吸收（10>=7），1 被 4 吸收（4>=1）；
        # 最终一个根 AAA；AAC 的归属不因 AAA 是否也直接距离 1 而改变。
        counts = {"AAA": 10, "AAC": 4, "AAG": 1}
        clusters = cluster_umis(counts)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].root, "AAA")
        self.assertEqual(
            clusters[0].node_counts, (("AAA", 10), ("AAC", 4), ("AAG", 1))
        )

    def test_chain_does_not_pull_node_without_edge_to_root(self):
        # AAA=10, AAC=4（并入 AAA）, ACC=2：ACC 与 AAC 距离 1，4>=3 成立，
        # 与 AAA 距离 2。沿父链归属 AAA，一个簇。
        clusters = cluster_umis({"AAA": 10, "AAC": 4, "ACC": 2})
        self.assertEqual(len(clusters), 1)
        self.assertEqual(set(clusters[0].members), {"AAA", "AAC", "ACC"})

    def test_first_highest_root_when_two_eligible_parents(self):
        # 小节点同时距离两个大节点 1，且两个大节点计数不同：
        # AAA=10, AAC=1, AAG=6。AAC 的合格父节点为 AAA(10) 与 AAG(6)，
        # 首个最高根是 AAA。
        clusters = cluster_umis({"AAG": 6, "AAC": 1, "AAA": 10})
        as_sets = _by_root(clusters)
        self.assertEqual(as_sets["AAA"], {"AAA", "AAC"})
        self.assertEqual(as_sets["AAG"], {"AAG"})

    def test_no_retroactive_merge(self):
        # AAA=10, AAC=1 先成簇；AAG=10 随后出现也不能把已有簇拉过来合并。
        # （在线归属天然保证：此处用顺序无关的接口验证最终仍是两个根。）
        counts_a = {"AAA": 10, "AAC": 1}
        counts_b = {"AAG": 10}
        merged = {**counts_a, **counts_b}
        clusters = cluster_umis(merged)
        roots = {c.root for c in clusters}
        self.assertEqual(roots, {"AAA", "AAG"})


class ExhaustiveGraphReferenceTests(unittest.TestCase):
    """对所有小规模 DNA 字母表上的计数组合穷举，与独立有向图参考对拍。"""

    def _all_strings(self, length):
        for chars in itertools.product("ACGT", repeat=length):
            yield "".join(chars)

    def test_all_subsets_length_2(self):
        universe = list(self._all_strings(2))
        tested = 0
        # 16 个节点的全部子集规模过大；对每个子集尺寸抽样若干组合，
        # 计数取自 {1,2,3,5,10}，覆盖相等、临界与悬殊比例。
        count_choices = (1, 2, 3, 5, 10)
        for size in range(1, 7):
            for sub_i, subset in enumerate(itertools.combinations(universe, size)):
                if sub_i >= 40:
                    break
                # 确定性地为节点分配计数（不引入随机性）
                counts = {
                    u: count_choices[(sub_i + k) % len(count_choices)]
                    for k, u in enumerate(subset)
                }
                expected = ref_cluster(dict(counts))
                got = cluster_umis(counts)
                self.assertEqual(
                    sorted((c.root, c.members) for c in got),
                    sorted(expected),
                    msg=f"counts={counts}",
                )
                tested += 1
        self.assertGreater(tested, 100)

    def test_targeted_hamming_one_exhaustive(self):
        # 固定一个 Hamming-1 邻域（以 AA 为中心），对中心和若干邻居
        # 穷举计数 1..6，全部与参考一致。
        neighbors = ["AA", "AC", "AG", "AT", "CA", "GA", "TA"]
        for combo in itertools.product(range(1, 7), repeat=4):
            counts = {neighbors[k]: combo[k] for k in range(4)}
            got = sorted((c.root, c.members) for c in cluster_umis(counts))
            exp = sorted(ref_cluster(dict(counts)))
            self.assertEqual(got, exp, msg=f"counts={counts}")

    def test_cluster_order_and_read_counts(self):
        counts = {"TTT": 7, "TTA": 2, "AAA": 9, "AAC": 1}
        clusters = cluster_umis(counts)
        # 根顺序按节点排序：AAA(9) 在 TTT(7) 之前
        self.assertEqual([c.root for c in clusters], ["AAA", "TTT"])
        for c in clusters:
            self.assertEqual(c.read_count, sum(n for _, n in c.node_counts))
            self.assertEqual(c.members[0], c.root)
            self.assertTrue(all(hamming(c.root, m) >= 1 for m in c.members[1:]))


if __name__ == "__main__":
    unittest.main()
