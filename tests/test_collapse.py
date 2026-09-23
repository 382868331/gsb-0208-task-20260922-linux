"""collapse 端到端测试：吸收 + 共识分子输出。"""

from __future__ import annotations

import unittest

from umi_consensus import Molecule, Record, ValidationError, collapse


def _mols_by_root(mols):
    return {m.root: m for m in mols}


class CollapseTests(unittest.TestCase):
    def test_single_molecule_fields(self):
        records = [
            Record("AA", "AC", (30, 30)),
            ("AA", "AC", (28, 25)),
            ("AT", "AC", (20, 22)),  # Hamming 1，1 条，被 AA=2 吸收
        ]
        mols = collapse(records)
        self.assertEqual(len(mols), 1)
        m = mols[0]
        self.assertIsInstance(m, Molecule)
        self.assertEqual(m.root, "AA")
        self.assertEqual(m.members, ("AA", "AT"))
        self.assertEqual(dict(m.node_counts), {"AA": 2, "AT": 1})
        self.assertEqual(m.read_count, 3)
        self.assertEqual(m.consensus, "AC")
        self.assertEqual(len(m.confidence), 2)
        self.assertEqual(len(m.scores), 2)
        self.assertTrue(all(c > 0 for c in m.confidence))

    def test_two_roots_stay_separate_through_bridge(self):
        # 丰度桥场景：AAC=1 同时距离 AAA=10 与 AAG=10 为 1，只归首个最高根。
        records = []
        for _ in range(10):
            records.append(("AAA", "AC", (35, 35)))
        for _ in range(10):
            records.append(("AAG", "GT", (35, 35)))
        records.append(("AAC", "AC", (20, 20)))
        mols = collapse(records)
        self.assertEqual(len(mols), 2)
        by_root = _mols_by_root(mols)
        self.assertEqual(by_root["AAA"].members, ("AAA", "AAC"))
        self.assertEqual(by_root["AAA"].read_count, 11)
        self.assertEqual(by_root["AAG"].members, ("AAG",))
        self.assertEqual(by_root["AAG"].read_count, 10)
        # 共识不受串簇影响
        self.assertEqual(by_root["AAA"].consensus, "AC")
        self.assertEqual(by_root["AAG"].consensus, "GT")

    def test_equal_roots_two_molecules(self):
        records = [("AA", "AC", (30, 30)), ("AC", "GT", (30, 30))]
        mols = collapse(records)
        self.assertEqual(len(mols), 2)
        self.assertEqual([m.root for m in mols], ["AA", "AC"])  # 字典序

    def test_consensus_uses_all_member_reads(self):
        # 桥读段携带的高置信证据必须进入根簇共识：
        # 根节点 2 条读段第二位都是低质量(Q2) A，桥 1 条是高质量(Q40) C。
        records = [
            ("AA", "AA", (40, 2)),
            ("AA", "AA", (40, 2)),
            ("AT", "AC", (20, 40)),  # 高置信 C，压倒两条低质量 A
        ]
        m = collapse(records)[0]
        self.assertEqual(m.members, ("AA", "AT"))
        self.assertEqual(m.consensus, "AC")

    def test_empty_and_validation(self):
        self.assertEqual(collapse([]), ())
        with self.assertRaises(ValidationError) as ctx:
            collapse([("AA", "AC", (30, 30)), ("AA", "AG", (30,))])
        self.assertEqual(ctx.exception.index, 1)
        self.assertIn("qualities", ctx.exception.field)

    def test_molecule_order_is_root_rank_order(self):
        records = [("TT", "A", (30,)), ("TT", "A", (30,)),
                   ("AA", "C", (30,)), ("AA", "C", (30,)), ("AA", "C", (30,))]
        mols = collapse(records)
        # AA=3 排在 TT=2 之前
        self.assertEqual([m.root for m in mols], ["AA", "TT"])


if __name__ == "__main__":
    unittest.main()
