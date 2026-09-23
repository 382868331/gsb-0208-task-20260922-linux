"""测试专用独立参考实现。

本文件刻意以与被测核心不同的写法实现同一契约：

* 吸收参考：O(n^2) 逐对枚举 Hamming 距离，显式构造有向吸收边，
  再沿“最早的入边”追溯根（核心用 3L 变体枚举 + 位置表）。
* 共识参考：逐条读段、逐真值碱基直接累加 log10 概率
  （核心用整数计数聚合 + math.fsum）。

不 import 被测核心，期望值全部在本文件内独立算出。
"""

from __future__ import annotations

import math

BASES = ("A", "C", "G", "T")


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def ref_cluster(counts: dict[str, int]) -> list[tuple[str, tuple[str, ...]]]:
    """穷举所有节点对构造有向图，按“首个最高根”归属，返回 (root, members)。"""
    order = sorted(counts, key=lambda u: (-counts[u], u))
    parent: dict[str, str | None] = {}
    for k, u in enumerate(order):
        cu = counts[u]
        eligible = [
            order[j]
            for j in range(k)  # 只看处理顺序更早的节点
            # 计数严格更大（相等计数互不为“大节点”）、距离 1、满足阈值
            if counts[order[j]] > cu
            and hamming(order[j], u) == 1
            and counts[order[j]] >= 2 * cu - 1
        ]
        parent[u] = eligible[0] if eligible else None  # 排序最靠前的入边

    def root(u: str) -> str:
        while parent[u] is not None:
            u = parent[u]  # type: ignore[assignment]
        return u

    groups: dict[str, list[str]] = {}
    for u in order:
        groups.setdefault(root(u), []).append(u)
    return [(r, tuple(groups[r])) for r in order if parent[r] is None]


def ref_consensus_scores(
    reads: list[tuple[str, tuple[int, ...]]],
) -> list[dict[str, float]]:
    """逐条读段直接计算每位置四碱基 log10 似然（Q=0 时 log10(0)=-inf）。"""
    length = len(reads[0][0])
    out: list[dict[str, float]] = []
    for pos in range(length):
        scores: dict[str, float] = {}
        for true_base in BASES:
            ll = 0.0
            for seq, qual in reads:
                p = 10.0 ** (-qual[pos] / 10.0)
                prob = 1.0 - p if seq[pos] == true_base else p / 3.0
                # Q=0 且观测碱基恰为候选真值时 prob=0 → log10(0) = -inf
                ll += float("-inf") if prob == 0.0 else math.log10(prob)
            scores[true_base] = ll
        out.append(scores)
    return out
