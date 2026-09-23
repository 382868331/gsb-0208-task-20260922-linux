"""测试专用的独立参考实现。

本文件**不导入 umicorrect**，用与核心实现完全不同的写法给出期望值：

* :func:`reference_clusters` —— 穷举全部节点对、Floyd-Warshall 传递闭包，
  再按全序分配首个最高根；
* :func:`reference_consensus` —— 逐读段、逐碱基直接累加对数似然。

它只允许在测试中使用，是“期望值”的独立来源，不是被测核心的包装
（往返一致测试只能作为补充）。
"""

from __future__ import annotations

import math

BASES = ("A", "C", "G", "T")


def reference_clusters(umi_counts: dict[str, int]) -> list[list[str]]:
    """返回成员 UMI 列表的列表；顺序：根出现序，簇内字典序。"""
    ordered = sorted(umi_counts, key=lambda u: (-umi_counts[u], u))
    n = len(ordered)

    # 1) 全对可达矩阵：对角线 + 满足规则的有向边（方向严格沿全序向前）
    reach = [[False] * n for _ in range(n)]
    for i in range(n):
        reach[i][i] = True
        ui = ordered[i]
        for j in range(i + 1, n):
            uj = ordered[j]
            dist = sum(a != b for a, b in zip(ui, uj))
            if dist == 1 and umi_counts[ui] >= 2 * umi_counts[uj] - 1:
                reach[i][j] = True

    # 2) Floyd-Warshall 传递闭包（独立于核心的 BFS 写法）
    for k in range(n):
        for i in range(n):
            if not reach[i][k]:
                continue
            row_i = reach[i]
            row_k = reach[k]
            for j in range(n):
                if row_k[j]:
                    row_i[j] = True

    # 3) 按全序建立根；只能拿到“尚未被更早的根拿走”的可达节点
    owner = [-1] * n
    groups: list[list[str]] = []
    for i in range(n):
        if owner[i] != -1:
            continue
        taken = [j for j in range(n) if reach[i][j] and owner[j] == -1]
        for j in taken:
            owner[j] = i
        groups.append(sorted(ordered[j] for j in taken))
    return groups


def reference_consensus(
    reads: list, length: int, indices: list[int]
) -> tuple[str, list[float]]:
    """直接按定义累计 log10 似然，返回 (共识, 逐位置 gap)。"""
    lls = [[0.0, 0.0, 0.0, 0.0] for _ in range(length)]
    for i in indices:
        r = reads[i]
        for p in range(length):
            q = r.qualities[p]
            err = 10.0 ** (-q / 10.0)
            called = r.sequence[p]
            for bi, b in enumerate(BASES):
                p_true = (1.0 - err) if b == called else err / 3.0
                # Q=0 时呼中碱基概率为 0：对数似然记 -inf（math.log10(0) 会抛错）
                lls[p][bi] += float("-inf") if p_true == 0.0 else math.log10(p_true)

    seq = []
    gaps = []
    for row in lls:
        best_val = max(row)
        winners = [bi for bi, v in enumerate(row) if v == best_val]
        if len(winners) > 1:
            seq.append("N")
            gaps.append(0.0)
        else:
            ordered_vals = sorted(row, reverse=True)
            seq.append(BASES[winners[0]])
            gaps.append(ordered_vals[0] - ordered_vals[1])
    return "".join(seq), gaps
