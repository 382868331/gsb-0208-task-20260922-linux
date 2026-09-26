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


def reference_edit_path_probability(u, v, err_profile):
    """编辑路径概率和的独立参考：显式递归枚举全部对齐路径。

    与核心的 DP 聚合写法完全不同：这里逐条枚举路径（对齐），每条路径
    独立计算 (编辑数, 概率)，最后取最小编辑数并对并列最短路径求和。
    路径模型（与核心契约一致）：替换/插入因子 e_j/3（j 为 v 的观测位置），
    删除因子 mean(err_profile)；indel 合计至多 1 次，总编辑数 <= 2。
    不可达返回 None，否则返回 (最小编辑数, 概率和)。
    """
    e_del = sum(err_profile) / len(err_profile)
    paths = []  # (edits, prob)

    def rec(i, j, indels, edits, prob):
        if edits > 2:
            return
        if i == len(u) and j == len(v):
            paths.append((edits, prob))
            return
        if i < len(u) and j < len(v):
            if u[i] == v[j]:
                rec(i + 1, j + 1, indels, edits, prob)
            else:
                rec(i + 1, j + 1, indels, edits + 1, prob * err_profile[j] / 3.0)
        if indels == 0:
            if i < len(u):
                rec(i + 1, j, 1, edits + 1, prob * e_del)
            if j < len(v):
                rec(i, j + 1, 1, edits + 1, prob * err_profile[j] / 3.0)

    rec(0, 0, 0, 0, 1.0)
    if not paths:
        return None
    best = min(e for e, _p in paths)
    return best, sum(p for e, p in paths if e == best)


def reference_clusters_edit(umi_counts, err_profiles, ambiguity_margin=0.0):
    """编辑距离聚类的独立参考：全对枚举候选 + Floyd-Warshall 归属。

    每个节点 v 的候选父节点为全序中在前、count 满足 2s-1 规则且编辑
    距离 <=2 的 u；得分 = log10(最短路径概率和)。最佳与次佳得分差
    <= ambiguity_margin 时 v 无父节点（歧义，保留未归并）。
    返回成员 UMI 列表的列表（根出现序，簇内字典序）。
    """
    ordered = sorted(umi_counts, key=lambda u: (-umi_counts[u], u))
    n = len(ordered)
    parent = [None] * n
    for j in range(n):
        cands = []
        for i in range(j):
            if umi_counts[ordered[i]] < 2 * umi_counts[ordered[j]] - 1:
                continue
            stats = reference_edit_path_probability(
                ordered[i], ordered[j], err_profiles[ordered[j]]
            )
            if stats is None:
                continue
            cands.append((i, math.log10(stats[1])))
        if not cands:
            continue
        cands.sort(key=lambda t: (-t[1], t[0]))
        if len(cands) > 1 and cands[0][1] - cands[1][1] <= ambiguity_margin:
            continue  # 歧义：保留未归并
        parent[j] = cands[0][0]

    reach = [[False] * n for _ in range(n)]
    for i in range(n):
        reach[i][i] = True
    for j in range(n):
        if parent[j] is not None:
            reach[parent[j]][j] = True
    for k in range(n):
        for i in range(n):
            if not reach[i][k]:
                continue
            for j in range(n):
                if reach[k][j]:
                    reach[i][j] = True

    owner = [-1] * n
    groups = []
    for i in range(n):
        if owner[i] != -1:
            continue
        taken = [j for j in range(n) if reach[i][j] and owner[j] == -1]
        for j in taken:
            owner[j] = i
        groups.append(sorted(ordered[j] for j in taken))
    return groups
