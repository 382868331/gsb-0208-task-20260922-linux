#!/usr/bin/env python3
"""固定输入演示：UMI 错误吸收与共识。

直接运行：``python demo.py``（Python 3.14.7，仅标准库，全程离线）。
示例数据由本文件内的确定性代码生成（固定种子/固定规格），真实计算并
打印：
  1) 一个正常批次：合并前/后分子数、每个分子的根、成员、共识与逐位置
     信心差（含一个并列报 N 的位置）；
  2) 两个实际触发的边界：空批次拒绝，以及搜索预算耗尽（RuntimeError，
     与“证明非法”的 ValueError 区分）。
不使用 sleep，不预录任何计算结果。
"""

from __future__ import annotations

import time

from umicorrect import (
    InvalidArgumentError,
    PositionedRead,
    Read,
    SearchBudgetExhaustedError,
    analyze_positioned_reads,
    analyze_reads,
)


def generate_demo_batch() -> list[Read]:
    """确定性构造一批 4 长度 UMI / 4 长度读段（无随机、无外部数据）。

    设计的 UMI 丰度结构：
      AAAA(20)  大根：19 条完美读 AAAA@30 + 1 条位置 3 的 Q7 错误读 AAAG
      AATT(20)  另一个大根，呼 AATT@28
      AAAT(1)   低丰度“桥”：与两大根各 Hamming=1，只能归字典序最先的根
      GGGG(2)   孤立小簇，位置 0 安排 A@10 vs C@10 的精确并列 -> 共识 N
    """
    reads: list[Read] = []

    def add(umi: str, seq: str, q: int) -> None:
        reads.append(Read(umi, seq, (q, q, q, q)))

    for _ in range(19):
        add("AAAA", "AAAA", 30)
    add("AAAA", "AAAG", 7)   # 唯一一条带测序错误的根读段（低质量）
    for _ in range(20):
        add("AATT", "AATT", 28)
    add("AAAT", "AAAA", 25)  # 桥：序列支持 AAAA 根
    add("GGGG", "AGGG", 10)
    add("GGGG", "CGGG", 10)
    return reads


def main() -> None:
    reads = generate_demo_batch()

    print("=" * 72)
    print("UMI 错误吸收与共识演示（离线、标准库、确定性输入）")
    print("=" * 72)
    print(f"读段总数: {len(reads)}（UMI 长度 = 读段长度 = 4）")

    t0 = time.perf_counter()
    result = analyze_reads(reads)
    elapsed = time.perf_counter() - t0

    print(
        f"\n[正常结果] 合并前分子数(去重 UMI 数) = {result.nodes_before}，"
        f"合并后分子数 = {result.molecules_after}，"
        f"吸收合并掉 {result.merged_count} 个错误 UMI"
        f"（真实计算耗时 {elapsed*1000:.2f} ms）"
    )
    for i, mol in enumerate(result.molecules, 1):
        print(f"  分子 {i}: 根={mol.root_umi} 成员={list(mol.members)} "
              f"读段数={len(mol.read_indices)}")
        print(f"          共识={mol.consensus_seq}")
        gaps = ", ".join(f"{g:.4f}" for g in mol.confidence_gap)
        print(f"          逐位置信心差(log10 Bayes 因子) = [{gaps}]")
    print("  说明: GGGG 簇位置 0 为等质量 A/C 精确并列，共识输出 N、gap=0；")
    print("        低丰度桥 AAAT 只并入字典序最先的大根 AAAA，两大根未被合并。")

    print("\n" + "-" * 72)
    print("[拒绝边界 1] 空批次：无法确定固定长度，必须拒绝")
    try:
        analyze_reads([])
    except InvalidArgumentError as exc:
        print(f"  实际抛出 {type(exc).__name__}: {exc}")

    print("\n[拒绝边界 2] 搜索预算耗尽：结果未知（RuntimeError），区别于非法输入")
    # 3 个节点共 3 对 Hamming 探测；预算 1 必然耗尽，且不返回截断结果
    try:
        analyze_reads(
            [
                Read("AAA", "AAA", (20, 20, 20)),
                Read("AAC", "AAC", (20, 20, 20)),
                Read("AAG", "AAG", (20, 20, 20)),
            ],
            probe_budget=1,
        )
    except SearchBudgetExhaustedError as exc:
        print(f"  实际抛出 {type(exc).__name__}: {exc}")
        print("  放大 probe_budget 后重算同一输入即可得到完整划分（预算耗尽≠无解）。")

    print("=" * 72)
    print("带显式参考起点的读段模式（变长读段，按参考坐标汇聚，无插缺比对）")
    print("=" * 72)
    # 确定性构造：大根 AAAA 的长读段 + 两端搭接短读段 + 中间空洞；
    # 错误 UMI AAAT 被吸收，其读段来自相邻参考区间。
    positioned = [
        PositionedRead("AAAA", 0, "AACCGG", (30, 30, 30, 30, 30, 30)),
        PositionedRead("AAAA", 0, "AAC", (25, 25, 25)),
        PositionedRead("AAAA", 8, "TTAA", (30, 30, 30, 30)),
        PositionedRead("AAAT", 10, "AG", (20, 20)),
    ]
    t1 = time.perf_counter()
    pres = analyze_positioned_reads(positioned)
    elapsed1 = time.perf_counter() - t1
    print(f"读段总数: {len(positioned)}（变长，带参考起点）")
    print(
        f"[正常结果] 合并前分子数 = {pres.nodes_before}，"
        f"合并后分子数 = {pres.molecules_after}"
        f"（真实计算耗时 {elapsed1*1000:.2f} ms）"
    )
    for mol in pres.molecules:
        print(f"  根={mol.root_umi} 成员={list(mol.members)} "
              f"参考区间=[{mol.ref_start}, {mol.ref_start + len(mol.consensus_seq)})")
        print(f"  共识={mol.consensus_seq}")
        print(f"  覆盖数={list(mol.coverage)}")
        gaps = ", ".join("缺失" if g is None else f"{g:.4f}" for g in mol.confidence_gap)
        print(f"  信心差=[{gaps}]")
    print("  说明: 坐标 6..7 无读段覆盖（空洞），共识 N 且信心差缺失；")
    print("        空洞不是低质量观测；短读段只在自身坐标投票，不错位。")

    print("[拒绝边界 3] 负参考起点：必须拒绝")
    try:
        PositionedRead("AA", -1, "AC", (20, 20))
    except InvalidArgumentError as exc:
        print(f"  实际抛出 {type(exc).__name__}: {exc}")

    print("\n演示完成。")


if __name__ == "__main__":
    main()
