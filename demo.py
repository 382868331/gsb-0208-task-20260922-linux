#!/usr/bin/env python3
"""固定输入演示：单端 UMI 错误吸收与双端 UMI 规范化成对。

直接运行：``python demo.py``（Python 3.14.7，仅标准库，全程离线）。
示例数据由本文件内的确定性代码生成（无随机、无外部数据），真实计算并
打印：
  1) 单端正常批次：合并前/后分子数、根、成员、共识与逐位置信心差
     （含一个并列报 N 的位置）；
  2) 单端两个边界：空批次拒绝，以及搜索预算耗尽（RuntimeError，
     与“证明非法”的 ValueError 区分）；
  3) 双端正常批次：双 UMI/坐标/链规范化、显式成对、保留链身份、
     反向互补相同但链不同 -> 两个分子，直接调用旧共识入口出共识；
  4) 双端两个实际错误：同 record_id 重复（RecordConflictError，
     不静默合并）与非法空区间坐标。
不使用 sleep，不预录任何计算结果。
"""

from __future__ import annotations

import time

from umicorrect import (
    InvalidArgumentError,
    PairedRecord,
    Read,
    RecordConflictError,
    SearchBudgetExhaustedError,
    analyze_paired_records,
    analyze_reads,
    reverse_complement,
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

    # ------------------------------------------------------------------
    # 双端 UMI + 坐标 + 链：规范化与显式成对（本次迭代新增能力）
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("双端 UMI 规范化 / 显式成对演示（链身份保留，直接调用旧共识入口）")
    print("=" * 72)

    paired = [
        # 分子 P：chr1 [100,104) + 链，双 UMI (AAA,AAC)；两条分子级重复观测
        PairedRecord("p1", "AAA", "AAC", "chr1", 100, 104, "+",
                     "ACGT", (30, 30, 30, 30)),
        PairedRecord("p2", "AAA", "AAC", "chr1", 100, 104, "+",
                     "ACGA", (30, 30, 30, 7)),  # 末位低质量测序错误
        # 分子 M：同一坐标、方向化后双 UMI 字面相同，但来自 - 链：
        # raw (GTT,TTT) = (rc(AAC), rc(AAA))，序列给 rc(ACGT)=ACGT。
        # 链是匹配键的一部分，P 与 M 必须保持为两个不同分子。
        PairedRecord("m1", "GTT", "TTT", "chr1", 100, 104, "-",
                     reverse_complement("ACGT"), (28, 28, 28, 28)),
        # 分子 Q：chr3 [50,55)，一条坐标按反向 (55,50) 给，规范化后成对
        PairedRecord("q1", "CCG", "GGA", "chr3", 50, 55, "+",
                     "TTAA", (25, 25, 25, 25)),
        PairedRecord("q2", "CCG", "GGA", "chr3", 55, 50, "+",
                     "TTAA", (25, 25, 25, 25)),
        # 分子 T：同分子两条记录位置 0 等质量呼 A/C -> 精确并列报 N
        PairedRecord("t1", "TTA", "ATC", "chr5", 0, 4, "+",
                     "ACGT", (20, 30, 30, 30)),
        PairedRecord("t2", "TTA", "ATC", "chr5", 0, 4, "+",
                     "CCGT", (20, 30, 30, 30)),
    ]
    print(f"双端记录总数: {len(paired)}")
    t0 = time.perf_counter()
    presult = analyze_paired_records(paired)
    elapsed = time.perf_counter() - t0
    print(
        f"\n[正常结果] 规范化成对：{presult.records_in} 条记录 -> "
        f"{presult.molecules_after} 个分子，吸收分子级重复 "
        f"{presult.duplicate_count} 条（真实计算耗时 {elapsed*1000:.2f} ms）"
    )
    for i, mol in enumerate(presult.molecules, 1):
        print(
            f"  分子 {i}: {mol.contig}:[{mol.start},{mol.end}) 链={mol.strand} "
            f"双UMI(参考方向)=({mol.umi_left},{mol.umi_right})"
        )
        print(f"          记录={list(mol.record_ids)} 共识={mol.consensus_seq}")
        gaps = ", ".join(f"{g:.4f}" for g in mol.confidence_gap)
        print(f"          逐位置信心差 = [{gaps}]")
    print("  说明: p1/p2 同键成对（低质量末位错误被共识吸收）；m1 虽方向化")
    print("        UMI 与 p 相同，但链='-'，保留为独立分子；q2 坐标反向给入")
    print("        仍与 q1 成对；t1/t2 位置 0 精确并列，共识报 N、gap=0。")

    print("\n[错误边界 1] 记录级重复（同一 record_id 出现两次）：显式拒绝，不静默合并")
    dup = [
        PairedRecord("same", "AAA", "AAC", "chr1", 0, 4, "+",
                     "ACGT", (30, 30, 30, 30)),
        PairedRecord("other", "TTT", "GGG", "chr1", 8, 12, "+",
                     "TTTT", (30, 30, 30, 30)),
        PairedRecord("same", "AAA", "AAC", "chr1", 0, 4, "+",
                     "ACGT", (30, 30, 30, 30)),
    ]
    try:
        analyze_paired_records(dup)
    except RecordConflictError as exc:
        print(f"  实际抛出 {type(exc).__name__}: {exc}")

    print("\n[错误边界 2] 非法坐标（空区间 start==end）：构造期明确报错")
    try:
        PairedRecord("bad", "AAA", "AAC", "chr1", 5, 5, "+",
                     "ACGT", (30, 30, 30, 30))
    except InvalidArgumentError as exc:
        print(f"  实际抛出 {type(exc).__name__}: {exc}")

    print("\n演示完成。")


if __name__ == "__main__":
    main()
