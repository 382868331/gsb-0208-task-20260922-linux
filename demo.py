#!/usr/bin/env python3
"""固定输入演示：UMI 错误吸收与共识（含双端 UMI + 坐标 + 链方向）。

直接运行：``python demo.py``（Python 3.14.7，仅标准库，全程离线）。
示例数据由本文件内的确定性代码生成（固定种子/固定规格），真实计算并
打印：

第一部分（单端，旧能力，保留不变）：
  1) 一个正常批次：合并前/后分子数、每个分子的根、成员、共识与逐位置
     信心差（含一个并列报 N 的位置）；
  2) 两个实际触发的边界：空批次拒绝，以及搜索预算耗尽（RuntimeError，
     与“证明非法”的 ValueError 区分）。

第二部分（本次迭代：双端 UMI + 坐标 + 链方向）：
  3) 正常批次：反链记录反向互补规范化后在同坐标与正链共存但链身份不合并、
     按 ID 折叠完全一致的重复提交、单碱基错误双端 UMI 的成对吸收；
  4) 边界：不同坐标即便规范化 UMI 完全相同也不成对；
  5) 实际错误：同一 record_id 内容冲突，显式拒绝静默合并。

不使用 sleep，不预录任何计算结果。
"""

from __future__ import annotations

import time

from umicorrect import (
    DuplicateRecordError,
    InvalidArgumentError,
    Read,
    SearchBudgetExhaustedError,
    TaggedRead,
    analyze_reads,
    analyze_tagged_reads,
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


def generate_tagged_batch():
    """确定性构造一批双端 UMI 记录（无随机、无外部数据）。

    返回 (records, notes)。结构（均可手工核对）：

    坐标 500、反链 '-'：1 条原始 UMI GT|AC（恰为 AC|GT 的双端反向互补），
      规范化键 AC|GT，链身份保留为 '-'。
    坐标 500、正链 '+'：
      AC|GT 10 条（大根）；AC|AT 2 条（umi2 端单碱基错误，10>=2*2-1 被吸收）；
      GC|GC 的记录 'dup-0' 完全一致地提交两次（按 ID 折叠，去重 1 条）。
    坐标 800、正链 '+'：AC|GT 3 条——规范化键与 500 的大根相同，但坐标不同，
      独立成分子、计数不跨坐标累加。
    """
    q4 = (30, 30, 30, 30)
    records: list[TaggedRead] = []
    notes = [
        "坐标500 '-'：GT|AC seq=ACGA（规范化键 AC|GT，seq=TCGT，质量随位置反转）",
        "坐标500 '+'：AC|GT x10（大根），AC|AT x2（umi2 单碱基差，被吸收）",
        "坐标500 '+'：GC|GC 的 dup-0 完全一致提交两次（按 ID 去重 1 条）",
        "坐标800 '+'：AC|GT x3（同键不同坐标，必须独立成分子）",
    ]

    # 反链记录放在首位，demo 用它手工展示规范化
    records.append(TaggedRead("GT", "AC", 500, "-", "ACGA", (7, 30, 30, 30), "m-0"))

    for k in range(10):
        records.append(TaggedRead("AC", "GT", 500, "+", "ACGT", q4, f"p-{k}"))
    for k in range(2):
        records.append(TaggedRead("AC", "AT", 500, "+", "ACGT",
                                  (10, 30, 30, 30), f"e-{k}"))
    # 同一记录完全一致提交两次：合法重复，按 ID 折叠
    for _ in range(2):
        records.append(TaggedRead("GC", "GC", 500, "+", "GGGG", q4, "dup-0"))

    for k in range(3):
        records.append(TaggedRead("AC", "GT", 800, "+", "ACGT", q4, f"x-{k}"))
    return records, notes


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

    # ======================================================================
    # 第二部分：双端 UMI + 坐标 + 链方向
    # ======================================================================
    print("\n" + "=" * 72)
    print("双端 UMI + 坐标 + 链方向：规范化、成对与按 ID 去重")
    print("=" * 72)
    tagged, notes = generate_tagged_batch()
    print(f"记录总数: {len(tagged)}（双端 UMI 各长 2，读段长 4）")
    print("输入设计（规范化键 = rc 后的 umi1|umi2）：")
    for note in notes:
        print(f"  - {note}")

    t0 = time.perf_counter()
    tres = analyze_tagged_reads(tagged)
    elapsed = time.perf_counter() - t0
    print(
        f"\n[正常结果] 去重 {tres.duplicates_removed} 条重复提交；"
        f"合并前双端 UMI 种数 = {tres.nodes_before}，合并后分子数 = "
        f"{tres.molecules_after}，吸收 {tres.merged_count} 个错误双端 UMI"
        f"（真实计算耗时 {elapsed*1000:.2f} ms）"
    )
    for i, mol in enumerate(tres.molecules, 1):
        print(
            f"  分子 {i}: 坐标={mol.coordinate} 链={mol.strand} 根={mol.root_umi} "
            f"成员={list(mol.members)} 记录={list(mol.record_ids)} "
            f"读段数={len(mol.read_indices)}"
        )
        print(f"          共识(正链空间)={mol.consensus_seq}")
    print("  说明: 坐标 500 的 '+' 与 '-' 规范化根键相同但链身份保留，是两个分子；")
    print("        坐标 500 内单碱基错误 AC|AT 被大根 AC|GT 成对吸收；")
    print("        坐标 800 是同键不同坐标，独立成分子（不跨坐标成对/累加计数）。")

    print("\n[边界] 手工核对一条反链记录的规范化（期望值直接由反向互补定义算出）")
    raw = tagged[0]  # '-' 记录，在生成函数中放在首位
    print(
        f"  原始: strand={raw.strand} umi1={raw.umi1} umi2={raw.umi2} "
        f"seq={raw.sequence} qual={raw.qualities}"
    )
    print(
        f"  规范化: umi1={reverse_complement(raw.umi1)} "
        f"umi2={reverse_complement(raw.umi2)} seq={reverse_complement(raw.sequence)} "
        f"qual={tuple(reversed(raw.qualities))}（质量随序列位置反转，链身份仍为 '-'）"
    )

    print("\n[实际错误] 同一 record_id 内容冲突：拒绝静默合并")
    conflicting = tagged + [
        # 复用已存在的 ID 'dup-0'，但 umi2 与首次提交不同
        TaggedRead("AC", "GG", 500, "+", "ACGT", (30, 30, 30, 30), "dup-0"),
    ]
    try:
        analyze_tagged_reads(conflicting)
    except DuplicateRecordError as exc:
        print(f"  实际抛出 {type(exc).__name__}: {exc}")
        print("  调用方必须修正 ID 或内容后重提；本库不选择任一版本、不做合并。")

    print("\n演示完成。")


if __name__ == "__main__":
    main()
