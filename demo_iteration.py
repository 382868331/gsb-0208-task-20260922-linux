#!/usr/bin/env python3
"""编辑距离吸收模型演示（本轮新增可选行为）。

直接运行：python demo_iteration.py（Python 3.14.7，仅标准库，全程离线，
真实计算，无 sleep、无预录结果，正常 1 秒内完成，远低于 90 秒）。

演示内容：
  1) 正常结果：edit 模式吸收"缺 1 碱基"与"2 次替换"的错误 UMI，
     并与默认 hamming 模式对照（默认行为不变）；
  2) 同聚物等价对齐：AAAA -> AAA 的 4 条等价删除路径概率和；
  3) 边界一：两个高丰度候选并列 -> 该节点保留未归并（不强行吸收）；
  4) 边界二：非法 error_model 被明确拒绝（可定位错误）。
"""

from __future__ import annotations

import time

from umicorrect import (
    InvalidArgumentError,
    InvalidReadError,
    Read,
    analyze_reads,
    cluster_umis,
    edit_path_probability,
)


def generate_batch() -> list[Read]:
    """确定性构造一批混合长度 UMI 的读段（无随机、无外部数据）。

    AAAA(20) 为大根；AAA(2) 是缺失 1 碱基的错误拷贝；AATT(1) 是
    2 次替换的错误拷贝。序列与各自 UMI 等长。
    """
    reads: list[Read] = []
    for _ in range(20):
        reads.append(Read("AAAA", "ACGT", (30, 30, 30, 30)))
    for _ in range(2):
        reads.append(Read("AAA", "ACG", (25, 25, 25)))
    reads.append(Read("AATT", "AGGT", (28, 28, 28, 28)))
    return reads


def main() -> None:
    print("=" * 72)
    print("编辑距离 UMI 吸收演示（error_model='edit'，离线、确定性输入）")
    print("=" * 72)

    reads = generate_batch()
    print(f"读段总数: {len(reads)}（UMI: AAAA x20, AAA x2, AATT x1）")

    t0 = time.perf_counter()
    edit_result = analyze_reads(reads, error_model="edit")
    elapsed = time.perf_counter() - t0
    print(
        f"\n[正常结果] edit 模式：合并前 {edit_result.nodes_before} 个 UMI -> "
        f"合并后 {edit_result.molecules_after} 个分子"
        f"（真实计算耗时 {elapsed*1000:.2f} ms）"
    )
    for mol in edit_result.molecules:
        print(f"  根={mol.root_umi} 成员={list(mol.members)} "
              f"共识={mol.consensus_seq}")
    print("  说明: 缺 1 碱基的 AAA 与 2 次替换的 AATT 都被吸收进根 AAAA；")
    print("        共识在根坐标系上对齐计算（插入列忽略、删除列不贡献）。")

    # 对照：默认 hamming 模式行为不变——变长批次仍被拒绝
    print("\n[对照] 默认 hamming 模式（原行为）：同一批次被拒绝")
    try:
        analyze_reads(reads)
    except InvalidReadError as exc:
        print(f"  实际抛出 {type(exc).__name__}: {exc}")

    print("\n" + "-" * 72)
    print("[同聚物路径模型] AAAA -> AAA 的最短编辑路径概率和")
    t0 = time.perf_counter()
    edits, prob = edit_path_probability("AAAA", "AAA", (0.001, 0.001, 0.001))
    elapsed = time.perf_counter() - t0
    print(f"  最小编辑数={edits}，概率和={prob:.6g}"
          f"（= 4 条等价删除对齐 x 1e-3，耗时 {elapsed*1000:.2f} ms）")
    print("  说明: 重复碱基的多种等价对齐按路径模型累计，不靠遍历顺序。")

    print("\n" + "-" * 72)
    print("[边界 1] 两个高丰度候选并列：保留未归并，避免强行吸收")
    counts = {"AAAA": 50, "TTTT": 50, "AATT": 5}
    t0 = time.perf_counter()
    clusters = cluster_umis(counts, error_model="edit")
    elapsed = time.perf_counter() - t0
    print(f"  计数 {counts}（AATT 到 AAAA/TTTT 同为 2 次替换，概率相等）")
    print(f"  结果（耗时 {elapsed*1000:.2f} ms）：")
    for cl in clusters:
        print(f"    根={cl.root} 成员={list(cl.members)}")
    print("  说明: AATT 的最佳/次佳候选差距为 0 <= 阈值，自成一根未被吸收。")

    print("\n[边界 2] 非法 error_model 被明确拒绝")
    try:
        cluster_umis({"AAA": 1}, error_model="levenshtein")
    except InvalidArgumentError as exc:
        print(f"  实际抛出 {type(exc).__name__}: {exc}")

    print("\n演示完成。")


if __name__ == "__main__":
    main()

