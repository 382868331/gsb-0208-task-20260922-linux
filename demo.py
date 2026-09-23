"""demo.py —— UMI 错误吸收与共识的固定输入演示（确定性、离线、纯计算）。

运行：python demo.py（约 1 秒内，无网络、无 sleep、无预录结果）。

数据由本文件内的确定性代码直接构造（无随机种子依赖外部状态），包含：
  * 两个等计数大根 AAAA / AAAT（距离 1，计数相等 → 不互相吸收）；
  * 低丰度桥 AAAG，同时距离两个大根为 1，只归入首个最高根 AAAA；
  * 吸收链 CCCC(8) -> GCCC(3) -> GGCC(1)；
  * 临界不吸收：TTTT(4) 与 ATTT(3)，4 < 2*3-1=5；
  * 低质量多数被高置信少数推翻（TATA：2×Q2-A vs 1×Q40-C）；
  * 严格并列 → N（GTGT：A@Q30 与 C@Q30 各一票）；
  * 一个实际触发的非法输入拒绝边界（质量 41，可定位到记录与字段）。
"""

from __future__ import annotations

import time

from umi_consensus import (
    ValidationError,
    cluster_umis,
    collapse,
    consensus,
)

UMI_LEN = 4
READ_LEN = 6


def build_records() -> list[tuple[str, str, tuple[int, ...]]]:
    records: list[tuple[str, str, tuple[int, ...]]] = []

    def add(umi: str, n: int, seq: str, quals: tuple[int, ...]) -> None:
        assert len(umi) == UMI_LEN and len(seq) == len(quals) == READ_LEN
        for _ in range(n):
            records.append((umi, seq, quals))

    # --- 大根 1：AAAA=12，真实序列 ACGTAC，质量 33..38 确定性轮换 ---
    for k in range(12):
        q = 33 + (k % 6)
        add("AAAA", 1, "ACGTAC", (q, q, q, q, q, q))
    add("AAAG", 1, "ACGTAC", (25, 26, 24, 25, 23, 22))  # 桥：距离 AAAT 也为 1

    # --- 大根 2：AAAT=12（与 AAAA 距离 1 且计数相等 → 保持独立）---
    for k in range(12):
        q = 34 + (k % 7)
        add("AAAT", 1, "GTACGT", (q, q, q, q, q, q))

    # --- 吸收链：CCCC(8) -> GCCC(3) -> GGCC(1) ---
    for k in range(8):
        q = 30 + (k % 8)
        add("CCCC", 1, "TTAAGG", (q, q, q, q, q, q))
    add("GCCC", 1, "TTAAGG", (22, 23, 24, 25, 26, 22))
    add("GCCC", 1, "TTAAGG", (24, 25, 26, 23, 22, 25))
    add("GCCC", 1, "TTAAGG", (26, 22, 23, 24, 25, 26))
    add("GGCC", 1, "TTAAGG", (18, 19, 20, 21, 22, 18))

    # --- 临界不吸收：4 不满足 >= 2*3-1=5 ---
    add("TTTT", 4, "CAGTCA", (35, 36, 34, 35, 37, 36))
    add("ATTT", 3, "CAGTCA", (20, 21, 19, 20, 22, 21))

    # --- 低质量多数 vs 高置信少数：第一位 2×Q2 的 A，1×Q40 的 C ---
    add("TATA", 1, "ACGTTT", (2, 35, 35, 35, 35, 35))
    add("TATA", 1, "ACGTTT", (2, 33, 34, 35, 33, 34))
    add("TATA", 1, "CCGTTT", (40, 35, 35, 35, 35, 35))

    # --- 严格并列：第一位 A@Q30 与 C@Q30 等似然 → N ---
    add("GTGT", 1, "ACGTAC", (30, 30, 30, 30, 30, 30))
    add("GTGT", 1, "CCGTAC", (30, 30, 30, 30, 30, 30))

    return records


def fmt_conf(values: tuple[float, ...]) -> str:
    return "[" + ", ".join("inf" if x == float("inf") else f"{x:.4f}" for x in values) + "]"


def main() -> None:
    t0 = time.perf_counter()
    records = build_records()

    counts: dict[str, int] = {}
    for umi, _, _ in records:
        counts[umi] = counts.get(umi, 0) + 1

    print("=" * 72)
    print("UMI 错误吸收与共识演示（固定输入，确定性生成，仅标准库）")
    print("=" * 72)
    print(f"读段总数 {len(records)}，不同 UMI 节点（合并前分子数）= {len(counts)}")
    print("节点计数表（计数降序/字典序）：")
    for umi, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {umi} : {n}")

    # 先用纯吸收接口展示分簇
    clusters = cluster_umis(counts)
    print(f"\n吸收后簇数（cluster_umis）= {len(clusters)}")

    molecules = collapse(records)
    print(f"collapse 输出分子数（合并后分子数）= {len(molecules)}")
    print(f"  合并前 {len(counts)}  →  合并后 {len(molecules)}"
          f"（减少 {len(counts) - len(molecules)} 个错误分子）")

    print("\n各分子明细：")
    for i, mol in enumerate(molecules, 1):
        members = ", ".join(f"{u}({n})" for u, n in mol.node_counts)
        print(f"  [{i}] 根 {mol.root}  读段 {mol.read_count}  成员[{len(mol.members)}]: {members}")
        print(f"      共识 = {mol.consensus}")
        print(f"      各位置信心差(log10) = {fmt_conf(mol.confidence)}")

    print("\n关键场景核对：")
    by_root = {m.root: m for m in molecules}
    print("  丰度桥：AAAG 同时距离 AAAA(12)、AAAT(12) 为 1")
    print(f"    -> AAAG 成员归属 = {[m for m in by_root['AAAA'].members]}")
    print(f"    -> AAAT 成员仍只有 = {list(by_root['AAAT'].members)}（两大根未被桥合并）")
    print(f"  吸收链：CCCC 簇成员 = {list(by_root['CCCC'].members)}")
    print(f"  临界不吸收：TTTT={by_root['TTTT'].read_count} 与"
          f" ATTT={by_root['ATTT'].read_count} 是两个分子")
    print(f"  低质量多数：TATA 第一位共识 = {by_root['TATA'].consensus[0]}"
          "（2×Q2 的 A 被 1×Q40 的 C 推翻）")
    print(f"  严格并列：GTGT 共识 = {by_root['GTGT'].consensus}（第一位 N，信心差 0）")

    # --- 实际触发的拒绝边界（不是预录文案，是真实抛出的异常）---
    print("\n边界：非法输入必须被拒绝并可定位")
    bad_batch = [
        ("AAAA", "ACGTAC", (35, 35, 35, 35, 35, 35)),
        ("AAAA", "ACGTAC", (35, 35, 35, 35, 35, 41)),  # Q=41 超出 0..40
    ]
    try:
        collapse(bad_batch)
    except ValidationError as exc:
        print(f"  collapse -> {type(exc).__name__}: {exc}")
        print(f"  定位：index={exc.index} field={exc.field!r}")

    try:
        consensus([])  # 空读段组无法定义共识
    except ValidationError as exc:
        print(f"  consensus([]) -> {type(exc).__name__}: {exc}")

    print(f"\n实耗计算时间 {time.perf_counter() - t0:.3f} 秒（无 sleep、无网络、无录播）。")


if __name__ == "__main__":
    main()
