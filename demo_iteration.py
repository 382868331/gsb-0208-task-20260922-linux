#!/usr/bin/env python3
"""编辑距离吸收模型的离线演示：真实计算，无 sleep、无预录结果。"""

from __future__ import annotations

import time

from umicorrect import (
    EditDistanceOptions,
    InvalidReadError,
    Read,
    analyze_reads,
)


def main() -> None:
    reads = []
    reads.extend(
        Read("AAAA", "AAAA", (30, 30, 30, 30), (30, 30, 30, 30))
        for _ in range(10)
    )
    reads.extend(
        Read("TAAA", "TTTT", (30, 30, 30, 30), (30, 30, 30, 30))
        for _ in range(10)
    )
    reads.append(Read("AAA", "AAAA", (30, 30, 30, 30), (30, 30, 30)))
    reads.append(Read("CAAA", "CCCC", (20, 20, 20, 20), (20, 20, 20, 20)))

    options = EditDistanceOptions(max_edit_distance=2, min_log10_gap=0.5, default_quality=20)

    print("=" * 72)
    print("可选编辑距离 UMI 吸收演示（Python 标准库，离线实时计算）")
    print("=" * 72)
    t0 = time.perf_counter()
    result = analyze_reads(reads, edit_options=options)
    elapsed = time.perf_counter() - t0
    print(f"读段数={len(reads)}，去重 UMI={result.nodes_before}，最终分子={result.molecules_after}")
    print(f"计算耗时 {elapsed * 1000:.3f} ms；阈值 gap={options.min_log10_gap}")

    for assignment in result.edit_assignments:
        print(f"\nUMI={assignment.umi} 状态={assignment.state}")
        if assignment.assigned_root is not None:
            print(f"  归入根={assignment.assigned_root}，最佳/次佳 log10 概率差={assignment.best_gap:.6f}")
        for candidate in assignment.candidates:
            print(
                f"  候选 {candidate.root}: 编辑数={candidate.edit_distance}, "
                f"log10概率和={candidate.log10_probability:.6f}, 最短路径数={candidate.path_count}"
            )

    print("\n说明: AAA 相对 AAAA 有 4 个同聚体等价删除位置，路径概率求和；CAAA")
    print("      对两个高丰度候选精确并列，证据差为 0，因此保留为独立分子。")

    print("\n" + "-" * 72)
    print("[边界] UMI 长度差为 2：超过一次插入或删除，明确拒绝")
    bad = reads + [Read("A", "AAAA", (20, 20, 20, 20), (20,))]
    try:
        analyze_reads(bad, edit_options=options)
    except InvalidReadError as exc:
        print(f"  实际抛出 {type(exc).__name__}: {exc}")

    print("\n演示完成。")


if __name__ == "__main__":
    main()
