"""umicorrect —— 离线 UMI 测序分子错误吸收与共识库（标准库实现）。

两层能力：

* 单端层（:func:`analyze_reads` 等）：固定长度单 UMI 的有向吸收聚类与
  质量加权对数似然共识。
* 双端层（:func:`analyze_paired_records` 等）：双端 UMI + 坐标 + 链的
  参考方向规范化与显式成对，链身份保留，成对后直接调用单端层
  :func:`consensus`。

快速入口::

    from umicorrect import Read, analyze_reads

    result = analyze_reads([
        Read("AAA", "ACG", (30, 25, 20)),
        Read("ATA", "ATG", (5, 25, 20)),
    ])
    print(result.nodes_before, result.molecules_after)
    for mol in result.molecules:
        print(mol.root_umi, mol.members, mol.consensus_seq, mol.confidence_gap)

双端快速入口::

    from umicorrect import PairedRecord, analyze_paired_records

    presult = analyze_paired_records([
        PairedRecord("r1", "AAA", "AAC", "chr1", 100, 104, "+",
                     "ACGT", (30, 30, 30, 30)),
        PairedRecord("r2", "AAA", "AAC", "chr1", 100, 104, "+",
                     "ACGA", (30, 30, 30, 7)),
    ])
    print(presult.molecules_after, presult.duplicate_count)
    for mol in presult.molecules:
        print(mol.key, mol.record_ids, mol.consensus_seq)
"""

from __future__ import annotations

from .core import (
    BASES,
    DEFAULT_PROBE_BUDGET,
    MAX_READS,
    MAX_READ_LENGTH,
    MAX_UMI_LENGTH,
    MIN_READ_LENGTH,
    MIN_UMI_LENGTH,
    AnalysisResult,
    Cluster,
    Molecule,
    Read,
    analyze_reads,
    cluster_umis,
    consensus,
)
from .errors import (
    InvalidArgumentError,
    InvalidReadError,
    LimitExceededError,
    RecordConflictError,
    SearchBudgetExhaustedError,
    UMICorrectError,
)
from .paired import (
    STRANDS,
    NormalizedRecord,
    PairedAnalysisResult,
    PairedMolecule,
    PairedRecord,
    analyze_paired_records,
    group_paired_records,
    normalize_record,
    reverse_complement,
)

__all__ = [
    "BASES",
    "DEFAULT_PROBE_BUDGET",
    "MAX_READS",
    "MAX_READ_LENGTH",
    "MAX_UMI_LENGTH",
    "MIN_READ_LENGTH",
    "MIN_UMI_LENGTH",
    "Read",
    "Cluster",
    "Molecule",
    "AnalysisResult",
    "analyze_reads",
    "cluster_umis",
    "consensus",
    "PairedRecord",
    "NormalizedRecord",
    "PairedMolecule",
    "PairedAnalysisResult",
    "STRANDS",
    "reverse_complement",
    "normalize_record",
    "group_paired_records",
    "analyze_paired_records",
    "UMICorrectError",
    "InvalidArgumentError",
    "InvalidReadError",
    "RecordConflictError",
    "LimitExceededError",
    "SearchBudgetExhaustedError",
]
