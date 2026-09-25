"""umicorrect —— 离线 UMI 测序分子错误吸收与共识库（标准库实现）。

快速入口::

    from umicorrect import Read, analyze_reads

    result = analyze_reads([
        Read("AAA", "ACG", (30, 25, 20)),
        Read("ATA", "ATG", (5, 25, 20)),
    ])
    print(result.nodes_before, result.molecules_after)
    for mol in result.molecules:
        print(mol.root_umi, mol.members, mol.consensus_seq, mol.confidence_gap)
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
    AnchoredAnalysisResult,
    AnchoredMolecule,
    AnchoredRead,
    AnalysisResult,
    Cluster,
    Molecule,
    Read,
    analyze_anchored_reads,
    analyze_reads,
    anchored_consensus,
    cluster_umis,
    consensus,
)
from .errors import (
    InvalidArgumentError,
    InvalidReadError,
    LimitExceededError,
    SearchBudgetExhaustedError,
    UMICorrectError,
)

__all__ = [
    "BASES",
    "DEFAULT_PROBE_BUDGET",
    "MAX_READS",
    "MAX_READ_LENGTH",
    "MAX_UMI_LENGTH",
    "MIN_READ_LENGTH",
    "MIN_UMI_LENGTH",
    "AnchoredRead",
    "AnchoredMolecule",
    "AnchoredAnalysisResult",
    "Read",
    "Cluster",
    "Molecule",
    "AnalysisResult",
    "analyze_anchored_reads",
    "analyze_reads",
    "anchored_consensus",
    "cluster_umis",
    "consensus",
    "UMICorrectError",
    "InvalidArgumentError",
    "InvalidReadError",
    "LimitExceededError",
    "SearchBudgetExhaustedError",
]
