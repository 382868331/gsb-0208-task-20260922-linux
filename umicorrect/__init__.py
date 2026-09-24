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
    MAX_COORDINATE,
    MAX_READS,
    MAX_READ_LENGTH,
    MAX_UMI_LENGTH,
    MIN_COORDINATE,
    MIN_READ_LENGTH,
    MIN_UMI_LENGTH,
    STRANDS,
    AnalysisResult,
    Cluster,
    Molecule,
    Read,
    TaggedAnalysisResult,
    TaggedMolecule,
    TaggedRead,
    analyze_reads,
    analyze_tagged_reads,
    cluster_umis,
    consensus,
    reverse_complement,
)
from .errors import (
    DuplicateRecordError,
    InvalidArgumentError,
    InvalidReadError,
    LimitExceededError,
    SearchBudgetExhaustedError,
    UMICorrectError,
)

__all__ = [
    "BASES",
    "STRANDS",
    "DEFAULT_PROBE_BUDGET",
    "MAX_COORDINATE",
    "MAX_READS",
    "MAX_READ_LENGTH",
    "MAX_UMI_LENGTH",
    "MIN_COORDINATE",
    "MIN_READ_LENGTH",
    "MIN_UMI_LENGTH",
    "Read",
    "TaggedRead",
    "Cluster",
    "Molecule",
    "TaggedMolecule",
    "AnalysisResult",
    "TaggedAnalysisResult",
    "analyze_reads",
    "analyze_tagged_reads",
    "cluster_umis",
    "consensus",
    "reverse_complement",
    "UMICorrectError",
    "InvalidArgumentError",
    "InvalidReadError",
    "DuplicateRecordError",
    "LimitExceededError",
    "SearchBudgetExhaustedError",
]
