"""umi_consensus —— UMI 测序分子的错误吸收与共识（离线、仅标准库）。

公开接口见模块 __all__ 与 README。核心做两件事：

1. 错误吸收（有向、按计数降序+字典序的在线归属）：
   节点按 count 降序、再按 UMI 字典序排列；处理节点 u 时，在它之前
   （即“大到小”方向）且 Hamming 距离为 1 的节点 v 中，若
   ``count[v] >= 2*count[u] - 1``，则 u 可被 v 吸收；u 归属其中排序最靠前
   （计数最高、字典序最小）的节点所在的根，即“首个最高根”。低丰度桥只能
   加入一个根，不会把两个大根合并。

2. 簇内逐碱基对数似然共识：
   独立错误率 p = 10**(-Q/10)；P(观测=c | 真值=b) = 1-p（b==c）或 p/3。
   每位置对 A/C/G/T 累加 log10 似然；唯一最高者为共识碱基，最高并列输出 N；
   信心差 = 最高与次高 log10 似然之差。

本模块只实现题目契约的有限子集，不宣称兼容任何行业标准 UMI 工具。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

__all__ = [
    "BASES",
    "LIMITS",
    "Read",
    "Record",
    "Cluster",
    "Molecule",
    "ValidationError",
    "cluster_umis",
    "consensus",
    "collapse",
]

# ---------------------------------------------------------------------------
# 公开常量与数据结构
# ---------------------------------------------------------------------------

BASES: tuple[str, ...] = ("A", "C", "G", "T")
_BASE_SET = frozenset(BASES)

#: 接口边界（题目上限）。短演示不需要达到这些规模。
LIMITS: Mapping[str, int] = {
    "umi_length_min": 1,
    "umi_length_max": 16,
    "read_length_min": 1,
    "read_length_max": 500,
    "max_records": 100_000,  # collapse 单次批处理读段条数上限
    "max_nodes": 100_000,  # cluster_umis 不同 UMI 节点数上限
    "max_count": 100_000,  # 单个 UMI 计数上限
    "max_quality": 40,
}

_NEG_INF = float("-inf")


class ValidationError(ValueError):
    """非法输入错误。

    所有非法输入都抛出本异常（不静默纠正），并尽量给出可定位信息：

    * ``index``：批次中第几条记录（0 起）；计数映射场景为出问题的 UMI 键；
    * ``field``：出错字段，如 ``"umi"``、``"sequence"``、``"qualities[3]"``；
    * ``message``：不含定位前缀的原始说明。
    """

    def __init__(
        self,
        message: str,
        *,
        index: int | str | None = None,
        field: str | None = None,
    ) -> None:

        self.message = message
        self.index = index
        self.field = field
        location = self._location(index, field)
        super().__init__(f"{location}: {message}" if location else message)

    @staticmethod
    def _location(index: int | str | None, field: str | None) -> str:
        parts: list[str] = []
        if index is not None:
            parts.append(f"record[{index}]" if isinstance(index, int) else f"key[{index!r}]")
        if field is not None:
            parts.append(field)
        return " ".join(parts)


@dataclass(frozen=True)
class Read:
    """一条等长 DNA 读段。

    ``sequence`` 只含 A/C/G/T（小写或 N 一律拒绝）；``qualities`` 为与序列
    等长的整数 Phred 质量，每位 0..40。
    """

    sequence: str
    qualities: tuple[int, ...]


@dataclass(frozen=True)
class Record:
    """一条带定长 UMI 的输入记录，字段顺序为 ``(umi, sequence, qualities)``。"""

    umi: str
    sequence: str
    qualities: tuple[int, ...]


@dataclass(frozen=True)
class Cluster:
    """UMI 吸收结果：根、全部成员（含根，按节点排序）及每个成员的计数。"""

    root: str
    members: tuple[str, ...]
    node_counts: tuple[tuple[str, int], ...]
    read_count: int


@dataclass(frozen=True)
class Molecule:
    """一个共识分子。

    ``confidence`` 为每位置最高与次高 log10 似然之差；并列时为 0.0；
    若只有一个碱基似然非零（Q=0 极端情形），该位为 +inf。
    ``scores`` 为每位置 A/C/G/T 的 log10 似然（可能含 -inf，不含 NaN）。
    """

    root: str
    members: tuple[str, ...]
    node_counts: tuple[tuple[str, int], ...]
    read_count: int
    consensus: str
    confidence: tuple[float, ...]
    scores: tuple[dict[str, float], ...]


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _is_real_int(value: Any) -> bool:
    """整数参数拒绝 bool（bool 是 int 子类，必须显式排除）。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _check_dna(value: Any, field_name: str, index: int | str | None) -> str:
    if not isinstance(value, str):
        raise ValidationError(
            f"{field_name} 必须是 str，实际为 {type(value).__name__}",
            index=index,
            field=field_name,
        )
    if len(value) == 0:
        raise ValidationError(f"{field_name} 不允许为空字符串", index=index, field=field_name)
    for pos, ch in enumerate(value):
        if ch not in _BASE_SET:  # 小写、N、其他字符一律拒绝，不静默纠正
            raise ValidationError(
                f"{field_name}[{pos}] 字符 {ch!r} 非法，只允许 A/C/G/T",
                index=index,
                field=f"{field_name}[{pos}]",
            )
    return value


def _check_length(
    length: int,
    field_name: str,
    index: int | str | None,
    *,
    minimum: int,
    maximum: int,
) -> None:
    if length < minimum:
        raise ValidationError(
            f"{field_name} 长度 {length} 小于下限 {minimum}", index=index, field=field_name
        )
    if length > maximum:
        raise ValidationError(
            f"{field_name} 长度 {length} 超过上限 {maximum}", index=index, field=field_name
        )


def _check_qualities(value: Any, length: int, *, index: int | str | None) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValidationError(
            f"qualities 必须是 list/tuple 整数，实际为 {type(value).__name__}",
            index=index,
            field="qualities",
        )
    for pos, q in enumerate(value):
        # 拒绝 bool；拒绝浮点（含 NaN/Infinity——质量位按契约只接受整数）
        if not _is_real_int(q):
            kind = "bool" if isinstance(q, bool) else type(q).__name__
            raise ValidationError(
                f"qualities[{pos}] 必须是 0..40 的整数，实际为 {kind}: {q!r}",
                index=index,
                field=f"qualities[{pos}]",
            )
        if not 0 <= q <= LIMITS["max_quality"]:
            raise ValidationError(
                f"qualities[{pos}] = {q} 超出范围 0..40",
                index=index,
                field=f"qualities[{pos}]",
            )
    if len(value) != length:
        raise ValidationError(
            f"qualities 长度 {len(value)} 与 sequence 长度 {length} 不一致",
            index=index,
            field="qualities",
        )
    return tuple(value)


def _as_record(item: Any, index: int) -> Record:
    if isinstance(item, Record):
        umi, sequence, qualities = item.umi, item.sequence, item.qualities
    elif isinstance(item, tuple) and len(item) == 3:
        umi, sequence, qualities = item
    else:
        raise ValidationError(
            "每条记录必须是 Record 或 (umi, sequence, qualities) 三元组",
            index=index,
        )
    umi = _check_dna(umi, "umi", index)
    sequence = _check_dna(sequence, "sequence", index)
    qualities = _check_qualities(qualities, len(sequence), index=index)
    return Record(umi=umi, sequence=sequence, qualities=qualities)


def _normalize_records(records: Any) -> list[Record]:
    if isinstance(records, (str, bytes, Record, Read)) or not hasattr(records, "__iter__"):
        raise ValidationError(
            f"records 必须是可迭代记录对象，实际为 {type(records).__name__}",
            field="records",
        )
    normalized: list[Record] = []
    umi_length: int | None = None
    read_length: int | None = None
    for index, item in enumerate(records):
        if index >= LIMITS["max_records"]:
            raise ValidationError(
                f"批处理读段数超过上限 {LIMITS['max_records']}",
                index=index,
                field="records",
            )
        record = _as_record(item, index)
        if umi_length is None:
            _check_length(
                len(record.umi), "umi", index,
                minimum=LIMITS["umi_length_min"], maximum=LIMITS["umi_length_max"],
            )
            _check_length(
                len(record.sequence), "sequence", index,
                minimum=LIMITS["read_length_min"], maximum=LIMITS["read_length_max"],
            )
            umi_length = len(record.umi)
            read_length = len(record.sequence)
        else:
            if len(record.umi) != umi_length:
                raise ValidationError(
                    f"UMI 长度 {len(record.umi)} 与首条记录 {umi_length} 不一致",
                    index=index, field="umi",
                )
            if len(record.sequence) != read_length:
                raise ValidationError(
                    f"读段长度 {len(record.sequence)} 与首条记录 {read_length} 不一致",
                    index=index, field="sequence",
                )
        normalized.append(record)
    return normalized


def _normalize_reads(reads: Any) -> list[Read]:
    if isinstance(reads, (str, bytes)) or not hasattr(reads, "__iter__"):
        raise ValidationError(
            f"reads 必须是可迭代读段对象，实际为 {type(reads).__name__}", field="reads"
        )
    result: list[Read] = []
    length: int | None = None
    for index, item in enumerate(reads):
        if isinstance(item, (Read, Record)):
            sequence, qualities = item.sequence, item.qualities
        elif isinstance(item, tuple) and len(item) == 2:
            sequence, qualities = item
        else:
            raise ValidationError(
                "每条读段必须是 Read/Record 或 (sequence, qualities) 二元组",
                index=index,
            )
        sequence = _check_dna(sequence, "sequence", index)
        qualities = _check_qualities(qualities, len(sequence), index=index)
        _check_length(
            len(sequence), "sequence", index,
            minimum=LIMITS["read_length_min"], maximum=LIMITS["read_length_max"],
        )
        if length is not None and len(sequence) != length:
            raise ValidationError(
                f"读段长度 {len(sequence)} 与首条 {length} 不一致",
                index=index, field="sequence",
            )
        length = len(sequence)
        result.append(Read(sequence=sequence, qualities=qualities))
    if not result:
        raise ValidationError("至少需要一条读段才能构建共识", field="reads")
    return result


# ---------------------------------------------------------------------------
# 1) UMI 错误吸收
# ---------------------------------------------------------------------------


def _hamming1_variants(umi: str) -> Iterator[str]:
    """枚举与 umi 的 Hamming 距离恰为 1 的全部 3*L 个字符串（穷举、无遗漏）。"""
    for pos, current in enumerate(umi):
        prefix, suffix = umi[:pos], umi[pos + 1 :]
        for base in BASES:
            if base != current:
                yield prefix + base + suffix


def _validate_counts(counts: Any) -> dict[str, int]:
    if not isinstance(counts, Mapping):
        raise ValidationError(
            f"counts 必须是 Mapping[str, int]，实际为 {type(counts).__name__}",
            field="counts",
        )
    if len(counts) > LIMITS["max_nodes"]:
        raise ValidationError(
            f"UMI 节点数 {len(counts)} 超过上限 {LIMITS['max_nodes']}", field="counts"
        )
    length: int | None = None
    cleaned: dict[str, int] = {}
    for umi, count in counts.items():
        umi = _check_dna(umi, "umi", umi)
        _check_length(
            len(umi), "umi", umi,
            minimum=LIMITS["umi_length_min"], maximum=LIMITS["umi_length_max"],
        )
        if length is None:
            length = len(umi)
        elif len(umi) != length:
            raise ValidationError(
                f"UMI 长度 {len(umi)} 与首个键的长度 {length} 不一致",
                index=umi, field="umi",
            )
        if not _is_real_int(count):
            kind = "bool" if isinstance(count, bool) else type(count).__name__
            raise ValidationError(
                f"计数必须是正整数，实际为 {kind}: {count!r}", index=umi, field="count"
            )
        if not 1 <= count <= LIMITS["max_count"]:
            raise ValidationError(
                f"计数 {count} 不在 1..{LIMITS['max_count']} 范围内",
                index=umi, field="count",
            )
        cleaned[umi] = count
    return cleaned


def cluster_umis(counts: Mapping[str, int]) -> tuple[Cluster, ...]:
    """按题目吸收规则把 UMI 计数表聚成簇。

    纯函数：输入不被修改，校验失败时不会产生任何部分结果。

    邻接关系通过枚举每个节点的 3*L 个 Hamming-1 变体并查计数表得到，
    是穷举精确的（不是启发式搜索），因此不存在“搜索预算耗尽”；
    每个节点必然自成根或归入更早的根，也不存在“证明无解”。
    """
    cleaned = _validate_counts(counts)
    if not cleaned:
        return ()

    # 计数降序，再字典序升序——这是唯一的处理顺序。
    items = sorted(cleaned.items(), key=lambda kv: (-kv[1], kv[0]))
    position = {umi: i for i, (umi, _) in enumerate(items)}

    root_of: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    roots_in_order: list[str] = []

    for i, (umi, count) in enumerate(items):
        parent_pos: int | None = None
        # 只在“大到小”方向找可吸收自己的节点：计数必须严格更大
        # （相等计数互不为“大节点”，相等根保持独立），Hamming 距离为 1，
        # 且满足 count[v] >= 2*count[u]-1；取排序最靠前的一个——首个最高根。
        for variant in _hamming1_variants(umi):
            j = position.get(variant)
            if j is None or j >= i:
                continue
            if items[j][1] > count and items[j][1] >= 2 * count - 1:
                if parent_pos is None or j < parent_pos:
                    parent_pos = j
        if parent_pos is None:
            root = umi
            roots_in_order.append(root)
        else:
            root = root_of[items[parent_pos][0]]
        root_of[umi] = root
        groups.setdefault(root, []).append(umi)

    clusters: list[Cluster] = []
    for root in roots_in_order:
        members = tuple(groups[root])  # 组成顺序即节点排序，根在第一位
        node_counts = tuple((member, cleaned[member]) for member in members)
        clusters.append(
            Cluster(
                root=root,
                members=members,
                node_counts=node_counts,
                read_count=sum(c for _, c in node_counts),
            )
        )
    return tuple(clusters)


# ---------------------------------------------------------------------------
# 2) 簇内对数似然共识
# ---------------------------------------------------------------------------


def _build_consensus(
    reads: Sequence[Read],
) -> tuple[str, tuple[float, ...], tuple[dict[str, float], ...]]:
    length = len(reads[0].sequence)
    base_index = {base: i for i, base in enumerate(BASES)}
    max_q = LIMITS["max_quality"]

    # 精确聚合：stats[pos][called_base][quality] = 条数（整数计数，无浮点误差）。
    # 似然只取决于各 (碱基, 质量) 的计数，因此结果与读段输入顺序无关。
    stats: list[list[list[int]]] = [
        [[0] * (max_q + 1) for _ in BASES] for _ in range(length)
    ]
    for read in reads:
        for pos, (called, quality) in enumerate(zip(read.sequence, read.qualities)):
            stats[pos][base_index[called]][quality] += 1

    match_ll: list[float] = [_NEG_INF]  # Q=0：p=1，与观测一致的真值似然恰为 0
    mismatch_ll: list[float] = [math.log10(1.0 / 3.0)]  # Q=0：log10(1/3)
    for q in range(1, max_q + 1):
        p_error = 10.0 ** (-q / 10.0)
        match_ll.append(math.log10(1.0 - p_error))
        mismatch_ll.append(math.log10(p_error / 3.0))

    consensus_bases: list[str] = []
    confidence: list[float] = []
    score_tables: list[dict[str, float]] = []
    for pos in range(length):
        totals = [0] * (max_q + 1)
        for column in stats[pos]:
            for q, c in enumerate(column):
                totals[q] += c  # 该位置质量为 q 的读段总数

        table: dict[str, float] = {}
        for bi, base in enumerate(BASES):
            # 每位置每碱基至多 82 个求和项；fsum 正确舍入，故数学上相等的
            # 似然多集会得到逐位相同的浮点数，并列判定可用精确 ==。
            terms: list[float] = []
            column = stats[pos][bi]
            for q in range(max_q + 1):
                n_q = totals[q]
                if n_q == 0:
                    continue
                n_match = column[q]
                if n_match:
                    terms.append(match_ll[q] * n_match)
                n_mismatch = n_q - n_match
                if n_mismatch:
                    terms.append(mismatch_ll[q] * n_mismatch)
            table[base] = math.fsum(terms) if terms else _NEG_INF

        values = [table[base] for base in BASES]
        best = max(values)
        winners = [base for base in BASES if table[base] == best]  # 精确判等
        ordered = sorted(values, reverse=True)
        if len(winners) > 1:
            # 最高并列 → N；包括四个似然都为 -inf（Q=0 且四种碱基都被测到）。
            consensus_bases.append("N")
            confidence.append(0.0)
        else:
            consensus_bases.append(winners[0])
            second = ordered[1]
            # best 有限、second == -inf 时按 IEEE-754 得 +inf（唯一非零似然）。
            confidence.append(best - second)
        score_tables.append(table)
    return "".join(consensus_bases), tuple(confidence), tuple(score_tables)


def consensus(
    reads: Any,
) -> tuple[str, tuple[float, ...], tuple[dict[str, float], ...]]:
    """对一组等长读段做逐碱基对数似然共识。

    返回 ``(共识序列, 每位置信心差, 每位置四碱基 log10 似然)``。
    共识碱基为 A/C/G/T，最高并列或全零似然时为 N。读段为空抛 ValidationError。
    """
    return _build_consensus(_normalize_reads(reads))


# ---------------------------------------------------------------------------
# 3) 一站式：吸收 + 共识
# ---------------------------------------------------------------------------


def collapse(records: Any) -> tuple[Molecule, ...]:
    """从 ``(umi, sequence, qualities)`` 记录直接得到共识分子列表。

    先完整校验全部输入再计算（纯函数，失败无任何部分变更）；分子按其根在
    节点排序中的先后顺序返回，成员按节点序排列，成员内读段按输入序参与累加。
    空批次返回空元组。
    """
    normalized = _normalize_records(records)
    if not normalized:
        return ()

    counts: dict[str, int] = {}
    reads_by_umi: dict[str, list[Read]] = {}
    for record in normalized:
        counts[record.umi] = counts.get(record.umi, 0) + 1
        reads_by_umi.setdefault(record.umi, []).append(
            Read(record.sequence, record.qualities)
        )

    molecules: list[Molecule] = []
    for cluster in cluster_umis(counts):
        cluster_reads: list[Read] = []
        for umi in cluster.members:
            cluster_reads.extend(reads_by_umi[umi])
        seq, conf, scores = _build_consensus(cluster_reads)
        molecules.append(
            Molecule(
                root=cluster.root,
                members=cluster.members,
                node_counts=cluster.node_counts,
                read_count=cluster.read_count,
                consensus=seq,
                confidence=conf,
                scores=scores,
            )
        )
    return tuple(molecules)
