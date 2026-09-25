"""带显式参考起点的读段模式：变长读段按参考坐标汇聚的共识。

与 umicorrect.core 的等长模式互补：

* 每条读段自带参考起点 start（0 基参考坐标），读段长度可以不同；
* 不做插缺比对：读段按声明的起点直接放置到参考坐标上，碱基只在
  自己覆盖的坐标投票，多条短读段不会把长读段的位置“顶错位”；
* UMI 吸收规则与等长模式完全一致（同一套 cluster_umis），同簇读段
  仍来自同一参考区间（簇输出区间为成员覆盖坐标的最小到最大值）；
* 簇输出 [ref_start, ref_end) 的共识、逐位置覆盖数与信心差；
  未被任何读段覆盖的坐标（空洞）没有投票，输出 'N' 且信心差为缺失
  （None）——空洞不是一条低质量观测，与 Q=0 观测严格区分；
* Q=0 与并列规则沿用等长模式的原对数似然契约（并列输出 'N'、
  gap=0.0；Q=0 呼中碱基似然为 -inf）。

公开接口
--------
PositionedRead              一条带参考起点的读段（UMI + start + DNA + Q）。
PositionedMolecule          一个定位共识分子（含逐位置覆盖数）。
PositionedAnalysisResult    批次结果。
analyze_positioned_reads    一步完成：校验 → 计数 → 吸收 → 坐标共识。
positioned_consensus        只做一簇的坐标汇聚共识。

兼容性：全部读段 start=0 且等长时，共识与信心差和
umicorrect.analyze_reads 完全一致（空洞不可能出现）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import (
    BASES,
    DEFAULT_PROBE_BUDGET,
    MAX_READ_LENGTH,
    MAX_READS,
    MAX_UMI_LENGTH,
    MIN_READ_LENGTH,
    MIN_UMI_LENGTH,
    _BASE_INDEX,
    _LL_TABLE,
    _reject_nonfinite,
    cluster_umis,
)
from .errors import (
    InvalidArgumentError,
    InvalidReadError,
    LimitExceededError,
)

__all__ = [
    "MAX_REF_START",
    "PositionedRead",
    "PositionedMolecule",
    "PositionedAnalysisResult",
    "positioned_consensus",
    "analyze_positioned_reads",
]

# 参考起点上限（接口边界；防止无界坐标导致内存失控）
MAX_REF_START = 10_000_000


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PositionedRead:
    """一条带显式参考起点的读段。

    umi 为仅含 A/C/G/T 的非空 DNA 串（批次内定长）；
    start 为 0 基参考起点（非负整数，拒绝 bool/浮点/NaN/Infinity）；
    sequence / qualities 等长，规则与 umicorrect.Read 相同，
    但不要求与 UMI 等长，也不要求批次内等长。
    """

    umi: str
    start: int
    sequence: str
    qualities: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.qualities, list):
            object.__setattr__(self, "qualities", tuple(self.qualities))
        _check_positioned_fields(self)


@dataclass(frozen=True)
class PositionedMolecule:
    """一个定位共识分子。

    Attributes:
        root_umi:      簇根 UMI（首个最高节点）。
        members:       成员 UMI 元组（含根，字典序升序）。
        read_indices:  成员读段在输入批次中的下标（升序）。
        ref_start:     簇输出区间的起始参考坐标（成员最小覆盖坐标）。
        consensus_seq: [ref_start, ref_start+len) 的共识序列；
                       空洞（覆盖数为 0）与并列位置均为 'N'。
        confidence_gap: 逐位置 (最高 ll - 次高 ll)，log10 域；并列位置
                       为 0.0；空洞位置为 None（缺失，不是 0.0）。
        coverage:      逐位置覆盖读段数；空洞位置为 0。
    """

    root_umi: str
    members: tuple[str, ...]
    read_indices: tuple[int, ...]
    ref_start: int
    consensus_seq: str
    confidence_gap: tuple[float | None, ...]
    coverage: tuple[int, ...]


@dataclass(frozen=True)
class PositionedAnalysisResult:
    """定位批次分析结果。molecules 按 (计数降序, 根 UMI 字典序) 排列。"""

    molecules: tuple[PositionedMolecule, ...]
    umi_length: int
    nodes_before: int
    molecules_after: int

    @property
    def merged_count(self) -> int:
        """吸收操作合并掉的 UMI 种数。"""
        return self.nodes_before - self.molecules_after


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------
def _check_start(start) -> int:
    """参考起点：非负整数；拒绝 bool、浮点（NaN/Infinity 专门报错）。"""
    if isinstance(start, bool):
        raise InvalidArgumentError(
            f"start 必须是非负整数且不能是 bool，得到 {start!r}"
        )
    if isinstance(start, float):
        _reject_nonfinite("start", start)  # NaN/Infinity 在此抛出专门错误
        raise InvalidArgumentError(f"start 必须是整数，得到 float {start!r}")
    if not isinstance(start, int):
        raise InvalidArgumentError(
            f"start 必须是非负整数，得到 {type(start).__name__}"
        )
    if start < 0:
        raise InvalidArgumentError(f"start = {start} 为负：参考起点必须 >= 0")
    if start > MAX_REF_START:
        raise LimitExceededError(f"start {start} 超过上限 {MAX_REF_START}")
    return start


def _check_positioned_fields(read: PositionedRead) -> None:
    """单条定位读段的字段级校验（构造与批次校验共用）。"""
    umi = read.umi
    seq = read.sequence
    qual = read.qualities

    if not isinstance(umi, str):
        raise InvalidArgumentError(f"umi 必须是 str，得到 {type(umi).__name__}")
    if not isinstance(seq, str):
        raise InvalidArgumentError(f"sequence 必须是 str，得到 {type(seq).__name__}")
    if not isinstance(qual, tuple):
        raise InvalidArgumentError(
            f"qualities 必须是 tuple（或 list），得到 {type(qual).__name__}"
        )

    if not umi:
        raise InvalidArgumentError(f"umi 为空串：UMI 长度必须 >= {MIN_UMI_LENGTH}")
    if not seq:
        raise InvalidArgumentError(
            f"sequence 为空串：读段长度必须 >= {MIN_READ_LENGTH}"
        )
    if len(umi) > MAX_UMI_LENGTH:
        raise LimitExceededError(f"umi 长度 {len(umi)} 超过上限 {MAX_UMI_LENGTH}")
    if len(seq) > MAX_READ_LENGTH:
        raise LimitExceededError(
            f"sequence 长度 {len(seq)} 超过上限 {MAX_READ_LENGTH}"
        )
    if len(qual) != len(seq):
        raise InvalidArgumentError(
            f"qualities 长度 {len(qual)} 与序列长度 {len(seq)} 不一致"
        )

    _check_start(read.start)
    if read.start + len(seq) - 1 > MAX_REF_START:
        raise LimitExceededError(
            f"读段末端坐标 {read.start + len(seq) - 1} 超过上限 {MAX_REF_START}"
        )

    for j, ch in enumerate(umi):
        if ch not in _BASE_INDEX:
            raise InvalidArgumentError(
                f"umi[{j}] = {ch!r} 非法，仅允许 A/C/G/T（拒绝 N/小写）"
            )
    for j, ch in enumerate(seq):
        if ch not in _BASE_INDEX:
            raise InvalidArgumentError(
                f"sequence[{j}] = {ch!r} 非法，仅允许 A/C/G/T（拒绝 N/小写）"
            )
    for j, q in enumerate(qual):
        if isinstance(q, bool) or not isinstance(q, int):
            if isinstance(q, float) and not isinstance(q, bool):
                _reject_nonfinite(f"qualities[{j}]", q)
            raise InvalidArgumentError(
                f"qualities[{j}] 必须是 0..40 的整数且不能是 bool，"
                f"得到 {type(q).__name__}"
            )
        if q < 0 or q > 40:
            raise InvalidArgumentError(f"qualities[{j}] = {q} 越界，允许范围 0..40")


def _validate_positioned_batch(reads) -> int:
    """整批校验（先全部校验、后构建，失败不留部分状态）。返回固定 UMI 长度。"""
    if not isinstance(reads, (list, tuple)):
        raise InvalidArgumentError(
            f"reads 必须是 list/tuple，得到 {type(reads).__name__}"
        )
    if len(reads) > MAX_READS:
        raise LimitExceededError(f"读段数 {len(reads)} 超过上限 {MAX_READS}")
    if len(reads) == 0:
        raise InvalidArgumentError(
            "reads 为空批次：无法确定固定 UMI 长度（空批次被拒绝）"
        )

    fixed_umi_len: int | None = None
    for i, read in enumerate(reads):
        if not isinstance(read, PositionedRead):
            raise InvalidReadError(
                f"reads[{i}] 必须是 PositionedRead，得到 {type(read).__name__}"
            )
        try:
            _check_positioned_fields(read)
        except InvalidArgumentError as exc:
            raise InvalidReadError(f"reads[{i}]: {exc}") from exc
        if fixed_umi_len is None:
            fixed_umi_len = len(read.umi)
        elif len(read.umi) != fixed_umi_len:
            raise InvalidReadError(
                f"reads[{i}].umi 长度 {len(read.umi)} 与批次固定 UMI 长度 "
                f"{fixed_umi_len} 不一致"
            )
    assert fixed_umi_len is not None
    return fixed_umi_len


# ---------------------------------------------------------------------------
# 坐标汇聚共识
# ---------------------------------------------------------------------------
def positioned_consensus(
    reads,
    read_indices,
) -> tuple[int, str, tuple[float | None, ...], tuple[int, ...]]:
    """对一簇定位读段按参考坐标计算对数似然共识。

    Args:
        reads: 批次读段序列（元素为 PositionedRead）。
        read_indices: 参与本簇共识的读段下标；非空。

    Returns:
        (ref_start, 共识序列, 逐位置信心差, 逐位置覆盖数)。
        空洞（覆盖数 0）位置共识为 'N'、信心差为 None；
        并列位置共识为 'N'、信心差为 0.0（沿用原似然规则）。
    """
    if read_indices is None or len(read_indices) == 0:
        raise InvalidArgumentError("共识簇至少需要 1 条读段（空簇被拒绝）")

    idxs: list[int] = []
    for i in read_indices:
        if isinstance(i, bool) or not isinstance(i, int):
            raise InvalidArgumentError("read_indices 必须全为 int 且不能是 bool")
        if i < 0 or i >= len(reads):
            raise InvalidArgumentError(
                f"read_indices 含越界下标 {i}（批次共 {len(reads)} 条）"
            )
        idxs.append(i)
    idxs.sort()  # 有序累加，保证求和顺序确定、与输入顺序无关

    ref_start = min(reads[i].start for i in idxs)
    ref_end = max(reads[i].start + len(reads[i].sequence) for i in idxs)
    n = ref_end - ref_start

    ll = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]
    coverage = [0] * n
    for i in idxs:
        read = reads[i]
        offset = read.start - ref_start
        seq = read.sequence
        qual = read.qualities
        for j in range(len(seq)):
            pos = offset + j
            coverage[pos] += 1
            b = _BASE_INDEX[seq[j]]
            log_correct, log_wrong = _LL_TABLE[qual[j]]
            row = ll[pos]
            for k in range(4):
                row[k] += log_correct if k == b else log_wrong

    out_bases: list[str] = []
    gaps: list[float | None] = []
    for pos in range(n):
        if coverage[pos] == 0:
            # 空洞：没有任何投票，不是低质量观测；信心差缺失
            out_bases.append("N")
            gaps.append(None)
            continue
        row = ll[pos]
        best = max(range(4), key=lambda k: row[k])
        best_val = row[best]
        tied = [k for k in range(4) if k != best and row[k] == best_val]
        if tied:
            out_bases.append("N")
            gaps.append(0.0)
        else:
            second = max(row[k] for k in range(4) if k != best)
            out_bases.append(BASES[best])
            gaps.append(best_val - second)
    return ref_start, "".join(out_bases), tuple(gaps), tuple(coverage)


# ---------------------------------------------------------------------------
# 顶层管线
# ---------------------------------------------------------------------------
def analyze_positioned_reads(
    reads,
    *,
    probe_budget: int = DEFAULT_PROBE_BUDGET,
) -> PositionedAnalysisResult:
    """对整批定位读段执行 校验 → UMI 计数 → 吸收聚类 → 坐标汇聚共识。

    校验先于任何状态构建并在整个批次上进行；任一读段非法则整批拒绝。
    UMI 吸收规则与等长模式完全一致；簇内按参考坐标汇聚，不做插缺比对。

    Returns:
        PositionedAnalysisResult。
    """
    umi_len = _validate_positioned_batch(reads)

    groups: dict[str, list[int]] = {}
    for i, read in enumerate(reads):
        groups.setdefault(read.umi, []).append(i)
    umi_counts = {u: len(v) for u, v in groups.items()}

    clusters = cluster_umis(umi_counts, probe_budget=probe_budget)

    molecules: list[PositionedMolecule] = []
    for cl in clusters:
        idxs = sorted(i for u in cl.members for i in groups[u])
        ref_start, seq, gaps, cov = positioned_consensus(reads, idxs)
        molecules.append(
            PositionedMolecule(
                root_umi=cl.root,
                members=cl.members,
                read_indices=tuple(idxs),
                ref_start=ref_start,
                consensus_seq=seq,
                confidence_gap=gaps,
                coverage=cov,
            )
        )

    molecules.sort(key=lambda m: (-len(m.read_indices), m.root_umi))
    return PositionedAnalysisResult(
        molecules=tuple(molecules),
        umi_length=umi_len,
        nodes_before=len(umi_counts),
        molecules_after=len(molecules),
    )
