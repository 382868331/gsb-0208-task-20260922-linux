"""双端 UMI、基因组坐标与链方向的规范化与显式成对（本次迭代新增）。

本模块在既有单端管线（:mod:`umicorrect.core` 的有向吸收聚类与质量加权
对数似然共识）之上，提供面向“双端 UMI + 坐标 + 链”记录的配对层：

* :class:`PairedRecord` —— 一条带双端 UMI、比对坐标、链方向、序列与
  逐碱基质量的记录（``record_id`` 在批次内必须唯一）。
* :func:`normalize_record` —— 把记录规范化为 :class:`NormalizedRecord`：
  坐标半开区间校验与升序规整、链定向双 UMI、参考方向序列/质量。
* :func:`group_paired_records` —— 按**显式匹配规则**把记录成对为
  :class:`PairedMolecule`：匹配键 = (contig, start, end, strand, 左UMI,
  右UMI)，全部相等才是同一分子的重复观测。
* :func:`analyze_paired_records` —— 配对后直接调用旧共识入口
  :func:`umicorrect.core.consensus` 产出每组的共识序列与逐位置信心差。

链身份与反向互补
----------------
UMI/序列的“参考方向化”只沿记录声明的链做一次：

* ``+`` 链：左 UMI = ``umi1``，右 UMI = ``umi2``，序列保持 5'→3'；
* ``-`` 链：双 UMI **交换并各自反向互补**，左 UMI = ``rc(umi2)``，
  右 UMI = ``rc(umi1)``；序列反向互补、质量序列反转。

**链（strand）本身是匹配键的一部分，规范化不会把两条链折叠到一起。**
因此“同一对 UMI 经反向互补后恰好相同、但观测自相反链”的两条记录键
并不相等（strand 不同），它们被判为**两个不同分子**，绝不合并——
即便 ``rc`` 后的双 UMI 串字面相等（测试覆盖此情形）。

重复与冲突
----------
* **分子级重复**（合法）：不同 ``record_id``、规范化匹配键相同——同一
  原始分子的多条 PCR/测序观测，进入同一 :class:`PairedMolecule`，并由
  旧共识入口做逐碱基质量加权对数似然共识。
* **记录级重复**（非法）：同一 ``record_id`` 在批次中出现两次——库无法
  判断是“重发同一观测”还是“两条不同观测误用同号”，一律抛
  :class:`~umicorrect.errors.RecordConflictError`，不静默去重、不合并。

原子性
------
所有校验（字段、坐标、链、ID 唯一性）在任何分组结果构建之前对整批
完成；任一记录非法即整批拒绝。本模块是无状态纯函数，不存在半更新状态。
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import (
    MAX_READS,
    MAX_READ_LENGTH,
    MAX_UMI_LENGTH,
    MIN_READ_LENGTH,
    MIN_UMI_LENGTH,
    _BASE_INDEX,
    _reject_nonfinite,
    consensus,
)
from .errors import (
    InvalidArgumentError,
    InvalidReadError,
    LimitExceededError,
    RecordConflictError,
)

__all__ = [
    "STRANDS",
    "PairedRecord",
    "NormalizedRecord",
    "PairedMolecule",
    "PairedAnalysisResult",
    "reverse_complement",
    "normalize_record",
    "group_paired_records",
    "analyze_paired_records",
]

STRANDS = ("+", "-")
_REV = {"A": "T", "C": "G", "G": "C", "T": "A"}


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PairedRecord:
    """一条双端 UMI + 坐标 + 链的记录。

    Attributes:
        record_id: 记录唯一标识（非空 str）；同批次重复出现即冲突。
        umi1 / umi2: 双端 UMI（读段 1 / 读段 2 上按各自 5'→3' 给出），
            仅含 A/C/G/T，长度 1..32，允许两端等长或不等长。
        contig: 比对参考序列名（非空 str）。
        start / end: 0-based 半开基因组坐标 ``[start, end)``，要求
            ``0 <= start < end``；允许以 ``end < start`` 的反向顺序传入，
            规范化时升序规整为 ``(min, max)``（不丢失区间身份，仅排序端点）。
        strand: 链方向，``"+"`` 或 ``"-"``。
        sequence: 该记录参与共识的 DNA 序列（仅 A/C/G/T，长度 1..1000），
            按记录自身 5'→3' 给出；``-`` 链记录在规范化时反向互补到参考
            方向。
        qualities: 逐碱基 Phred 质量（整数 0..40），与 ``sequence`` 等长。
    """

    record_id: str
    umi1: str
    umi2: str
    contig: str
    start: int
    end: int
    strand: str
    sequence: str
    qualities: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.qualities, list):
            object.__setattr__(self, "qualities", tuple(self.qualities))
        _check_paired_fields(self)


@dataclass(frozen=True)
class NormalizedRecord:
    """规范化后的记录：坐标升序、UMI/序列统一到参考（``+``）方向。

    ``strand`` 被原样保留——方向化只翻转序列表示，不改变链身份。
    """

    record_id: str
    umi_left: str
    umi_right: str
    contig: str
    start: int
    end: int
    strand: str
    sequence: str
    qualities: tuple[int, ...]
    index: int


@dataclass(frozen=True)
class PairedMolecule:
    """一个匹配键下的成对分子（同一原始分子的全部重复观测）。

    Attributes:
        key: 匹配键 ``(contig, start, end, strand, umi_left, umi_right)``。
        record_ids: 组内记录 ID（按批次下标升序）。
        record_indices: 组内记录在输入批次中的下标（升序）。
        strand / contig / start / end / umi_left / umi_right: 键字段展开。
        consensus_seq: 参考方向（``+`` 方向）共识序列；并列位置为 ``'N'``。
        confidence_gap: 逐位置 (最高 ll - 次高 ll)，log10 域；并列位置 0.0。
    """

    key: tuple
    record_ids: tuple[str, ...]
    record_indices: tuple[int, ...]
    strand: str
    contig: str
    start: int
    end: int
    umi_left: str
    umi_right: str
    consensus_seq: str
    confidence_gap: tuple[float, ...]


@dataclass(frozen=True)
class PairedAnalysisResult:
    """双端配对批次结果。``molecules`` 按匹配键字典序排列，确定可复现。"""

    molecules: tuple[PairedMolecule, ...]
    records_in: int
    molecules_after: int

    @property
    def duplicate_count(self) -> int:
        """被吸收为分子内重复的记录数（= 输入记录数 - 分子数）。"""
        return self.records_in - self.molecules_after


# ---------------------------------------------------------------------------
# 反向互补
# ---------------------------------------------------------------------------
def reverse_complement(seq: str) -> str:
    """DNA 串反向互补（A<->T、C<->G）；输入须为非空 A/C/G/T 串。

    长度保持不变；``rc(rc(s)) == s``。非法字符抛
    :class:`~umicorrect.errors.InvalidArgumentError`。
    """
    if not isinstance(seq, str):
        raise InvalidArgumentError(
            f"reverse_complement 输入必须是 str，得到 {type(seq).__name__}"
        )
    if not seq:
        raise InvalidArgumentError("reverse_complement 输入为空串")
    out = []
    for j, ch in enumerate(seq):
        if ch not in _REV:
            raise InvalidArgumentError(
                f"序列[{j}] = {ch!r} 非法，仅允许 A/C/G/T（拒绝 N/小写）"
            )
        out.append(_REV[ch])
    return "".join(reversed(out))


# ---------------------------------------------------------------------------
# 字段校验
# ---------------------------------------------------------------------------
def _check_umi(name: str, umi: object) -> None:
    if not isinstance(umi, str):
        raise InvalidArgumentError(f"{name} 必须是 str，得到 {type(umi).__name__}")
    if not umi:
        raise InvalidArgumentError(f"{name} 为空串：UMI 长度必须 >= {MIN_UMI_LENGTH}")
    if len(umi) > MAX_UMI_LENGTH:
        raise LimitExceededError(f"{name} 长度 {len(umi)} 超过上限 {MAX_UMI_LENGTH}")
    for j, ch in enumerate(umi):
        if ch not in _BASE_INDEX:
            raise InvalidArgumentError(
                f"{name}[{j}] = {ch!r} 非法，仅允许 A/C/G/T（拒绝 N/小写）"
            )


def _check_paired_fields(rec: PairedRecord) -> None:
    """单条双端记录的字段级校验（构造期即拒绝，路径可定位）。"""
    if not isinstance(rec.record_id, str) or not rec.record_id:
        raise InvalidArgumentError(
            f"record_id 必须是非空 str，得到 {rec.record_id!r}"
        )
    _check_umi("umi1", rec.umi1)
    _check_umi("umi2", rec.umi2)

    if not isinstance(rec.contig, str) or not rec.contig:
        raise InvalidArgumentError(f"contig 必须是非空 str，得到 {rec.contig!r}")

    for name in ("start", "end"):
        v = getattr(rec, name)
        if isinstance(v, bool) or not isinstance(v, int):
            if isinstance(v, float) and not isinstance(v, bool):
                # 与既有口径一致：浮点先专门拒绝 NaN/Infinity
                _reject_nonfinite(name, v)
            raise InvalidArgumentError(
                f"{name} 必须是整数且不能是 bool，得到 {type(v).__name__}"
            )
    if rec.start < 0 or rec.end < 0:
        raise InvalidArgumentError(
            f"坐标必须非负，得到 start={rec.start}, end={rec.end}"
        )
    if rec.start == rec.end:
        raise InvalidArgumentError(
            f"空区间 [start, end) 被拒绝：start == end == {rec.start}"
        )

    if rec.strand not in STRANDS:
        raise InvalidArgumentError(
            f"strand 必须是 '+' 或 '-'，得到 {rec.strand!r}"
        )

    seq = rec.sequence
    qual = rec.qualities
    if not isinstance(seq, str):
        raise InvalidArgumentError(f"sequence 必须是 str，得到 {type(seq).__name__}")
    if not isinstance(qual, tuple):
        raise InvalidArgumentError(
            f"qualities 必须是 tuple（或 list），得到 {type(qual).__name__}"
        )
    if not seq:
        raise InvalidArgumentError(f"sequence 为空串：读段长度必须 >= {MIN_READ_LENGTH}")
    if len(seq) > MAX_READ_LENGTH:
        raise LimitExceededError(
            f"sequence 长度 {len(seq)} 超过上限 {MAX_READ_LENGTH}"
        )
    if len(qual) != len(seq):
        raise InvalidArgumentError(
            f"qualities 长度 {len(qual)} 与序列长度 {len(seq)} 不一致"
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


# ---------------------------------------------------------------------------
# 规范化
# ---------------------------------------------------------------------------
def normalize_record(rec: PairedRecord, index: int = 0) -> NormalizedRecord:
    """把单条记录规范化为参考方向表示（不与其他记录比较）。

    Args:
        rec: 已通过字段校验的记录（:class:`PairedRecord` 构造期已校验；
            直接调用本函数时会再次触发字段读取但不绕过构造）。
        index: 该记录在批次中的下标（透传到结果，便于成对排序）。

    Returns:
        :class:`NormalizedRecord`。坐标升序；``-`` 链记录的双 UMI 交换并
        反向互补、序列反向互补、质量反转；``strand`` 原样保留。
    """
    if not isinstance(rec, PairedRecord):
        raise InvalidArgumentError(
            f"normalize_record 输入必须是 PairedRecord，得到 {type(rec).__name__}"
        )
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise InvalidArgumentError(f"index 必须是非负整数，得到 {index!r}")

    lo, hi = (rec.start, rec.end) if rec.start < rec.end else (rec.end, rec.start)

    if rec.strand == "+":
        umi_left, umi_right = rec.umi1, rec.umi2
        seq, qual = rec.sequence, rec.qualities
    else:
        # 反向链：读段2 的 UMI 在参考方向左侧。两端各自反向互补后交换位置。
        umi_left = reverse_complement(rec.umi2)
        umi_right = reverse_complement(rec.umi1)
        seq = reverse_complement(rec.sequence)
        qual = tuple(reversed(rec.qualities))

    return NormalizedRecord(
        record_id=rec.record_id,
        umi_left=umi_left,
        umi_right=umi_right,
        contig=rec.contig,
        start=lo,
        end=hi,
        strand=rec.strand,
        sequence=seq,
        qualities=qual,
        index=index,
    )


def _match_key(nr: NormalizedRecord) -> tuple:
    """显式匹配规则：六元组全等才是同一分子。

    (contig, start, end, strand, umi_left, umi_right)。strand 在键内，
    因此反向互补相同但链不同的记录键不同，不会合并。
    """
    return (nr.contig, nr.start, nr.end, nr.strand, nr.umi_left, nr.umi_right)


# ---------------------------------------------------------------------------
# 批次校验与成对
# ---------------------------------------------------------------------------
def _validate_paired_batch(records) -> list[PairedRecord]:
    if not isinstance(records, (list, tuple)):
        raise InvalidArgumentError(
            f"records 必须是 list/tuple，得到 {type(records).__name__}"
        )
    if len(records) > MAX_READS:
        raise LimitExceededError(f"记录数 {len(records)} 超过上限 {MAX_READS}")
    if len(records) == 0:
        raise InvalidArgumentError(
            "records 为空批次：双端配对拒绝空批次（无固定长度/坐标可推断）"
        )

    # ID 唯一性 + 显式字段复检：与单端 _validate_batch 同一口径——即使
    # 调用方绕过构造器安放了非法字段，批级路径仍能拒绝并加 records[i]
    # 定位前缀。全部检查在任何规范化/分组结果构建之前完成，保证原子失败。
    first_seen: dict[str, int] = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, PairedRecord):
            raise InvalidReadError(
                f"records[{i}] 必须是 PairedRecord，得到 {type(rec).__name__}"
            )
        try:
            _check_paired_fields(rec)
        except InvalidArgumentError as exc:
            raise InvalidReadError(f"records[{i}]: {exc}") from exc
        rid = rec.record_id
        if rid in first_seen:
            raise RecordConflictError(
                f"records[{i}].record_id={rid!r} 与 records[{first_seen[rid]}]"
                f" 重复：记录级重复被拒绝（同一 ID 不得出现两次；不同观测请"
                f"使用不同 record_id，分子级重复由匹配键自动归组）"
            )
        first_seen[rid] = i
    return list(records)


def _normalize_and_group(
    batch: list[PairedRecord],
) -> tuple[list[NormalizedRecord], dict[tuple, list[NormalizedRecord]]]:
    """规范化整批并按键归组，同时做组内一致性校验。

    同一匹配键的记录是“同一分子的重复观测”，规范化后序列长度必须一致
    （逐位置共识才有定义）；长度不一致属于不同观测间的冲突，显式拒绝，
    不截断、不静默合并。UMI 串本身是键的一部分，键相同即双 UMI 字面相等，
    无需另查。
    """
    normalized: list[NormalizedRecord] = [
        normalize_record(rec, index=i) for i, rec in enumerate(batch)
    ]
    groups: dict[tuple, list[NormalizedRecord]] = {}
    for nr in normalized:
        groups.setdefault(_match_key(nr), []).append(nr)

    for key, members in groups.items():
        first_len = len(members[0].sequence)
        for m in members[1:]:
            if len(m.sequence) != first_len:
                a, b = members[0], m
                raise InvalidArgumentError(
                    f"records[{b.index}] 与 records[{a.index}] 匹配键相同 "
                    f"{(key[0], key[1], key[2], key[3])} 但规范化后序列长度 "
                    f"{len(m.sequence)} != {first_len}：同分子重复观测长度不一致，"
                    f"拒绝静默合并"
                )
    return normalized, groups


def group_paired_records(records) -> tuple[PairedMolecule, ...]:
    """校验 → 规范化 → 按显式键成对（**不做**碱基共识，序列留空）。

    返回的 :class:`PairedMolecule` 的 ``consensus_seq`` 为空串、
    ``confidence_gap`` 为空元组；需要共识请用
    :func:`analyze_paired_records`。本入口便于只复用配对逻辑。

    组按匹配键字典序排列；组内 ``record_indices`` / ``record_ids`` 按
    批次下标升序。
    """
    batch = _validate_paired_batch(records)
    _, groups = _normalize_and_group(batch)

    molecules: list[PairedMolecule] = []
    for key in sorted(groups):
        members = groups[key]  # 已按批次下标升序（遍历顺序）
        contig, start, end, strand, ul, ur = key
        molecules.append(
            PairedMolecule(
                key=key,
                record_ids=tuple(m.record_id for m in members),
                record_indices=tuple(m.index for m in members),
                strand=strand,
                contig=contig,
                start=start,
                end=end,
                umi_left=ul,
                umi_right=ur,
                consensus_seq="",
                confidence_gap=(),
            )
        )
    return tuple(molecules)


# ---------------------------------------------------------------------------
# 顶层管线：配对 → 调用旧共识入口
# ---------------------------------------------------------------------------
def analyze_paired_records(records) -> PairedAnalysisResult:
    """对双端记录批次执行 校验 → 规范化 → 成对 → 逐组旧共识入口。

    共识直接复用 :func:`umicorrect.core.consensus`：同一匹配键下的全部
    记录序列在规范化后已统一到参考方向，按记录下标升序做逐碱基质量加权
    对数似然共识（并列位置输出 ``'N'``、gap=0.0），与单端管线同一套
    数值口径与误差约定。

    原子性：先整批校验（含 ID 唯一性），再规范化、分组、共识；任一记录
    非法即抛错，不返回任何部分结果。本函数无状态、不修改输入。

    Returns:
        :class:`PairedAnalysisResult`。
    """
    batch = _validate_paired_batch(records)
    normalized, groups = _normalize_and_group(batch)

    # 旧 consensus 入口只读取 .sequence 与 .qualities（见 core.consensus），
    # NormalizedRecord 恰好提供两者；直接复用 normalized 列表，不拷贝序列、
    # 不重新实现似然累加。read_indices 按下标升序，保持确定性求和次序。
    molecules: list[PairedMolecule] = []
    for key in sorted(groups):
        members = groups[key]
        contig, start, end, strand, ul, ur = key
        indices = tuple(m.index for m in members)
        read_len = len(members[0].sequence)
        seq, gaps = consensus(normalized, read_len, read_indices=indices)
        molecules.append(
            PairedMolecule(
                key=key,
                record_ids=tuple(m.record_id for m in members),
                record_indices=indices,
                strand=strand,
                contig=contig,
                start=start,
                end=end,
                umi_left=ul,
                umi_right=ur,
                consensus_seq=seq,
                confidence_gap=gaps,
            )
        )

    return PairedAnalysisResult(
        molecules=tuple(molecules),
        records_in=len(batch),
        molecules_after=len(molecules),
    )
