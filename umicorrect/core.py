"""UMI 测序分子的错误吸收与共识（题目核心算法，标准库实现）。

公开接口
--------
:class:`Read`          一条固定长度 UMI + 等长 DNA 读段 + 每碱基质量。
:class:`Molecule`      一个共识分子（根、成员、共识序列、逐位置信心差）。
:class:`AnalysisResult` 批次结果（合并前/后分子数、分子列表）。
:func:`analyze_reads`  一步完成：校验 → 按 UMI 计数 → 吸收聚类 → 逐簇共识。
:func:`cluster_umis`   只做吸收聚类（:class:`Cluster`）。
:func:`consensus`      只做一簇的加权对数似然共识。

算法（契约）
------------
1. 相同 UMI 的读段计为一个节点，count = 该 UMI 出现次数。
2. 节点排序：count 降序，count 相同按 UMI 字典序升序。
3. 构造有向图：对任意排序在前的大节点 *u* 与在后的小节点 *v*（“大/小”
   按“count 降序、再字典序”的全序），若
     * Hamming(u, v) 恰为 1；
     * count[u] >= 2 * count[v] - 1；
   则存在有向边 u -> v。
4. 按全序依次处理未归属节点，节点 *r* 未归属即建立根，从 *r* 沿有向边做
   确定性 BFS，把所有能到达且尚未归属的节点归入本簇（允许 R->X->Y 式
   传递吸收）。“按首个最高根归属”：能被多个根到达的节点归全序中最先
   处理的根。边总是从大节点指向小节点，因此低丰度节点即便与两个大根
   都相邻（“桥”），也不存在 桥->大根 方向的边，两个大根永远不会被
   合并——本算法不需要任何特例来禁止桥接。
5. 每个簇内逐位置对 A/C/G/T 累计对数似然
   ``log10 p(base|Q)``，其中
     * 与该位置碱基相同：``1 - 10**(-Q/10)``
     * 其余三种碱基：``10**(-Q/10) / 3``
   Q 为 0..40 的整数。对数似然最高者为共识碱基；并列（最高 ll 完全相等）
   输出 ``'N'``，且该位置 confidence_gap = 0.0。
6. 置信差 = 最大对数似然减次大对数似然（log10 域，即 Bayes 因子的数量级）。

数值与误差口径
--------------
* Hamming 距离、计数、吸收判定全部为整数精确运算。
* 对数似然仅在 ``log10`` 与一次浮点除法处产生 IEEE-754 双精度舍入；
  累计求和按“成员在输入列表中的出现顺序”做有序求和，确定可复现。
  单碱基概率的相对误差 < 1e-15；在声明上限内（见下）累计误差远小于 1
  个 log10 单位。并列判定使用精确 ``==``：能产生精确相等的场景（如完全
  对称数据）必须报 N，不用容差“猜”碱基。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import (
    InvalidArgumentError,
    InvalidReadError,
    LimitExceededError,
    SearchBudgetExhaustedError,
)

__all__ = [
    "Read",
    "Molecule",
    "Cluster",
    "AnalysisResult",
    "EditDistanceOptions",
    "EditCandidate",
    "EditAssignment",
    "EditClusteringResult",
    "DEFAULT_PROBE_BUDGET",
    "MAX_UMI_LENGTH",
    "MAX_READ_LENGTH",
    "MAX_READS",
    "MIN_UMI_LENGTH",
    "MIN_READ_LENGTH",
    "BASES",
    "cluster_umis",
    "consensus",
    "analyze_reads",
]

BASES = ("A", "C", "G", "T")
_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}

# ---- 公开输入上限（接口边界；短演示不必触及） -----------------------------
MIN_UMI_LENGTH = 1
MAX_UMI_LENGTH = 32
MIN_READ_LENGTH = 1
MAX_READ_LENGTH = 1000
MAX_READS = 1_000_000
# Hamming 邻居探测次数预算 = 需要评估的有向节点对数（上界 K(K-1)/2）。
# 默认 2e8 覆盖约 2 万个去重 UMI 的最坏两两比较；典型 UMI 数据邻接稀疏，
# 实际探测远少于此。更大 K 需显式放大预算，耗尽时报错而非截断结果。
DEFAULT_PROBE_BUDGET = 200_000_000


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Read:
    """一条输入读段。

    ``umi`` / ``sequence`` 为仅含 A/C/G/T 的等长 DNA 字符串；
    ``qualities`` 为逐碱基 Phred 质量分（整数 0..40），长度与序列相同。

    构造时即做完整字段校验（非法输入抛出
    :class:`~umicorrect.errors.InvalidArgumentError`）；批次接口
    :func:`analyze_reads` 会额外加上 ``reads[i]`` 定位前缀并检查跨读段的
    固定长度一致性。
    """

    umi: str
    sequence: str
    qualities: tuple[int, ...]
    umi_qualities: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.qualities, list):
            object.__setattr__(self, "qualities", tuple(self.qualities))
        if isinstance(self.umi_qualities, list):
            object.__setattr__(self, "umi_qualities", tuple(self.umi_qualities))
        _check_read_fields(self)


@dataclass(frozen=True)
class Cluster:
    """UMI 吸收簇。

    Attributes:
        root: 根 UMI。
        members: 成员 UMI（含根，字典序升序）。
        root_count: 根节点计数。
        total_count: 簇内全部成员计数之和。
    """

    root: str
    members: tuple[str, ...]
    root_count: int
    total_count: int


@dataclass(frozen=True)
class Molecule:
    """一个共识分子。

    Attributes:
        root_umi:      簇根 UMI（首个最高节点）。
        members:       成员 UMI 元组（含根，字典序升序）。
        read_indices:  成员读段在输入批次中的下标（升序）。
        consensus_seq: 共识序列；逐位置最大对数似然并列时该位置为 ``'N'``。
        confidence_gap: 逐位置 (最高 ll - 次高 ll)，log10 域；并列位置为 0.0。
    """

    root_umi: str
    members: tuple[str, ...]
    read_indices: tuple[int, ...]
    consensus_seq: str
    confidence_gap: tuple[float, ...]


@dataclass(frozen=True)
class AnalysisResult:
    """批次分析结果。``molecules`` 按 (count 降序, 根 UMI 字典序) 排列。"""

    molecules: tuple[Molecule, ...]
    umi_length: int
    read_length: int
    nodes_before: int
    molecules_after: int
    edit_assignments: tuple["EditAssignment", ...] = ()

    @property
    def merged_count(self) -> int:
        """吸收操作合并掉的 UMI 种数（= 合并前分子数 - 合并后分子数）。"""
        return self.nodes_before - self.molecules_after


@dataclass(frozen=True)
class EditDistanceOptions:
    """可选短 UMI 编辑距离吸收模型的参数。

    模型只处理最小编辑距离不超过 ``max_edit_distance`` 的候选，且候选仍须
    沿丰度全序从高丰度指向低丰度并满足原有 ``2*child_count-1`` 丰度规则。

    Attributes:
        max_edit_distance: 最多编辑数；当前契约支持 1 或 2。长度差大于 1
            时不可能在两次编辑内完成，直接不是候选。
        min_log10_gap: 最佳与次佳候选的 log10 概率差阈值；严格小于该值
            时低丰度 UMI 保持独立。
        default_quality: 某条读段未提供 ``umi_qualities`` 时使用的 UMI Q。
    """

    max_edit_distance: int = 2
    min_log10_gap: float = 1.0
    default_quality: int = 20


@dataclass(frozen=True)
class EditCandidate:
    """一个高丰度根对低丰度观测 UMI 的编辑路径证据。"""

    root: str
    edit_distance: int
    log10_probability: float
    path_count: int


@dataclass(frozen=True)
class EditAssignment:
    """低丰度 UMI 相对于全部可行高丰度候选的归并/保留决定。"""

    umi: str
    state: str  # "merged"、"ambiguous" 或 "independent"
    assigned_root: str | None
    candidates: tuple[EditCandidate, ...]
    best_gap: float | None


@dataclass(frozen=True)
class EditClusteringResult:
    """编辑距离模型的聚类结果和逐低丰度节点证据。"""

    clusters: tuple[Cluster, ...]
    assignments: tuple[EditAssignment, ...]


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------
def _reject_nonfinite(name: str, value) -> float:
    """拒绝 NaN/Infinity 的数值输入；bool 不当作数值。返回 float 值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidArgumentError(f"{name} 必须是实数，得到 {type(value).__name__}")
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise InvalidArgumentError(f"{name} 拒绝 NaN/Infinity，得到 {value!r}")
    return fv


def _strict_int(name: str, value) -> int:
    """整数参数拒绝 bool（True/False 不是 1/0）。

    浮点参数先按 NaN/Infinity 专门拒绝（即使类型也不对），给出可定位的
    具体错误；有限浮点仍按“必须是整数”拒绝。
    """
    if isinstance(value, bool):
        raise InvalidArgumentError(
            f"{name} 必须是整数且不能是 bool，得到 {value!r}"
        )
    if isinstance(value, float):
        _reject_nonfinite(name, value)  # NaN/Infinity 在此抛出专门错误
        raise InvalidArgumentError(f"{name} 必须是整数，得到 float {value!r}")
    if not isinstance(value, int):
        raise InvalidArgumentError(
            f"{name} 必须是整数且不能是 bool，得到 {type(value).__name__}"
        )
    return value


def _check_read_fields(read: Read) -> None:
    """单条读段的字段级校验（``Read`` 构造与批次校验共用）。

    覆盖：类型、非空、等长、A/C/G/T 字母表、质量 0..40 整数（拒绝 bool）、
    长度上限。错误为 :class:`InvalidArgumentError`（字段路径如
    ``"qualities[2]"`` 可定位）。
    """
    umi = read.umi
    seq = read.sequence
    qual = read.qualities
    umi_qual = read.umi_qualities

    if not isinstance(umi, str):
        raise InvalidArgumentError(f"umi 必须是 str，得到 {type(umi).__name__}")
    if not isinstance(seq, str):
        raise InvalidArgumentError(f"sequence 必须是 str，得到 {type(seq).__name__}")
    if not isinstance(qual, tuple):
        raise InvalidArgumentError(
            f"qualities 必须是 tuple（或 list），得到 {type(qual).__name__}"
        )
    if umi_qual is not None and not isinstance(umi_qual, tuple):
        raise InvalidArgumentError(
            "umi_qualities 必须是 tuple（或 list）或 None，"
            f"得到 {type(umi_qual).__name__}"
        )

    if not umi:
        raise InvalidArgumentError(f"umi 为空串：UMI 长度必须 >= {MIN_UMI_LENGTH}")
    if not seq:
        raise InvalidArgumentError(f"sequence 为空串：读段长度必须 >= {MIN_READ_LENGTH}")
    if len(umi) > MAX_UMI_LENGTH:
        raise LimitExceededError(f"umi 长度 {len(umi)} 超过上限 {MAX_UMI_LENGTH}")
    if len(seq) > MAX_READ_LENGTH:
        raise LimitExceededError(f"sequence 长度 {len(seq)} 超过上限 {MAX_READ_LENGTH}")
    if umi_qual is None and len(seq) != len(umi):
        raise InvalidArgumentError(
            f"读段长度 {len(seq)} 与 UMI 长度 {len(umi)} 不相等；"
            "变长 UMI 需同时提供等长 umi_qualities"
        )
    if len(qual) != len(seq):
        raise InvalidArgumentError(
            f"qualities 长度 {len(qual)} 与序列长度 {len(seq)} 不一致"
        )

    for j, ch in enumerate(umi):
        if ch not in _BASE_INDEX:
            raise InvalidArgumentError(f"umi[{j}] = {ch!r} 非法，仅允许 A/C/G/T（拒绝 N/小写）")
    for j, ch in enumerate(seq):
        if ch not in _BASE_INDEX:
            raise InvalidArgumentError(
                f"sequence[{j}] = {ch!r} 非法，仅允许 A/C/G/T（拒绝 N/小写）"
            )
    for j, q in enumerate(qual):
        if isinstance(q, bool) or not isinstance(q, int):
            if isinstance(q, float) and not isinstance(q, bool):
                # 浮点质量分：先给明确的 NaN/Infinity 拒绝，再拒绝类型
                _reject_nonfinite(f"qualities[{j}]", q)
            raise InvalidArgumentError(
                f"qualities[{j}] 必须是 0..40 的整数且不能是 bool，得到 {type(q).__name__}"
            )
        if q < 0 or q > 40:
            raise InvalidArgumentError(f"qualities[{j}] = {q} 越界，允许范围 0..40")
    if umi_qual is not None:
        if len(umi_qual) != len(umi):
            raise InvalidArgumentError(
                f"umi_qualities 长度 {len(umi_qual)} 与 UMI 长度 {len(umi)} 不一致"
            )
        for j, q in enumerate(umi_qual):
            if isinstance(q, bool) or not isinstance(q, int):
                raise InvalidArgumentError(
                    f"umi_qualities[{j}] 必须是 0..40 的整数且不能是 bool，"
                    f"得到 {type(q).__name__}"
                )
            if q < 0 or q > 40:
                raise InvalidArgumentError(
                    f"umi_qualities[{j}] = {q} 越界，允许范围 0..40"
                )


def _validate_batch(reads, *, variable_umi: bool = False) -> tuple[int, int]:
    """校验整个批次（先全部校验、后构建，失败不留部分状态）。

    返回 (固定 UMI 长度, 固定读段长度)。所有错误消息带 ``reads[i]`` 前缀。
    """
    if not isinstance(reads, (list, tuple)):
        raise InvalidArgumentError(
            f"reads 必须是 list/tuple，得到 {type(reads).__name__}"
        )
    if len(reads) > MAX_READS:
        raise LimitExceededError(f"读段数 {len(reads)} 超过上限 {MAX_READS}")
    if len(reads) == 0:
        # 空批次：无固定长度可推断，明确拒绝而不是静默返回空结果。
        raise InvalidArgumentError(
            "reads 为空批次：无法确定固定 UMI/读段长度（空批次被拒绝）"
        )

    fixed_umi_len: int | None = None
    min_umi_len: int | None = None
    max_umi_len = 0
    read_len = 0
    for i, read in enumerate(reads):
        if not isinstance(read, Read):
            raise InvalidReadError(
                f"reads[{i}] 必须是 Read，得到 {type(read).__name__}"
            )
        try:
            _check_read_fields(read)
        except InvalidArgumentError as exc:
            # 加上批次内定位前缀，重新抛出同类异常
            raise InvalidReadError(f"reads[{i}]: {exc}") from exc
        if fixed_umi_len is None:
            fixed_umi_len = len(read.umi)
            min_umi_len = len(read.umi)
            max_umi_len = len(read.umi)
            read_len = len(read.sequence)
        elif not variable_umi and len(read.umi) != fixed_umi_len:
            raise InvalidReadError(
                f"reads[{i}].umi 长度 {len(read.umi)} 与批次固定 UMI 长度 "
                f"{fixed_umi_len} 不一致"
            )
        elif variable_umi:
            assert min_umi_len is not None
            min_umi_len = min(min_umi_len, len(read.umi))
            max_umi_len = max(max_umi_len, len(read.umi))
            if max_umi_len - min_umi_len > 1:
                raise InvalidReadError(
                    f"reads[{i}].umi 长度使批次 UMI 长度差超过 1；"
                    "编辑距离模型最多允许一次插入或删除"
                )
            if read.umi_qualities is None:
                raise InvalidReadError(
                    f"reads[{i}]: 编辑距离模型处理长度为 {len(read.umi)} 的 UMI 时，"
                    "必须提供与 UMI 等长的 umi_qualities"
                )
            if len(read.sequence) != read_len:
                raise InvalidReadError(
                    f"reads[{i}].sequence 长度 {len(read.sequence)} 与固定读段长度 {read_len} 不一致；"
                    "编辑距离模型只允许 UMI 长度变化，不允许读段长度变化"
                )
    assert fixed_umi_len is not None
    return fixed_umi_len, read_len


# ---------------------------------------------------------------------------
# 第一步：按 UMI 计数
# ---------------------------------------------------------------------------
def _group_by_umi(reads) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, read in enumerate(reads):
        groups.setdefault(read.umi, []).append(i)
    return groups


# ---------------------------------------------------------------------------
# 第二步：错误吸收聚类
# ---------------------------------------------------------------------------
def cluster_umis(
    umi_counts: dict[str, int],
    *,
    probe_budget: int = DEFAULT_PROBE_BUDGET,
) -> tuple[Cluster, ...]:
    """按题面规则把 UMI 节点吸收成簇（不做碱基共识）。

    Args:
        umi_counts: UMI -> 计数（正整数）。键必须为等长 A/C/G/T 串。
        probe_budget: Hamming 比较（邻居探测）次数上限；耗尽抛
            :class:`SearchBudgetExhaustedError`（结果未知，区别于无解）。

    Returns:
        Cluster 元组，按 (簇总计数降序, 根 UMI 字典序) 排列。
    """
    probe_budget, ordered = _prepare_umi_counts(umi_counts, probe_budget, variable_umi=False)
    adjacency, _ = _build_default_adjacency(umi_counts, ordered, probe_budget)
    return _clusters_from_adjacency(umi_counts, ordered, adjacency)


def _prepare_umi_counts(umi_counts, probe_budget: int, *, variable_umi: bool):
    probe_budget = _strict_int("probe_budget", probe_budget)
    if probe_budget < 0:
        raise InvalidArgumentError("probe_budget 必须为非负整数")
    if not isinstance(umi_counts, dict):
        raise InvalidArgumentError(
            f"umi_counts 必须是 dict，得到 {type(umi_counts).__name__}"
        )
    fixed_len: int | None = None
    min_len: int | None = None
    max_len = 0
    for umi, count in umi_counts.items():
        if not isinstance(umi, str):
            raise InvalidArgumentError(f"umi_counts 键必须是 str，得到 {type(umi).__name__}")
        if not umi:
            raise InvalidArgumentError("umi_counts 含空 UMI 键")
        if len(umi) > MAX_UMI_LENGTH:
            raise LimitExceededError(f"UMI 长度 {len(umi)} 超过上限 {MAX_UMI_LENGTH}")
        if not variable_umi:
            if fixed_len is None:
                fixed_len = len(umi)
            elif len(umi) != fixed_len:
                raise InvalidArgumentError(
                    f"UMI {umi!r} 长度 {len(umi)} 与固定长度 {fixed_len} 不一致"
                )
        else:
            if min_len is None:
                min_len = len(umi)
                max_len = len(umi)
            else:
                assert min_len is not None
                min_len = min(min_len, len(umi))
                max_len = max(max_len, len(umi))
                if max_len - min_len > 1:
                    raise InvalidArgumentError("编辑距离模型中 UMI 长度差不得超过 1")
        for j, ch in enumerate(umi):
            if ch not in _BASE_INDEX:
                raise InvalidArgumentError(f"umi_counts 键 {umi!r} 位置 {j} 含非法碱基 {ch!r}")
        if isinstance(count, bool) or not isinstance(count, int):
            raise InvalidArgumentError(
                f"umi_counts[{umi!r}] 计数必须是正整数且不能是 bool，"
                f"得到 {type(count).__name__}"
            )
        if count <= 0:
            raise InvalidArgumentError(f"umi_counts[{umi!r}] 计数必须为正整数，得到 {count}")
    if not umi_counts:
        return probe_budget, ()

    # count 降序、再 UMI 字典序升序 —— 全局确定全序
    ordered = sorted(umi_counts, key=lambda u: (-umi_counts[u], u))
    return probe_budget, ordered


def _build_default_adjacency(umi_counts, ordered, probe_budget: int):
    probes = 0
    adjacency: dict[str, list[str]] = {u: [] for u in ordered}
    for ai in range(len(ordered)):
        u = ordered[ai]
        cu = umi_counts[u]
        for bi in range(ai + 1, len(ordered)):
            v = ordered[bi]
            probes += 1
            if probes > probe_budget:
                raise SearchBudgetExhaustedError(
                    f"Hamming 邻居探测预算耗尽（已用 {probes}，预算 {probe_budget}）；"
                    "吸收结果未知，请放大 probe_budget 后重算",
                    probes_used=probes,
                    budget=probe_budget,
                )
            if abs(len(u) - len(v)) == 0 and _hamming1(u, v) and cu >= 2 * umi_counts[v] - 1:
                adjacency[u].append(v)
    return adjacency, probes


def _clusters_from_adjacency(umi_counts, ordered, adjacency):

    # owner[u]：u 已归属的根；None 表示未归属。
    owner: dict[str, str | None] = {u: None for u in ordered}
    members: dict[str, list[str]] = {}

    for root in ordered:
        if owner[root] is not None:
            continue  # 已被更早（更高）的根可达并吸收
        owner[root] = root
        mine = [root]
        queue = [root]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adjacency[u]:  # 已按全序排列
                if owner[v] is None:
                    owner[v] = root
                    mine.append(v)
                    queue.append(v)
        members[root] = mine

    clusters: list[Cluster] = []
    for root, mem in members.items():
        clusters.append(
            Cluster(
                root=root,
                members=tuple(sorted(mem)),
                root_count=umi_counts[root],
                total_count=sum(umi_counts[u] for u in mem),
            )
        )
    clusters.sort(key=lambda c: (-c.total_count, c.root))
    return tuple(clusters)


def _hamming1(a: str, b: str) -> bool:
    """Hamming 距离是否恰为 1（a、b 等长，调用前已保证）。"""
    diff = 0
    for x, y in zip(a, b):
        if x != y:
            diff += 1
            if diff > 1:
                return False
    return diff == 1


def _validate_edit_options(options: EditDistanceOptions | None) -> EditDistanceOptions:
    if options is None:
        options = EditDistanceOptions()
    if not isinstance(options, EditDistanceOptions):
        raise InvalidArgumentError(
            f"edit_options 必须是 EditDistanceOptions，得到 {type(options).__name__}"
        )
    max_d = _strict_int("max_edit_distance", options.max_edit_distance)
    default_q = _strict_int("default_quality", options.default_quality)
    if max_d < 1 or max_d > 2:
        raise InvalidArgumentError("max_edit_distance 当前仅支持 1 或 2")
    if default_q < 0 or default_q > 40:
        raise InvalidArgumentError("default_quality 必须在 0..40")
    gap = _reject_nonfinite("min_log10_gap", options.min_log10_gap)
    if gap < 0:
        raise InvalidArgumentError("min_log10_gap 必须是非负数")
    return EditDistanceOptions(max_d, gap, default_q)


_NEG_INF = float("-inf")


def _log_add(x: float, y: float) -> float:
    if x == _NEG_INF:
        return y
    if y == _NEG_INF:
        return x
    if x < y:
        x, y = y, x
    return x + math.log1p(10.0 ** (y - x)) / math.log(10.0)


def _edit_log_terms(q: int) -> tuple[float, float, float, float]:
    err = 10.0 ** (-q / 10.0)
    correct = 1.0 - err
    return (
        _NEG_INF if correct == 0.0 else math.log10(correct),
        math.log10(err / 9.0),
        math.log10(err / 12.0),
        math.log10(err / 3.0),
    )


_EDIT_TERM_TABLE = tuple(_edit_log_terms(q) for q in range(41))


def shortest_edit_evidence(root: str, observed: str, qualities: tuple[int, ...]):
    """返回 root 生成 observed 的 (最小编辑数, 全部最短路径 log10 概率和, 路径数)。"""
    for name, value in (("root", root), ("observed", observed)):
        if not isinstance(value, str):
            raise InvalidArgumentError(f"{name} 必须是 str")
        for pos, base in enumerate(value):
            if base not in _BASE_INDEX:
                raise InvalidArgumentError(f"{name}[{pos}] 非法，仅允许 A/C/G/T")
        if not value:
            raise InvalidArgumentError(f"{name} 不能为空")
        if len(value) > MAX_UMI_LENGTH:
            raise LimitExceededError(f"{name} 长度超过上限 {MAX_UMI_LENGTH}")
    if len(qualities) != len(observed):
        raise InvalidArgumentError("UMI qualities 必须与观测 UMI 等长")
    for pos, quality in enumerate(qualities):
        if isinstance(quality, bool) or not isinstance(quality, int):
            raise InvalidArgumentError(f"qualities[{pos}] 必须是 0..40 的整数且不能是 bool")
        if quality < 0 or quality > 40:
            raise InvalidArgumentError(f"qualities[{pos}] = {quality} 越界，允许范围 0..40")
    m, n = len(root), len(observed)
    dist = [[10**9] * (n + 1) for _ in range(m + 1)]
    logp = [[_NEG_INF] * (n + 1) for _ in range(m + 1)]
    paths = [[0] * (n + 1) for _ in range(m + 1)]
    dist[0][0] = 0
    logp[0][0] = 0.0
    paths[0][0] = 1

    def relax(i, j, previous_distance, previous_log, previous_paths, added_distance, term):
        candidate_distance = previous_distance + added_distance
        candidate_log = previous_log + term
        if candidate_distance < dist[i][j]:
            dist[i][j] = candidate_distance
            logp[i][j] = candidate_log
            paths[i][j] = previous_paths
        elif candidate_distance == dist[i][j]:
            logp[i][j] = _log_add(logp[i][j], candidate_log)
            paths[i][j] += previous_paths

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            if i > 0 and j > 0:
                q = qualities[j - 1]
                log_match, log_sub, _, _ = _EDIT_TERM_TABLE[q]
                if root[i - 1] == observed[j - 1]:
                    relax(i, j, dist[i-1][j-1], logp[i-1][j-1], paths[i-1][j-1], 0, log_match)
                else:
                    relax(i, j, dist[i-1][j-1], logp[i-1][j-1], paths[i-1][j-1], 1, log_sub)
            if i > 0:
                anchor_q = qualities[min(j, n - 1)]
                _, _, _, log_delete = _EDIT_TERM_TABLE[anchor_q]
                relax(i, j, dist[i-1][j], logp[i-1][j], paths[i-1][j], 1, log_delete)
            if j > 0:
                _, _, log_insert, _ = _EDIT_TERM_TABLE[qualities[j - 1]]
                relax(i, j, dist[i][j-1], logp[i][j-1], paths[i][j-1], 1, log_insert)
    return dist[m][n], logp[m][n], paths[m][n]


def _quality_records_for(umi, groups, quality_records, default_quality):
    if quality_records is None or umi not in quality_records:
        count = len(groups[umi]) if groups is not None else 1
        return [tuple([default_quality] * len(umi)) for _ in range(count)]
    records = list(quality_records[umi])
    if groups is not None and len(records) != len(groups[umi]):
        raise InvalidArgumentError(f"UMI {umi!r} 的 UMI 质量记录数与读段数不一致")
    out = []
    for k, record in enumerate(records):
        if record is None:
            out.append(tuple([default_quality] * len(umi)))
        else:
            if len(record) != len(umi):
                raise InvalidArgumentError(f"UMI {umi!r} 第 {k} 条 UMI 质量长度不一致")
            out.append(tuple(record))
    return out


def cluster_umis_with_edit_distance(
    umi_counts: dict[str, int],
    quality_records=None,
    *,
    edit_options: EditDistanceOptions | None = None,
    probe_budget: int = DEFAULT_PROBE_BUDGET,
) -> EditClusteringResult:
    """用短编辑距离和碱基质量执行可选吸收。"""
    options = _validate_edit_options(edit_options)
    probe_budget, ordered = _prepare_umi_counts(umi_counts, probe_budget, variable_umi=True)
    probes = 0
    adjacency: dict[str, list[str]] = {u: [] for u in ordered}
    assignments: list[EditAssignment] = []

    for vi, child in enumerate(ordered):
        candidates = []
        expected_count = umi_counts[child]
        child_records = []
        if vi:
            if quality_records is None or child not in quality_records:
                child_records = [tuple([options.default_quality] * len(child)) for _ in range(expected_count)]
            else:
                child_records = _quality_records_for(child, None, quality_records, options.default_quality)
                if len(child_records) != expected_count:
                    raise InvalidArgumentError(f"UMI {child!r} 的 UMI 质量记录数与计数不一致")
        for root in ordered[:vi]:
            if umi_counts[root] < 2 * expected_count - 1:
                continue
            probes += 1
            if probes > probe_budget:
                raise SearchBudgetExhaustedError(
                    f"编辑距离候选探测预算耗尽（已用 {probes}，预算 {probe_budget}）",
                    probes_used=probes,
                    budget=probe_budget,
                )
            if abs(len(root) - len(child)) > 1:
                continue
            total_log = 0.0
            distance = 10**9
            path_count = 1
            valid = True
            for qualities in child_records:
                distance, one_log, one_paths = shortest_edit_evidence(root, child, qualities)
                if distance > options.max_edit_distance:
                    valid = False
                    break
                total_log += one_log
                path_count *= one_paths
            if valid:
                candidates.append(EditCandidate(root, distance, total_log, path_count))

        candidates.sort(key=lambda c: (-c.log10_probability, c.edit_distance, c.root))
        state = "independent"
        assigned = None
        gap = None
        if candidates:
            best = candidates[0]
            second_log = candidates[1].log10_probability if len(candidates) > 1 else _NEG_INF
            gap = best.log10_probability - second_log
            if gap >= options.min_log10_gap:
                state = "merged"
                assigned = best.root
                adjacency[best.root].append(child)
            else:
                state = "ambiguous"
        assignments.append(EditAssignment(child, state, assigned, tuple(candidates), gap))

    for edges in adjacency.values():
        edges.sort(key=lambda u: (-umi_counts[u], u))
    clusters = _clusters_from_adjacency(umi_counts, ordered, adjacency)
    return EditClusteringResult(clusters=clusters, assignments=tuple(assignments))


# ---------------------------------------------------------------------------
# 第三步：逐簇对数似然共识
# ---------------------------------------------------------------------------
def _log10_prob(q: int) -> tuple[float, float]:
    """返回 (log10(1-e), log10(e/3))，e = 10^(-Q/10)。

    Q=0 时 e=1，正确碱基概率为 0，对数似然记为 -inf
    （math.log10(0.0) 在 Python 中抛 ValueError，需显式处理）。
    """
    err = 10.0 ** (-q / 10.0)
    correct = 1.0 - err
    log_correct = float("-inf") if correct == 0.0 else math.log10(correct)
    return log_correct, math.log10(err / 3.0)


# Q=0..40 的似然项预计算（确定性、无平台差异）
_LL_TABLE: tuple[tuple[float, float], ...] = tuple(_log10_prob(q) for q in range(41))


def consensus(
    reads,
    read_length: int,
    *,
    read_indices: tuple[int, ...] | None = None,
) -> tuple[str, tuple[float, ...]]:
    """对一簇读段计算对数似然共识。

    Args:
        reads: 批次读段序列。
        read_length: 固定读段长度。
        read_indices: 参与本簇共识的读段下标；非空。

    Returns:
        (共识序列, 逐位置信心差)。最大 ll 并列位置输出 ``'N'``、gap=0.0。
    """
    read_length = _strict_int("read_length", read_length)
    if read_length < MIN_READ_LENGTH or read_length > MAX_READ_LENGTH:
        raise InvalidArgumentError(
            f"read_length 越界（{MIN_READ_LENGTH}..{MAX_READ_LENGTH}），得到 {read_length}"
        )
    if read_indices is None or len(read_indices) == 0:
        raise InvalidArgumentError("共识簇至少需要 1 条读段（空簇被拒绝）")

    n = read_length
    # ll[pos][base] 累计 log10 似然
    ll = [[0.0, 0.0, 0.0, 0.0] for _ in range(n)]

    # 有序累加（按下标升序），保证求和顺序确定
    for i in sorted(read_indices):
        if isinstance(i, bool) or not isinstance(i, int):
            raise InvalidArgumentError(f"read_indices 必须全为 int 且不能是 bool")
        if i < 0 or i >= len(reads):
            raise InvalidArgumentError(f"read_indices 含越界下标 {i}（批次共 {len(reads)} 条）")
        read = reads[i]
        for pos in range(n):
            b = _BASE_INDEX[read.sequence[pos]]
            q = read.qualities[pos]
            log_correct, log_wrong = _LL_TABLE[q]
            row = ll[pos]
            for k in range(4):
                row[k] += log_correct if k == b else log_wrong

    out_bases: list[str] = []
    gaps: list[float] = []
    for pos in range(n):
        row = ll[pos]
        best = max(range(4), key=lambda k: row[k])
        best_val = row[best]
        # 并列检测：是否存在另一个碱基 ll 与最高完全相等
        tied = [k for k in range(4) if k != best and row[k] == best_val]
        if tied:
            out_bases.append("N")
            gaps.append(0.0)
        else:
            second = max(row[k] for k in range(4) if k != best)
            out_bases.append(BASES[best])
            gaps.append(best_val - second)
    return "".join(out_bases), tuple(gaps)


# ---------------------------------------------------------------------------
# 顶层管线
# ---------------------------------------------------------------------------
def analyze_reads(
    reads,
    *,
    probe_budget: int = DEFAULT_PROBE_BUDGET,
    edit_options: EditDistanceOptions | None = None,
) -> AnalysisResult:
    """对整批读段执行 校验 → UMI 计数 → 吸收聚类 → 逐簇共识。

    校验先于任何状态构建并在整个批次上进行；任一读段非法则整批拒绝
    （不产生部分结果）。吸收阶段耗尽预算时同样不返回截断结果。

    Returns:
        :class:`AnalysisResult`。
    """
    edit_mode = edit_options is not None
    if edit_mode:
        validated_edit_options = _validate_edit_options(edit_options)

    # 1) 整批先校验（原子失败语义）
    umi_len, read_len = _validate_batch(reads, variable_umi=edit_mode)

    # 2) 分组计数
    groups = _group_by_umi(reads)
    umi_counts = {u: len(v) for u, v in groups.items()}

    # 3) 吸收聚类
    edit_assignments = ()
    if edit_mode:
        quality_records = {}
        for read in reads:
            quality_records.setdefault(read.umi, []).append(read.umi_qualities)
        edit_result = cluster_umis_with_edit_distance(
            umi_counts,
            quality_records,
            edit_options=validated_edit_options,
            probe_budget=probe_budget,
        )
        clusters = edit_result.clusters
        edit_assignments = edit_result.assignments
    else:
        clusters = cluster_umis(umi_counts, probe_budget=probe_budget)
    nodes_before = len(umi_counts)

    # 4) 逐簇共识
    molecules: list[Molecule] = []
    for cl in clusters:
        idxs = sorted(i for u in cl.members for i in groups[u])
        seq, gaps = consensus(reads, read_len, read_indices=tuple(idxs))
        molecules.append(
            Molecule(
                root_umi=cl.root,
                members=cl.members,
                read_indices=tuple(idxs),
                consensus_seq=seq,
                confidence_gap=gaps,
            )
        )

    # 与 cluster 排序一致（成员重计后顺序不变，保持显式排序）
    molecules.sort(
        key=lambda m: (-sum(1 for _ in m.read_indices), m.root_umi)
    )
    return AnalysisResult(
        molecules=tuple(molecules),
        umi_length=umi_len,
        read_length=read_len,
        nodes_before=nodes_before,
        molecules_after=len(molecules),
        edit_assignments=edit_assignments,
    )
