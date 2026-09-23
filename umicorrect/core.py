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

    def __post_init__(self) -> None:
        if isinstance(self.qualities, list):
            object.__setattr__(self, "qualities", tuple(self.qualities))
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

    @property
    def merged_count(self) -> int:
        """吸收操作合并掉的 UMI 种数（= 合并前分子数 - 合并后分子数）。"""
        return self.nodes_before - self.molecules_after


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
        raise InvalidArgumentError(f"sequence 为空串：读段长度必须 >= {MIN_READ_LENGTH}")
    if len(umi) > MAX_UMI_LENGTH:
        raise LimitExceededError(f"umi 长度 {len(umi)} 超过上限 {MAX_UMI_LENGTH}")
    if len(seq) > MAX_READ_LENGTH:
        raise LimitExceededError(f"sequence 长度 {len(seq)} 超过上限 {MAX_READ_LENGTH}")
    if len(seq) != len(umi):
        raise InvalidArgumentError(
            f"读段长度 {len(seq)} 与 UMI 长度 {len(umi)} 不相等（契约要求等长）"
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


def _validate_batch(reads) -> tuple[int, int]:
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
            read_len = len(read.sequence)
        elif len(read.umi) != fixed_umi_len:
            raise InvalidReadError(
                f"reads[{i}].umi 长度 {len(read.umi)} 与批次固定 UMI 长度 "
                f"{fixed_umi_len} 不一致"
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
    probe_budget = _strict_int("probe_budget", probe_budget)
    if probe_budget < 0:
        raise InvalidArgumentError("probe_budget 必须为非负整数")
    if not isinstance(umi_counts, dict):
        raise InvalidArgumentError(
            f"umi_counts 必须是 dict，得到 {type(umi_counts).__name__}"
        )

    # 独立校验键值（cluster_umis 可被直接复用）
    fixed_len: int | None = None
    for umi, count in umi_counts.items():
        if not isinstance(umi, str):
            raise InvalidArgumentError(f"umi_counts 键必须是 str，得到 {type(umi).__name__}")
        if not umi:
            raise InvalidArgumentError("umi_counts 含空 UMI 键")
        if len(umi) > MAX_UMI_LENGTH:
            raise LimitExceededError(f"UMI 长度 {len(umi)} 超过上限 {MAX_UMI_LENGTH}")
        if fixed_len is None:
            fixed_len = len(umi)
        elif len(umi) != fixed_len:
            raise InvalidArgumentError(
                f"UMI {umi!r} 长度 {len(umi)} 与固定长度 {fixed_len} 不一致"
            )
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
        return ()

    # count 降序、再 UMI 字典序升序 —— 全局确定全序
    ordered = sorted(umi_counts, key=lambda u: (-umi_counts[u], u))

    # ---- 构造有向边 u -> v（u 在全序中先于 v）----
    # adjacency[u] 按全序排列，保证 BFS 确定性。
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
            if _hamming1(u, v) and cu >= 2 * umi_counts[v] - 1:
                adjacency[u].append(v)

    # ---- 按全序建立根，沿有向边确定性 BFS 传递吸收 ----
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
) -> AnalysisResult:
    """对整批读段执行 校验 → UMI 计数 → 吸收聚类 → 逐簇共识。

    校验先于任何状态构建并在整个批次上进行；任一读段非法则整批拒绝
    （不产生部分结果）。吸收阶段耗尽预算时同样不返回截断结果。

    Returns:
        :class:`AnalysisResult`。
    """
    # 1) 整批先校验（原子失败语义）
    umi_len, read_len = _validate_batch(reads)

    # 2) 分组计数
    groups = _group_by_umi(reads)
    umi_counts = {u: len(v) for u, v in groups.items()}

    # 3) 吸收聚类
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
    )
