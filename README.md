# umicorrect —— UMI 测序分子的错误吸收与共识

离线、纯标准库实现的可复用题库，提供两层能力：

1. **单端层**：对固定长度 UMI + 等长 DNA 读段的批次做
   **UMI 错误吸收聚类** 与 **逐碱基质量加权对数似然共识**；
2. **双端层（本次迭代新增）**：对带**双端 UMI、基因组坐标、链方向**的
   记录做参考方向规范化，在**显式匹配规则**下成对，并**直接调用旧共识
   入口** 产出每组共识；链身份保留，反向互补相同但链不同的记录仍是两个
   分子；记录级重复按 ID 显式拒绝，冲突不静默合并。

- **Python 版本**：Python 3.14.7（Linux 原生开发验证；3.10+ 语法即可运行，但验收以 3.14.7 为准）
- **依赖**：无第三方依赖，仅标准库（`math`、`dataclasses`）。安装 Python 后直接运行，无需联网、无需账号/密钥/数据库/中间件/Docker。
- **示例数据**：由 `demo.py` 内确定性代码生成，不读取任何外部附件或在线语料。

## 准确命令

```bash
# 全部测试（74 个用例：旧 42 个手算似然/穷举小有向图参考/丰度桥/相等根/
# 低质量多数/空读段拒绝等，加新 32 个双端规范化/链分离/ID 冲突/坐标与
# 成对边界；旧测试一字未改）
python -m unittest discover -s tests -v

# 固定输入演示（约 0.04 秒，远小于 90 秒；真实计算，无 sleep、无预录结果）
python demo.py
```

## 公开接口

```python
from umicorrect import Read, analyze_reads, cluster_umis, consensus

read = Read(umi="AAA", sequence="ACG", qualities=(30, 25, 20))
result = analyze_reads([read, ...], probe_budget=200_000_000)

result.nodes_before      # 合并前分子数 = 去重 UMI 种数
result.molecules_after   # 合并后分子数（簇数）
result.merged_count      # = nodes_before - molecules_after
for mol in result.molecules:
    mol.root_umi         # 分子根 UMI（首个最高节点）
    mol.members          # 成员 UMI 元组（含根，字典序升序）
    mol.read_indices     # 成员读段在输入批次中的下标（升序）
    mol.consensus_seq    # 共识序列；逐位置最大对数似然并列时为 'N'
    mol.confidence_gap   # 逐位置 (最高ll-次高ll)，log10 域；并列位置为 0.0
```

分层接口（均可单独复用）：

| 接口 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `cluster_umis(umi_counts, *, probe_budget=...)` | `dict[str, int]`（UMI→正整数计数） | `tuple[Cluster, ...]` | 只做有向吸收聚类 |
| `consensus(reads, read_length, *, read_indices)` | 批次 + 一簇读段下标 | `(共识序列, 逐位置gap)` | 只做对数似然共识 |
| `analyze_reads(reads, *, probe_budget=...)` | `list[Read]` | `AnalysisResult` | 校验→计数→吸收→共识 |

`Cluster` 字段：`root`、`members`、`root_count`、`total_count`。

## 双端 UMI / 坐标 / 链：公开接口

```python
from umicorrect import PairedRecord, analyze_paired_records

# record_id, umi1, umi2, contig, start, end, strand, sequence, qualities
rec = PairedRecord("r1", "AAA", "AAC", "chr1", 100, 104, "+",
                   "ACGT", (30, 30, 30, 30))
result = analyze_paired_records([rec, ...])

result.records_in        # 输入记录数
result.molecules_after   # 成对后分子数（匹配键种数）
result.duplicate_count   # = records_in - molecules_after（分子级重复条数）
for mol in result.molecules:
    mol.key              # (contig, start, end, strand, umi_left, umi_right)
    mol.strand           # 链身份原样保留（'+' / '-'）
    mol.umi_left, mol.umi_right   # 参考方向双 UMI
    mol.contig, mol.start, mol.end  # 坐标（端点已升序规整，半开区间）
    mol.record_ids       # 组成员记录 ID（按下标升序）
    mol.record_indices   # 组成员批次下标（升序）
    mol.consensus_seq    # 参考方向共识；并列位置 'N'（复用旧 consensus）
    mol.confidence_gap   # 逐位置 log10 Bayes 因子数量级（旧 consensus 口径）
```

分层入口：

| 接口 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `reverse_complement(seq)` | A/C/G/T 串 | 反向互补串 | 纯函数，非法字符拒绝 |
| `normalize_record(rec, index=0)` | 单条 `PairedRecord` | `NormalizedRecord` | 只做单记录参考方向规范化 |
| `group_paired_records(records)` | `list[PairedRecord]` | `tuple[PairedMolecule, ...]` | 只校验+规范化+成对，不做共识（共识字段为空） |
| `analyze_paired_records(records)` | `list[PairedRecord]` | `PairedAnalysisResult` | 校验→规范化→成对→**调用旧 `consensus`** |

### 双端规范化与匹配规则（契约）

1. **坐标**：`start/end` 为 0-based 半开区间，要求非负且 `start != end`；
   允许反向顺序（`end < start`）传入，规范化时升序规整为 `(min,max)`。
2. **链方向 + 双 UMI**（UMI 按各读段自身 5'→3' 给出）：
   - `+` 链：`umi_left=umi1`，`umi_right=umi2`，序列保持；
   - `-` 链：两端交换并各自反向互补，`umi_left=rc(umi2)`，
     `umi_right=rc(umi1)`，序列反向互补、质量序列反转。
   - 双 UMI 长度 1..32，**两端可不等长**，也不必与插入序列等长。
3. **显式匹配键**：`(contig, start, end, strand, umi_left, umi_right)`
   六元组**全部相等**才是同一分子的重复观测；UMI Hamming=1、坐标差 1、
   contig 不同都**不**成对（本层不做 UMI 错误吸收，也不做跨链折叠）。
4. **保留正反链身份**：`strand` 是键的一部分。一条 `+` 记录与一条 `-`
   记录即使规范化后双 UMI 字面完全相同（含双 UMI 自身回文的更强情形）、
   坐标相同，也判为**两个不同分子**，绝不合并。
5. **成对后调用旧共识入口**：同键记录序列已统一到参考方向，按记录下标
   升序调用 `umicorrect.consensus`，逐碱基质量加权对数似然、并列报 N、
   gap 口径与单端层完全一致。

### 重复与冲突语义（不静默合并）

- **分子级重复（合法）**：不同 `record_id`、匹配键相同 → 同一原始分子
  的多条 PCR/测序观测，自动归组进共识（计入 `duplicate_count`）。
- **记录级重复（非法）**：同一 `record_id` 出现两次 →
  `RecordConflictError`（`InvalidArgumentError` 子类），消息定位两次
  出现的批次下标；即使内容完全一致也拒绝（库无法区分“重发”与“两条
  不同观测误用同号”），不去重、不合并。
- **组内长度冲突（非法）**：同键记录规范化后序列长度不一致 →
  `InvalidArgumentError`，定位两条记录，不截断、不静默合并。
- **原子性**：字段、坐标、链、ID 唯一性等校验在任何分组/共识结果构建
  之前对整批完成；任一记录非法即整批拒绝，无半更新状态（纯函数）。

`PairedRecord` 字段约束：`record_id`/`contig` 为非空 `str`；
`strand ∈ {'+','-'}`；坐标为非负 `int`（拒绝 bool，浮点先拒 NaN/Inf
再拒类型）；序列仅 A/C/G/T、长度 1..1000；质量为 0..40 整数、与序列
等长。批次数上限沿用 `MAX_READS`（1,000,000），UMI 上限 32。

## 算法与契约

### 1. UMI 节点与排序
相同 UMI 的读段合并为一个节点，`count` 为出现次数。节点全序为
**count 降序，count 相同按 UMI 字典序升序**。

### 2. 有向吸收图
对全序中在前的大节点 `u` 与在后的小节点 `v`，当且仅当

1. `Hamming(u, v) == 1`；
2. `count[u] >= 2 * count[v] - 1`；

时建立有向边 `u -> v`。按全序依次取未归属节点为根，沿有向边做确定性
BFS，所有可达且未归属的节点归入该簇（允许 `R→X→Y` 式传递吸收）。

**低丰度桥不会合并两大根**：边总是从大节点指向小节点。若桥 `B`
（count=1）同时与两个大根 `R1`、`R2` Hamming=1，只存在 `R1→B`、
`R2→B` 方向的候选边；`B→R1`、`B→R2` 因方向（count 下降）不可能存在，
因此两个大根之间永不可达。桥按“首个最高根”归属字典序/计数序最先
处理的根。

### 3. 逐簇对数似然共识
每个位置独立。对质量分 `Q ∈ {0,...,40}`，错误率 `e = 10^(-Q/10)`：

- 读段呼中碱基的概率：`1 - e`；
- 其余三种碱基各：`e / 3`。

对 A/C/G/T 累计 `log10` 似然，取最大者为共识碱基；**最高似然精确
并列时输出 `N`**，该位置 `confidence_gap = 0.0`；否则
`confidence_gap = 最高ll - 次高ll`（log10 Bayes 因子的数量级）。
`Q=0` 时 `e=1`，呼中碱基概率为 0（ll = -∞）：单条 Q0 读段在三个未呼中
碱基间并列，输出 N。

## 输入上限（接口边界）

| 参数 | 上限/范围 |
|---|---|
| UMI 长度 | 1..32 |
| 读段长度 | 1..1000，且与 UMI 等长 |
| 批次数 | ≤ 1,000,000 条 |
| 单碱基质量 Q | 整数 0..40 |
| 字母表 | UMI 与序列均仅允许大写 A/C/G/T（拒绝 N 与小写） |
| `probe_budget` | 非负整数，默认 200,000,000（覆盖约 2 万个去重 UMI 的最坏两两比较） |

超过长度/批次上限抛 `LimitExceededError`。上限是接口边界，`demo.py`
不穷举上限规模。

聚类复杂度：去重 UMI 数为 K、UMI 长 L 时，时间 O(K²·L)，最坏边数
O(K²)。

## 错误语义

所有异常继承 `UMICorrectError`：

- `InvalidArgumentError(ValueError)`：参数被**证明非法**——类型错误、
  空批次、空 UMI/序列、长度不等或跨读段不一致、非法碱基、Q 越界、
  **数值为 NaN/Infinity**、**整数参数传 bool**（`True/False` 不会被当作
  1/0）等。字段级非法在 `Read(...)` 构造期即拒绝；批级错误消息带
  `reads[i].field[j]` 形式的可定位前缀，不做静默纠正。
- `InvalidReadError(InvalidArgumentError)`：批次中某条读段非法，带
  `reads[i]` 前缀。
- `RecordConflictError(InvalidArgumentError)`：双端批次中同一
  `record_id` 重复出现（记录级重复），消息定位两次出现下标；与合法的
  分子级重复（不同 ID、同匹配键）严格区分，不静默去重/合并。
- `LimitExceededError`：超过上表硬上限。
- `SearchBudgetExhaustedError(RuntimeError)`：Hamming 探测预算耗尽，
  **结果未知**（异常对象带 `probes_used`、`budget`），放大
  `probe_budget` 重算即可。它与“证明非法/无解”严格分开：合法输入的
  吸收过程必然产出一个划分（最差为每节点各自成根），本问题不存在
  “证明无解”的返回。

**原子性**：`analyze_reads` 为无状态纯函数，先对整批完成校验再构建任何
结果；任一读段非法或预算耗尽都直接抛错，不存在未说明的部分结果。

## 浮点结果与误差口径

- 聚类（计数、Hamming、吸收判定、根归属）全部为**整数精确运算**，无浮点。
- 浮点仅出现在共识：`10**(-Q/10)`、一次除法与 `math.log10`，以及簇内
  逐读段的**有序**累加（按读段下标升序），同输入同次序结果可逐位复现。
- 单碱基概率为 IEEE-754 双精度舍入，相对误差 < 1e-15；在声明上限内
  （单簇至多 10^6 条 × 1000 位置）累计误差远小于 1 个 log10 单位，
  不用宽容差掩盖边界。测试对独立参考用 1e-12 绝对容差比较，并列判定
  使用精确 `==`（对称数据必须报 N，不容差“猜”碱基）。
- 仅改变读段输入顺序会改变同簇浮点求和次序，gap 可能有 ulp 级差异
（测试覆盖并按 1e-12 容差比较）；共识碱基与簇结构不受影响。

## 测试与独立性

- `tests/test_likelihood.py`：手算质量似然（Q20→log10(297) 等）、
  精确并列报 N、Q0 三向并列、**低质量多数被高质量单读推翻**、多位置独立性。
- `tests/test_clustering.py`：
  - **穷举小有向图**：5 个固定 UMI、计数 ∈ {1,2,5} 的全部 3⁵=243 种赋值；
    另加 300 个长度 4 的确定性随机实例；
  - 丰度桥（含桥后再挂下游错误 UMI）、相等根（2 与 2 不合、1 与 1 合）、
    吸收阈值 `2s-1` 的恰好边界两侧、传递链、成员/分子排序、预算语义。
- `tests/test_pipeline.py`：端到端分子数变化、40 个随机批次对独立参考、
  输入重排确定性、空批次/空字段拒绝与定位、原子失败、上限异常类型、
  预算耗尽≠无解。
- `tests/test_paired.py`（新增）：
  - 手算反向互补（回文、`rc(rc(s))=s` 全枚举 4^1..4^4）与拒绝路径；
  - 手算规范化：`+` 链保序、`-` 链交换+双 RC、序列/质量参考方向化、
    坐标反向端点升序；
  - **反向互补相同但链不同 → 两个分子**（含双 UMI 自身回文、原始串也
    相同的更强边界）；同链 `-` 重复成对；坐标反向给入成对；
  - 显式精确匹配：UMI Hamming=1 / 坐标差 1 / contig 不同均不成对；
  - 同组并列报 N、低质量多数被高质量单读推翻、输入重排确定性；
  - 60 条确定性随机记录对测试内独立匹配键/独立似然参考；
  - **记录级重复按 ID 拒绝**（定位两次下标、属 `InvalidArgumentError`、
    原子失败）、同键长度冲突不静默合并；空批次/非记录/空 ID/非法 UMI、
    坐标（负、空区间、float/bool/NaN）、非法链、非法碱基/质量等边界。
- `tests/reference.py` 是**独立参考**：穷举节点对 + Floyd-Warshall
  传递闭包做聚类、按定义逐读段累计似然做共识。它不导入 `umicorrect`，
  期望值不来自被测核心；往返/划分完备性测试仅作补充。双端测试的匹配键
  与序列方向化用测试内独立手写函数（`indep_rc`/`indep_normalized`），
  共识期望复用 `reference_consensus`，均不调用被测规范化实现。

## 设计取舍

1. **有向边 + 传递可达** 而非“根只直接吸收一跳”：错误 UMI 可以链式
   出现（R→X→Y），传递闭包符合真实 UMI 纠错方向；同时边的单向性天然
   杜绝低丰度桥合并两大根，无需特判。
2. **全序 + 有序 BFS** 保证“首个最高根归属”完全确定，且不依赖输入顺序。
3. 并列碱基输出 N 而非按字母表/字典序猜测，避免用确定性假象掩盖证据
   不足；信心差同时给出每个位置的可解释置信度。
4. 质量模型采用对称三备选（每个非呼中碱基 `e/3`），是题面给定的有限
   数学子集；**不宣称**与任何行业标准（如具体测序仪的上下文相关错误
   模型、UMI-tools 等）兼容，也未调用第三方同类实现。
5. 双端层把“方向规范化”与“分子身份”分开：参考方向化只翻转表示，
   **链身份始终保留在匹配键中**，因此不会把正反链上反向互补相同的
   UMI 错误折叠；配对采用显式精确键，吸收/容错仍归单端层职责。
6. 记录级重复与分子级重复分开处理：前者按 `record_id` 显式拒绝
   （冲突不能静默合并），后者才是进共识的合法 PCR 重复。

## 已知限制

- 仅支持固定长度 UMI 与等长读段（单端层）；不支持变长 UMI、插入/缺失（indel）
  造成的长度变化。双端层双 UMI 长度彼此独立（各 1..32），也不要求与插入
  序列等长，但同一匹配键内记录的规范化序列长度必须一致。
- 错误模型为“每位置独立、错误均匀分给其余 3 种碱基”；不建模上下文
  相关错误率。
- 双端层的配对是**显式精确匹配**：不做 UMI Hamming-1 吸收（那是单端
  :func:`cluster_umis` 的职责，两层未自动串联），不做碰撞概率推断、
  不做多分子自动拆分，也不跨链折叠——这是本次迭代刻意限定的范围。
- 双端层不校验比对坐标是否“生物学合理”（如是否越出某参考序列），
  `contig` 仅作不透明名字；坐标只做半开区间与非负校验。
- 共识序列不输出 IUPAC 模糊码，证据并列一律 N；Q=0 呼中碱基似然为 -∞。
- 默认 `probe_budget=2e8` 覆盖约 2 万个去重 UMI 的最坏两两比较（仅
  单端层）；去重 UMI 数更大时需显式放大预算，耗尽会显式抛错而不是返回
  截断结果。
- 无并行/外存实现，K（去重 UMI 数）非常大时 O(K²) 邻接构建可能较慢；
  题面上限内为内存内计算。

## 仓库结构

```
umicorrect/          本题库
  errors.py          异常体系（含新增 RecordConflictError）
  core.py            单端核心：校验、有向吸收聚类、对数似然共识
  paired.py          双端规范化与显式成对（新增；共识直接调用 core.consensus）
tests/               unittest 用例 + 独立参考 reference.py
  test_likelihood.py / test_clustering.py / test_pipeline.py  旧测试（未改）
  test_paired.py     双端新功能测试（新增）
demo.py              固定输入演示（单端 + 双端两段）
requirements.txt     无第三方依赖（仅注释说明）
```
