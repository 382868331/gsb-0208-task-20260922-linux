# umicorrect —— UMI 测序分子的错误吸收与共识

离线、纯标准库实现的可复用题库，提供两套入口：

1. **单端 UMI**：对固定长度 UMI + 等长 DNA 读段的批次做 **UMI 错误吸收
   聚类** 与 **逐碱基质量加权对数似然共识**（首轮能力，契约不变）。
2. **双端 UMI + 坐标 + 链方向**（本次迭代）：规范化双端 UMI、坐标与链
   方向，**保留正反链身份**，在明确的匹配规则下成对，并复用旧的吸收
   聚类与共识入口；重复记录按 `record_id` 处理，冲突显式报错、绝不静默
   合并。

- **Python 版本**：Python 3.14.7（Linux 原生开发验证；3.10+ 语法即可运行，但验收以 3.14.7 为准）
- **依赖**：无第三方依赖，仅标准库（`math`、`types`、`dataclasses`）。安装 Python 后直接运行，无需联网、无需账号/密钥/数据库/中间件/Docker。
- **示例数据**：由 `demo.py` 内确定性代码生成，不读取任何外部附件或在线语料。

## 准确命令

```bash
# 必要测试（70 个用例：手算似然、穷举小有向图参考、丰度桥、相等根、
# 低质量多数、空读段拒绝，以及双端 UMI 规范化/成对/去重/冲突等）
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

### 双端 UMI + 坐标 + 链方向入口（本次迭代）

```python
from umicorrect import TaggedRead, analyze_tagged_reads

rec = TaggedRead(
    umi1="AC", umi2="GT",          # 两端等长 UMI（各 1..32，长度彼此相等）
    coordinate=500,                # 基因组坐标，整数 0..2^31-1（只做相等/分组）
    strand="-",                    # '+' 正链 / '-' 反链；链身份保留，不跨链合并
    sequence="ACGA",               # 读段（1..1000，可与 UMI 不等长）
    qualities=(7, 30, 30, 30),     # 0..40
    record_id="m-0",               # 非空稳定 ID；重复记录按 ID 处理
)
result = analyze_tagged_reads([rec, ...], probe_budget=200_000_000)

result.duplicates_removed   # 按 ID 折叠掉的“内容完全一致”重复提交条数
result.nodes_before         # 各 (坐标,链) 分区内规范化双端 UMI 种数之和
result.molecules_after      # 分子总数（坐标或链不同即为不同分子）
for mol in result.molecules:
    mol.coordinate, mol.strand   # 分子所属坐标与链（链身份不合并）
    mol.root_umi                 # 规范化根双端 UMI："umi1|umi2"（正链空间）
    mol.members                  # 成员规范化双端 UMI（含根，字典序）
    mol.record_ids               # 成员记录 ID（按输入下标升序）
    mol.read_indices             # 去重后成员记录在输入批次中的下标
    mol.consensus_seq            # 正链空间共识（规则与旧共识入口完全一致）
    mol.confidence_gap           # 逐位置 log10 信心差
```

| 接口 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `analyze_tagged_reads(records, *, probe_budget=...)` | `list[TaggedRead]` | `TaggedAnalysisResult` | 校验→按 ID 去重→规范化→分区→旧吸收聚类→旧共识 |
| `reverse_complement(seq)` | A/C/G/T 串 | 反向互补串 | 公开的规范化原语，可单独复用 |

**规范化（反链记录）**：`strand == '-'` 的记录，其 `umi1`、`umi2`、`sequence`
分别取反向互补，质量元组随序列位置整体反转，使所有碱基进入统一的正链
坐标空间；原始链身份仍记录在 `strand` 字段并作为分区键保留。正链记录
原样使用。`reverse_complement` 是对合变换（`rc(rc(x)) == x`）。

**明确的成对（匹配）规则**：两条记录可能进入同一分子，当且仅当

1. `coordinate` 相同；
2. `strand` 相同（**正反链永不成对**，即便规范化后双端 UMI 完全相同）；
3. 规范化双端 UMI 组合键 `n1 + "|" + n2` 之间 **Hamming 距离恰为 1**
   ——即恰好任一端相差一个碱基；两端各差一个（组合距离 2）不成对；
4. 满足旧的丰度阈值与有向吸收规则（`c >= 2s - 1`、按全序首个最高根
   BFS 归属、低丰度桥不合并两大根、传递可达）。

聚类在每个 `(coordinate, strand)` 分区内独立进行，内部调用的正是旧
`cluster_umis` 的同一吸收核心与旧 `consensus` 共识入口；`probe_budget`
为所有分区**共享累计**的探测预算，任一分区耗尽即整批抛
`SearchBudgetExhaustedError`（结果未知），不返回跨分区的截断结果。

**重复记录按 ID 处理**：

- 同一 `record_id` 多次出现且全部字段（umi1/umi2/coordinate/strand/
  sequence/qualities）完全一致：视为同一记录的重复提交，折叠到首次
  出现（计入 `duplicates_removed`，丰度只算一次）。
- 同一 `record_id` 内容冲突：抛 `DuplicateRecordError`，消息给出两次
  出现下标（`records[i] 与 records[j]`）与首个冲突字段及两边取值，
  **不选择任一版本、不静默合并**。

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

单端入口（`Read` / `analyze_reads`）：

| 参数 | 上限/范围 |
|---|---|
| UMI 长度 | 1..32 |
| 读段长度 | 1..1000，且与 UMI 等长 |
| 批次数 | ≤ 1,000,000 条 |
| 单碱基质量 Q | 整数 0..40 |
| 字母表 | UMI 与序列均仅允许大写 A/C/G/T（拒绝 N 与小写） |
| `probe_budget` | 非负整数，默认 200,000,000（覆盖约 2 万个去重 UMI 的最坏两两比较） |

双端入口（`TaggedRead` / `analyze_tagged_reads`）：

| 参数 | 上限/范围 |
|---|---|
| `umi1` / `umi2` 长度 | 各 1..32，且两端长度必须相等 |
| `coordinate` | 整数 0..2,147,483,647（拒绝 bool） |
| `strand` | 仅 `'+'` / `'-'` |
| 读段长度 | 1..1000（**不要求**与 UMI 等长） |
| `record_id` | 非空 `str` |
| 批次数 / 质量 / 字母表 / `probe_budget` | 同单端（预算跨所有坐标/链分区共享累计） |

超过长度/批次上限抛 `LimitExceededError`。上限是接口边界，`demo.py`
不穷举上限规模。

聚类复杂度：单分区内去重 UMI 数为 K、UMI 长 L 时，时间 O(K²·L)，最坏边数
O(K²)；双端入口在每个 `(coordinate, strand)` 分区内分别满足此界。

## 错误语义

所有异常继承 `UMICorrectError`：

- `InvalidArgumentError(ValueError)`：参数被**证明非法**——类型错误、
  空批次、空 UMI/序列、长度不等或跨读段不一致、非法碱基、Q 越界、
  **数值为 NaN/Infinity**、**整数参数传 bool**（`True/False` 不会被当作
  1/0）等。字段级非法在 `Read(...)` 构造期即拒绝；批级错误消息带
  `reads[i].field[j]` 形式的可定位前缀，不做静默纠正。
- `InvalidReadError(InvalidArgumentError)`：批次中某条读段/记录非法，带
  `reads[i]` / `records[i]` 前缀。
- `DuplicateRecordError(InvalidReadError)`：双端批次中同一 `record_id`
  的记录内容冲突；消息定位到两次出现的下标与首个冲突字段。内容完全一致
  的重复提交不是错误，按 ID 折叠并计入 `duplicates_removed`。
- `LimitExceededError`：超过上表硬上限。
- `SearchBudgetExhaustedError(RuntimeError)`：Hamming 探测预算耗尽，
  **结果未知**（异常对象带 `probes_used`、`budget`），放大
  `probe_budget` 重算即可。它与“证明非法/无解”严格分开：合法输入的
  吸收过程必然产出一个划分（最差为每节点各自成根），本问题不存在
  “证明无解”的返回。

**原子性**：`analyze_reads` / `analyze_tagged_reads` 为无状态纯函数，
先对整批完成校验（双端还包括按 ID 的冲突检测）再构建任何结果；任一读段
非法、ID 冲突或预算耗尽都直接抛错，不修改输入、不存在未说明的部分结果。

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
- `tests/test_tagged.py`（本次迭代）：
  - `reverse_complement` 手算（`AACG→CGTT`、对合性）；
  - **反链规范化**：反链双端 UMI/序列/质量进入正链空间，共识与逐位置
    信心差按公式手算/独立参考核对；
  - **链身份保留**：同坐标正反链规范化键完全相同仍是两个分子；
  - **反向互补相同但不同分子**：同正链下 `AC|GT` 与其整体反向互补键
    `GT|AC` Hamming=4，不合并；
  - 成对规则：单端单碱基差吸收、两端各差一个（距离 2）不合、阈值
    `2s-1` 边界两侧、不同坐标不合、不同链不合、手算实例对独立参考；
  - 反链簇内成对吸收并对独立参考；
  - 按 ID：完全一致重复折叠（保留下标/计数语义）、逐字段冲突定位、
    冲突失败的原子性与输入不变；
  - 校验：空批次、非 TaggedRead、双端 UMI 不等长、非法碱基、空 ID、
    非法 strand、coordinate 越界与 bool、质量越界、跨记录长度不一致、
    上限异常、probe_budget 类型/NaN；
  - 预算跨分区共享耗尽≠无解、输入重排确定性、40 个随机批次（独立
    规范化 + Floyd 参考 + 独立似然参考，期望值不来自被测核心）。
- `tests/reference.py` 是**独立参考**：穷举节点对 + Floyd-Warshall
  传递闭包做聚类、按定义逐读段累计似然做共识。它不导入 `umicorrect`，
  期望值不来自被测核心；往返/划分完备性测试仅作补充。

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
5. **双端 UMI 的“成对”只按 Hamming=1 的错误吸收建模**：组合键
   `umi1|umi2` 的单碱基差即“任一端一个错误”；不实现接头结构解析、
   UMI 换位（swapped UMI）、碰撞概率推断或多分子自动拆分（本次迭代明确
   不做）。坐标是不透明整数分组键，本库不解释 0/1-based 体系或区间重叠；
   正反链视为不同分子身份，不做跨链证据合并。

## 已知限制

- 单端入口仅支持固定长度 UMI 与等长读段；双端入口两端 UMI 等长、读段
  长度独立；均不支持变长 UMI、插入/缺失（indel）造成的长度变化。
- 错误模型为“每位置独立、错误均匀分给其余 3 种碱基”；不建模上下文相关
  错误率或 UMI 接头结构。双端入口只对双端 UMI 与读段做反向互补规范化、
  保留并按链分区；**不**合并正反链证据，也不推断碰撞概率、不自动拆分
  同一规范化键下的多个真实分子。
- 坐标为不透明相等键：不解释 0/1-based、不做区间/邻近匹配，坐标不同即
  不成对；`coordinate` 上限为 2^31-1。
- 共识序列不输出 IUPAC 模糊码，证据并列一律 N；Q=0 呼中碱基似然为 -∞。
- 默认 `probe_budget=2e8` 为所有分区共享累计，覆盖单分区约 2 万个去重
  UMI 的最坏两两比较；分区多或去重 UMI 数更大时需显式放大预算，耗尽会
  显式抛错而不是返回截断结果。
- 无并行/外存实现，单分区 K（去重 UMI 数）非常大时 O(K²) 邻接构建可能
  较慢；题面上限内为内存内计算。

## 仓库结构

```
umicorrect/            本题库（errors.py 异常；core.py 核心算法与双端管线）
tests/reference.py     独立参考（穷举边 + Floyd 闭包；逐读段似然）
tests/test_likelihood.py / test_clustering.py / test_pipeline.py  首轮测试
tests/test_tagged.py   本次迭代：双端 UMI/坐标/链/ID 测试
demo.py                固定输入演示（单端 + 双端两段）
requirements.txt       无第三方依赖（仅注释说明）
```
