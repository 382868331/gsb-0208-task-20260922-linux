# umicorrect —— UMI 测序分子的错误吸收与共识

离线、纯标准库实现的可复用题库：对固定长度 UMI + 等长 DNA 读段的批次做
**UMI 错误吸收聚类** 与 **逐碱基质量加权对数似然共识**。
本轮新增**可选编辑距离吸收模型**（一次插入/删除 + 替换，总编辑数 ≤ 2，
路径概率按质量加权并对全部最短路径求和；默认关闭，原行为不变）。

- **Python 版本**：Python 3.14.7（Linux 原生开发验证；3.10+ 语法即可运行，但验收以 3.14.7 为准）
- **依赖**：无第三方依赖，仅标准库（`math`、`dataclasses`）。安装 Python 后直接运行，无需联网、无需账号/密钥/数据库/中间件/Docker。
- **示例数据**：由 `demo.py` 内确定性代码生成，不读取任何外部附件或在线语料。

## 准确命令

```bash
# 必要测试（61 个用例：手算似然、穷举小有向图参考、丰度桥、相等根、编辑路径枚举核对、歧义并列等）
python -m unittest discover -s tests -v

# 固定输入演示（约 0.03 秒，远小于 8 秒；真实计算，无 sleep、无预录结果）
python demo.py
```

# 本轮新增：编辑距离吸收模型演示（正常 + 边界，真实计算 < 1 秒，远低于 90 秒上限）
python demo_iteration.py
```

## 环境准备（venv）

```bash
python3.14 -m venv .venv        # 任意 Python 3.10+ 均可，验收以 3.14.7 为准
source .venv/bin/activate       # 可选；库无第三方依赖，系统 Python 也能直接跑
python -m unittest discover -s tests -v
```

无需 pip install 任何包（requirements.txt 仅为说明性注释）。

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

### 可选编辑距离模型（本轮新增，默认关闭）

`cluster_umis` 与 `analyze_reads` 新增关键字参数（默认值 = 原行为，
不传参的调用结果与上一版逐位一致）：

| 参数 | 取值 | 说明 |
|---|---|---|
| `error_model` | `"hamming"`（默认）/ `"edit"` | 吸收判定模型 |
| `umi_error_profiles` | `dict[str, tuple[float, ...]]` 或 `None` | 仅 `cluster_umis` 的 edit 模式：每个 UMI 的逐位置错误率 e_j∈[0,1]；`None` 时统一用 `DEFAULT_EDIT_ERROR_RATE = 0.01`。`analyze_reads` 总是从批次质量分自动计算（组内 10^(-Q/10) 的逐位置均值） |
| `ambiguity_margin` | 非负有限实数，默认 `0.0` | 仅 edit 模式：最佳与次佳候选的路径概率对数差（log10 域）`<=` 该阈值时判歧义，节点保留未归并 |

另新增可复用函数 `edit_path_probability(u, v, err_profile)`：返回
`(最小编辑数, 最短编辑路径概率和)`，模型内不可达返回 `None`。

```python
from umicorrect import analyze_reads, cluster_umis, edit_path_probability

cluster_umis({"AAAA": 100, "AAA": 10}, error_model="edit")   # 缺 1 碱基也被吸收
edit_path_probability("AAAA", "AAA", (0.001, 0.001, 0.001))  # -> (1, 0.004)：4 条等价删除路径之和
result = analyze_reads(reads, error_model="edit", ambiguity_margin=1.0)
```

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

### 4. 可选编辑距离吸收模型（error_model="edit"）

在原有 Hamming==1 规则之外，可选启用编辑距离模型，契约如下：

1. **路径模型**：允许的编辑为替换、一次插入、一次删除；总编辑数 ≤ 2，
   且插入/删除合计至多 1 次（不允许两次 indel）。一条“路径”是一个
   对齐（编辑列序列）。同聚物等重复碱基产生的多种等价对齐是不同路径，
   概率**全部累计**（DP 状态内对并列最小编辑数的路径概率求和），
   归属不由候选遍历顺序决定。只累计**最短**编辑路径的概率和。
2. **概率因子**（err_profile 为被观测小节点 v 的逐位置错误率 e_j）：
   替换（观测位置 j）= e_j/3；插入（观测位置 j）= e_j/3；
   删除（无观测碱基）= mean(err_profile)。
3. **候选与归属**：先按丰度方向（全序在前的大节点 → 在后的小节点，
   仍要求 `count[u] >= 2*count[v] - 1`）找出编辑距离 ≤2 的全部候选；
   再按各候选的最短路径概率和（log10 域）取最佳者建边。最佳与次佳
   差距 `<= ambiguity_margin` 时判歧义，该节点**保留未归并**（自成一根），
   避免强行吸收；默认 `ambiguity_margin=0.0`，即仅精确并列判歧义。
   建边后仍按原全序 BFS 传递吸收，低丰度桥同样不可能合并两大根。
4. **变长输入**：edit 模式下 `cluster_umis` 接受长度相差 1 的 UMI 键，
   `analyze_reads` 接受变长 UMI 批次（每条读段仍须序列与自身 UMI 等长）；
   共识在**簇根坐标系**上计算：每条读段用自身质量画像求到根的代表
   对齐（最短路径中概率最大者，确定性），匹配/替换列计入对应根位置，
   插入列忽略、删除列不贡献。
5. **默认不变**：不传 `error_model` 时行为与上一版完全一致（hamming 模式，
   变长 UMI 依旧被拒绝）。

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

edit 模式下允许同一批次内 UMI 长度相差 1（indel 错误拷贝）；其余上限不变。

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
- `tests/test_edit_distance.py`（本轮新增）：同聚物等价对齐概率和手算
  （4 条删除路径、4 条插入路径、删除+替换组合）、长度差 1 吸收、
  两个高丰度候选并列保留未归并、歧义阈值两侧、质量画像决定归属
  （与遍历顺序无关）、输入置换不变性、4000 对短串对独立枚举参考核对、
  60 个随机实例对独立聚类参考、默认模式回归。
- `tests/reference.py` 是**独立参考**：穷举节点对 + Floyd-Warshall
  传递闭包做聚类、按定义逐读段累计似然做共识。它不导入 `umicorrect`，
  期望值不来自被测核心；往返/划分完备性测试仅作补充。
  本轮新增 `reference_edit_path_probability`（递归枚举全部对齐路径）与
  `reference_clusters_edit`（全对枚举候选 + Floyd-Warshall），同样不导入
  被测核心。

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

## 已知限制

- 默认模式仅支持固定长度 UMI 与等长读段；插入/缺失（indel）造成的
  长度变化仅在 `error_model="edit"` 下支持，且编辑模型限定为
  “至多一次 indel + 替换，总编辑数 ≤ 2”，**不处理任意长度的一般比对**。
- edit 模式的删除概率用观测节点画像均值近似（删除的碱基无质量分可读）；
  变长共识采用“对齐到簇根”的代表对齐，插入碱基不进入共识。
- 错误模型为“每位置独立、错误均匀分给其余 3 种碱基”；不建模上下文
  相关错误率、链特异性、双重 UMI 或 UMI 接头结构。
- 共识序列不输出 IUPAC 模糊码，证据并列一律 N；Q=0 呼中碱基似然为 -∞。
- 默认 `probe_budget=2e8` 覆盖约 2 万个去重 UMI 的最坏两两比较；去重
  UMI 数更大时需显式放大预算，耗尽会显式抛错而不是返回截断结果。
- 无并行/外存实现，K（去重 UMI 数）非常大时 O(K²) 邻接构建可能较慢；
  题面上限内为内存内计算。

## 仓库结构

```
umicorrect/          本题库（errors.py 异常；core.py 核心算法）
tests/               unittest 用例 + 独立参考 reference.py
demo.py              固定输入演示
demo_iteration.py    本轮新增：编辑距离模型演示（正常 + 边界）
requirements.txt     无第三方依赖（仅注释说明）
```
