"""UMI 错误吸收与共识库 —— 异常体系。

所有本库抛出的异常都继承 :class:`UMICorrectError`，便于调用方统一捕获。
错误分两类，严格区分：

* :class:`InvalidArgumentError` / :class:`InvalidReadError`
  —— 输入被**证明**违反契约（确定性拒绝，与搜索过程无关）。
* :class:`SearchBudgetExhaustedError`
  —— 搜索预算耗尽，结果**未知**（不是“证明无解”）。本问题的吸收过程
  对任意合法输入都必然产出一个划分（最差也是每节点各自成根），不存在
  “证明无解”这一结果；预算与无解因此在类型层面就不会混淆。
* :class:`LimitExceededError` —— 超过 README 声明的硬输入上限。
"""

from __future__ import annotations


class UMICorrectError(Exception):
    """本库所有异常的公共基类。"""


class InvalidArgumentError(UMICorrectError, ValueError):
    """标量/参数非法（类型、取值、NaN/Infinity、bool 冒充整数等）。"""


class InvalidReadError(UMICorrectError, ValueError):
    """批次中某条读段/记录非法。消息带 ``reads[i]`` / ``records[i]`` 前缀。"""


class DuplicateRecordError(InvalidReadError):
    """同一 record_id 出现内容不同的记录（ID 冲突）。

    相同 ID 必须指向同一条记录：内容完全一致的重复提交按 ID 去重（保留首
    次出现），内容冲突直接抛错，消息给出两次出现的下标与首个冲突字段，
    绝不静默合并。继承 :class:`InvalidReadError`（批级、可定位），后者再继承
    :class:`InvalidArgumentError`。
    """


class LimitExceededError(UMICorrectError):
    """超过公开契约声明的输入规模上限。"""


class SearchBudgetExhaustedError(UMICorrectError, RuntimeError):
    """Hamming 邻居探测预算耗尽；吸收结果未知，需放大预算重算。"""

    def __init__(self, message: str, *, probes_used: int, budget: int) -> None:
        super().__init__(message)
        self.probes_used = probes_used
        self.budget = budget
