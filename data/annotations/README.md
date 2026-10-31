# 跨意图多操作语料及导出

[`mixed_intents.jsonl`](mixed_intents.jsonl) 用于先保存一句话包含多个不同意图的原始输入与完整标注。
目前有 **40 条人工构造样本**：`accounting + reminder`、`accounting + todo`、`reminder + todo` 各 12 条，
三类同时出现 4 条。它是预备语料，尚未指定训练/验证划分，也没有模型评测结果。
JSONL 是标注源数据；默认导出到 [`../labeling/mixed_intents_data.txt`](../labeling/mixed_intents_data.txt)，
随后使用 `-i data/labeling` 训练时会读取这份多标签文本，JSONL 本身不直接传给训练脚本。

## 文件格式

使用 UTF-8 JSONL：每行一个完整 JSON 对象，不加外层数组、注释或空行。例如：

```json
{"text":"记录午饭30元，顺便提醒我明天开会","intents":["accounting","reminder"],"routing_label":"accounting"}
```

| 字段 | 类型 | 约定 |
| --- | --- | --- |
| `text` | 非空字符串 | 保留用户原始整句，包括语序、省略、口语和标点；不预先拆句 |
| `intents` | 字符串数组 | 完整的意图标注，至少包含两个不同类别；每个标签只写一次 |
| `routing_label` | 字符串或 `null`，可省略 | 暂存的单路由提示；非空值必须属于 `intents`，不代表全部意图 |

当前类别取值为 `accounting`、`reminder`、`todo`，数组按该顺序保存，顺序不表示用户操作顺序或类别优先级。
多次创建账户仍只对应一个 `accounting` 标签；操作数量与标签数量分别统计。
`unknown` 是已支持类别之外的兜底，不与具体类别拼成多标签。包含尚未定义类别的句子应先审阅、明确类别后再加入。

## 标注边界与单路由提示

按独立、有效的请求标注，不能只根据句中出现的关键词增加标签：

- “记录午饭30元，并提醒我明天开会”包含 `accounting`、`reminder`。
- “把吃药加入待办，并设置早上八点的吃药提醒”明确要求两个操作，包含 `reminder`、`todo`。
- “提醒我明天记账”只有 `reminder`；“把创建支付宝账户加入待办”只有 `todo`。这些单意图句不放入本文件。
- 否定、引用或咨询中提及某类操作，不自动构成该类请求；拒绝创建提醒也不等于取消一个已有提醒。

本批样本沿用此前讨论的“含独立记账请求时，可优先交给记账入口”作为暂存提示：
包含 `accounting` 时，`routing_label` 为 `accounting`；仅有 `reminder + todo` 时暂为 `null`，等待确定单路由策略。
这是标注源数据的提示约定；本次导出没有重训现有模型或改变运行中的分流行为。
`null` 表示尚未确定单路由，**不表示 `unknown`、没有意图或缺少完整标注**。
路由策略以后可以重新生成，完整意图标注始终以 `intents` 为准。

## 导出多标签 fastText 文本

从项目根目录运行独立命令，默认读取 `data/annotations/mixed_intents.jsonl`，
生成 `data/labeling/mixed_intents_data.txt`：

```bash
make export-mixed-intents
# 覆盖输入和输出路径；路径包含空格时用引号包住变量值
make export-mixed-intents MIXED_INPUT=data/annotations/mixed_intents.jsonl MIXED_OUTPUT=/tmp/mixed_intents.txt
# 直接运行脚本，默认路径与 make 命令相同
python3 scripts/export_mixed_intents.py
python3 scripts/export_mixed_intents.py --input data/annotations/mixed_intents.jsonl --output /tmp/mixed_intents.txt
```

脚本仅使用 Python 标准库，无需安装或加载 fastText；`-i` / `-o` 分别是 `--input` / `--output` 的简写。
Make 目标每次执行都会重新校验并导出，不依赖输出文件的修改时间。
导出只生成文本，不触发训练；默认输出直接位于 `data/labeling/`，随后目录训练会加载这个新 `.txt`。
如仅需检查导出，可通过 `MIXED_OUTPUT` 或 `--output` 指定目录之外的路径。

每条原句只输出一次，保留 `intents` 的全部标签，按 `accounting`、`reminder`、`todo` 的固定顺序排列。
上面的 JSONL 示例对应输出：

```text
__label__accounting __label__reminder 记录午饭30元，顺便提醒我明天开会
```

`routing_label` 不决定导出的标签，值为 `null` 的样本也保留全部 `intents`。例如：

```json
{"text":"提醒我明早吃药，再把买药加入待办","intents":["reminder","todo"],"routing_label":null}
```

```text
__label__reminder __label__todo 提醒我明早吃药，再把买药加入待办
```

导出前会校验全部记录：文本必须非空、可编码为 UTF-8，且不含换行、`__label__` 标签注入或独立的句尾 token `</s>`；
`intents` 至少包含两个不同的受支持标签，重复标签、未知标签与格式错误会报错。
空文件、空行和重复 JSON 字段也会报错。重复原句会报错；Unicode NFKC 与连续空白折叠后相同的文本同样视为重复。
查重保留标点，以区分 `3.5` / `35`、`A-1` / `A1` 等不同内容；导出保留原句，不使用查重后的文本。
全部记录校验成功后才原子替换输出文件；失败时返回非零状态并保留已有输出，避免留下半份语料。

JSONL 是完整标注的源数据，导出文本是生成文件；当前 **40 条源样本对应 40 行输出**，不作为额外 40 条语料重复计数。
默认训练目录共有 2042 条 `.txt` 样本（2002 条存量单标签 + 40 条混合），涉及 4 类标签。
两份各 400 条的提醒/待办 `.text` 单意图语料仍不会自动加载；连同它们，五份语料共 2842 条。
这份语料尚未划分训练/验证集；正式使用前应按原句及改写来源分组，避免近似改写跨集合，
并与现有单意图语料做重叠检查。正式多标签训练前还需迁移旧 `unknown` 中的提醒和待办标签、补齐评估样本，
并显式使用 `--loss ova`；本次只导出，没有运行训练。

当前脚本只导出多标签文本。单标签分流导出留待后续明确路由策略：遇到 `routing_label: null` 时，
应要求明确策略或跳过并报告，不能静默取第一个意图或当作 `unknown`。
