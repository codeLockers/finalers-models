# 跨意图多操作语料及导出

[`mixed_intents.jsonl`](mixed_intents.jsonl) 保存一句话包含多个独立意图时的原始输入和完整标注。当前有 **68 条**人工样本：账单、资产账户、提醒、待办的组合均有覆盖，其中新增了账单与资产账户同时出现的句子。

JSONL 是标注源数据；默认导出到 [`../labeling/mixed_intents_data.txt`](../labeling/mixed_intents_data.txt)，随后使用 `-i data/labeling` 训练时会读取这份多标签文本。

## 文件格式

使用 UTF-8 JSONL，每行一个完整 JSON 对象：

```json
{"text":"今天吃早饭花了50元，明早提醒我带饭盒","intents":["bill","reminder"],"routing_label":"bill"}
```

| 字段 | 类型 | 约定 |
| --- | --- | --- |
| `text` | 非空字符串 | 保留用户原始整句，包括语序、省略、口语和标点；不预先拆句 |
| `intents` | 字符串数组 | 完整意图标注，至少包含两个不同类别；每个标签只写一次 |
| `routing_label` | 字符串或 `null` | 暂存的单路由提示；非空值必须属于 `intents`，不代表全部意图 |

支持的类别是 `asset_account`（记账本资产/负债账户的增删改查和管理）、`bill`（已发生的收支账单记录）、`reminder` 和 `todo`。数组按 `asset_account`、`bill`、`reminder`、`todo` 的固定顺序保存，顺序不表示用户操作顺序。

“创建账户、查询余额、更新余额、删除账户”等立即执行的账户操作标 `asset_account`；“早餐花了50元、工资到账8000元、补记停车费”等已发生收支记录标 `bill`。只有同一句包含两个独立请求时才同时标两个类别。

## 标注边界

- “记录午饭30元，并提醒我明天开会”包含 `bill`、`reminder`。
- “在账本新建支付宝账户，再提醒我核对余额”包含 `asset_account`、`reminder`。
- “新建现金账户，同时记录晚饭支出”包含 `asset_account`、`bill`。
- “把创建支付宝账户加入待办”只有 `todo`；“提醒我明天记账”只有 `reminder`，这类单意图句不放入本文件。
- 咨询账单统计、支付状态或如何操作软件，不自动标为 `bill`；咨询如何创建资产账户也不自动标为 `asset_account`。

`routing_label` 只做暂存的分流提示：账单句使用 `bill`，账户操作句使用 `asset_account`；仅有 `reminder + todo` 时可为 `null`。它不决定导出的标签，`null` 也不表示 unknown。

## 导出多标签 fastText 文本

从项目根目录运行：

```bash
make export-mixed-intents
python3 scripts/export_mixed_intents.py --input data/annotations/mixed_intents.jsonl --output /tmp/mixed_intents.txt
```

脚本仅使用 Python 标准库。每条原句只输出一次，保留 `intents` 的全部标签，并按 `asset_account`、`bill`、`reminder`、`todo` 排序：

```text
__label__bill __label__reminder 今天吃早饭花了50元，明早提醒我带饭盒
```

导出前会校验字段、标签、重复文本、控制字符和 fastText 标签注入；全部记录成功后才原子替换输出文件。默认训练目录目前包含 68 行混合语料；JSONL 与导出文本不重复计数。
