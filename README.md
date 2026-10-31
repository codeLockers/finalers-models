# fastText 文本分类训练脚本

`scripts/fasttext_training_script.py` 是一个生产可用的 fastText 有监督分类命令行工具，覆盖
**环境自检 → 数据检查 → 分层切分 → 训练/自动调参 → 评估 → 推理 → JSON 报告** 的完整流程。

本文命令均从项目根目录执行；显式传入的相对路径按当前工作目录解析。

## 1. 环境准备

```bash
python3 -m pip install --user -r requirements.txt
```

`requirements.txt`（内含详细的踩坑说明，节选）：

```text
fasttext==0.9.3
numpy<2

# 若 fasttext==0.9.3 的 macOS wheel 无法修复, 可改用:
# fasttext-wheel==0.9.2
```

> 之所以锁 `numpy<2`：fasttext 0.9.3 的 Python 封装使用了 NumPy 2.x 中已移除的
> `np.array(x, copy=False)`。本机选择的是「保留 numpy 2.x + 打补丁 + 脚本内做运行时回退」
> 方案，两条路都能走通，见下一节。

检查环境是否可用（会用内置极小样例训练并预测一次；只检查数据请用 `--dry-run`）：

```bash
python3 scripts/fasttext_training_script.py --check-env
```

### Needle 虚拟环境

```bash
make run-needle-virtual      # 创建 needle-virtual，已存在则复用
make install-needle-virtual  # 自动创建环境，并在其中安装 cactus-needle
```

安装命令直接使用虚拟环境里的 Python；如果还要在当前终端使用该环境，执行
`source needle-virtual/bin/activate`。可通过 `NEEDLE_VENV=目录` 自定义环境位置，
通过 `PYTHON=python3.11` 指定创建环境的 Python 版本。

### Needle 3 中文工具调用演示与评测

`tests/test_needle_toolchain.py` 测试「中文一句话 → 资产账户增删改查工具调用 → 内存中的账户 JSON」。
安装 `cactus-needle` 后，运行内置的 38 条案例：创建 13 条、查询 5 条、修改 11 条、删除 2 条、不调用工具 7 条。

```bash
make test-needle
# 等价于
needle-virtual/bin/python tests/test_needle_toolchain.py
```

四个工具始终同时注册，由模型选择；每句最多调用一次。它们按账户名称定位目标：

| 工具 | 中文用例 | 参数与结果 |
| --- | --- | --- |
| `create_asset_account` | 帮我创建一个支付宝账户，余额12.50元 | `name`、`type` 必填，返回完整 `FCAssetAccountModel` JSON |
| `query_asset_account` | 查询我的微信余额 | `name` 可省略以查全部，`type` 可筛选类型；返回账户 JSON 数组 |
| `update_asset_account_info` | 把工行卡的余额改成500元 | `name` 是原名称，`newName` 用于改名；返回修改后的完整账户 JSON |
| `delete_asset_account` | 从记账应用中删除工行卡账户 | `name` 必须明确且唯一；返回删除状态和被移除的账户 |

案例覆盖 `cash`、`eWallet`、`credit`、`debit`、`investment`、`loan`、`receivable`、`other` 八类账户，
以及中文金额、币种差异、可空字段、账号前导零和默认账户。否定操作、缺少必要参数、收支记录和真实转账请求不应调用工具。
模型提取的 `balance` 必须是最小货币单位整数，例如人民币 12.50 元为 1250、1234 日元为 1234、
科威特第纳尔 1.234 为 1234；创建时未提供的 `currency`、`balance`、`note`、`number` 在 JSON 中为 `null`。
`accountId` 的 UUID 和 `createdAt`/`updatedAt` 的 UTC 时间由宿主程序生成，不要求模型编造。

修改时省略字段或传 `None` 表示不改；明确清空可空字段使用 `clearFields` 数组，支持 `currency`、`balance`、`note`、`number`。
余额为 0 和默认账户为 `false` 都是有效修改。原余额非空时，改变币种必须同时指定新余额或清空余额；
设为默认账户会取消其他账户的默认标记，更新保留原 UUID 和创建时间。

每条案例重置内存账户，并使用独立上下文。`--prompt` 预置微信（123.45 元，默认账户）、工行卡（5000 元）、
招行信用卡（欠款 1500 元）和现金（300 元）；支付宝未预置，可以新建。所有初始金额均为模拟数据。
脚本通过 `complete` 保留首轮真实工具调用及参数，再由宿主校验并执行工具；
打印原始参数、完整结果 JSON、执行后模拟账户，以及工具选择、参数、业务结果和最终状态与预期的差异。
余额换算由模型完成，宿主不会按预期答案修正；非法枚举或带否定标记的调用会被拒绝。
逻辑测试独立验证宿主内存操作，**38 条案例尚未运行真实模型验证**。本测试不落库、不接入真实账本或支付。
`test-needle` 使用 `NEEDLE_VENV` 中现有的 Python，不会自动安装依赖或触发训练。

```bash
# 只看案例和预期字段，无需 Needle SDK 或模型
python3 tests/test_needle_toolchain.py --dry-run
# 测试自定义的一句话
needle-virtual/bin/python tests/test_needle_toolchain.py --prompt '帮我创建一个支付宝账户，余额12.50元'
# 只运行查询案例，仍注册全部四个工具用于识别
needle-virtual/bin/python tests/test_needle_toolchain.py --tool query_asset_account
# 只使用已有缓存，缺少引擎或权重则报错
needle-virtual/bin/python tests/test_needle_toolchain.py --offline
```

首次真实运行仍可能由 SDK 自动下载本机引擎与模型权重；当前环境访问 Hugging Face 的 TLS 问题仍需另行修复。
安装 pip 包本身不代表引擎和权重已缓存。可通过 `--max-new-tokens` 设置单次推理上限。
`--tool` 可选上述四个工具名，不能与 `--prompt` 同用。

`scripts/toolchain_traning_script.py` 还保留可选的简洁演示模式：

```bash
needle-virtual/bin/python scripts/toolchain_traning_script.py --demo
```

`--demo` 用 `@needle.tool` 定义 `add_expense` 和 `query_expense` 两个工具，通过 `agent.run(..., max_steps=4)`
处理四句中文示例，打印原始返回值和模拟工具调用记录；可用 `--demo --prompt "午饭花了30块"` 自定义一句。
工具仅操作内存中的模拟数据，每句独立上下文，不接入真实账本或支付，不计算评测分数，也不写报告文件。
`python3 scripts/toolchain_traning_script.py --demo --dry-run` 只列出示例，不导入 SDK 或加载模型。

`scripts/toolchain_traning_script.py` 默认使用中文输入和中文工具说明，评测
`data/needle/chinese_tool_cases.json` 中的 27 条人工用例。它覆盖收支记录、资产账户查询和管理、
中文金额、缺失参数、多工具调用与依赖前一步结果的工具链；这是一份小规模人工集合，不能当作生产基准。

直接运行脚本可使用这套独立评测：

```bash
needle-virtual/bin/python scripts/toolchain_traning_script.py
needle-virtual/bin/python scripts/toolchain_traning_script.py --schema-language both
needle-virtual/bin/python scripts/toolchain_traning_script.py --prompt "查询我的微信余额"
```

无需手工下载模型或指定 `--weights`：SDK 使用本地缓存，缺少模型或本机引擎时会从 Hugging Face Hub 自动获取；
推理在本机完成，无需 API key。默认评测中的工具返回值由 mock 提供，业务工具均为模拟，不接入真实账本或支付。
`both` 用同一批中文输入分别对比中文和英文工具说明，
输入始终为中文；`--prompt` 是单句观察模式，不计入准确率。

只校验用例和查看预期调用时，无需安装 Needle 或加载模型：

```bash
python3 scripts/toolchain_traning_script.py --dry-run
```

已有 Needle 环境时，也可运行 `needle-virtual/bin/python scripts/toolchain_traning_script.py --dry-run`。
`--offline` 只使用本地模型和引擎缓存，`--weights /path/to/model.cact` 指定已有模型；
`--case 用例ID` 可重复指定用例，`--limit N` 只运行前 N 条。

评分严格比较工具数量、名称、顺序及全部参数，并校验参数类型和约束；依赖工具链只有前一步完全匹配，
才注入 mock 结果继续。`no_call` 只表示没有调用工具，不代表已正确澄清或回答。
完整报告默认写入 `models/needle_chinese.report.json`，包含原始响应（raw）、confidence、suppressed_calls、
加载与推理耗时和逐条比较结果，可用 `--report` 改路径。此处尚未运行模型评测，没有模型分数。

### 本机（macOS + Python 3.9）踩过的两个坑

`fasttext 0.9.3` 的预编译 wheel 在新版 macOS + NumPy 2.x 下有两个真实阻塞问题，已在
site-packages 内就地修复（原文件备份为 `*.orig`，重装 fasttext 后需要重新执行）：

1. **`fasttext_pybind.cpython-39-darwin.so` 链接了不存在的 `@rpath/libc++.1.dylib`**
    → `import fasttext` 直接报 `Library not loaded`。修复：

   ```bash
   SP=~/Library/Python/3.9/lib/python/site-packages
   install_name_tool -change @rpath/libc++.1.dylib /usr/lib/libc++.1.dylib \
       "$SP/fasttext_pybind.cpython-39-darwin.so"
   ```

2. **`fasttext/FastText.py` 使用了 NumPy 2.x 已移除的 `np.array(x, copy=False)`**
    → 预测时报 `ValueError: Unable to avoid copy while creating an array as requested.`。修复：

   ```bash
   python3 - <<'PY'
   import pathlib
   p = pathlib.Path.home() / "Library/Python/3.9/lib/python/site-packages/fasttext/FastText.py"
   src = p.read_text()
   p.with_suffix(".py.orig").write_text(src)
   p.write_text(src.replace("np.array(x, copy=False)", "np.asarray(x)"))
   PY
   ```

   脚本内的 `predict_one()` 还实现了**运行时回退**：一旦捕获到该 `ValueError`，会改走底层
   `model.f.predict(...)`，因此即使没有打上面的补丁，预测功能依然可用。

> 只想避免这些坑的话，也可以直接用 fastText 的 CLI（`./fasttext supervised ...`）或改用
> `pip install fasttext-wheel`。

## 2. 数据格式

标准 fastText 有监督格式：`__label__标签 文本`（多标签时一行写多个 `__label__`）。

```text
__label__accounting 今天午饭花了30块
__label__unknown 明天上午十点提醒我开会
```

仓库自带示例数据放在 `data/labeling/`，**按标签拆成每个分类一个文件**（文件名 `<标签>_data.txt`）；
`-i data/labeling` 会把目录里的 `*.txt` 按文件名排序合并后训练：

| 文件 | 标签 | 含义 | 条数 |
| --- | --- | --- | --- |
| `accounting_data.txt` | `__label__accounting` | 记录/陈述已发生的收支，或查询/管理资产与负债账户 | 859 |
| `unknown_data.txt` | `__label__unknown` | 未知（其余全部兜底到这一类） | 983 |

两个文件合计 **1842 条、2 个标签**，每行都保留 `__label__` 前缀（拆分文件本身就是合法的 fastText
数据），也可以只取其中一个文件训练单类别模型。标签名刻意用英文，避免中文标签在代码 / JSON / 日志里
出现编码与转义问题。数据是**按「说法覆盖维度」铺出来的**（不是按条数堆），判定规则、边界约定、
覆盖矩阵和实测指标都写在 [`data/LABELING.md`](data/LABELING.md)。

新增的 [`data/labeling/reminder_data.text`](data/labeling/reminder_data.text) 包含 **320 条**
`__label__reminder` 待接入语料，覆盖提醒创建、查询、修改、取消和周期/条件触发等说法。
[`data/labeling/todo_data.text`](data/labeling/todo_data.text) 另含 **320 条** `__label__todo` 待接入语料，
覆盖待办创建、查询、修改、完成、删除、清单管理和多个待办；截止时间与提醒通知分开标注。
目录模式只读取 `*.txt`，这些 `.text` 尚未参与默认训练；正式接入前需统一扩展名，并迁移现有训练及留出集中
仍标为 `unknown` 的提醒和待办样本。当前二分类的数量、规则和历史评估仍按原数据口径保留。

### 按标签拆分数据

手上有完整的一份数据时，用 `--split-by-label` 按标签拆成多个文件（对已经拆好的目录再拆一次是幂等的）：

```bash
python3 scripts/fasttext_training_script.py -i /path/to/full.txt --split-by-label data/labeling
# 等价写法：make split-data SRC=/path/to/full.txt LABEL_DIR=data/labeling
```

输出 `data/labeling/<标签>_data.txt`，每行原样保留 `__label__` 前缀；目录里其它 `*.txt` 不会被删除，
但会被 `-i` 一起读入训练，所以发现不是本次生成的历史文件时脚本会**直接报错中止（不写任何文件）**，
按提示删掉或移走这些残留再重跑即可。

另外带一份**独立留出测试集** `data/probe/`（499 条 = accounting 206 / unknown 293），同样按标签拆成
`accounting_data.txt` / `unknown_data.txt`，刻意与训练集**零重叠**，用来量「换说法的泛化」，
而不是在调参数据上自评：

```bash
python3 scripts/fasttext_training_script.py -i data/labeling -t data/probe -o /tmp/m.bin
```

`-t/--test`、`--valid`、`--predict-file` 都支持**目录**：目录里的 `*.txt` 按文件名排序合并后使用
（fastText 只认单文件，脚本会自动合并到临时文件，结束后清理）。所以也可以只挑一个类单独看：
`-t data/probe/accounting_data.txt`。

2026-10-27 恢复资产账户相关样本：微信/支付宝/银行卡余额查询、账户创建、余额/额度修改、账户列表、
改名和归档等均属于 `accounting`。同时修正了旧数据中 21 条余额/额度/账户查询负例。
随后又针对 16 个线上 badcase（账户余额/额度/净资产/账户列表查询，以及算数、充值、免息期、
转账/还款、银行办理、消费统计）补齐成对边界样本，训练集扩到 **1842 条**（accounting 859 /
unknown 983），端侧模型已用当前数据重训，在 499 条独立留出集上 **499/499**。下文同时保留
v2、v3 的历史结果，当前 v4 指标见下文「v4（1842 条）实测」。

> 这是一份**二分类**示例：收支记录及资产账户查询/管理标 `__label__accounting`，其余标 `__label__unknown`。
> 消费统计、账单明细、提醒、实际支付/转账、注册登录账号及实体卡办理等仍属于 `unknown`，边界见标注规范。
> 想换标签名（例如 `bookkeeping` / `other`）直接全量替换即可：
>
> ```bash
> sed -i '' 's/__label__accounting/__label__bookkeeping/g; s/__label__unknown/__label__other/g' \
>     data/labeling/*_data.txt
> ```
>
> 当前训练集正负样本约 1:1.14（v1 是 1:4），评估时重点看 `accounting` 的 P/R，不要只看整体准确率
> （全判 unknown 也有约 0.53）。

> **常见疑问：没命中的不就是 unknown 吗？为什么还要给 unknown 训练数据？**
> fastText 是有监督的**判别式**模型，输出空间 = 训练时见过的标签集合。只喂 accounting 时模型
> 只有 1 个标签，softmax 概率恒为 `1.000`（实测），v1 数据里的 48 条 unknown 会被判成 accounting 48/48——
> 「不命中就 unknown」这个兜底**并不存在**，`--threshold` 也完全失效。
> 「不命中默认 unknown」是规则引擎的语义，要靠自己写关键词表、或换成 `--loss ova` 再给概率加阈值。
> 同一份测试集、20 个随机划分实测的 accounting F1（在 v1 的 60 条数据上）：只训正样本 + `ova` + 调参集选阈值 = 0.583
> （阈值在 0.1~0.7 之间乱跳、不稳定），固定 0.5 = 0.375，关键词规则 = 0.849，
> 而**正样本 + 8 条 unknown = 0.826、正样本 + 32 条 unknown = 0.920**。
> 结论：unknown 不需要和 accounting 配平，但**不能为 0**——它提供的是「不是记账」的分布，
> 尤其是「记一下张三的电话138xxxx1234」这类含数字但非记账的句子。

> **中文提示**：fastText 按空格分词，中文整句会被当成一个 token，实际全靠 `--min-n/--max-n` 字符
> n-gram 学习。**中文未分词时务必用 `--min-n 1`**（脚本默认值就是 `--min-n 1 --max-n 3`），
> 否则单个字的线索（如「花」「元」「块」）会被丢掉，正类可能直接学不出来，详见
> [§4 调参建议](#调参建议)。

## 3. 快速开始

```bash
# 最简：不传任何数据参数, 默认用 data/labeling, 训练并评估(输出 model.bin)
python3 scripts/fasttext_training_script.py

# 显式指定数据与模型输出路径, 自动切分 20% 做验证集
python3 scripts/fasttext_training_script.py -i data/labeling -o models/intent.bin

# 只看流程不训练（打印环境/数据/参数）
python3 scripts/fasttext_training_script.py -i data/labeling --dry-run

# 只体检环境：用 4 条内置样例真的训练+预测一次
python3 scripts/fasttext_training_script.py --check-env

# 自动调参（60s 上限，用验证集选超参数）
python3 scripts/fasttext_training_script.py -i data/labeling -o models/tuned.bin \
    --valid-ratio 0.2 --autotune --autotune-duration 60

# 训练 + 立即对文本/文件预测
python3 scripts/fasttext_training_script.py -i data/labeling -o models/intent.bin \
    --predict "午饭花了30块" --predict "帮我记一下明天十点开会"
python3 scripts/fasttext_training_script.py -i data/labeling -o models/intent.bin \
    --predict-file data/probe > preds.jsonl   # 目录或单个文件都行，JSONL 走 stdout
python3 scripts/fasttext_training_script.py -i data/labeling -o models/intent.bin -I   # 交互模式

# 只用已有模型推理，不训练
python3 scripts/fasttext_training_script.py -m models/intent.bin --predict "午饭花了30块"
python3 scripts/fasttext_training_script.py -m models/intent.bin -t data/probe --predict-file data/probe

# 训练一个给手机端用的小模型：76.5 MB -> 3.9 MB -> 量化后 0.52 MB（留出集指标不变）
python3 scripts/fasttext_training_script.py -i data/labeling -o models/intent_mobile.ftz \
    -t data/probe --dim 20 --bucket 50000 --quantize

# 把已经训练好的 .bin 量化导出成 .ftz（推荐：源 .bin 原样保留，导出后自动校验）
python3 scripts/fasttext_training_script.py -m models/intent.bin --quantize -t data/probe
#   -> 默认输出 models/intent.ftz；也可以只写 -o models/intent.ftz（脚本会自动补上 --quantize）
```

## 4. 关键参数

| 参数 | 说明 |
| --- | --- |
| `-i/--input` | 训练数据：单个文件，或**按标签拆分的目录**（默认 `data/labeling`，目录里的 `*.txt` 按文件名排序合并） |
| `--split-by-label DIR` | 按标签把 `-i` 的数据拆成 `DIR/<标签>_data.txt` 后退出（不训练），可直接 `make split-data SRC=...` |
| `-o/--output` | 模型输出路径（默认 `model.bin`，同时生成 `<模型>.report.json`）；**`-m` 模式下是量化模型的输出路径**，不给则自动推导为 `<源模型>.ftz` |
| `-m/--model` | 载入已有模型（`.bin`/`.ftz`），跳过训练，只做评估/推理；**配合 `--quantize` 则把已有 `.bin` 量化导出 `.ftz`**（不再要求 `--test` 等参数） |
| `-t/--test` | 独立测试集：单个文件，或**按标签拆分的目录**（如 `data/probe`，目录内 `*.txt` 合并读取），优先用于评估 |
| `--valid` / `--valid-ratio` | 验证集路径（同样支持目录）/ 自动切分比例（默认 0.2） |
| `--seed` | 随机种子，保证切分可复现（默认 42） |
| `--dim --epoch --lr --ws --word-ngrams --min-count --loss --neg --bucket --lr-update-rate --thread` | fastText 超参数 |
| `--min-n` / `--max-n` | 字符 n-gram 长度范围，**默认 `1` / `3`**（中文未分词建议保持 `--min-n 1`） |
| `--autotune --autotune-duration --autotune-metric --autotune-predictions` | 自动调参 |
| `--autotune-max-bucket` | autotune 结果 bucket 上限（**默认跟随 `--bucket`**，即默认配置下 200000；0 = 不限制） |
| `--quantize` / `--quantize-dsub` | 乘积量化（PQ）压缩模型：训练后压缩再落盘，或配合 `-m` 把已有 `.bin` 量化导出 `.ftz`；体积/内存约降到 `1/(dim/dsub+1)`（默认 `dsub=2`，dim=100 时约 1/8）。`-o` 以 `.ftz` 结尾时会自动打开 |
| `--quantize-qnorm` / `--quantize-retrain` | 归一化后再量化 / 量化后微调一轮（实测本示例数据都无收益，默认关闭） |
| `-k/--k` / `--threshold` | 评估与预测时取前 k 个 / 概率阈值 |
| `--metrics` / `--max-failures` | 评估输出：`--metrics` 额外打印 P/R/F1 与每标签指标；`--max-failures N` 控制失败样本最多列多少条（默认 20，0 = 全部） |
| `--predict --predict-file -I/--interactive` | 推理方式（可组合） |
| `--work-dir --keep-work-dir` | 切分中间文件目录（默认临时目录，结束自动清理） |
| `--dry-run` / `--check-env` | 只演练数据流程 / 只体检环境 |
| `-q/--quiet` / `--verbose` | 日志级别 / fastText 训练输出级别 |

### 模型体积与端侧部署（实测）

模型体积几乎只由输入矩阵 `(词表 + bucket) × dim × 4 字节` 决定（输出层只有 `标签数 × dim`，
可忽略），所以既能解析估算，也能解析地往小里压：

| dim / bucket | 估算 | 实测文件 |
| --- | --- | --- |
| 100 / 200000（脚本默认） | 76.5 MB | 76.50 MB |
| 100 / 50000 | 19.2 MB | 19.28 MB |
| 100 / 20000 | 7.8 MB | 7.84 MB |
| 20 / 50000 | 3.9 MB | 3.87 MB |
| 20 / 2000 | 0.16 MB | 0.21 MB |
| 100 / 2000000（fastText 官方默认 bucket） | 762 MB | — |

脚本把默认 `--bucket` 从官方的 `2000000` 调到了 `200000`（官方默认在小数据集上会直接产出几百 MB
的模型），并在保存后超过 300 MB 时打印提示。

#### 降 `--bucket` 到 50000 会掉点吗：实测不会

bucket 只决定**哈希碰撞概率**，而碰撞取决于「语料里出现过多少个不同的字符 n-gram」，与样本条数无关：

| `--min-n/--max-n` | 不同字符 n-gram 数 | bucket=50000 的占用 |
| --- | --- | --- |
| 1 / 3（脚本默认） | **6607** | 13% |
| 1 / 4 | 10089 | 20% |
| 1 / 6 | 15988 | 32% |

下表是 v2（631 条训练；留出集 60 条）的历史扫描，用来比较 dim/bucket 的相对趋势；v3 的生产配置
复测见下文「v3（1441 条）历史实测」，当前 v4（1842 条训练 / 499 条留出）见「v4（1842 条）实测」：

| dim / bucket | 体积 | 留出集 P@1 | 8:2 5-seed 均值 | 单 seed 范围 |
| --- | --- | --- | --- | --- |
| 100 / 200000（脚本默认） | 76.50 MB | 1.0000 | 0.9667 | 0.9603 ~ 0.9841 |
| 100 / 50000 | 19.28 MB | 1.0000 | 0.9683 | 0.9524 ~ 0.9841 |
| 100 / 20000 | 7.84 MB | 1.0000 | 0.9714 | 0.9603 ~ 0.9762 |
| 50 / 50000 | 9.65 MB | 1.0000 | 0.9683 | 0.9524 ~ 0.9841 |
| 20 / 50000 | 3.87 MB | 1.0000 | 0.9683 | 0.9524 ~ 0.9841 |
| 25 / 20000 | 1.97 MB | 1.0000 | 0.9699 | 0.9603 ~ 0.9762 |
| 10 / 10000 | 0.42 MB | 1.0000 | 0.9667 | 0.9524 ~ 0.9762 |

76.5 MB 与 0.42 MB 之间的差异落在 seed 噪声里（同一配置换 seed 波动 ±0.03），也就是说
**这份数据撑不起大模型**，`--dim 20 --bucket 50000` 已经是「够用且小」的甜点。

bucket 的下限（固定 dim=20）：

| bucket | 体积 | 留出集 P@1 | 8:2 5-seed 均值 |
| --- | --- | --- | --- |
| 50000 | 3.87 MB | 1.0000 | 0.9683 |
| 20000 | 1.58 MB | 1.0000 | 0.9667 |
| 10000 | 0.82 MB | 1.0000 | 0.9714 |
| 5000 | 0.44 MB | 1.0000 | 0.9683 |
| 2000 | 0.21 MB | 0.9833 | 0.9572 |
| 1000 | 0.13 MB | 1.0000 | 0.9413 |

经验规则：**bucket ≈ 5~10 倍「不同 n-gram 数」**（v2 数据 6607 → 30000 起步）。示例数据上 5000 还很稳、
2000 开始抖、1000 明显掉点（碰撞对数 ~2 万对）。数据再涨 10 倍之前 `50000` 都够用。

`--max-n` 同样在改 n-gram 数量：`maxn=6` 时有 15988 个，此时把 bucket 从 10000 加到 200000 也几乎无差别
（0.9555 / 0.9524），但都比 `maxn=3` 的 0.9683 差 —— 想省体积不要动 `--min-n/--max-n`，先把 `1/3` 定住。

#### 再进一步：乘积量化 `--quantize`（约 1/8，实测无损）

> 下面两张压缩表基于 v2（60 条留出集）；v3 的历史复测见下文「v3（1441 条）历史实测」，
> 当前 v4（1842 条训练 / 499 条留出）的 `dim 20 / bucket 50000 / quantize` 体积约 **0.56 MB**，
> 留出集 499/499，见「v4（1842 条）实测」。

脚本新增 `--quantize`：把每个 subvector 压成 1 字节（乘积量化），体积公式（实测吻合）
`(词表 + bucket) × (nsubq + 1) 字节 + 码本`，其中 `nsubq` 是 fastText 实际切出的 subvector 段数
（`dim` 是 `dsub` 的整数倍时就是 `dim/dsub`；`dim/dsub` 只是「整除时的写法」）：

| dim / bucket | 量化前 | 量化后 | 留出集 P@1（前 → 后） | 8:2 5-seed 均值（前 → 后） |
| --- | --- | --- | --- | --- |
| 100 / 200000 | 76.56 MB | 9.69 MB | 1.0000 → 1.0000 | 0.9683 → 0.9746 |
| 100 / 50000 | 19.34 MB | 2.53 MB | 1.0000 → 1.0000 | 0.9683 → 0.9778 |
| 20 / 50000 | 3.88 MB | 0.52 MB | 1.0000 → 1.0000 | 0.9667 → 0.9730 |

前后用的是同一批切分、可直接对比；这张表的切分方式与上面的 5-seed 扫描不同，**绝对值不要跨表比较**。

* `--quantize-dsub` 越大压得越狠。同一份 dim=20 源模型（未量化 3.87 MB）实测：

  | dsub | 实际切分（nsubq 段） | 体积 | 留出集 P@1 | 逐条一致率 |
  | --- | --- | --- | --- | --- |
  | 1 | 20 段 × 1 维 | 1.00 MB | 1.0000 | 50/50 |
  | 2 | 10 段 × 2 维 | 0.52 MB | 1.0000 | 50/50 |
  | 4 | 5 段 × 4 维 | 0.28 MB | 1.0000 | 50/50 |
  | 8 | 2 段 × 8 维 + 末段 4 维 | 0.18 MB | 1.0000 | 50/50 |
  | 20 / 32 | 1 段 × 20 维（**压缩下限**） | 0.08 MB | 1.0000 | 50/50 |

  示例数据已饱和（留出集 60 条全是 1.0000），**这张表只能说明「压到 0.08 MB 也没崩、也没退化成乱猜」，
  不代表一般任务无损**——用自己的数据一定要用 `-t` 复测。压到下限后码本（`dim × 256 × 4` 字节，
  dim=20 时 20 KB）反而是体积大头。
* `dsub` **不必是 `dim` 的因数，fastText 也不会报错**：`src/productquantizer.cc` 里
  `nsubq = dim / dsub`、末段是 `dim % dsub` 维（**维度不会被丢掉**），`dsub ≥ dim` 时退化成
  1 个「整维度」subvector（每行 1 字节，即压缩下限）。这种取值脚本会打一条提示，说明实际段数与
  每行字节数（`--dim 20 --quantize-dsub 8` → 「3 段(8 + 8 + 4 维), 每行 3 字节」）。
* 代价是**一次性离线时间**：PQ 训练是单线程的，实测 `dim=100/bucket=50000` 约 17s、
  `dim=100/bucket=200000` 约 24s、`dim=20/bucket=50000` 约 3.4s，随 bucket 线性增长。
* `--quantize-qnorm` / `--quantize-retrain` 实测在本示例数据上都没有收益（体积略增或持平、指标不变），
  脚本默认关闭。
* **别用 fastText 的 `cutoff`**：实测 `cutoff=100` 直接抛
  `ValueError: Matrix too small for quantization, must have at least 256 rows`；`cutoff=1000` 会把字典
  从 632 个词重写成 29 个词、模型缩到 39 KB，体积与 bucket 完全脱钩、行为不可预期。脚本固定 `cutoff=0`
  （见常量 `QUANTIZE_CUTOFF`）。

#### 把已有 `.bin` 量化导出 `.ftz`（端侧导出流程）

训练时加 `--quantize`，落盘的就只有量化模型（未量化模型不存）。想「先留一份完整 `.bin`，再随时导出端侧
`.ftz`」，用 `-m` 复用同一套量化实现即可——**源 `.bin` 一个字节都不会被改动**：

```bash
# 源 .bin 不动, 默认输出同名 .ftz(models/intent.ftz)
python3 scripts/fasttext_training_script.py -m models/intent.bin --quantize -t data/probe

# 也可以显式指定输出路径; 后缀是 .ftz 时脚本会自动补上 --quantize
python3 scripts/fasttext_training_script.py -m models/intent.bin -o models/intent.ftz
```

实测（v2 历史：源模型 `models/intent.bin` = dim 100 / bucket 200000 / 76.50 MB，评估集 `data/probe`）：

| 指标 | 结果 |
| --- | --- |
| 量化耗时 | 约 24s（同样 dsub=2；dim=20/bucket=50000 只要 3.4s） |
| 体积 | 76.50 MB → **9.68 MB**（约 1/7.9，省 66.8 MB） |
| 逐条预测一致率（留出集前 50 条，k=1） | **50/50** |
| 留出集 60 条 | 命中 **60/60 → 60/60**（P@1 = 1.0000 → 1.0000，`--metrics` 可看） |

导出后脚本会自动做**三重校验**，任一项不过就非 0 退出：

1. **重新载入** `.ftz`（`load_model`）——证明端侧真的能加载；
2. `is_quantized()` 必须为 `True`，且词表 / 标签与量化前一致；
3. 给了评估集（`-t`/`--valid`）时，额外输出**预测一致率**与**量化前后命中数对比**（如 `60/60 (100.00%) -> 60/60 (100.00%)`），一起写进报告。

注意事项：

* **不要对量化模型再量化**：实测 `fasttext.load_model("x.ftz").quantize()` 会让进程**直接段错误**
  （不是异常，`try/except` 拦不住），脚本先用 `is_quantized()` 拦截，并直接给出「换未量化源模型 /
  重训更小模型」两条路。注意**看后缀没用**：`models/intent_mobile.ftz` 就是
  `--dim 20 --bucket 50000 --quantize` 训出来的量化模型（0.52 MB，早期产物叫 `.bin`，只是文件名）；
  量化后权重以 uint8 码 + 码本存储，文件里已经没有浮点权重可以再压了。
* **只是改文件名/复制量化模型的话，用 `mv` / `cp` 就行**，不必经过脚本。实测把同一个文件复制成
  `.ftz` / `.bin` / `.txt` / 无后缀 4 个名字，字节内容完全相同、`is_quantized()` 全是 `True`、
  预测结果逐条一致——决定「是不是量化模型」的是文件内容里的矩阵类型，不是后缀。
* 输出路径与源模型相同时**报错**（防止手滑覆盖源模型）；目标文件已存在时只提示「将被覆盖」。
* `.ftz` 只是**业界约定**，fastText 载入时并不看后缀——实测把未量化模型改名成 `.ftz` 照样能加载，
  `is_quantized()` 仍是 `False`。所以脚本看到 `-o xxx.ftz` 就自动开启 `--quantize`，
  避免出现「名字像量化模型、其实是全精度」的文件。
* `-m ... --quantize` 可以和 `--predict/--predict-file/-I` 组合：导出后直接用量化模型推理，验证端侧行为。
* `--quantize-retrain` 在这个流程里需要原训练数据（`-i`），缺文件会直接报错；量化失败时**不产出**
  `.ftz`（不会留下半成品）。
* 报告走独立分支 `mode: "quantize_only"`，比训练报告多了 `source_model`、`compression`、
  `verification`（含 `prediction_agreement` 与前后指标）。

#### 手机端怎么选（下面三档是 v2 在留出集 60 条上的历史结果；v3/v4 的结果见下文）

| 目标 | 命令 | 体积 |
| --- | --- | --- |
| 手机首选（小且稳） | `--dim 20 --bucket 50000` | 3.9 MB |
| 手机极限（压进 1 MB） | `--dim 20 --bucket 50000 --quantize` | 0.52 MB |
| 只求最小（`--quantize-dsub 20`，每行 1 字节的下限） | `--dim 20 --bucket 50000 --quantize --quantize-dsub 20` | 0.08 MB |
| 想保留 dim=100 的容量 | `--dim 100 --bucket 50000 --quantize` | 2.53 MB |
| 已经有 76 MB 的 `.bin`，只想导出端侧 | `-m models/intent.bin --quantize` | 9.68 MB |

先训练再导出也行：`--dim 20 --bucket 50000` 出一个 3.9 MB 的 `.bin`，之后再 `-m that.bin --quantize`
拿到 0.52 MB 的 `.ftz`，两步之间可以反复换 `--quantize-dsub` 重导（源 `.bin` 不变）。

端侧三点注意：

* **文件体积 ≈ 常驻内存**：权重是整体载入的（量化模型载入的是 uint8 码 + 码本），加载耗时也随体积下降。
* `--thread` 只影响训练；手机端只做推理，不受它影响。
* 若把 `--bucket` 压到 50000 以下，训练时要留意「`Encountered NaN.`」那节的条件：bucket < 50000
  **且** 线程数小于 CPU 核数会必现；把 `--thread` 拉满、或让脚本自动调大 bucket 重试即可。

> **注意**：`autotune` 只优化指标、完全不看体积，实测在本示例数据上会选出
> `bucket=10000000`（≈4.9 GB），因此脚本用 `--autotune-max-bucket` 兜底（**默认跟随 `--bucket`**，
> 超出时用同一批超参数把 bucket 压低重训，并在日志里说明）。
>
> 另外 autotune **不会**帮你找 `--min-n`：fastText 的搜索空间里没有 `1`，实测会给出一组
> `minn=0, maxn=0, bucket=0` 的「无字符 n-gram」配置——21 KB 的模型在示例数据上把
> accounting 全判成 unknown。所以脚本会把 `--min-n/--max-n` 固定后传给 autotune
> （实测会打印 `minn is manually set to a specific value...`），只让 autotune 搜索
> dim/epoch/lr/wordNgrams/bucket/loss。
>
> autotune 搜出的 `dim/epoch` 往往偏大（示例数据上 `dim=131, epoch=81`，模型约 100 MB），
> 且 `--autotune-max-bucket` 只限制 bucket、不限制 dim，需要小模型就手动传
> `--dim 20 --bucket 50000`（必要时再加 `--quantize`），或直接用 `--epoch/--lr` 手调。
> 注意 `--autotune` 会**忽略**你传的 `--bucket`（它要自己搜），小模型只能靠上限约束：
> 上限默认跟随 `--bucket`，所以 `--bucket 50000 --autotune` 出来的模型也不再是 76 MB。
>
> **示例数据上不要迷信 autotune**：验证集只有 15 条（其中 accounting 3 条）时，autotune 8s
> 选出的 `dim=42, epoch=15, lr=0.0873, bucket=200000`（32 MB）在同一份验证集上
> accounting 只有 P=0.50 / R=0.33 / F1=0.40，而默认参数（`--min-n 1 --max-n 3`）是
> accounting F1=1.00 —— autotune 要在小数据上可信，验证集至少得每个标签几百条。

### 踩坑：`RuntimeError: Encountered NaN.`

触发条件是**一组参数组合**，不是某个单独参数。实测固定
`dim=140, epoch=81, lr=0.2917, minn=1, maxn=3, thread=1`（示例数据）时：

| bucket | thread | 结果 |
| --- | --- | --- |
| 200000 | 1 | ❌ `Encountered NaN.` |
| 50000 / 100000 / 400000 / 906213 / 1000000 | 1 | ✅ 正常 |
| 200000 | 12（CPU 核数） | ✅ 正常 |
| 200000 | 1（只把 `lr`/`epoch`/`dim` 换成别的值） | ✅ 正常 |

也就是说：**换任意一个参数通常就能跳出去**，bucket 越大越稳这个直觉并不成立。
更早的网格扫描还发现 `bucket=1000/2000/20000` 搭配 `--thread < CPU 核数` 时必现 NaN，
所以「小 bucket + 线程不足」是高危组合，但并不是唯一触发条件。

脚本对此做了三层保护：

1. **事前提醒**：`--bucket < 50000` 且 `--thread < CPU 核数` 时给出警告与建议；
   `--min-n 0 --max-n 0`（关闭字符 n-gram）时也会警告；
2. **事后自愈**：捕获 NaN 后按阶梯自动重试，最多 6 次尝试——
   `bucket 提到 ≥50000` → `thread 提到 CPU 核数` → `lr 减半` → `lr 再减半` → `bucket 翻倍`；
   每次调整都会打印日志（例如实测 autotune 场景就是这样从 `thread=1` 自动切到 `thread=12` 才训成功）；
3. **不静默改结果**：真正生效的参数写进日志与 JSON 报告
   （`model.summary` 是真实训练参数，autotune 场景下还有 `model.params.retrained_after_autotune`）；
   如果压低 bucket 重训仍然失败，会保留 autotune 原模型并明确告警，而不是让整个流程失败。

### 调参建议

#### `--min-n`：中文二分类最关键的一个参数（实测）

示例数据（accounting vs unknown）上做 10 轮随机划分、`--thread 1 --seed 42`：

| 配置 | val_acc | accounting P | accounting R | accounting F1 | 出现「全判 unknown」的轮数 |
| --- | --- | --- | --- | --- | --- |
| `--min-n 3 --max-n 6`（脚本旧默认） | 0.950 | 0.800 | 0.700 | 0.733 | **2/10** |
| `--min-n 1 --max-n 3`（**现默认**） | **1.000** | **1.000** | **1.000** | **1.000** | 0/10 |
| `--min-n 1 --max-n 3 --dim 20 --bucket 50000`（小模型） | 0.975 | 1.000 | 0.850 | 0.900 | 0/10 |
| `--min-n 1 --max-n 2` | 1.000 | 1.000 | 1.000 | 1.000 | 0/10 |

原因：中文不分词时整句只是一个 token，判别信息全在字符 n-gram 里；`--min-n 3` 会丢掉「花」「元」
「块」这类**单字/双字线索**，正类样本又少，模型容易退化成「全判 unknown」。所以脚本默认改成了
`--min-n 1 --max-n 3`（要恢复旧行为就显式传 `--min-n 3 --max-n 6`）。
注意 v1 的 60 条数据里 `accounting` 只有 12 条，任何指标都波动很大（当前已扩到 859 条，见 §2 数据格式）。

#### 数据量：每类要多少条（实测）

**v1（60 条）诊断：12 条为什么不够用**——瓶颈是「说法覆盖面」而不是条数：

* 12 条正样本 **12/12 含阿拉伯数字**、8/12 含「元/块」，而 48 条负样本里只有 **1 条**含金额单位
  （8 条含数字）——正负类在「有没有金额单位」这个特征上几乎完全可分，模型很容易学成
  「有数字 + 元/块 = 记账」的捷径。
* 12 条上 `P@1=F1=1.000` 不代表能用：12/12 全对时真实召回率的 95% 置信区间下界只有 **0.757**；
  0 错误时漏检率上界 ≈ 3/n —— n=12 → **25%**、n=60 → 5%、n=300 → 1%。
* 学习曲线（正样本 3→10 条 × 50 个随机划分，负样本按 4 倍配，`--thread 1 --seed 42`）：
  n=3 召回 0.929 → n=6 0.963 → n=8 0.985 → n=10 0.990，**已接近饱和**；
  继续堆同款句子收益很小，补「新说法」才有效。
* 手写 15 条新说法 + 12 条硬负例实测：召回 **13/15**（漏「出差住宿680」「看电影两张票76元」，
  概率只有 0.54~0.69），误报 **1/12**（「这个月一共花了多少钱」被判 accounting 0.643）。

**v2（631 条）实测**——按下面的清单把示例数据铺满之后的数字（命令见本文末，规则见
[`data/LABELING.md`](data/LABELING.md)）：

| 口径 | 结果 |
| --- | --- |
| 独立留出集 60 条（`-t data/probe`，与训练集零重叠） | P@1 = R@1 = F1 = **1.0000**（60/60） |
| 手写全新说法 32 条（两轮 20 + 12 条，均不在任何训练文件里） | **32/32** 正确 |
| 8:2 切分（seed 1 / 7 / 42 / 2024） | P@1 = 0.9603 / 0.9683 / 0.9603 / 0.9762 |
| 「含元/块 = 记账」单特征捷径的准确率 | 0.807（v1 约 0.98，捷径已被大幅削弱） |
| 训练集内「含金额单位但非记账」的硬负例 | 34 条（v1 只有 1 条） |

**v3（1441 条）历史实测**——当时针对 7 个线上 badcase 补了「充值 / 转账 / 收款码 / 补录 / 物业费 /
折扣」的成对边界样本（判定规则见 [`data/LABELING.md`](data/LABELING.md) §2）：

| 口径 | 结果 |
| --- | --- |
| 独立留出集 417 条（`-t data/probe`，与训练集 0 重叠） | P@1 = R@1 = F1 = **1.0000**（417/417） |
| v3 新写同族改写 210 条（92 + 118，临时回归集，未入仓；已过滤与当时训练集 / probe 的完全重复） | **210/210** 正确（参与过边界迭代，仅作回归参考） |
| 8:2 切分（seed 1 / 7 / 42 / 2024，thread 1） | P@1 = 0.9861 / 0.9722 / 0.9861 / 0.9618（均值 0.9766） |
| 单特征捷径「含元/块 = 记账」 | 训练集 0.7988 / 独立留出集 0.8993（模型 1.0000） |
| 捷径会判错的子集 | probe 中「unknown 且含元/块」16/16；probe 中「accounting 且不含元/块」26/26 |
| 训练集内「含金额单位但非记账」的硬负例 | 103 条（v2 34 条） |
| 数据体检：重复 / 冲突 / 训练-留出完全重叠 | 0 / 0 / 0 |

**v4（1842 条）实测**——在资产账户数据基础上，针对 16 个线上 badcase 补了成对边界样本（账户余额/额度/净资产/账户列表查询；算数、充值、免息期、转账/还款、银行办理、消费统计），并用 `--dim 20 --bucket 50000 --quantize --thread 1 --seed 42` 重训端侧模型：

| 口径 | 结果 |
| --- | --- |
| 独立留出集 499 条（`-t data/probe`，与训练集 0 重叠） | **499/499 = 1.0000** |
| 16 个线上 badcase | **16/16**，预测概率 0.9600~0.9999 |
| 新写同族改写 32 条（仅覆盖这 16 类，未入仓、与训练/probe 零重叠） | **32/32** |
| v3 遗留临时回归集 92 + 118 条 | **91/92 + 118/118**（fresh2 的「给舅舅转过去了一千」同时出现在两个标签下，属回归集自身标注冲突） |
| 8:2 切分（seed 1 / 7 / 42 / 2024，thread 1） | P@1 = 0.9729 / 0.9648 / 0.9675 / 0.9594（均值 0.9662） |
| 训练集自评（1842 条） | 1827/1842 = 0.9919；15 条残余见 §5 |
| 数据体检：重复 / 冲突 / 训练-留出完全重叠 | 0 / 0 / 0 |
| 模型体积 / SHA-256 | 0.5631 MB / `425ed5efba1ca1687b4f3db05daad1103a61eec0c18c726bf701782babfbb7a9` |

> 注意：499 条 probe 在 v4 迭代中被用于挑错，**它已经不再是严格的一次性留出集**；32 条新写改写和 v3 遗留回归集才是更接近「换说法」的泛化检查。正式上线前请再留一批不参与任何调参的线上样本。

但别把「留出集满分」当成万事大吉：v4 在 1842 条训练集上仍有 **15 条**自相矛盾/边界抖动，主要集中在：

* **只差一个完成标记的最小对**（「给朋友转账了三千元」vs「给朋友转账三千元」，「把信用卡欠的钱还上了」vs「把信用卡欠的钱直接还上」）；
* **「转生活费」这类带用途的裸转账**（probe 标 `accounting`）与「给朋友转账三千元」这类裸转账（`unknown`）字面几乎同形；
* **资产账户余额 vs 费用余额/消费统计**（「微信零钱还剩多少」`accounting` vs「查一下水费余额」「用微信零钱买东西一共用了多少」`unknown`）；
* **账本内账户 vs 实体银行账户/游戏账号**（「在账本里新建账户」`accounting` vs「到银行办理新账户」「新建游戏小号」`unknown`）。

这些样本的概率常在 0.5~0.9，单靠 dim=20 的小模型无法稳定区分。业务侧如果对这几类特别敏感，建议加一层规则（完成标记词典、账户/费用关键词、银行/游戏实体词）或人工确认，不要只靠提高模型容量。明细见 [`data/LABELING.md`](data/LABELING.md) §5。

结论与做法：

* 初期参考规模：正样本 **100~200 条**，unknown 类 **300~500 条**（它要覆盖「其余全部」），
  并留一份独立测试集（别用同一份数据既调参又报指标）。当前示例数据已扩到 **859 / 983 条**，
  并配了独立留出集 `data/probe`。
* 扩数据按**说法维度**补，不是按条数补：谁花的（我/孩子/公司）、金额写法（`30` / `30元` /
  `32.5` / `1,200` / `三百`）、动词（花/买/交/付/充/订/到账/支出）、口语 vs 书面、是否带时间词。
* **优先补硬负例**（当前最大的坑）：含金额或数字但不记账的句子——查询「这个月一共花了多少钱」、
  算数「128 的 20% 是多少」、查账单「信用卡还款日是几号」、提醒「提醒我明天交电费350」。
* 低成本来源：线上真实日志抽样（最靠谱）> 对现有样本做受控改写（换动词/换金额写法/口语化）
  > 主动学习（用模型给线上句子打概率，只人工标注 0.5~0.7 的边界样本，单条信息量最大）。
* 数据到量之前，可用 `-k 2 --threshold` 提高召回（代价是误报上升），先保证不漏；数据到量之后，
  概率落在 0.5~0.7 的边界句（见上一段的残余错误家族）仍建议用阈值或人工确认兜底。

#### 其他

* `--thread > 1` 时 fastText 是**异步 SGD**：即使固定 `--seed`，多次训练指标也会有波动（线程越多
  越明显，模型参数本身也不一致）。要完全可复现请用 `--thread 1`。
* 当前正负样本约 1:1.14，不要只看整体准确率（全判 `unknown` 就有约 0.53），重点看 `accounting` 的
  P/R（加 `--metrics` 就能看到脚本打印的「每标签指标」）。
* 指标波动大时优先补数据（每个标签几百条起步）；示例数据已扩到 1842 条，
  请用 `-t` 独立留出集评估新数据的效果；旧版指标中的波动幅度仅供参考。

### 用留出集自评（推荐）

```bash
python3 scripts/fasttext_training_script.py -i data/labeling -t data/probe -o /tmp/m.bin
```

这条命令会用**全部 1842 条**训练、在**从未参与训练的 499 条**上评估；当前 v4 结果为 **499/499**。
`--report` 里也会把 `data.test` / `data.test_stats` 一起写进 JSON。

## 5. 输出说明

* **评估结果默认只给「条目数 / 成功 / 失败 / 失败明细」**，一眼就能看出错在哪：

  ```
  评估数据   : data/probe (10 条)
  测试条目   : 10 条
  成功       : 9 条  (成功率 90.00%)
  失败       : 1 条  (失败率 10.00%)
  失败明细   :
     1. 转给同事300块
        真实: accounting
        预测: unknown(0.5497)
  ```

  成功 = 预测标签命中真实标签（单标签、`k=1` 时就是「猜对」）。失败样本默认最多列 20 条，
  超出会提示还有多少条未显示；要一次看全部用 `--max-failures 0`。
  需要 P/R/F1 与每标签指标（判断是「漏报」还是「误报」）时加 `--metrics`。

* **日志全部走 stderr**；`--predict-file` 的 JSONL 结果走 **stdout**，可以直接 `> preds.jsonl`
  或管道给下游：

  ```json
  {"text": "打车花了25", "predictions": [{"label": "accounting", "prob": 0.9512649178504944}], "true_labels": ["accounting"], "hit": true}
  {"text": "提醒我明天交电费", "predictions": [{"label": "unknown", "prob": 0.9632157683372498}], "true_labels": ["unknown"], "hit": true}
  ```

  待预测文件本身带 `__label__` 时，会自动把真实标签写进 `true_labels` 并给出 `hit`（是否命中），
  可以直接拿去做 badcase 分析；只有标签没有文本的行会被跳过并提示。
* **JSON 报告**（`<模型>.report.json`）包含 `environment`（Python / 平台 / fasttext / numpy 版本）、
  `arguments`、`data`（样本数、标签分布、切分信息）、`model`（真实超参数 + 体积 + 训练耗时，
  量化时还带 `quantization`：`dsub/cutoff/qnorm/retrain/seconds/estimated_size_before_mb`，
  以及 `verification`：落盘后重新载入的自检结果）、
  `evaluation`（默认口径的 `judged_samples/correct/failed/success_rate/failure_rate` + `failures`
  失败明细；P/R/F1 与 `per_label` 无论加不加 `--metrics` 都会写进报告，因为报告是给机器读的）。
  所有指标都做了 `nan` 归一化，是严格合法的 JSON。
  用 `-m/--model` 只推理时，报告走 `mode: "inference_only"` 分支；`-m --quantize` 导出 `.ftz` 时
  走 `mode: "quantize_only"` 分支，额外带 `source_model`、`compression`
  （`size_before_mb/size_after_mb/ratio/saved_mb`）与 `verification`
  （`is_quantized/vocab_size_same/labels_same/prediction_agreement`，有评估集时还有前后指标与 `precision_delta`）。
* 评估集给**目录**（如 `data/probe`）时，脚本会先把目录内 `*.txt` 按文件名排序合并成临时文件再交给
  fastText（当前数据合并为 499 条、2 个文件），结束后自动清理。
* **多标签数据**会自动检测并用 `loss=ova` 提示；单标签数据在 `--metrics` 里会说明 `P@1 即 accuracy`。
* 切分时若某个标签样本太少，会打印实际切分比例与目标比例的偏差提醒。
* autotune 时若验证集同时用于选超参数，会提示指标偏乐观，建议另给 `--test`。

## 6. 产出文件

```
scripts/fasttext_training_script.py  主脚本
scripts/toolchain_traning_script.py  Needle 3 中文工具调用评测脚本
tests/test_needle_toolchain.py       Needle 资产账户增删改查测试（38 条案例，make test-needle 入口）
tests/test_asset_account_conversion.py  资产账户构造、内存增删改查及工具分发逻辑测试（无需模型）
requirements.txt                   依赖
README.md                          本文件
data/needle/chinese_tool_cases.json  Needle 中文工具调用人工用例（27 条）
data/labeling/accounting_data.txt  示例数据-记账及资产账户类（`__label__accounting` 859 条）
data/labeling/unknown_data.txt     示例数据-未知类（`__label__unknown` 983 条）
data/probe/accounting_data.txt     独立留出测试集-记账及资产账户类（206 条，与训练集零重叠）
data/probe/unknown_data.txt        独立留出测试集-未知类（293 条，与训练集零重叠；`-t data/probe` 合并评估）
data/LABELING.md                   标注规范：判定顺序、边界约定、覆盖维度、实测指标
models/finalers.bin(.report.json)  默认命令的产物（模型产物默认不进 git；`models/` 已在 `.gitignore` 中忽略）
models/finalers_mobile.ftz(.report.json)  v4 端侧小模型（--dim 20 --bucket 50000 --quantize，0.56 MB，499/499）
models/intent.ftz(.report.json)    端侧导出示例：由 dim 100 / bucket 200000 的 .bin 量化而来（76.50 MB → 9.68 MB）
models/needle_chinese.report.json  Needle 中文工具调用评测报告（运行评测后生成）
```
