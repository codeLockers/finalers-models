#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fastText 文本分类模型训练脚本(增强版)。

相比最初几行的示例, 这里补齐了训练一个可用模型需要的完整流程:

  1. 环境检查 : 打印 fasttext / numpy 版本, 导入失败时给出可操作的修复提示
  2. 数据检查 : 统计样本数、标签分布, 校验 __label__ 前缀、空文本等格式问题
  3. 数据划分 : 按标签分层切分 训练集 / 验证集(可复现, 保证训练集覆盖所有标签)
  4. 模型训练 : 暴露 fastText 常用超参数, 固定随机种子便于复现; --quantize 可做端侧压缩
  5. 自动调参 : --autotune 调用 fastText 内置 autotune 搜索超参数
  6. 效果评估 : 整体 Precision / Recall / F1 + 每个标签的 P / R / F1
  7. 预测推理 : --predict 批量预测 / --predict-file 文件批量推理 / --interactive 交互调试
  8. 结果落盘 : 保存 .bin 模型 + .report.json 训练报告(含数据统计与评估指标)
  9. 端侧导出 : --quantize 对训练结果或已有 .bin 做乘积量化, 导出 .ftz 并重载校验

常用示例(从项目根目录执行；相对路径基于当前工作目录):
  # 用默认数据训练(默认 -i data/labeling, 该目录按标签拆成多个 <标签>_data.txt)
  python3 scripts/fasttext_training_script.py

  # 按标签拆分: 一份完整数据 -> 每个标签一个文件(写入 data/labeling/<标签>_data.txt)
  python3 scripts/fasttext_training_script.py -i /tmp/sample_intent_data.txt --split-by-label data/labeling

  # 指定数据与模型路径, 并用独立测试集评估(-i 可以是单文件, 也可以是拆分目录)
  python3 scripts/fasttext_training_script.py -i data/labeling -o models/intent.bin -t data/probe

  # 自动调参(未提供 --valid 时自动切分出验证集)
  python3 scripts/fasttext_training_script.py -i data/labeling --autotune --autotune-duration 60

  # 训练后直接推理 / 交互调试
  python3 scripts/fasttext_training_script.py -o models/intent.bin --predict "午饭花了30块" "提醒我明天交电费"
  python3 scripts/fasttext_training_script.py -o models/intent.bin --interactive

  # 端侧部署: 小模型 + 乘积量化(实测本示例数据 76 MB -> 3.9 MB -> 量化后 0.5 MB, 指标不变)
  python3 scripts/fasttext_training_script.py -o models/intent_mobile.ftz --dim 20 --bucket 50000 --quantize

  # 端侧导出: 把已经训练好的 .bin 量化成 .ftz(推荐流程, 源 .bin 原样保留)
  python3 scripts/fasttext_training_script.py -m models/intent.bin --quantize -t data/probe
  # 默认输出 models/intent.ftz, 想指定路径就加 -o; 报告里会记录前后体积/指标/预测一致率

  # 只检查数据不训练 / 只做环境自检(含一次极小样例训练+预测)
  python3 scripts/fasttext_training_script.py -i data/labeling --dry-run
  python3 scripts/fasttext_training_script.py --check-env

  # 用已有模型推理, 不重新训练
  python3 scripts/fasttext_training_script.py -m models/intent.bin --predict "午饭花了30块"
  python3 scripts/fasttext_training_script.py -m models/intent.bin -t data/probe \
      --predict-file data/probe

数据格式(fastText 监督分类标准格式, 每行一条):
  __label__accounting 午饭花了30块
  __label__unknown 提醒我明天交电费
  __label__a __label__b 一行也可以带多个标签        <-- 多标签样本
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, OrderedDict
from pathlib import Path

LABEL_PREFIX = "__label__"
# 默认训练数据: 按标签拆分好的目录(每个标签一个 <标签>_data.txt, 合并后就是完整训练集)
DEFAULT_INPUT = "data/labeling"
DEFAULT_LABEL_DIR = "data/labeling"
DATA_SUFFIX = ".txt"
SPLIT_FILE_SUFFIX = "_data.txt"
DEFAULT_OUTPUT = "model.bin"
# 实测(bucket 网格扫描): bucket < 50000 且线程数不足时, fastText 0.9.3 会必现
# "RuntimeError: Encountered NaN.", bucket >= 50000 则稳定可训。
SAFE_BUCKET = 50000
# 端侧压缩固定用 cutoff=0: fastText 的 --cutoff 非 0 时会重写字典(实测词表从 632 掉到 29,
# 模型缩到 39 KB 但行为不可预期), 且 cutoff < 256 直接抛
# "ValueError: Matrix too small for quantization, must have at least 256 rows"。
QUANTIZE_CUTOFF = 0
# 量化导出后做「行为一致性」自检时, 最多对比多少条样本的预测(取评估集/训练集前 N 条)
PROBE_TEXT_LIMIT = 50
SEPARATOR = "=" * 68

QUIET = False


# --------------------------------------------------------------------------- #
# 日志与终端输出
# --------------------------------------------------------------------------- #
def log(message: str = "", level: int = 0) -> None:
    """输出日志(走 stderr, 保证 stdout 只留给 --predict-file 的 JSONL 结果)。

    level 越大越啰嗦; --quiet 时丢弃 level > 0 的信息。
    """
    if QUIET and level > 0:
        return
    print(message, file=sys.stderr, flush=True)


def section(title: str) -> None:
    log("\n" + SEPARATOR)
    log(title)
    log(SEPARATOR)


def fmt_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s" if seconds < 60 else f"{seconds / 60:.2f}min"


# --------------------------------------------------------------------------- #
# 环境检查
# --------------------------------------------------------------------------- #
def import_fasttext():
    """导入 fasttext, 失败时给出可操作的修复提示。"""
    try:
        import fasttext  # 运行时导入, 便于捕获二进制加载失败(OSError)
    except Exception as exc:  # noqa: BLE001 - 需要把底层报错一起展示给用户
        log(f"[错误] 无法导入 fasttext: {exc}")
        log(
            "\n请先安装依赖:\n"
            "    python3 -m pip install --user -r requirements.txt\n"
            "常见问题:\n"
            "  1) macOS 报 'Library not loaded: @rpath/libc++.1.dylib'\n"
            "     PyPI 上的 fasttext==0.9.3 wheel 引用了构建机上的 libc++ 路径,\n"
            "     可改用 fasttext-wheel, 或用 install_name_tool 修正依赖路径(见 README.md)。\n"
            "  2) 报 NumPy 相关的 ValueError(copy/dimension 之类)\n"
            "     fasttext 0.9.3 与 NumPy 2.x 不兼容, 可降级 numpy<2 或改用 fasttext-wheel。"
        )
        raise SystemExit(2)
    return fasttext


def environment_info(fasttext) -> dict:
    """收集运行环境信息, 写入训练报告, 方便复现问题。"""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "fasttext": getattr(fasttext, "__version__", "unknown"),
    }
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except Exception:  # noqa: BLE001 - numpy 只用于展示版本
        info["numpy"] = "not installed"
    return info


def print_environment(fasttext) -> dict:
    info = environment_info(fasttext)
    log(f"Python     : {info['python']}")
    log(f"fasttext   : {info['fasttext']}")
    log(f"numpy      : {info['numpy']}")
    log(f"系统       : {info['platform']}")
    return info


def run_env_check(fasttext, env: dict) -> bool:
    """环境自检: 用极小的内置样例真的训练一次并预测, 确认 fasttext 二进制可用。

    只打印版本号是不够的 —— macOS 上 fasttext 0.9.3 常见的两类问题
    (libc++ 动态库路径错误 / NumPy 2.x 不兼容) 都只在真正调用 train/predict 时才暴露。
    """
    sample = [
        "__label__A 这是一条测试文本 alpha",
        "__label__A 这是另一条测试文本 beta",
        "__label__B 完全不同的内容 gamma",
        "__label__B 另一条完全不同的内容 delta",
    ]
    temp_dir = Path(tempfile.mkdtemp(prefix="fasttext_envcheck_"))
    sample_path = temp_dir / "sample.txt"
    try:
        write_lines(sample_path, sample)
        model = fasttext.train_supervised(
            input=str(sample_path), dim=10, bucket=1000, epoch=1, verbose=0
        )
        predictions = predict_one(model, "这是一条测试文本", 1, 0.0)
        if not predictions:
            log("[错误] 环境自检失败: 训练成功但预测结果为空。")
            return False
        log(f"训练自检   : 通过(4 条样例, dim=10, bucket=1000, epoch=1)")
        log(f"预测自检   : 通过 -> {format_predictions(predictions)}")
        log("[结果] 当前环境可以正常训练与预测。")
        return True
    except Exception as exc:  # noqa: BLE001 - 自检要把原始报错完整展示
        log(f"[错误] 环境自检失败: {type(exc).__name__}: {exc}")
        log(
            "       多为 fasttext 二进制与系统/NumPy 不兼容, 处理方式见 README.md 的"
            "'本机踩过的两个坑', 或改用 fasttext-wheel / fastText CLI。"
        )
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 数据读取、校验与统计
# --------------------------------------------------------------------------- #
def read_text_lines(path: Path):
    """读取单个文本文件, 统一处理缺失/编码问题; 返回去掉空行后的行列表(可能为空)。"""
    if not path.exists():
        raise SystemExit(f"[错误] 找不到数据文件: {path}\n请用 -i/--input 指定训练数据路径。")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit(
            f"[错误] {path} 不是 UTF-8 编码, 请先转换后重试, 例如:\n"
            f"    iconv -f GBK -t UTF-8 {path} > data.utf8.txt\n({exc})"
        )
    return [line.strip() for line in content.splitlines() if line.strip()]


def list_data_files(path: Path):
    """目录 -> 其中所有 *.txt(按文件名排序); 单文件 -> [该文件]。"""
    if not path.is_dir():
        return [path]
    return sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() == DATA_SUFFIX
    )


def load_lines(path: Path):
    """读取 fastText 格式数据, 支持「单个文件」和「按标签拆分的目录」。

    目录形态下会读取所有 *.txt(按文件名排序)并合并, 所以 -t/--valid/--predict-file
    既可以直接指向 `data/probe/` 这样的目录, 也可以指向其中某个具体文件。
    """
    path = path.expanduser()
    if not path.exists():
        raise SystemExit(
            f"[错误] 找不到数据路径: {path}\n请用 -i/--input 指定训练数据路径。"
        )
    if path.is_dir():
        files = list_data_files(path)
        if not files:
            raise SystemExit(
                f"[错误] 目录里没有 {DATA_SUFFIX} 数据文件: {path}\n"
                f"      可先用 --split-by-label 拆分, 或把数据文件放进该目录。"
            )
        lines = []
        for file in files:
            lines.extend(read_text_lines(file))
        if not lines:
            raise SystemExit(f"[错误] 目录里的数据文件都是空的: {path}")
        return lines
    lines = read_text_lines(path)
    if not lines:
        raise SystemExit(f"[错误] 数据文件为空: {path}")
    return lines


def load_dataset(path: Path):
    """读取训练数据, 支持「单个文件」和「按标签拆分的目录」两种形式。

    目录模式: 读取目录下所有 *.txt(按文件名排序)后合并, 返回 (lines, sources);
      sources = [{"path": str, "samples": int}, ...], 用于日志与训练报告。
    文件模式: 行为与 load_lines 一致, sources 只有一个条目。
    """
    path = path.expanduser()
    if not path.exists():
        raise SystemExit(f"[错误] 找不到数据路径: {path}\n请用 -i/--input 指定数据文件或目录。")
    if not path.is_dir():
        lines = load_lines(path)
        return lines, [{"path": str(path), "samples": len(lines)}]

    files = list_data_files(path)
    if not files:
        raise SystemExit(
            f"[错误] 目录里没有 {DATA_SUFFIX} 数据文件: {path}\n"
            f"      先用 --split-by-label 拆分, 或把数据文件放进该目录。"
        )

    lines, sources = [], []
    for file in files:
        part = load_lines(file)
        lines.extend(part)
        sources.append({"path": str(file), "samples": len(part)})
    return lines, sources


def safe_label_filename(label: str) -> str:
    """把标签名转成安全文件名(保留中文、字母、数字、下划线和短横线)。"""
    cleaned = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", label).strip("_")
    return cleaned or "unlabeled"


def split_by_label(input_path: Path, output_dir: Path) -> int:
    """按标签把数据拆成 output_dir/<标签>_data.txt, 每行保留 __label__ 前缀。

    拆分文件既可单独作为某个类别的样本集合, 也可以把整个目录交给
    `-i <output_dir>` 训练(合并后与原始数据等价)。
    """
    lines, sources = load_dataset(input_path)
    log(f"数据来源   : {input_path} ({len(lines)} 条, {len(sources)} 个文件)")
    for source in sources:
        log(f"  - {source['path']} ({source['samples']} 条)", level=1)

    buckets = OrderedDict()
    missing_label = 0
    multi_label = 0
    for line in lines:
        labels, _ = parse_line(line)
        if not labels:
            missing_label += 1
            continue
        if len(labels) > 1:
            multi_label += 1
        buckets.setdefault(labels[0], []).append(line)

    if missing_label:
        log(f"[警告] 跳过 {missing_label} 行没有 {LABEL_PREFIX} 标签的数据(无法归类)")

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"输出目录   : {output_dir}")

    # 先算好每个标签对应的文件名(重名时追加序号), 写盘前做残留检查
    planned = OrderedDict()
    for label in sorted(buckets, key=lambda name: (-len(buckets[name]), name)):
        base = safe_label_filename(label)
        file_name = base + SPLIT_FILE_SUFFIX
        index = 2
        while file_name in planned:
            # 不同标签归一化成同一个文件名(如 "a/b" 与 "a_b")时追加序号, 避免互相覆盖
            file_name = f"{base}_{index}{SPLIT_FILE_SUFFIX}"
            index += 1
        planned[file_name] = label

    # 目录里的其它 *.txt 会被 -i 一起读入训练, 有残留就先中止, 避免把旧副本混进新拆分
    stale = sorted(
        item.name
        for item in output_dir.iterdir()
        if item.is_file() and item.suffix.lower() == DATA_SUFFIX and item.name not in planned
    )
    if stale:
        listed = "\n".join(f"    {output_dir / name}" for name in stale)
        raise SystemExit(
            f"[错误] 输出目录里有不是本次生成的数据文件, 已中止(未写入任何文件):\n{listed}\n"
            "       这些文件会被 -i <目录> 一起读入训练, 请先删除或移走到别处, 再重新执行。"
        )

    for file_name, label in planned.items():
        items = buckets[label]
        write_lines(output_dir / file_name, items)
        log(f"  {file_name:<24} {len(items):>5} 条  ({label})")

    if multi_label:
        log(f"[提示] {multi_label} 行是多标签样本, 按第一个标签归类(其余标签仍保留在行内)")

    total = sum(len(items) for items in buckets.values())
    log(f"拆分结果   : {len(planned)} 个标签 / {total} 条 -> {output_dir}")
    return 0


def parse_line(line: str):
    """拆分一行, 返回 (labels, words)。

    fastText 会把任意以 __label__ 开头的 token 当作标签(不限于行首),
    这里保持一致, 避免统计口径和实际训练结果不符。
    """
    labels, words = [], []
    for token in line.split():
        if token.startswith(LABEL_PREFIX):
            labels.append(token[len(LABEL_PREFIX):])
        else:
            words.append(token)
    return labels, words


def inspect_dataset(lines, name: str = "数据"):
    """统计标签分布并收集格式问题, 返回 (stats, problems)。"""
    label_counter = Counter()
    problems = []
    multi_label = 0
    no_text = 0
    numeric_labels = 0

    for index, line in enumerate(lines, start=1):
        labels, words = parse_line(line)
        if not labels:
            problems.append(f"第 {index} 行缺少 {LABEL_PREFIX} 标签, 该行不会被有效训练: {line[:40]}")
        if not words:
            no_text += 1
            problems.append(f"第 {index} 行只有标签没有文本, 对训练基本没有帮助: {line[:40]}")
        if len(labels) > 1:
            multi_label += 1
        for label in labels:
            label_counter[label] += 1
            if label.isdigit():
                numeric_labels += 1

    stats = {
        "source": name,
        "samples": len(lines),
        "labels": OrderedDict(sorted(label_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "label_count": len(label_counter),
        "multi_label_samples": multi_label,
        "no_text_samples": no_text,
    }
    if numeric_labels:
        problems.append(
            f"发现 {numeric_labels} 个纯数字标签(如 __label__1), 通常是把标签写成了编号, "
            "建议改成可读的标签名"
        )
    return stats, problems


def print_dataset_stats(stats: dict) -> None:
    log(f"数据文件   : {stats['source']}")
    log(f"样本总数   : {stats['samples']}")
    log(f"标签总数   : {stats['label_count']}")
    log(f"多标签样本 : {stats['multi_label_samples']}")
    log("标签分布   :")
    total = max(stats["samples"], 1)
    for label, count in stats["labels"].items():
        bar = "#" * max(1, int(round(count / total * 40)))
        log(f"  {label:<14} {count:>5} 条  {bar}")


def print_problems(problems, limit: int = 10) -> None:
    if not problems:
        log("格式检查   : 通过, 未发现明显问题")
        return
    log(f"格式检查   : 发现 {len(problems)} 个问题")
    for problem in problems[:limit]:
        log(f"  - {problem}")
    if len(problems) > limit:
        log(f"  - ... 其余 {len(problems) - limit} 个问题已省略")


def warn_degenerate_dataset(stats: dict) -> None:
    """对样本过少 / 标签不均衡给出提醒, 避免训练出看起来 100% 准确的假模型。"""
    samples = stats["samples"]
    labels = stats["label_count"]
    if labels <= 1:
        log("[警告] 数据里只有一个标签, 模型无法做出有意义的区分, 请补充其它标签的样本。")
    if samples < 100:
        log(
            f"[警告] 样本数只有 {samples} 条, 评估指标波动会很大, "
            "正式使用前建议每个标签至少准备几百条样本。"
        )
    if samples and labels:
        per_label = samples / labels
        if per_label < 5:
            log(f"[警告] 平均每个标签只有 {per_label:.1f} 条样本, 建议适当扩充数据。")
    counts = list(stats["labels"].values())
    if len(counts) > 1 and max(counts) > 5 * min(counts):
        log("[警告] 标签分布不均衡(最多/最少相差 5 倍以上), 可通过过采样或补充数据缓解。")


# --------------------------------------------------------------------------- #
# 数据划分
# --------------------------------------------------------------------------- #
def stratified_split(lines, valid_ratio: float, seed: int):
    """按首个标签分层切分出 训练集/验证集, 保证训练集中每个标签至少有 1 条样本。"""
    buckets = OrderedDict()
    for line in lines:
        labels, _ = parse_line(line)
        key = labels[0] if labels else ""
        buckets.setdefault(key, []).append(line)

    rng = random.Random(seed)
    train, valid = [], []
    for key in sorted(buckets):
        items = list(buckets[key])
        rng.shuffle(items)
        if len(items) < 2:
            # 只有 1 条样本的标签全部留在训练集, 否则该标签模型完全学不到
            train.extend(items)
            continue
        n_valid = int(round(len(items) * valid_ratio))
        n_valid = max(1, min(n_valid, len(items) - 1))
        valid.extend(items[:n_valid])
        train.extend(items[n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def write_lines(path: Path, lines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


# --------------------------------------------------------------------------- #
# 模型训练
# --------------------------------------------------------------------------- #
def build_train_kwargs(args, train_path: Path) -> dict:
    """把命令行参数整理成 fasttext.train_supervised 的关键字参数。"""
    return {
        "input": str(train_path),
        "epoch": args.epoch,
        "lr": args.lr,
        "dim": args.dim,
        "ws": args.ws,
        "wordNgrams": args.word_ngrams,
        "minCount": args.min_count,
        "minCountLabel": args.min_count_label,
        "minn": args.min_n,
        "maxn": args.max_n,
        "loss": args.loss,
        "neg": args.neg,
        "bucket": args.bucket,
        "lrUpdateRate": args.lr_update_rate,
        "thread": args.thread,
        "seed": args.seed,
        "verbose": args.verbose,
    }


def warn_risky_bucket(args) -> None:
    """提前提醒 bucket/线程的危险组合(实测会直接抛 Encountered NaN.)。"""
    cpu_count = os.cpu_count() or 1
    if args.bucket < SAFE_BUCKET and args.thread < cpu_count:
        log(
            f"[警告] --bucket={args.bucket} 偏小且 --thread={args.thread} 小于 CPU 核数({cpu_count}): "
            f"实测该组合会让 fastText 0.9.3 直接抛 'Encountered NaN.'。"
            f"建议 --bucket 不小于 {SAFE_BUCKET}(默认 200000), 或把 --thread 提到 {cpu_count}; "
            "训练失败时脚本会自动调大 bucket 重试。"
        )


def warn_risky_subword(args) -> None:
    """提醒关闭字符 n-gram 的危险组合(中文未分词时模型基本不可用)。"""
    if args.min_n == 0 and args.max_n == 0:
        log(
            "[警告] --min-n=0 --max-n=0 表示关闭字符 n-gram: 中文未分词时整句只是一个 token, "
            "模型只能记住整句, 换一个说法就预测不出来。建议保持 --min-n 1 --max-n 3。"
        )


def nan_retry_plan(kwargs) -> "list[tuple[str, dict]]":
    """按实测有效的顺序, 生成 NaN 失败后的参数调整方案(逐级叠加)。

    fastText 0.9.3 在小数据上抛 "Encountered NaN." 时, 触发条件是一组
    **参数组合**而不是单个参数。实测样例(dim=140, epoch=81, lr=0.2917,
    minn=1, maxn=3, thread=1):

      bucket=200000 -> NaN;  bucket=50000 / 400000 / 906213 -> OK
      thread=12 (CPU 核数) -> OK;  lr=0.036 / 0.5 -> OK;  dim=20/100 -> OK

    也就是说随便动哪一个参数都能跳出去, 因此重试阶梯覆盖多个方向:
    bucket -> thread -> lr 减半 -> lr 再减半 -> bucket 翻倍(换哈希桶)。
    """
    cpu_count = os.cpu_count() or 1
    plan = []
    current = dict(kwargs)

    bucket = current.get("bucket", 0)
    if bucket < SAFE_BUCKET:
        target = max(bucket * 10, SAFE_BUCKET)
        current = {**current, "bucket": target}
        plan.append((f"把 bucket 从 {bucket} 提到 {target}", current))

    thread = current.get("thread", 1)
    if thread < cpu_count:
        current = {**current, "thread": cpu_count}
        plan.append((f"把 thread 从 {thread} 提到 CPU 核数({cpu_count})", current))

    for _ in range(2):
        lr = current.get("lr", 0.5)
        current = {**current, "lr": round(lr / 2, 6)}
        plan.append((f"把 lr 从 {lr} 降到 {current['lr']}", current))

    bucket = current.get("bucket", 0)
    target = max(bucket, SAFE_BUCKET) * 2
    current = {**current, "bucket": target}
    plan.append((f"把 bucket 从 {bucket} 翻倍到 {target}(换一批哈希桶打散数值退化)", current))

    return plan


def train_with_retry(fasttext, kwargs, max_attempts: int = 6):
    """调用 fasttext.train_supervised; 遇到数值溢出(NaN)时按阶梯自动调整参数重试。

    返回 (model, 实际生效的参数), 真实使用的参数会进日志与 JSON 报告, 不会静默改变结果。
    """
    plan = [(None, dict(kwargs))] + nan_retry_plan(kwargs)
    plan = plan[:max_attempts]
    for index, (note, params) in enumerate(plan):
        try:
            return fasttext.train_supervised(**params), params
        except RuntimeError as exc:
            if "NaN" not in str(exc):
                raise
            if index + 1 >= len(plan):
                raise SystemExit(
                    f"[错误] 训练失败({exc}): 已按阶梯调整参数重试 {len(plan) - 1} 次仍然溢出。\n"
                    f"       最后一轮参数: bucket={params.get('bucket')}, lr={params.get('lr')}, "
                    f"dim={params.get('dim')}, epoch={params.get('epoch')}, thread={params.get('thread')}\n"
                    f"       建议: 补充更多训练数据, 或手动换一组参数(如 --dim/--lr/--bucket/--thread), "
                    f"NaN 的触发取决于参数组合, 换任意一个通常就能跳出去。"
                ) from exc
            log(
                f"[警告] 训练出现 NaN({exc}); 自动{plan[index + 1][0]}, "
                f"重试(第 {index + 2}/{len(plan)} 次)"
            )
    raise SystemExit("[错误] 训练失败: 重试次数已用尽。")


def train_model(fasttext, args, train_path: Path, valid_path):
    """训练模型。返回 (model, 实际使用的参数/调参信息)。"""
    if args.autotune:
        if valid_path is None:
            raise SystemExit("[错误] 使用 --autotune 时必须提供验证集(--valid 或允许自动切分)。")
        log(f"调参模式   : autotune, 上限 {args.autotune_duration}s, 指标 {args.autotune_metric}")
        log(f"验证集     : {valid_path}")
        log("说明       : autotune 只搜索 dim/epoch/lr/wordNgrams/bucket/loss 等, 手动指定的这些参数会被忽略")
        log(
            f"说明       : --min-n/--max-n 会固定为 {args.min_n}/{args.max_n} 交给 autotune"
            "(fastText 的搜索空间里没有 1, 中文未分词时会退化成无字符 n-gram 的模型)"
        )
        log(
            f"说明       : autotune 只优化指标不看体积, 若 bucket 超过 {args.autotune_max_bucket} "
            "会用同一批超参数压低 bucket 重训"
        )
        params = {
            "autotuneValidationFile": str(valid_path),
            "autotuneMetric": args.autotune_metric,
            "autotuneDuration": args.autotune_duration,
            "autotunePredictions": args.autotune_predictions,
            "minn": args.min_n,
            "maxn": args.max_n,
            "thread": args.thread,
            "seed": args.seed,
        }
        warn_risky_subword(args)
        model = fasttext.train_supervised(input=str(train_path), **params)
        return model, {"autotune": True, **params}

    kwargs = build_train_kwargs(args, train_path)
    log("训练参数   : " + ", ".join(f"{key}={value}" for key, value in kwargs.items() if key != "input"))
    warn_risky_bucket(args)
    warn_risky_subword(args)
    model, used_kwargs = train_with_retry(fasttext, kwargs)
    return model, {"autotune": False, **used_kwargs}


def estimate_model_mb(vocab_size: int, bucket: int, dim: int) -> float:
    """解析估算未量化模型的体积: 输入矩阵 (词表 + bucket) x dim x 4 字节。

    输出层只有 标签数 x dim, 相对输入矩阵可忽略。实测与保存后的文件大小一致(误差 < 1%),
    例如 词表=474, bucket=200000, dim=100 -> 76.5 MB。
    """
    return (vocab_size + bucket) * dim * 4 / 1024 / 1024


def apply_autotune_bucket_cap(fasttext, args, model, params, train_path: Path):
    """防止 autotune 选出超大 bucket 导致模型文件暴涨(实测小数据集上可达数 GB)。

    autotune 只优化准确率相关指标, 不关心模型体积, 因此这里在结果超出
    --autotune-max-bucket 时, 用同一批超参数把 bucket 压到上限后重训一次。
    """
    cap = args.autotune_max_bucket
    tuned = model_summary(model)
    if cap <= 0 or tuned["bucket"] <= cap:
        return model, params

    estimated_mb = estimate_model_mb(tuned["vocab_size"], tuned["bucket"], tuned["dim"])
    retrain_kwargs = {
        "input": str(train_path),
        "dim": tuned["dim"],
        "epoch": tuned["epoch"],
        "lr": tuned["lr"],
        "ws": tuned["ws"],
        "wordNgrams": tuned["wordNgrams"],
        "minCount": tuned["minCount"],
        "minCountLabel": args.min_count_label,
        "minn": tuned["minn"],
        "maxn": tuned["maxn"],
        "loss": tuned["loss"],
        "neg": tuned["neg"],
        "bucket": cap,
        "lrUpdateRate": args.lr_update_rate,
        "thread": args.thread,
        "seed": args.seed,
        "verbose": args.verbose,
    }
    try:
        capped_model, used_kwargs = train_with_retry(fasttext, retrain_kwargs)
    except SystemExit as exc:
        # 压低 bucket 后重训仍可能撞上 NaN(触发条件取决于参数组合), 此时保留 autotune
        # 原模型: 模型大一点总比整个流程失败好, 并且真实体积会在日志与报告里写明。
        log(
            f"[警告] autotune 选出的 bucket={tuned['bucket']} 超过上限 {cap}, 但压低 bucket 重训失败, "
            f"保留 autotune 原模型(约 {estimated_mb:.0f} MB)。原因: {str(exc).splitlines()[0]}"
        )
        updated = dict(params)
        updated["retrain_after_autotune"] = "failed"
        return model, updated

    log(
        f"[警告] autotune 选出的 bucket={tuned['bucket']}(模型预计约 {estimated_mb:.0f} MB), "
        f"已按 --autotune-max-bucket={cap} 用同一批超参数重新训练为 "
        f"bucket={used_kwargs.get('bucket')}(实际体积以日志中的『模型已保存』为准)"
    )
    updated = dict(params)
    updated["retrained_after_autotune"] = used_kwargs
    return capped_model, updated


MODEL_CONFIG_FIELDS = (
    "dim",
    "epoch",
    "lr",
    "ws",
    "wordNgrams",
    "minCount",
    "minn",
    "maxn",
    "loss",
    "bucket",
    "neg",
    "thread",
)


def read_model_config(model):
    """读取模型超参数, 返回 (config, 是否来自模型文件)。

    训练得到的模型可以直接取属性; 而 fasttext.load_model() 载入的模型没有这些属性,
    只能退回底层 f.getArgs() —— 其中 lr/thread 反映的是当前进程的默认值, 仅供参考。
    """
    from_file = False
    raw_args = None
    config = {}
    for field in MODEL_CONFIG_FIELDS:
        try:
            value = getattr(model, field)
        except Exception:  # noqa: BLE001 - 载入的模型属性缺失时回退到 getArgs()
            if raw_args is None:
                raw_args = getattr(getattr(model, "f", None), "getArgs", lambda: None)()
            value = getattr(raw_args, field, None) if raw_args is not None else None
            from_file = True
        config[field] = value
    return config, from_file


def model_summary(model) -> dict:
    """整理模型的关键参数, 用于打印和写入报告(autotune 后的真实取值也在这里体现)。"""
    config, from_file = read_model_config(model)
    loss = config.pop("loss")
    return {
        **config,
        "loss": getattr(loss, "name", str(loss)),
        "config_source": "model_file" if from_file else "training",
        "vocab_size": len(model.words),
        "labels": list(model.labels),
    }


def print_model_summary(summary: dict) -> None:
    log(f"词表大小   : {summary['vocab_size']}")
    if summary["config_source"] == "model_file":
        log(
            "模型参数   : dim={dim}, epoch={epoch}, wordNgrams={wordNgrams}, minCount={minCount}, "
            "minn={minn}, maxn={maxn}, loss={loss}, bucket={bucket}".format(**summary)
        )
        log("说明       : 参数读取自模型文件, 其中 lr/thread 无法从文件还原, 这里是当前环境默认值")
        return
    log(
        "模型参数   : dim={dim}, epoch={epoch}, lr={lr}, wordNgrams={wordNgrams}, "
        "minCount={minCount}, minn={minn}, maxn={maxn}, loss={loss}, bucket={bucket}".format(**summary)
    )


def save_model(model, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output))
    size_mb = output.stat().st_size / 1024 / 1024
    log(f"模型已保存 : {output} ({size_mb:.2f} MB)")
    try:
        is_quantized = bool(model.is_quantized())
    except Exception:  # noqa: BLE001 - 老版本可能没有该方法, 不影响保存
        is_quantized = False
    if is_quantized and output.suffix.lower() != ".ftz":
        log(
            "[提示] 这是量化模型, 端侧约定用 .ftz 扩展名(如 -o models/intent.ftz), "
            "便于区分是否量化(fastText 载入时不看后缀, 改名不影响可用性)"
        )
    if size_mb > 300:
        log(
            f"[提示] 模型体积偏大({size_mb:.0f} MB), 主要来自 bucket×dim 的词向量矩阵。"
            "若用于端侧/内存受限场景, 可降低 --bucket / --dim, 或加 --quantize 做乘积量化。"
        )
    return size_mb


def quantize_layout(dim: int, dsub: int) -> tuple[int, int]:
    """复刻 fastText ProductQuantizer 的切分规则, 返回 (subvector 个数, 末段维度)。

    源码 src/productquantizer.cc:

        nsubq_(dim / dsub), dsub_(dsub), lastdsub_ = dim_ % dsub;
        if (lastdsub_ == 0) { lastdsub_ = dsub_; } else { nsubq_++; }

    也就是说:
      * dim 是 dsub 的整数倍 -> nsubq = dim/dsub 段, 每段都是 dsub 维;
      * dim 不是整数倍 -> 前面 nsubq 段是 dsub 维, **末段变短**为 dim % dsub 维(维度不会丢);
      * dsub >= dim -> 退化成 1 段、维度就是 dim(即每行只有 1 个 code, 压缩下限)。
    """
    nsubq, lastdsub = divmod(dim, dsub)
    if lastdsub == 0:
        lastdsub = dsub
    else:
        nsubq += 1
    return nsubq, lastdsub


def log_quantize_layout(dim: int, dsub: int) -> None:
    """dsub 与 dim 不整除/超过 dim 时, 提前说明 fastText 实际怎么切、体积会是多少。"""
    nsubq, lastdsub = quantize_layout(dim, dsub)
    if dsub > dim:
        log(
            f"[提示] --quantize-dsub({dsub}) 大于 --dim({dim}): fastText 会退化成 1 个 subvector, "
            f"等价于 dsub={dim}(每行 {nsubq} 字节 + 码本 {dim * 256 * 4 / 1024:.0f} KB), "
            "这已是压缩下限, 再调大体积基本不变"
        )
    elif dim % dsub != 0:
        segments = " + ".join([str(dsub)] * (nsubq - 1) + [str(lastdsub)])
        log(
            f"[提示] --dim({dim}) 不能被 --quantize-dsub({dsub}) 整除: 实际切成 {nsubq} 段"
            f"({segments} 维), 每行 {nsubq} 字节; 想让体积公式好估算, dsub 取 dim 的因数"
        )


def apply_quantization(model, args, train_path: Path):
    """对训练好的模型做乘积量化(PQ), 端侧体积与内存约降到 1/(nsubq + 1)。

    体积公式(实测): (词表 + bucket) x (nsubq + 1) 字节 + 码本(dim x 256 x 4 字节,
    dim=20 时 20 KB)。nsubq 是 fastText 实际切出的 subvector 个数(见 quantize_layout):
    dim 是 dsub 的整数倍时 nsubq = dim/dsub, 例如 dim=100/dsub=2 约是未量化的 1/8
    (实测 76.5 MB -> 9.7 MB)。量化是一次性离线开销(PQ 训练为单线程, 实测
    dim=100/bucket=50000 约 17s, dim=20/bucket=50000 约 3.4s, 随 bucket 线性增长)。
    cutoff 固定用 QUANTIZE_CUTOFF(0), 原因见该常量处的注释。
    失败时保留未量化模型并给出警告, 不让整个流程失败。
    """
    pre_summary = model_summary(model)
    before_mb = estimate_model_mb(pre_summary["vocab_size"], pre_summary["bucket"], pre_summary["dim"])
    kwargs = {"dsub": args.quantize_dsub, "cutoff": QUANTIZE_CUTOFF}
    if args.quantize_qnorm:
        kwargs["qnorm"] = True
    if args.quantize_retrain:
        kwargs["retrain"] = True
        kwargs["input"] = str(train_path)
    log_quantize_layout(pre_summary["dim"], args.quantize_dsub)
    started = time.time()
    try:
        model.quantize(**kwargs)
    except Exception as exc:  # noqa: BLE001 - 量化失败不应丢掉已经训好的模型
        log(f"[警告] 量化失败({type(exc).__name__}: {exc}), 保留未量化模型(约 {before_mb:.2f} MB)")
        return model, None
    seconds = time.time() - started
    info = {
        "dsub": args.quantize_dsub,
        "cutoff": QUANTIZE_CUTOFF,
        "qnorm": bool(args.quantize_qnorm),
        "retrain": bool(args.quantize_retrain),
        "seconds": round(seconds, 3),
        "estimated_size_before_mb": round(before_mb, 4),
    }
    log(
        "量化       : "
        + ", ".join(f"{key}={value}" for key, value in kwargs.items() if key != "input")
        + f" (耗时 {fmt_seconds(seconds)}, 未量化约 {before_mb:.2f} MB, 量化后体积见『模型已保存』)"
    )
    if args.quantize_retrain:
        log("说明       : retrain 会在量化后再微调一轮, 实测本示例数据无收益, 只是更慢")
    return model, info


def default_ftz_output(model_path: Path) -> Path:
    """由源模型名推导量化模型输出路径: models/intent.bin -> models/intent.ftz。"""
    if model_path.suffix.lower() == ".ftz":
        return model_path.with_name(model_path.stem + ".quantized.ftz")
    return model_path.with_suffix(".ftz")


def resolve_quantize_output(args, model_path: Path) -> Path:
    """决定量化模型的输出路径: 显式 -o 优先, 否则由源模型名推导出 .ftz。

    与源文件同名会覆盖输入模型, 属于必须拦截的操作; 目标已存在时只做提示。
    """
    output = Path(args.output).expanduser() if args.output_given else default_ftz_output(model_path)
    if output.resolve() == model_path.resolve():
        raise SystemExit(
            f"[错误] 量化输出与源模型是同一个文件({output}), 会覆盖源模型, 请用 -o 换个路径"
        )
    if output.exists():
        log(f"[提示] {output} 已存在, 将被覆盖")
    return output


def collect_probe_texts(candidates, limit: int = PROBE_TEXT_LIMIT):
    """从候选文件里取前 limit 条纯文本(去掉 __label__), 用于量化前后的一致性对比。"""
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.exists():
            continue
        # 训练数据可能是目录(--split-by-label 的产物), 这里同样支持
        if path.is_dir():
            files = sorted(
                item
                for item in path.iterdir()
                if item.is_file() and item.suffix.lower() == DATA_SUFFIX
            )
        else:
            files = [path]
        texts = []
        for file in files:
            for raw in load_lines(file):
                labels, words = parse_line(raw)
                text = " ".join(words) if labels else raw.strip()
                if text:
                    texts.append(text)
                if len(texts) >= limit:
                    break
            if len(texts) >= limit:
                break
        if texts:
            return path, texts
    return None, []


def compare_predictions(model_a, model_b, texts, k: int, threshold: float) -> dict:
    """逐条比较两个模型的预测标签是否一致(量化有没有改变行为最直接的证据)。

    只比标签不比概率: 量化必然带来轻微数值扰动, 概率不会逐位相同。
    """
    same = 0
    for text in texts:
        labels_a = [label for label, _ in predict_one(model_a, text, k, threshold)]
        labels_b = [label for label, _ in predict_one(model_b, text, k, threshold)]
        if labels_a == labels_b:
            same += 1
    total = len(texts)
    return {
        "samples": total,
        "same": same,
        "rate": round(same / total, 6) if total else None,
        "k": k,
    }


def verify_quantized_export(
    fasttext,
    args,
    output_path: Path,
    summary_before: dict,
    source_path: Path = None,
    eval_path: Path = None,
):
    """量化模型落盘后的自检: 能否重新载入 + 是否真是量化模型 + 与量化前是否一致。

    只给 output_path(训练流程)时做「载入 + 量化标记 + 词表/标签一致」校验;
    再给 source_path(从 .bin 导出流程)时额外做逐条预测一致率与指标前后对比。
    返回 (verification, 载入的量化模型); 载入失败时模型为 None。
    """
    verification = {"reloaded": False}
    try:
        quantized = fasttext.load_model(str(output_path))
    except Exception as exc:  # noqa: BLE001 - 载入失败要给出可读结论而不是抛栈
        message = f"{type(exc).__name__}: {exc}"
        log(f"[错误] 量化模型重新载入失败: {message}")
        verification["error"] = message
        return verification, None

    verification.update(
        {
            "reloaded": True,
            "is_quantized": bool(quantized.is_quantized()),
            "vocab_size": len(quantized.words),
            "vocab_size_same": len(quantized.words) == summary_before["vocab_size"],
            "labels_same": list(quantized.labels) == list(summary_before["labels"]),
        }
    )
    log(
        f"量化校验   : 重新载入 OK, is_quantized={verification['is_quantized']}, "
        f"词表 {verification['vocab_size']}(与量化前一致: {verification['vocab_size_same']}), "
        f"标签一致: {verification['labels_same']}"
    )
    if not verification["is_quantized"]:
        log("[警告] 落盘文件没有被标记为量化模型, 请检查 fastText 版本")

    if source_path is not None:
        source = fasttext.load_model(str(source_path))
        probe_path, texts = collect_probe_texts([eval_path, args.test, args.valid, args.input])
        if texts:
            agreement = compare_predictions(source, quantized, texts, args.k, args.threshold)
            agreement["source"] = str(probe_path)
            verification["prediction_agreement"] = agreement
            log(
                f"预测一致率 : {agreement['same']}/{agreement['samples']} "
                f"(k={args.k}, 基于 {probe_path}, 最多对比前 {PROBE_TEXT_LIMIT} 条)"
            )
        if eval_path is not None:
            before = evaluate(source, eval_path, args.k, args.threshold, max_failures=0)
            after = evaluate(quantized, eval_path, args.k, args.threshold, max_failures=0)
            verification["evaluation"] = {
                "path": str(eval_path),
                "source_model": before,
                "quantized_model": after,
                "precision_delta": round(after["precision"] - before["precision"], 6),
                "recall_delta": round(after["recall"] - before["recall"], 6),
            }
            log(
                f"命中对比   : 量化前 {before['correct']}/{before['judged_samples']}"
                f" ({before['success_rate'] * 100:.2f}%) -> 量化后 {after['correct']}/{after['judged_samples']}"
                f" ({after['success_rate'] * 100:.2f}%)  (基于 {eval_path})"
            )
    return verification, quantized


# --------------------------------------------------------------------------- #
# 效果评估
# --------------------------------------------------------------------------- #
def safe_metric(value) -> float:
    """fastText 在某些标签上会返回 nan(例如该标签没有被预测到), 统一归零避免写出非法 JSON。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def scan_eval(model, eval_path: Path, k: int, threshold: float, max_failures=None) -> dict:
    """逐条预测评估集, 统计成功/失败条数并收集判错的样本。

    成功 = 预测的前 k 个标签里命中了真实标签(单标签时就是"猜对了")。
    失败明细最多收集 max_failures 条(None 表示不限制, 0 表示只统计不收集明细),
    失败总数仍然精确统计。
    """
    total = 0
    correct = 0
    unlabeled = 0
    failures = []
    for raw_line in load_lines(eval_path):
        true_labels, words = parse_line(raw_line)
        text = " ".join(words) if true_labels else raw_line
        if not text:
            continue
        total += 1
        if not true_labels:
            unlabeled += 1
            continue
        predictions = predict_one(model, text, k, threshold)
        predicted_labels = [strip_prefix(label) for label, _ in predictions]
        if set(predicted_labels) & set(true_labels):
            correct += 1
        elif max_failures is None or len(failures) < max_failures:
            failures.append(
                {
                    "text": text,
                    "true": true_labels,
                    "predicted": [
                        {"label": strip_prefix(label), "prob": round(prob, 6)}
                        for label, prob in predictions
                    ],
                }
            )
    judged = total - unlabeled
    return {
        "total": total,
        "judged": judged,
        "correct": correct,
        "failed": judged - correct,
        "failures": failures,
        "failures_complete": len(failures) == judged - correct,
    }


def materialize_eval(eval_path: Path):
    """fastText 的 test()/test_label() 只接受文件路径, 目录形态的评估集先合并成临时文件。

    返回 (实际读取的文件路径, 需要清理的临时目录或 None)。
    """
    if not eval_path.is_dir():
        return eval_path, None
    lines = load_lines(eval_path)
    temp_dir = Path(tempfile.mkdtemp(prefix="fasttext_eval_"))
    target = temp_dir / "eval_all.txt"
    write_lines(target, lines)
    log(
        f"合并评估集 : {eval_path} ({len(lines)} 条, {len(list_data_files(eval_path))} 个文件)",
        level=1,
    )
    return target, temp_dir


def evaluate(
    model,
    eval_path: Path,
    k: int,
    threshold: float,
    with_failures: bool = True,
    max_failures=None,
) -> dict:
    """在评估集上计算整体与单标签指标, 以及"成功/失败"的逐条统计。

    fastText 返回的 precision/recall 是"所有标签平均"的口径; 对于单标签数据,
    P@1 就等价于准确率(accuracy)。
    """
    eval_target, eval_temp = materialize_eval(eval_path)
    try:
        samples, precision, recall = model.test(str(eval_target), k=k, threshold=threshold)
        raw = {}
        try:
            raw = model.test_label(str(eval_target), k=k, threshold=threshold) or {}
        except Exception as exc:  # noqa: BLE001 - 单标签指标失败不应该影响主流程
            log(f"[警告] 单标签指标计算失败: {exc}", level=1)
    finally:
        if eval_temp is not None:
            shutil.rmtree(eval_temp, ignore_errors=True)

    precision = safe_metric(precision)
    recall = safe_metric(recall)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    per_label = {}
    for label, scores in sorted(raw.items()):
        label_p = safe_metric(scores.get("precision", 0.0))
        label_r = safe_metric(scores.get("recall", 0.0))
        label_f1 = scores.get("f1score")
        if label_f1 is None:
            label_f1 = 0.0 if label_p + label_r == 0 else 2 * label_p * label_r / (label_p + label_r)
        per_label[label] = {
            "precision": round(label_p, 6),
            "recall": round(label_r, 6),
            "f1": round(safe_metric(label_f1), 6),
        }

    result = {
        "samples": samples,
        "k": k,
        "threshold": threshold,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "per_label": per_label,
    }
    if with_failures:
        scan = scan_eval(model, eval_path, k, threshold, max_failures=max_failures)
        judged = scan["judged"]
        result.update(
            {
                "judged_samples": judged,
                "correct": scan["correct"],
                "failed": scan["failed"],
                "success_rate": round(scan["correct"] / judged, 6) if judged else 0.0,
                "failure_rate": round(scan["failed"] / judged, 6) if judged else 0.0,
                "failures": scan["failures"],
                "failures_complete": scan["failures_complete"],
            }
        )
    return result


def print_evaluation(
    result: dict,
    eval_path: Path,
    is_multilabel: bool,
    note: str = "",
    max_failures: int = 0,
    show_metrics: bool = False,
) -> None:
    """打印评估结果: 先给条目数/成功率/失败率, 再逐条列出判错的样本。

    max_failures <= 0 表示全部列出(默认), >0 时只列前 N 条并提示剩余条数。
    逐标签 P/R/F1 属于"细节", 默认不打印, 需要时加 --metrics。
    """
    total = result.get("judged_samples")
    log(f"评估数据   : {eval_path} ({result['samples']} 条)")
    if total is not None:
        log(f"测试条目   : {total} 条")
        log(f"成功       : {result['correct']} 条  (成功率 {result['success_rate'] * 100:.2f}%)")
        log(f"失败       : {result['failed']} 条  (失败率 {result['failure_rate'] * 100:.2f}%)")
        failures = result.get("failures") or []
        if failures:
            shown = failures if max_failures <= 0 else failures[:max_failures]
            log(f"失败明细   :")
            for index, item in enumerate(shown, 1):
                truth = ", ".join(item["true"]) or "(无标签)"
                predicted = (
                    ", ".join(f"{p['label']}({p['prob']:.4f})" for p in item["predicted"])
                    or "(无预测结果)"
                )
                log(f"  {index:>2}. {item['text']}")
                log(f"      真实: {truth}")
                log(f"      预测: {predicted}")
            if len(shown) < result["failed"]:
                log(
                    f"      ... 还有 {result['failed'] - len(shown)} 条失败未显示"
                    "(用 --max-failures 0 或去掉该参数即可查看全部)"
                )
    if note:
        log(f"[注意] {note}")
    if show_metrics:
        k = result["k"]
        log(f"指标详情   : P@{k}={result['precision']:.4f}  R@{k}={result['recall']:.4f}  F1={result['f1']:.4f}")
        if not is_multilabel:
            if k == 1:
                log("说明       : 单标签数据下 P@1 = R@1 = F1 = 准确率(accuracy)")
            else:
                log(
                    f"说明       : 单标签数据下 R@{k} 表示真实标签落在前 {k} 个预测里的比例(相当于 top-{k} 准确率); "
                    f"P@{k} 会随 k 增大而下降, 想看分类准确率请用 -k 1"
                )
        if result["per_label"]:
            scope = "多标签数据按标签" if is_multilabel else "单标签数据按标签"
            log(f"每标签指标 : ({scope}, k={k}, 未被预测到的标签 P/R/F1 记为 0)")
            log(f"  {'标签':<16}{'P':>10}{'R':>10}{'F1':>10}")
            for label, scores in result["per_label"].items():
                log(f"  {label:<16}{scores['precision']:>10.4f}{scores['recall']:>10.4f}{scores['f1']:>10.4f}")


# --------------------------------------------------------------------------- #
# 预测推理
# --------------------------------------------------------------------------- #
def predict_one(model, text: str, k: int, threshold: float):
    """对单条文本预测, 返回 [(label, prob), ...]。

    兼容性处理: fasttext==0.9.3 的 Python 封装里 predict() 用了
    np.array(probs, copy=False), 在 NumPy 2.x 下会抛 ValueError;
    这里捕获后直接调用底层接口, 保证在两种环境下都能预测。
    """
    if not text.strip():
        return []
    try:
        labels, probs = model.predict(text, k=k, threshold=threshold)
        return [(label, float(prob)) for label, prob in zip(labels, probs)]
    except ValueError as exc:
        if "copy" not in str(exc):
            raise
        raw = model.f.predict(text + "\n", k, threshold, "strict")
        return [(label, float(prob)) for prob, label in raw]


def strip_prefix(label: str) -> str:
    return label[len(LABEL_PREFIX):] if label.startswith(LABEL_PREFIX) else label


def format_predictions(predictions) -> str:
    if not predictions:
        return "(无预测结果, 请检查输入文本是否为空)"
    return ", ".join(f"{strip_prefix(label)}({prob:.4f})" for label, prob in predictions)


def print_predictions(model, texts, k: int, threshold: float) -> None:
    heading = f"预测结果 (k={k}, threshold={threshold})"
    log("\n" + heading)
    log("-" * len(heading))
    for text in texts:
        log(f"{text}")
        log(f"  -> {format_predictions(predict_one(model, text, k, threshold))}")


def predict_file(model, path: Path, k: int, threshold: float) -> int:
    """对文件里的每一行文本做预测。

    stdout 输出 JSONL(每行一条, 便于程序消费), stderr 输出人类可读日志。
    若待预测文件本身带 __label__ 标签(例如数据文件/测试集), 会把真实标签一并写进
    JSONL 的 true_labels 字段, 方便直接拿去做 badcase 分析。
    """
    if not path.exists():
        raise SystemExit(f"[错误] 找不到待预测文件: {path}")
    if path.is_dir():
        files = sorted(
            item for item in path.iterdir() if item.is_file() and item.suffix.lower() == DATA_SUFFIX
        )
        if not files:
            raise SystemExit(f"[错误] 目录里没有 {DATA_SUFFIX} 文件: {path}")
    else:
        files = [path]

    count = 0
    for file in files:
        if len(files) > 1:
            log(f"预测文件   : {file}", level=1)
        with open(file, "r", encoding="utf-8") as handle:
            for line in handle:
                raw_line = line.strip()
                if not raw_line:
                    continue
                true_labels, words = parse_line(raw_line)
                text = " ".join(words) if true_labels else raw_line
                if not text:
                    log(f"[跳过] 该行只有标签没有文本: {raw_line[:40]}", level=1)
                    continue
                predictions = predict_one(model, text, k, threshold)
                record = {
                    "text": text,
                    "predictions": [{"label": strip_prefix(label), "prob": prob} for label, prob in predictions],
                }
                if true_labels:
                    record["true_labels"] = true_labels
                    hit = bool({strip_prefix(label) for label, _ in predictions} & set(true_labels))
                    record["hit"] = hit
                print(json.dumps(record, ensure_ascii=False), flush=True)
                suffix = ""
                if true_labels:
                    suffix = f"  [真实: {','.join(true_labels)} {'命中' if hit else '未命中'}]"
                log(f"{text} -> {format_predictions(predictions)}{suffix}")
                count += 1
    return count


def interactive_loop(model, k: int, threshold: float) -> None:
    """交互式预测, 方便边调数据边验证效果。输入 exit / quit / Ctrl-D 退出。"""
    log("\n进入交互模式: 输入文本回车查看预测, 输入 exit / quit 或按 Ctrl-D 退出")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            log("\n退出交互模式")
            return
        if not text:
            continue
        if text.lower() in {"exit", "quit", ":q"}:
            log("退出交互模式")
            return
        log(f"  -> {format_predictions(predict_one(model, text, k, threshold))}")


# --------------------------------------------------------------------------- #
# 命令行参数
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fastText 监督分类模型训练脚本(数据检查 + 训练 + 评估 + 推理)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("数据")
    data.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT,
        help=(
            "训练数据路径: 单个文件, 或按标签拆分的目录"
            f"(默认 {DEFAULT_INPUT}, 目录里的 *{DATA_SUFFIX} 会按文件名排序后合并)"
        ),
    )
    data.add_argument("-t", "--test", default=None, help="独立测试集路径, 提供后用整个训练集训练")
    data.add_argument("--valid", default=None, help="验证集路径, 主要用于 --autotune")
    data.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="未提供 --test/--valid 时自动切分出的验证集比例, 设为 0 表示不切分",
    )
    data.add_argument("--seed", type=int, default=42, help="随机种子(切分与模型初始化)")
    data.add_argument(
        "--split-by-label",
        default=None,
        metavar="DIR",
        help=(
            f"按标签把 -i 的数据拆成 DIR/<标签>{SPLIT_FILE_SUFFIX} 后退出(不训练); "
            "DIR 里其它 *.txt 不会被删除, 但会参与读取, 有残留副本时报错中止"
        ),
    )

    output = parser.add_argument_group("输出")
    output.add_argument(
        "-o",
        "--output",
        default=argparse.SUPPRESS,
        help=(
            f"模型保存路径(默认 {DEFAULT_OUTPUT}); 未量化建议 .bin, 量化模型建议 .ftz; "
            "-m/--model 模式下是量化模型的输出路径(默认由源模型名推导为 <源模型>.ftz)"
        ),
    )
    output.add_argument(
        "-m",
        "--model",
        default=None,
        help="载入已有模型(.bin/.ftz), 跳过训练, 只做评估/推理; 配合 --quantize 可量化导出 .ftz",
    )
    output.add_argument("--report", default=None, help="训练报告 JSON 路径, 默认 <模型>.report.json")
    output.add_argument("--no-report", action="store_true", help="不写训练报告")
    output.add_argument("--work-dir", default=None, help="存放切分后中间文件的目录, 默认用临时目录")
    output.add_argument("--keep-work-dir", action="store_true", help="保留中间文件(临时目录时不自动清理)")

    hyper = parser.add_argument_group("超参数")
    hyper.add_argument("--epoch", type=int, default=25, help="训练轮数")
    hyper.add_argument("--lr", type=float, default=0.5, help="学习率")
    hyper.add_argument("--dim", type=int, default=100, help="词向量维度")
    hyper.add_argument("--ws", type=int, default=5, help="上下文窗口大小")
    hyper.add_argument("--word-ngrams", type=int, default=2, help="word n-gram 最大阶数(>=2 时用上词组特征)")
    hyper.add_argument("--min-count", type=int, default=1, help="词频低于该值的词直接丢弃")
    hyper.add_argument("--min-count-label", type=int, default=1, help="标签最低出现次数")
    hyper.add_argument("--min-n", type=int, default=1, help="subword 最小长度(中文未分词建议 1)")
    hyper.add_argument("--max-n", type=int, default=3, help="subword 最大长度(中文未分词建议 3)")
    hyper.add_argument(
        "--loss",
        choices=["softmax", "ova", "hs"],
        default="softmax",
        help="损失函数: 单标签用 softmax, 多标签/类别很多用 ova 或 hs(层次 softmax)",
    )
    hyper.add_argument("--neg", type=int, default=5, help="负采样数量(仅 loss=ns 时生效, 保留以兼容)")
    hyper.add_argument(
        "--bucket",
        type=int,
        default=200000,
        help="subword 哈希桶数量, 模型体积 ≈ (词表 + bucket) × dim × 4 字节, 小数据集可调小",
    )
    hyper.add_argument("--lr-update-rate", type=int, default=100, help="学习率衰减更新频率")
    hyper.add_argument("--thread", type=int, default=os.cpu_count() or 1, help="训练线程数")

    quant = parser.add_argument_group("体积 / 端侧部署")
    quant.add_argument(
        "--quantize",
        action="store_true",
        help=(
            "乘积量化(product quantization): 训练后压缩再落盘, 或配合 -m/--model 把已有 .bin 量化导出 .ftz; "
            "体积与内存约降到 1/(dim/dsub+1)(dim=100/dsub=2 时约 1/8)"
        ),
    )
    quant.add_argument(
        "--quantize-dsub",
        type=int,
        default=2,
        help="量化时每个 subvector 的维度, 越大压得越狠(把握不住的先留 2)",
    )
    quant.add_argument(
        "--quantize-qnorm",
        action="store_true",
        help="量化前做 L2 归一化并单独量化模长(体积略增, 实测本示例数据无收益)",
    )
    quant.add_argument(
        "--quantize-retrain",
        action="store_true",
        help="量化后用训练数据再微调一轮(更慢, 实测本示例数据无收益)",
    )

    tune = parser.add_argument_group("自动调参")
    tune.add_argument("--autotune", action="store_true", help="启用 fastText 内置 autotune 搜索超参数")
    tune.add_argument("--autotune-duration", type=int, default=60, help="autotune 时长上限(秒)")
    tune.add_argument(
        "--autotune-metric",
        choices=["f1", "accuracy", "precisionAtRecall", "recallAtPrecision"],
        default="f1",
        help="autotune 优化指标",
    )
    tune.add_argument("--autotune-predictions", type=int, default=1, help="autotune 评估时的 k")
    tune.add_argument(
        "--autotune-max-bucket",
        type=int,
        default=None,
        help="autotune 结果中 bucket 的上限(默认跟随 --bucket, 超过则用同一批超参数压低 bucket 重训; 0 表示不限制)",
    )

    run = parser.add_argument_group("评估与推理")
    run.add_argument("-k", type=int, default=1, help="评估/预测时取前 k 个标签")
    run.add_argument("--threshold", type=float, default=0.0, help="预测概率阈值(>0 时只保留高于阈值的标签)")
    run.add_argument(
        "--max-failures",
        type=int,
        default=0,
        metavar="N",
        help="评估结果里最多列出多少条失败样本(默认 0 = 全部列出, 给正数则只列前 N 条)",
    )
    run.add_argument(
        "--metrics",
        action="store_true",
        help="额外打印 P/R/F1 与每标签指标(默认只打印条目数/成功率/失败率/失败明细)",
    )
    run.add_argument("--predict", nargs="+", default=None, metavar="TEXT", help="训练后对给定文本做预测")
    run.add_argument("--predict-file", default=None, help="训练后对文件逐行预测, 输出 JSONL")
    run.add_argument("-I", "--interactive", action="store_true", help="训练后进入交互式预测")
    run.add_argument("--dry-run", action="store_true", help="只做数据检查与切分, 不训练")
    run.add_argument("--check-env", action="store_true", help="只做环境自检(含一次极小样例训练+预测)")
    run.add_argument("-q", "--quiet", action="store_true", help="精简输出(fastText verbose=0)")
    run.add_argument("--verbose", type=int, default=2, help="fastText 训练日志级别(0 最安静)")

    args = parser.parse_args(argv)

    # -o 用 SUPPRESS 的默认值(而不是 model.bin), 才能区分「用户显式给了 -o」——-m 量化导出时要靠它
    raw_output = getattr(args, "output", None)
    args.output_given = raw_output is not None
    args.output = raw_output if raw_output is not None else DEFAULT_OUTPUT

    if args.autotune_max_bucket is None:
        # 默认跟随 --bucket: 想给端侧出小模型时, autotune 不会偷偷把 bucket 拉回默认的 200000
        args.autotune_max_bucket = args.bucket

    args.auto_quantize_from_extension = False
    if not args.quantize and args.output_given and Path(args.output).suffix.lower() == ".ftz":
        # 文件名写成 .ftz 却不量化, 会得到一个「看起来像量化模型」的未量化文件, 直接替用户打开
        args.quantize = True
        args.auto_quantize_from_extension = True

    if not 0 <= args.valid_ratio < 1:
        parser.error("--valid-ratio 需要落在 [0, 1) 区间")
    if args.k < 1:
        parser.error("-k 必须大于等于 1")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold 需要落在 [0, 1] 区间")
    if args.epoch < 1:
        parser.error("--epoch 必须大于等于 1")
    if args.autotune and args.valid is None and args.valid_ratio <= 0:
        parser.error("--autotune 需要验证集: 请提供 --valid, 或让 --valid-ratio 大于 0")
    if args.model and (args.autotune or args.dry_run):
        parser.error("使用 -m/--model 时不会训练, 请去掉 --autotune/--dry-run")
    if args.split_by_label and args.model:
        parser.error("--split-by-label 只处理数据文件, 不能和 -m/--model 一起使用")
    if args.min_n > args.max_n:
        parser.error(f"--min-n({args.min_n}) 不能大于 --max-n({args.max_n})")
    if args.quantize_dsub < 1:
        parser.error("--quantize-dsub 必须大于等于 1")
    if args.model and not (
        args.test or args.valid or args.predict or args.predict_file or args.interactive or args.quantize
    ):
        parser.error(
            "使用 -m/--model 时请至少指定一项: --test/--valid/--predict/--predict-file/-I, "
            "或加 --quantize 把已有 .bin 量化导出 .ftz"
        )
    return args


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def prepare_work_dir(args):
    """创建存放切分文件的目录, 返回 (work_dir, 是否为临时目录)。"""
    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser()
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir, False
    return Path(tempfile.mkdtemp(prefix="fasttext_split_")), True


def resolve_data(args, input_path: Path, train_lines, work_dir):
    """准备训练/验证文件。

    返回 dict: train_path / valid_path / valid_lines / split_info
    - 提供 --valid: 直接使用该文件作为验证集
    - 未提供 --valid 且 --valid-ratio > 0: 分层切分训练数据
    - 否则: 不切分, 全部数据用于训练
    """
    # 目录输入(按标签拆分的产物)不能被 fastText 直接读取, 先合并成一个文件
    train_path = input_path
    if input_path.is_dir():
        train_path = work_dir / "train_all.txt"
        write_lines(train_path, train_lines)
        log(f"合并训练集 : {len(train_lines)} 条 -> {train_path} (目录输入需先合并成单文件)")

    if args.valid:
        valid_source = Path(args.valid).expanduser()
        valid_lines = load_lines(valid_source)
        valid_path = valid_source
        if valid_source.is_dir():
            # 目录形态的验证集(按标签拆分)也要合并成单文件, 自动调参和 fastText 都只认文件
            valid_path = work_dir / "valid_all.txt"
            write_lines(valid_path, valid_lines)
            log(f"合并验证集 : {len(valid_lines)} 条 -> {valid_path} (目录输入需先合并成单文件)")
        log(f"验证集     : {valid_source} ({len(valid_lines)} 条, 来自 --valid)")
        return {
            "train_path": train_path,
            "valid_path": valid_path,
            "valid_lines": valid_lines,
            "valid_eval_path": valid_source,
            "split_info": None,
        }

    if args.valid_ratio <= 0:
        log("数据划分   : 未切分(--valid-ratio=0), 全部数据用于训练")
        return {
            "train_path": train_path,
            "valid_path": None,
            "valid_lines": None,
            "valid_eval_path": None,
            "split_info": None,
        }

    train_part, valid_part = stratified_split(train_lines, args.valid_ratio, args.seed)
    train_path = work_dir / "train.txt"
    valid_path = work_dir / "valid.txt"
    write_lines(train_path, train_part)
    write_lines(valid_path, valid_part)
    log(
        f"数据划分   : 训练 {len(train_part)} 条 / 验证 {len(valid_part)} 条 "
        f"(valid_ratio={args.valid_ratio}, seed={args.seed})"
    )
    actual_ratio = len(valid_part) / max(len(train_part) + len(valid_part), 1)
    if actual_ratio > args.valid_ratio * 1.5 + 0.05:
        log(
            f"说明       : 为保证每个标签至少有 1 条验证样本, 实际验证集占比为 {actual_ratio:.2f}, "
            "样本很少时建议多补数据"
        )
    log(f"切分文件   : {train_path}, {valid_path}", level=1)
    return {
        "train_path": train_path,
        "valid_path": valid_path,
        "valid_lines": valid_part,
        "valid_eval_path": valid_path,
        "split_info": {
            "train_samples": len(train_part),
            "valid_samples": len(valid_part),
            "valid_ratio": args.valid_ratio,
            "seed": args.seed,
            "files": [str(train_path), str(valid_path)],
        },
    }


def build_report(args, env, stats, test_stats, data_info, model_info, eval_info, duration, input_sources=None):
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": round(duration, 3),
        "environment": env,
        "arguments": vars(args),
        "data": {
            "input": str(args.input),
            "input_files": input_sources,
            "input_stats": stats,
            "test": str(args.test) if args.test else None,
            "test_stats": test_stats,
            "split": data_info["split_info"],
        },
        "model": model_info,
        "evaluation": eval_info,
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    log(f"训练报告   : {path}")


# --------------------------------------------------------------------------- #
# 仅推理模式 (-m/--model)
# --------------------------------------------------------------------------- #
def load_existing_model(fasttext, model_path: Path):
    if not model_path.exists():
        raise SystemExit(f"[错误] 找不到模型文件: {model_path}")
    if model_path.is_dir():
        raise SystemExit(f"[错误] {model_path} 是目录, -m/--model 需要指向 .bin 文件。")
    try:
        model = fasttext.load_model(str(model_path))
    except Exception as exc:  # noqa: BLE001 - 载入失败要把原因告诉用户
        raise SystemExit(f"[错误] 载入模型失败: {type(exc).__name__}: {exc}")
    size_mb = model_path.stat().st_size / 1024 / 1024
    log(f"模型载入   : {model_path} ({size_mb:.2f} MB)")
    return model, size_mb


def run_inference_only(args, fasttext, env, started: float) -> int:
    """-m/--model 模式: 跳过训练, 直接用已有模型做评估/推理。"""
    section("载入模型")
    model_path = Path(args.model).expanduser()
    model, size_mb = load_existing_model(fasttext, model_path)
    summary = model_summary(model)
    print_model_summary(summary)

    eval_result = None
    eval_path = None
    if args.test:
        eval_path = Path(args.test).expanduser()
    elif args.valid:
        eval_path = Path(args.valid).expanduser()

    if eval_path is not None:
        section("效果评估")
        eval_lines = load_lines(eval_path)
        eval_stats, eval_problems = inspect_dataset(eval_lines, name=str(eval_path))
        print_dataset_stats(eval_stats)
        print_problems(eval_problems, limit=3)
        eval_result = evaluate(model, eval_path, args.k, args.threshold)
        eval_result["path"] = str(eval_path)
        eval_result["is_test_set"] = args.test is not None
        print_evaluation(
            eval_result,
            eval_path,
            eval_stats["multi_label_samples"] > 0,
            max_failures=args.max_failures,
            show_metrics=args.metrics,
        )

    if args.predict:
        section("预测(命令行文本)")
        print_predictions(model, args.predict, args.k, args.threshold)

    if args.predict_file:
        section("预测(文件)")
        count = predict_file(model, Path(args.predict_file).expanduser(), args.k, args.threshold)
        log(f"共预测 {count} 条, JSONL 结果已输出到 stdout")

    if args.interactive:
        section("交互式预测")
        interactive_loop(model, args.k, args.threshold)

    if args.report:
        report_path = Path(args.report).expanduser()
        write_report(
            report_path,
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(time.time() - started, 3),
                "environment": env,
                "arguments": vars(args),
                "mode": "inference_only",
                "model": {"path": str(model_path), "size_mb": round(size_mb, 4), "summary": summary},
                "evaluation": eval_result,
            },
        )

    section("完成")
    if eval_result is not None:
        log(
            f"评估结果   : {eval_result['judged_samples']} 条, 成功 {eval_result['correct']} "
            f"({eval_result['success_rate'] * 100:.2f}%), 失败 {eval_result['failed']} "
            f"({eval_result['failure_rate'] * 100:.2f}%)"
        )
    log(f"总耗时     : {fmt_seconds(time.time() - started)}")
    return 0


def run_quantize_only(args, fasttext, env, started: float) -> int:
    """-m/--model + --quantize: 载入已有 .bin, 乘积量化后导出 .ftz(源模型文件的字节不动)。

    与训练流程里的 --quantize 差别:
      * 不改动源模型, 只产出量化模型(所以可以「先留一份 .bin, 再导出 .ftz」);
      * 导出后自动重新载入 .ftz 校验, 并与源模型比预测一致率与指标(有评估集时)。
    """
    section("载入模型")
    model_path = Path(args.model).expanduser()
    model, size_before_mb = load_existing_model(fasttext, model_path)
    summary_before = model_summary(model)
    print_model_summary(summary_before)

    if model.is_quantized():
        raise SystemExit(
            f"[错误] {model_path} 已经是量化模型(文件里存的是 uint8 码 + 码本, 不再是浮点权重), "
            "重复量化会让 fastText 直接段错误崩溃(不是抛异常, try/except 拦不住), 脚本主动拦截。\n"
            "        想更小只有两条路:\n"
            "          1) 用未量化的源模型重新导出: -m <源 .bin> --quantize --quantize-dsub 4\n"
            "             (dsub 越大压得越狠; dsub >= dim 时到下限 1 字节/行, 例如 dim=20 约 0.08 MB)\n"
            "          2) 训练一个更小的模型: --dim 20 --bucket 50000(3.9 MB, 再量化 0.52 MB)\n"
            "        如果你只是想改后缀/复制这个文件, 用 mv/cp 即可(后缀不影响加载, 见 README)"
        )

    output_path = resolve_quantize_output(args, model_path)

    train_path = None
    retrain_temp_dir = None
    if args.quantize_retrain:
        train_path = Path(args.input).expanduser()
        if not train_path.exists():
            raise SystemExit(
                f"[错误] --quantize-retrain 需要原训练数据做微调, 但找不到: {train_path}(可用 -i 指定)"
            )
        if train_path.is_dir():
            # 目录输入(按标签拆分的产物): 先合并成临时文件再交给 fastText 微调
            merged_lines, _ = load_dataset(train_path)
            retrain_temp_dir = Path(tempfile.mkdtemp(prefix="fasttext_retrain_"))
            train_path = retrain_temp_dir / "train_all.txt"
            write_lines(train_path, merged_lines)
            log(f"合并训练集 : {len(merged_lines)} 条 -> {train_path} (目录输入需先合并成单文件)")

    section("乘积量化")
    log(f"输出路径   : {output_path}")
    model, quant_info = apply_quantization(model, args, train_path)
    if retrain_temp_dir is not None:
        shutil.rmtree(retrain_temp_dir, ignore_errors=True)
    if quant_info is None:
        log("[错误] 量化失败, 未生成 .ftz 文件(源模型未被修改)")
        return 2

    section("保存量化模型")
    size_mb = save_model(model, output_path)
    ratio = size_before_mb / size_mb if size_mb else 0.0
    log(
        f"压缩效果   : {size_before_mb:.2f} MB -> {size_mb:.2f} MB "
        f"(约 1/{ratio:.1f}, 省 {size_before_mb - size_mb:.2f} MB, 端侧常驻内存近似等于文件体积)"
    )

    eval_path = None
    if args.test:
        eval_path = Path(args.test).expanduser()
    elif args.valid:
        eval_path = Path(args.valid).expanduser()

    section("校验量化模型")
    verification, quantized = verify_quantized_export(
        fasttext, args, output_path, summary_before, source_path=model_path, eval_path=eval_path
    )
    if quantized is None or not verification.get("is_quantized"):
        log("[错误] 量化模型校验未通过, 请按上面的错误信息排查")
        return 2

    if not args.no_report:
        report_path = Path(args.report).expanduser() if args.report else Path(str(output_path) + ".report.json")
        write_report(
            report_path,
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(time.time() - started, 3),
                "environment": env,
                "arguments": vars(args),
                "mode": "quantize_only",
                "source_model": {
                    "path": str(model_path),
                    "size_mb": round(size_before_mb, 4),
                    "summary": summary_before,
                    "is_quantized": False,
                },
                "model": {
                    "path": str(output_path),
                    "size_mb": round(size_mb, 4),
                    "summary": model_summary(quantized),
                    "quantization": quant_info,
                },
                "compression": {
                    "size_before_mb": round(size_before_mb, 4),
                    "size_after_mb": round(size_mb, 4),
                    "ratio": round(ratio, 4),
                    "saved_mb": round(size_before_mb - size_mb, 4),
                },
                "verification": verification,
                "evaluation": verification.get("evaluation"),
            },
        )

    if args.predict:
        section("量化模型预测(命令行文本)")
        print_predictions(quantized, args.predict, args.k, args.threshold)

    if args.predict_file:
        section("量化模型预测(文件)")
        count = predict_file(quantized, Path(args.predict_file).expanduser(), args.k, args.threshold)
        log(f"共预测 {count} 条, JSONL 结果已输出到 stdout")

    if args.interactive:
        section("交互式预测(量化模型)")
        interactive_loop(quantized, args.k, args.threshold)

    section("完成")
    log(f"源模型     : {model_path} ({size_before_mb:.2f} MB)")
    log(f"端侧模型   : {output_path} ({size_mb:.2f} MB)")
    if verification.get("evaluation"):
        delta = verification["evaluation"]["precision_delta"]
        log(f"量化影响   : P@{args.k} 变化 {delta:+.4f} (基于 {verification['evaluation']['path']})")
    log(f"总耗时     : {fmt_seconds(time.time() - started)}")
    return 0


def main(argv=None) -> int:
    global QUIET

    args = parse_args(argv)
    QUIET = args.quiet
    if args.quiet:
        args.verbose = 0

    started = time.time()

    # 纯数据整理: 不需要 fastText, 放在环境检查之前, 环境没装好也能用
    if args.split_by_label:
        section("按标签拆分数据")
        return split_by_label(
            Path(args.input).expanduser(), Path(args.split_by_label).expanduser()
        )

    section("1/6 环境检查")
    fasttext = import_fasttext()
    env = print_environment(fasttext)

    if args.check_env:
        section("环境自检")
        return 0 if run_env_check(fasttext, env) else 2

    if getattr(args, "auto_quantize_from_extension", False):
        log(f"[提示] 输出路径 {args.output} 以 .ftz 结尾, 已自动开启 --quantize(量化模型约定用 .ftz 扩展名)")

    if args.model:
        if args.quantize:
            return run_quantize_only(args, fasttext, env, started)
        return run_inference_only(args, fasttext, env, started)

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    section("2/6 数据检查")
    train_lines, input_sources = load_dataset(input_path)
    stats, problems = inspect_dataset(train_lines, name=str(input_path))
    print_dataset_stats(stats)
    if input_path.is_dir():
        log(f"合并文件   : 目录下 {len(input_sources)} 个 *{DATA_SUFFIX} 文件")
        for source in input_sources:
            log(f"  - {source['path']} ({source['samples']} 条)")
    print_problems(problems)
    warn_degenerate_dataset(stats)
    is_multilabel = stats["multi_label_samples"] > 0

    test_path = Path(args.test).expanduser() if args.test else None
    test_stats = None
    if test_path is not None:
        test_lines = load_lines(test_path)
        test_stats, test_problems = inspect_dataset(test_lines, name=str(test_path))
        log("")
        log("[独立测试集]")
        if test_path.is_dir():
            files = list_data_files(test_path)
            log(f"合并文件   : 目录下 {len(files)} 个 *{DATA_SUFFIX} 文件")
            for file in files:
                log(f"  - {file.name} ({len(read_text_lines(file))} 条)", level=1)
        print_dataset_stats(test_stats)
        print_problems(test_problems, limit=3)
    section("3/6 数据划分")
    work_dir, is_temp_dir = prepare_work_dir(args)
    data_info = resolve_data(args, input_path, train_lines, work_dir)
    if is_temp_dir:
        log(f"中间目录   : {work_dir} (临时目录, 结束自动清理)")

    eval_path = test_path if test_path is not None else data_info["valid_eval_path"]
    if eval_path is None:
        log("[警告] 既没有 --test 也没有切分验证集, 将跳过评估(建议保留 --valid-ratio 或提供 --test)")

    if args.dry_run:
        log("\n--dry-run: 数据检查与切分完成, 未训练模型。")
        if is_temp_dir and not args.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
        return 0

    section("4/6 模型训练")
    training_started = time.time()
    model, params = train_model(fasttext, args, data_info["train_path"], data_info["valid_path"])
    if args.autotune:
        model, params = apply_autotune_bucket_cap(
            fasttext, args, model, params, data_info["train_path"]
        )
    training_seconds = time.time() - training_started
    log(f"训练耗时   : {fmt_seconds(training_seconds)}")
    quant_info = None
    verification = None
    summary_before_quantize = None
    if args.quantize:
        summary_before_quantize = model_summary(model)
        model, quant_info = apply_quantization(model, args, data_info["train_path"])
    summary = model_summary(model)
    print_model_summary(summary)

    section("5/6 保存模型")
    size_mb = save_model(model, output_path)
    if quant_info:
        ratio = quant_info["estimated_size_before_mb"] / size_mb if size_mb else 0.0
        log(
            f"量化体积   : 约 {quant_info['estimated_size_before_mb']:.2f} MB -> {size_mb:.2f} MB "
            f"(约 1/{ratio:.1f}, 端侧常驻内存近似等于文件体积)"
        )
        # 落盘后立刻重新载入一遍: 确认文件真的能加载、且确实是量化模型
        verification, _ = verify_quantized_export(
            fasttext, args, output_path, summary_before_quantize or summary
        )

    section("6/6 效果评估")
    eval_result = None
    if eval_path is not None:
        tuning_note = ""
        if args.autotune and data_info["valid_path"] is not None and eval_path == data_info["valid_eval_path"]:
            tuning_note = "该验证集同时用于 autotune 选择超参数, 指标偏乐观, 建议用 --test 提供独立测试集"
        eval_result = evaluate(model, eval_path, args.k, args.threshold)
        eval_result["path"] = str(eval_path)
        eval_result["is_test_set"] = test_path is not None and eval_path == test_path
        print_evaluation(
            eval_result,
            eval_path,
            is_multilabel,
            tuning_note,
            max_failures=args.max_failures,
            show_metrics=args.metrics,
        )
    else:
        log("已跳过")

    duration = time.time() - started
    model_info = {
        "path": str(output_path),
        "size_mb": round(size_mb, 4),
        "training_seconds": round(training_seconds, 3),
        "params": params,
        "summary": summary,
        "quantization": quant_info,
        "verification": verification,
    }
    if not args.no_report:
        report_path = Path(args.report).expanduser() if args.report else Path(str(output_path) + ".report.json")
        report = build_report(
            args, env, stats, test_stats, data_info, model_info, eval_result, duration, input_sources
        )
        write_report(report_path, report)

    if args.predict:
        section("预测(命令行文本)")
        print_predictions(model, args.predict, args.k, args.threshold)

    if args.predict_file:
        section("预测(文件)")
        count = predict_file(model, Path(args.predict_file).expanduser(), args.k, args.threshold)
        log(f"共预测 {count} 条, JSONL 结果已输出到 stdout")

    if args.interactive:
        section("交互式预测")
        interactive_loop(model, args.k, args.threshold)

    if is_temp_dir:
        if args.keep_work_dir:
            log(f"中间目录已保留: {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    section("完成")
    log(f"模型文件   : {output_path} ({size_mb:.2f} MB)")
    if eval_result is not None:
        log(
            f"评估结果   : {eval_result['judged_samples']} 条, 成功 {eval_result['correct']} "
            f"({eval_result['success_rate'] * 100:.2f}%), 失败 {eval_result['failed']} "
            f"({eval_result['failure_rate'] * 100:.2f}%)"
            f"  (基于 {eval_result['path']})"
        )
    log(f"总耗时     : {fmt_seconds(time.time() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
