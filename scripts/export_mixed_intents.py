#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将跨意图 JSONL 标注导出为 fastText 多标签文本，仅依赖 Python 标准库。

默认读取 data/annotations/mixed_intents.jsonl，生成 data/labeling/mixed_intents_data.txt。
每条原句只导出一行，标签来自全部 intents；routing_label 仅做格式校验。
所有记录校验成功后才原子替换输出，不修改源文件。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from pathlib import Path


DEFAULT_INPUT = "data/annotations/mixed_intents.jsonl"
DEFAULT_OUTPUT = "data/labeling/mixed_intents_data.txt"
INTENT_ORDER = ("accounting", "reminder", "todo")
LABEL_PREFIX = "__label__"
ALLOWED_FIELDS = {"text", "intents", "routing_label"}


class ConversionError(ValueError):
    """输入标注或输出路径不符合转换要求。"""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConversionError(f"JSON 字段重复: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ConversionError(f"不是有效的 JSON 常量: {value}")


def normalize_text(text: str) -> str:
    """仅用于查重，保留小数点、正负号等标点；不改变导出的原句。"""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def validate_record(record):
    if not isinstance(record, dict):
        raise ConversionError("每行必须是一个 JSON 对象")
    missing = {"text", "intents"} - record.keys()
    if missing:
        raise ConversionError(f"缺少字段: {', '.join(sorted(missing))}")
    extra = record.keys() - ALLOWED_FIELDS
    if extra:
        raise ConversionError(f"未知字段: {', '.join(sorted(extra))}")

    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        raise ConversionError("text 必须是非空字符串")
    if any(
        (unicodedata.category(char) == "Cc" and char != "\t")
        or char in "\u2028\u2029"
        for char in text
    ):
        raise ConversionError("text 不能包含换行或控制字符（允许制表符）")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ConversionError("text 包含无法编码为 UTF-8 的字符") from exc
    if any(token.startswith(LABEL_PREFIX) for token in text.split()):
        raise ConversionError("text 不能包含以 __label__ 开头的标签 token")
    if "</s>" in text.split():
        raise ConversionError("text 不能包含 fastText 句尾 token </s>")

    intents = record["intents"]
    if not isinstance(intents, list) or len(intents) < 2:
        raise ConversionError("intents 必须是至少包含两个不同意图的数组")
    if any(not isinstance(label, str) for label in intents):
        raise ConversionError("intents 中的每个标签必须是字符串")
    unsupported = set(intents) - set(INTENT_ORDER)
    if unsupported:
        raise ConversionError(f"不支持的意图标签: {', '.join(sorted(unsupported))}")
    if len(intents) != len(set(intents)):
        raise ConversionError("intents 中存在重复标签")

    route = record.get("routing_label")
    if route is not None and (not isinstance(route, str) or route not in intents):
        raise ConversionError("routing_label 必须是 intents 中的一个标签或 null")
    ordered = tuple(label for label in INTENT_ORDER if label in intents)
    return text, ordered


def load_output_lines(source: Path):
    output_lines = []
    seen = {}
    with source.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            try:
                if not raw_line.strip():
                    raise ConversionError("不允许空行，每行必须是一个 JSON 对象")
                record = json.loads(
                    raw_line,
                    object_pairs_hook=unique_object,
                    parse_constant=reject_constant,
                )
                text, intents = validate_record(record)
                key = normalize_text(text)
                if key in seen:
                    previous_line, previous_intents = seen[key]
                    detail = "，且意图标注冲突" if intents != previous_intents else ""
                    raise ConversionError(f"文本与第 {previous_line} 行重复{detail}")
                seen[key] = (line_number, intents)
                labels = " ".join(LABEL_PREFIX + label for label in intents)
                output_lines.append(f"{labels} {text}")
            except (ConversionError, json.JSONDecodeError) as exc:
                raise ConversionError(f"{source}: 第 {line_number} 行: {exc}") from exc
    if not output_lines:
        raise ConversionError(f"{source}: 输入文件为空，没有可导出的样本")
    return output_lines


def ensure_distinct_paths(source: Path, output: Path) -> None:
    same_path = source.resolve() == output.resolve()
    same_file = source.exists() and output.exists() and source.samefile(output)
    if same_path or same_file:
        raise ConversionError("输入和输出不能是同一个文件（包括软链接或硬链接）")


def write_atomic(output: Path, payload: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp",
            dir=output.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="源 JSONL 路径")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="输出 fastText 文本路径")
    args = parser.parse_args(argv)
    source = Path(args.input).expanduser()
    output = Path(args.output).expanduser()
    try:
        ensure_distinct_paths(source, output)
        lines = load_output_lines(source)
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        write_atomic(output, payload)
    except (ConversionError, OSError, UnicodeError, RuntimeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    print(f"已导出 {len(lines)} 条多标签样本: {source} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
