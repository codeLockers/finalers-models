"""Black-box checks for exporting mixed intents without losing labels or data."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_mixed_intents.py"


class ExportMixedIntentsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mixed-intents-test-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.source = self.work / "input.jsonl"
        self.output = self.work / "output.txt"

    def write_records(self, records, path=None):
        path = self.source if path is None else path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, arguments)],
            cwd=self.work,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )

    def export(self, output=None):
        return self.run_cli(
            "--input", self.source,
            "--output", self.output if output is None else output,
        )

    def assert_source_line(self, result, line):
        self.assertRegex(
            result.stderr,
            rf"(?:第\s*{line}\s*行|line\s*{line}\b|:\s*{line}\b)",
            msg=result.stderr,
        )

    def test_exports_all_intents_in_canonical_order_and_preserves_sentences(self):
        records = [
            {
                "text": "记录午饭30元。明天上午九点提醒我开会！",
                "intents": ["reminder", "accounting"],
                "routing_label": "accounting",
            },
            {
                "text": "把买牛奶加入待办；今晚八点提醒我取快递。",
                "intents": ["todo", "reminder"],
                "routing_label": "reminder",
            },
            {
                "text": "记早餐12元，把买书加入任务，晚上提醒我关窗。",
                "intents": ["todo", "accounting", "reminder"],
                "routing_label": "accounting",
            },
        ]
        self.write_records(records)
        original = self.source.read_bytes()

        result = self.export()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.output.read_text(encoding="utf-8"),
            "__label__accounting __label__reminder " + records[0]["text"] + "\n"
            "__label__reminder __label__todo " + records[1]["text"] + "\n"
            "__label__accounting __label__reminder __label__todo "
            + records[2]["text"] + "\n",
        )
        self.assertEqual(self.source.read_bytes(), original)

    def test_accepts_null_or_absent_routing_label_without_dropping_intents(self):
        self.write_records([
            {
                "text": "记午饭32元，再把买纸巾加入待办。",
                "intents": ["todo", "accounting"],
                "routing_label": None,
            },
            {
                "text": "明天提醒我复诊，再添加一个整理书架的任务。",
                "intents": ["reminder", "todo"],
            },
        ])

        result = self.export()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.output.read_text(encoding="utf-8").splitlines(),
            [
                "__label__accounting __label__todo 记午饭32元，再把买纸巾加入待办。",
                "__label__reminder __label__todo 明天提醒我复诊，再添加一个整理书架的任务。",
            ],
        )

    def test_default_paths_work_from_the_current_directory(self):
        source = self.work / "data" / "annotations" / "mixed_intents.jsonl"
        self.write_records([
            {"text": "记停车15元，明早提醒我带伞。", "intents": ["accounting", "reminder"]},
        ], path=source)

        result = self.run_cli()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.work / "data" / "labeling" / "mixed_intents_data.txt").read_text(encoding="utf-8"),
            "__label__accounting __label__reminder 记停车15元，明早提醒我带伞。\n",
        )
        self.assertFalse(source.with_name("mixed_intents_data.txt").exists())

    def test_creates_missing_output_parents_and_supports_short_options(self):
        self.write_records([
            {"text": "添加买书待办，记晚饭25元。", "intents": ["todo", "accounting"]},
        ])
        original = self.source.read_bytes()
        nested = self.work / "new" / "nested" / "export.txt"

        result = self.run_cli("-i", self.source, "-o", nested)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            nested.read_text(encoding="utf-8"),
            "__label__accounting __label__todo 添加买书待办，记晚饭25元。\n",
        )
        self.assertEqual(self.source.read_bytes(), original)

    def test_rejects_invalid_records_with_line_number_and_reason(self):
        good = {"text": "记咖啡18元，提醒我明天复诊。", "intents": ["accounting", "reminder"]}
        base = {"text": "添加整理书架任务，晚上提醒我取快递。", "intents": ["todo", "reminder"]}
        cases = [
            ("malformed JSON", "{broken", r"JSON|json"),
            ("empty record", "", r"空|blank|empty|JSON"),
            (
                "duplicate JSON keys",
                '{"text":"first","text":"second","intents":["todo","reminder"]}',
                r"重复|duplicate|键|key",
            ),
            (
                "nonstandard JSON NaN",
                '{"text":"two actions","intents":["todo","reminder"],"extra":NaN}',
                r"NaN|JSON|json",
            ),
            ("nonobject record", [], r"对象|object|字典|dict"),
            ("missing text", {"intents": base["intents"]}, r"text|文本"),
            ("null text", {**base, "text": None}, r"text|文本"),
            ("nonstr text", {**base, "text": 123}, r"text|文本"),
            ("blank text", {**base, "text": "  \t "}, r"text|文本|空"),
            ("linefeed in text", {**base, "text": "记午饭30元\n提醒我开会"}, r"text|文本|行|newline"),
            ("carriage return in text", {**base, "text": "记午饭30元\r提醒我开会"}, r"text|文本|行|newline"),
            ("Unicode line separator", {**base, "text": "记午饭30元\u2028提醒我开会"}, r"text|文本|行|字符|newline"),
            ("Unicode paragraph separator", {**base, "text": "记午饭30元\u2029提醒我开会"}, r"text|文本|行|字符|newline"),
            ("NUL in text", {**base, "text": "记午饭30元\x00提醒我开会"}, r"text|文本|字符|NUL"),
            ("lone surrogate", {**base, "text": "记午饭30元\ud800提醒我开会"}, r"text|文本|字符|surrogate|UTF"),
            ("embedded label", {**base, "text": "加任务 __label__unknown 明天开会"}, r"label|标签|保留"),
            ("missing intents", {"text": base["text"]}, r"intents|意图|标签"),
            ("null intents", {**base, "intents": None}, r"intents|意图|标签"),
            ("string intents", {**base, "intents": "todo,reminder"}, r"intents|意图|标签"),
            ("empty intents", {**base, "intents": []}, r"intents|意图|标签"),
            ("single intent", {**base, "intents": ["todo"]}, r"intents|意图|标签"),
            ("nonstr intent", {**base, "intents": ["todo", 12]}, r"intents|意图|标签"),
            ("unsupported intent", {**base, "intents": ["todo", "unknown"]}, r"intents|意图|标签|unknown"),
            ("duplicate intent", {**base, "intents": ["todo", "reminder", "todo"]}, r"重复|duplicate"),
        ]
        first_line = json.dumps(good, ensure_ascii=False) + "\n"
        for name, invalid, reason in cases:
            with self.subTest(name=name):
                second_line = invalid if isinstance(invalid, str) else json.dumps(invalid, ensure_ascii=True)
                self.source.write_text(first_line + second_line + "\n", encoding="utf-8")
                original = self.source.read_bytes()
                self.output.unlink(missing_ok=True)

                result = self.export()

                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assert_source_line(result, 2)
                self.assertRegex(result.stderr, reason)
                self.assertFalse(self.output.exists(), result.stderr)
                self.assertEqual(self.source.read_bytes(), original)

    def test_rejects_normalized_duplicates_including_conflicting_intents(self):
        first = {
            "text": "记午饭30元，明天 提醒我开会。",
            "intents": ["accounting", "reminder"],
        }
        cases = [
            ("exact", {**first}),
            ("repeated whitespace", {**first, "text": "  记午饭30元，明天 \t  提醒我开会。  "}),
            ("fullwidth digits", {**first, "text": "记午饭３０元，明天 提醒我开会。"}),
            ("conflicting intents", {**first, "intents": ["reminder", "todo"]}),
        ]
        for name, second in cases:
            with self.subTest(name=name):
                self.write_records([first, second])
                original = self.source.read_bytes()
                self.output.write_text("previous verified export\n", encoding="utf-8")

                result = self.export()

                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assert_source_line(result, 2)
                self.assertRegex(result.stderr, r"重复|duplicate|冲突|conflict")
                self.assertEqual(self.output.read_text(encoding="utf-8"), "previous verified export\n")
                self.assertEqual(self.source.read_bytes(), original)

    def test_duplicate_detection_preserves_meaningful_punctuation(self):
        records = [
            {"text": "记咖啡3.5元，明天提醒我喝水。", "intents": ["accounting", "reminder"]},
            {"text": "记咖啡35元，明天提醒我喝水。", "intents": ["accounting", "reminder"]},
            {"text": "把检查A-1设备加入待办，今晚提醒我带工具。", "intents": ["todo", "reminder"]},
            {"text": "把检查A1设备加入待办，今晚提醒我带工具。", "intents": ["todo", "reminder"]},
        ]
        self.write_records(records)

        result = self.export()

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = self.output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 4)
        for line, record in zip(lines, records):
            self.assertTrue(line.endswith(" " + record["text"]), line)

    def test_late_validation_error_preserves_existing_export_and_source(self):
        self.write_records([
            {"text": "记午饭20元，提醒我带钥匙。", "intents": ["accounting", "reminder"]},
            {"text": "添加修椅子任务，记录早餐9元。", "intents": ["todo", "accounting"]},
            {"text": "提醒我取书，把写论文加到待办。", "intents": ["reminder", "todo"]},
        ])
        with self.source.open("a", encoding="utf-8") as stream:
            stream.write("{unfinished JSON\n")
        original = self.source.read_bytes()
        previous = b"existing export\nwith multiple lines\n"
        self.output.write_bytes(previous)

        result = self.export()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assert_source_line(result, 4)
        self.assertEqual(self.output.read_bytes(), previous)
        self.assertEqual(self.source.read_bytes(), original)

    def test_rejects_same_file_via_identical_relative_symlink_and_hardlink_paths(self):
        for alias in ("identical", "relative", "symlink", "hardlink"):
            with self.subTest(alias=alias):
                source = self.work / (alias + ".jsonl")
                self.write_records([
                    {"text": "记晚饭28元，明早提醒我带电脑。", "intents": ["accounting", "reminder"]},
                ], path=source)
                if alias == "identical":
                    output = source
                elif alias == "relative":
                    output = "./" + source.name
                else:
                    output = self.work / (alias + "-alias.txt")
                    if alias == "symlink":
                        output.symlink_to(source)
                    else:
                        os.link(source, output)
                original = source.read_bytes()

                result = self.run_cli("-i", source, "-o", output)

                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertRegex(result.stderr, r"相同|同一|same|覆盖|alias")
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual((self.work / output).read_bytes(), original)

    def test_rejects_fasttext_sentence_boundary_without_overwriting_output(self):
        self.write_records([
            {"text": "记早餐12元，再加一个取书待办。", "intents": ["accounting", "todo"]},
            {"text": "记午饭30元 </s> 提醒我明天开会", "intents": ["accounting", "reminder"]},
        ])
        original = self.source.read_bytes()
        previous = b"keep existing verified export\n"
        self.output.write_bytes(previous)

        result = self.export()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assert_source_line(result, 2)
        self.assertIn("</s>", result.stderr)
        self.assertEqual(self.output.read_bytes(), previous)
        self.assertEqual(self.source.read_bytes(), original)

    def test_rejects_missing_or_empty_input_without_overwriting_output(self):
        for name, content in (("missing", None), ("empty", ""), ("blank", " \n\t\n")):
            with self.subTest(name=name):
                self.source.unlink(missing_ok=True)
                if content is not None:
                    self.source.write_text(content, encoding="utf-8")
                previous = b"keep this export\n"
                self.output.write_bytes(previous)

                result = self.export()

                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertTrue(result.stderr.strip())
                self.assertEqual(self.output.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
