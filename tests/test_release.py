import ast
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_python_syntax(self):
        for path in ROOT.glob('*.py'):
            with self.subTest(file=path.name):
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_example_input_contract(self):
        for line in (ROOT / 'examples/tasks.sample.jsonl').read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            self.assertIsInstance(item['generated_prompt'], str)
            self.assertTrue(item['generated_prompt'].strip())

    def test_saved_config_does_not_persist_keys(self):
        tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'save_config')
        class UI:
            def toast(self, *args, **kwargs): pass
            def error(self, message): raise AssertionError(message)
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'config.json'
            scope = {'json': json, 'CONFIG_FILE': str(output), 'st': UI()}
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<save_config>', 'exec'), scope)
            scope['save_config']({'judge_api_key': 'test-only-value', 'target_api_key': 'test-only-value', 'concurrency': 4})
            saved = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(saved['judge_api_key'], '')
            self.assertEqual(saved['target_api_key'], '')
            self.assertEqual(saved['concurrency'], 4)

    def test_private_config_excluded(self):
        self.assertFalse((ROOT / 'config_presets.json').exists())
        self.assertFalse((ROOT / '.env').exists())

    def test_full_benchmark_contract(self):
        rows = [json.loads(line) for line in
                (ROOT / 'data/edu_eval.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip()]
        self.assertEqual(len(rows), 226)
        self.assertEqual(len({row['id'] for row in rows}), 226)
        self.assertEqual(len({row['generated_prompt'] for row in rows}), 226)
        self.assertEqual(Counter(row['original_data']['科目'] for row in rows),
                         {'数学': 78, '语文': 44, '英语': 52, '物理': 52})
        self.assertEqual(Counter(row['original_data']['学段1'] for row in rows),
                         {'小学': 78, '初中': 96, '高中': 52})


if __name__ == '__main__':
    unittest.main()
