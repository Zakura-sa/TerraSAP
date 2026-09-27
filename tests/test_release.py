"""Standard-library checks for experiment configs, splits, and Python syntax."""
import ast
import csv
import json
from pathlib import Path
import unittest
from utils.support import SupportSampler

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    'nwpu': ([25, 5, 5, 5, 5], 22455, 9045),
    'ucm': ([12, 3, 3, 3], 765, 1335),
    'mstar': ([4, 1, 1, 1, 1, 1, 1], 1093, 2426),
}


def read_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


class ReleaseTests(unittest.TestCase):
    def test_configs_follow_paper_training_schedule(self):
        expected = {
            'tuned_epoch': 40, 'fs_epoch': 0, 'init_lr': 0.007,
            'batch_size': 16, 'weight_decay': 0.0005, 'optimizer': 'sgd',
            'prompt_token_num': 6, 'TIP_init': 'zero', 'kshot': 5,
            'shuffle': False, 'use_layered_lr': False, 'kl_weight': 0.0,
            'paper_protocol': True,
        }
        for dataset, (sizes, _, _) in PROTOCOL.items():
            with self.subTest(dataset=dataset):
                config = read_json(ROOT / 'configs' / f'{dataset}.json')
                for key, value in expected.items():
                    self.assertEqual(config[key], value, key)
                self.assertEqual(config['init_cls'], sizes[0])
                self.assertEqual(config['increment'], sizes[1])

    def test_class_orders_and_file_inventories(self):
        for dataset, (sizes, train, test) in PROTOCOL.items():
            with self.subTest(dataset=dataset):
                info = read_json(ROOT / 'splits' / f'{dataset}.json')
                classes = info['class_order']
                self.assertEqual(len(classes), len(set(classes)))
                self.assertEqual([len(s) for s in info['sessions']], sizes)
                self.assertEqual([c for s in info['sessions'] for c in s], classes)
                self.assertEqual([info['imagefolder_classes'][i] for i in info['new_to_imagefolder']], classes)
                with (ROOT / 'splits' / f'{dataset}_files.csv').open(encoding='utf-8', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), train + test)
                self.assertEqual(len({r['path'] for r in rows}), len(rows))
                self.assertEqual(sum(r['split'] == 'train' for r in rows), train)
                self.assertEqual(sum(r['split'] == 'test' for r in rows), test)
                counts = {c: {'train': 0, 'test': 0} for c in classes}
                for row in rows:
                    name = row['class_name']
                    self.assertIn(name, classes)
                    self.assertFalse(Path(row['path']).is_absolute())
                    self.assertTrue(row['path'].startswith('data/'))
                    counts[name][row['split']] += 1
                self.assertEqual(counts, info['class_counts'])
                if dataset != 'nwpu':
                    for session in info['sessions'][1:]:
                        for name in session:
                            self.assertEqual(counts[name]['train'], 5)

    def test_support_is_reused_across_views(self):
        sampler = SupportSampler(2025)
        first = sampler.select(25, range(499), 5)
        second = sampler.select(25, range(499), 5)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)
        first.clear()
        self.assertEqual(sampler.select(25, range(499), 5), second)

    def test_support_seed_is_reproducible(self):
        one, two = SupportSampler(2025), SupportSampler(2025)
        for label in (25, 26, 27):
            self.assertEqual(one.select(label, range(499), 5), two.select(label, range(499), 5))

    def test_invalid_support_pool_is_rejected(self):
        sampler = SupportSampler(2025)
        with self.assertRaises(ValueError):
            sampler.select(25, range(4), 5)
        with self.assertRaises(ValueError):
            sampler.select(25, range(10), 0)
        sampler.select(25, range(10), 5)
        with self.assertRaises(ValueError):
            sampler.select(25, range(10, 20), 5)

    def test_python_syntax(self):
        files = [ROOT / 'main.py', ROOT / 'trainer.py']
        for directory in ('backbone', 'models', 'utils', 'tests', 'scripts'):
            files.extend((ROOT / directory).glob('*.py'))
        for path in files:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.name)


if __name__ == '__main__':
    unittest.main()
