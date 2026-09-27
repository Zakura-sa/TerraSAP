"""Dependency-free contract tests using extracted source and lightweight mocks.

These exercise selected functions, not full model imports or tensor computation.
"""
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch
import warnings

from utils.pretrained import pretrained_model_name
from utils.support import SupportSampler

ROOT = Path(__file__).resolve().parents[1]


def extract(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    node = next(n for n in ast.walk(tree) if getattr(n, 'name', None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class Array(list):
    """Only the indexing/comparison subset needed by get_dataset."""
    def __getitem__(self, key):
        if isinstance(key, list):
            return Array(list.__getitem__(self, index) for index in key)
        value = list.__getitem__(self, key)
        return Array(value) if isinstance(key, slice) else value

    def __eq__(self, label):
        return [value == label for value in self]

    def __ge__(self, label):
        return [value >= label for value in self]

    def __lt__(self, label):
        return [value < label for value in self]

    def tolist(self):
        return list(self)


def flatnonzero(values):
    return Array(i for i, value in enumerate(values) if value)


class CompatibilityTests(unittest.TestCase):
    def test_checkpoint_identifier_contract(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertEqual(pretrained_model_name({}), 'vit_base_patch16_224')
        self.assertIn('not verified paper weights', str(caught[0].message))
        for value in (None, '', '  ', 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pretrained_model_name({'pretrained_model_name': value})
        self.assertEqual(pretrained_model_name({'pretrained_model_name': ' custom.weight '}), 'custom.weight')

    def test_spatial_and_baseline_use_same_checkpoint(self):
        for path, function in [('backbone/asp_backbone.py', 'build_promptmodel'),
                               ('utils/inc_net.py', 'get_backbone')]:
            for config, expected in [({}, 'vit_base_patch16_224'),
                                     ({'pretrained_model_name': 'custom.weight'}, 'custom.weight'),
                                     ({'pretrained_model_name': None}, None)]:
                with self.subTest(function=function, config=config):
                    basic_model = Mock()
                    basic_model.state_dict.return_value = {'head.weight': 1, 'head.bias': 2}
                    timm = SimpleNamespace(create_model=Mock(return_value=basic_model))
                    namespace = {'timm': timm, 'pretrained_model_name': pretrained_model_name,
                                 'VPT_ViT': Mock(), 'torch': SimpleNamespace(nn=SimpleNamespace(Identity=Mock()))}
                    build = extract(path, function, namespace)
                    args = {'backbone_type': 'vit_base_patch16_224', **config}
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        if expected is None:
                            with self.assertRaises(ValueError):
                                build(args=args)
                            timm.create_model.assert_not_called()
                        else:
                            build(args=args)
                            self.assertEqual(timm.create_model.call_args.args[0], expected)
                            self.assertTrue(timm.create_model.call_args.kwargs['pretrained'])

    def test_inference_timer_syncs_only_the_active_cuda_device(self):
        measure = extract('models/asp.py', '_measure_single_image_inference_time', {'logging': Mock()})
        for device_type in ('cpu', 'cuda'):
            with self.subTest(device=device_type):
                device = SimpleNamespace(type=device_type)
                synchronize = Mock()
                torch = SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(synchronize=synchronize))
                numpy = SimpleNamespace(mean=lambda values: 0, median=lambda values: 0,
                                        percentile=lambda values, q: 0, std=lambda values: 0)
                learner = SimpleNamespace(_device=device, _network=Mock(return_value={'logits': None}))
                sample = MagicMock()
                with patch.dict('sys.modules', {'torch': torch, 'numpy': numpy}):
                    measure(learner, [(None, sample, None)])
                if device_type == 'cpu':
                    synchronize.assert_not_called()
                else:
                    self.assertEqual(synchronize.call_count, 201)
                    synchronize.assert_called_with(device)

    def test_all_remote_sensing_datasets_still_dispatch(self):
        names = {'nwpu': 'NWPU_RESISC45', 'ucmerced': 'UCMerced', 'mstar': 'MSTAR',
                 'siri_whu': 'SIRIWHU', 'so2sat': 'So2SatLCZ42',
                 'paviau': 'PaviaUniversity', 'grss2013': 'Houston2013'}
        namespace = {value: Mock(return_value=value) for value in names.values()}
        dispatch = extract('utils/data_manager.py', '_get_idata', namespace)
        for key, value in names.items():
            self.assertEqual(dispatch(key, {}), value)
        for config in (ROOT / 'exps/simplified_ablation').glob('*.json'):
            self.assertIn(json.loads(config.read_text(encoding='utf-8'))['dataset'], names)
        for script in ('prepare_paviau.py', 'prepare_grss2013.py', 'prepare_so2sat.py'):
            self.assertTrue((ROOT / 'scripts' / script).is_file())

    def test_imagefolder_root_and_mapping_validation(self):
        train = SimpleNamespace(classes=['a', 'b'], class_to_idx={'a': 0, 'b': 1})
        test = SimpleNamespace(classes=['a', 'b'], class_to_idx={'a': 0, 'b': 1})
        factory = Mock(side_effect=[train, test])
        load = extract('utils/data.py', '_load_scene_folders',
                       {'Path': Path, 'datasets': SimpleNamespace(ImageFolder=factory)})
        load({'data_root': 'custom'}, 'MSTAR', 2)
        self.assertEqual(Path(factory.call_args_list[0].args[0]), Path('custom/MSTAR/train'))
        factory.side_effect = [train, test]
        load({'mstar_data_root': 'direct'}, 'MSTAR', 2)
        self.assertEqual(Path(factory.call_args_list[-2].args[0]), Path('direct/train'))
        test.class_to_idx = {'a': 1, 'b': 0}
        factory.side_effect = [train, test]
        with self.assertRaisesRegex(ValueError, 'mappings differ'):
            load({}, 'MSTAR', 2)
        test.class_to_idx = train.class_to_idx
        factory.side_effect = [train, test]
        with self.assertRaisesRegex(ValueError, 'expected 10'):
            load({}, 'MSTAR', 10)

    def test_manager_reuses_supports_and_preserves_full_tests_and_appendent(self):
        numpy = SimpleNamespace(
            flatnonzero=flatnonzero, where=lambda values: (flatnonzero(values),),
            logical_and=lambda left, right: [a and b for a, b in zip(left, right)],
            concatenate=lambda arrays: Array(item for array in arrays for item in array),
        )
        transform = SimpleNamespace(Compose=lambda operations: operations,
                                    RandomHorizontalFlip=lambda **kwargs: 'flip')
        namespace = {'np': numpy, 'transforms': transform, 'random': random,
                     'DummyDataset': lambda images, labels, trsf, use_path:
                     SimpleNamespace(images=images, labels=labels)}
        manager_type = extract('utils/data_manager.py', 'DataManager', namespace)
        manager = manager_type.__new__(manager_type)
        manager.args = {'init_cls': 1}
        manager.support_sampler = SupportSampler(2025)
        manager._train_data = Array(f'train-{i}' for i in range(12))
        manager._test_data = Array(f'test-{i}' for i in range(12))
        manager._train_targets = Array([0] * 4 + [1] * 8)
        manager._test_targets = Array([0] * 4 + [1] * 8)
        manager._train_trsf = manager._test_trsf = manager._common_trsf = []
        manager.use_path = True
        first = manager.get_dataset([1], 'train', 'train', kshot=5)
        second = manager.get_dataset([1], 'train', 'test', kshot=5)
        self.assertEqual(list(first.images), list(second.images))
        self.assertEqual(len(first.images), 5)
        self.assertEqual(len(manager.get_dataset([1], 'test', 'test', kshot=5).images), 8)
        self.assertEqual(len(manager.get_dataset([0], 'train', 'train', kshot=1).images), 4)
        data, targets, dataset = manager.get_dataset([], 'train', 'test',
                                                     appendent=(Array(['extra']), Array([2])), ret_data=True)
        self.assertEqual(list(data), ['extra'])
        self.assertEqual(list(targets), [2])
        self.assertEqual(list(dataset.images), ['extra'])


if __name__ == '__main__':
    unittest.main()
