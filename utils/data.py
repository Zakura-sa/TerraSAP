import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import torch
import torch.nn.functional as F
import random
import math
from PIL import Image


class MixUp:
    """MixUp数据增强"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = batch.size(0)
        index = torch.randperm(batch_size)

        mixed_batch = lam * batch + (1 - lam) * batch[index, :]
        return mixed_batch, index, lam


class CutMix:
    """CutMix数据增强"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = batch.size(0)
        index = torch.randperm(batch_size)

        # 计算裁剪区域
        W, H = batch.size(2), batch.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 随机选择裁剪位置
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 应用CutMix
        batch[:, :, bbx1:bbx2, bby1:bby2] = batch[index, :, bbx1:bbx2, bby1:bby2]

        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        return batch, index, lam


class RemoteSensingAugmentation:
    """
    遥感图像专用数据增强策略

    特点：
    1. 全角度旋转不变性
    2. 多尺度变换
    3. 光谱增强
    4. 任务自适应强度
    """

    def __init__(self, task_id=0, difficulty_level='medium', performance_mode='balanced'):
        """
        初始化遥感数据增强

        Args:
            task_id: 任务ID，用于自适应调整
            difficulty_level: 难度级别 ('easy', 'medium', 'hard')
            performance_mode: 性能模式 ('fast', 'balanced', 'quality')
        """
        self.task_id = task_id
        self.difficulty_level = difficulty_level
        self.performance_mode = performance_mode

        # 根据任务难度调整增强强度
        self.strength_multiplier = self._get_strength_multiplier()

        # MixUp和CutMix增强器
        self.mixup = MixUp(alpha=0.2 * self.strength_multiplier)
        self.cutmix = CutMix(alpha=1.0 * self.strength_multiplier)

    def _get_strength_multiplier(self):
        """根据任务难度获取增强强度倍数"""
        base_strength = {
            'easy': 0.7,
            'medium': 1.0,
            'hard': 1.3
        }

        # 随着任务进行，逐渐增加增强强度
        task_factor = 1.0 + (self.task_id * 0.1)
        return base_strength.get(self.difficulty_level, 1.0) * task_factor

    def get_train_transforms(self):
        """获取训练时的变换序列（支持性能模式）"""
        # 核心几何变换（所有模式都包含）
        core_transforms = [
            # 多尺度裁剪（保持原有功能）
            transforms.RandomResizedCrop(
                224,
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            # 水平翻转
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        # 遥感特有增强（根据性能模式调整）
        rs_specific_transforms = []

        # 根据性能模式决定是否包含旋转变换
        if self.performance_mode == 'fast':
            # 快速模式：跳过旋转变换，只保留基础增强
            pass
        else:
            # balanced和quality模式：包含旋转变换
            if self.performance_mode == 'balanced':
                # 平衡模式：限制旋转角度
                if self.strength_multiplier <= 0.8:  # easy难度
                    rotation_degrees = (-30, 30)
                elif self.strength_multiplier <= 1.1:  # medium难度
                    rotation_degrees = (-45, 45)
                else:  # hard难度
                    rotation_degrees = (-90, 90)
            else:  # quality模式
                # 质量模式：全角度旋转
                if self.strength_multiplier <= 0.8:  # easy难度
                    rotation_degrees = (-45, 45)
                elif self.strength_multiplier <= 1.1:  # medium难度
                    rotation_degrees = (-90, 90)
                else:  # hard难度
                    rotation_degrees = (-180, 180)

            rs_specific_transforms.append(
                transforms.RandomRotation(
                    degrees=rotation_degrees,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0
                )
            )

        # 垂直翻转（遥感图像特有）- 所有模式都包含
        vflip_prob = min(0.2 + (self.strength_multiplier - 0.7) * 0.2, 0.4)
        rs_specific_transforms.append(
            transforms.RandomVerticalFlip(p=max(0.1, vflip_prob))
        )

        # 颜色增强 - 根据性能模式调整强度
        if self.performance_mode == 'fast':
            color_strength = min(self.strength_multiplier * 0.1, 0.15)
        else:
            color_strength = min(self.strength_multiplier * 0.15, 0.2)

        rs_specific_transforms.append(
            transforms.ColorJitter(
                brightness=color_strength,
                contrast=color_strength,
                saturation=color_strength * 0.7,
                hue=color_strength * 0.3
            )
        )

        # 高级增强 - 仅quality模式和medium及以上难度
        if self.performance_mode == 'quality' and self.strength_multiplier >= 1.0:
            # 随机灰度化（低概率）
            rs_specific_transforms.append(
                transforms.RandomGrayscale(p=0.05 * self.strength_multiplier)
            )

        # 组合变换
        all_transforms = core_transforms + rs_specific_transforms

        return all_transforms

    def get_test_transforms(self):
        """获取测试时的变换序列"""
        return [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]

    def apply_mixup_cutmix(self, batch, labels, use_mixup=True, use_cutmix=True):
        """
        应用MixUp和CutMix增强

        Args:
            batch: 输入批次
            labels: 标签
            use_mixup: 是否使用MixUp
            use_cutmix: 是否使用CutMix

        Returns:
            augmented_batch, mixed_labels, lambda_value
        """
        # 随机选择增强方式
        augmentation_choice = random.random()

        if use_mixup and use_cutmix:
            if augmentation_choice < 0.5:
                # 使用MixUp
                mixed_batch, index, lam = self.mixup(batch)
                return mixed_batch, (labels, labels[index], lam), 'mixup'
            else:
                # 使用CutMix
                mixed_batch, index, lam = self.cutmix(batch)
                return mixed_batch, (labels, labels[index], lam), 'cutmix'
        elif use_mixup:
            mixed_batch, index, lam = self.mixup(batch)
            return mixed_batch, (labels, labels[index], lam), 'mixup'
        elif use_cutmix:
            mixed_batch, index, lam = self.cutmix(batch)
            return mixed_batch, (labels, labels[index], lam), 'cutmix'
        else:
            return batch, labels, 'none'


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

def build_transform_coda_prompt(is_train, args):
    if is_train:        
        transform = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
        return transform

    t = []
    if args["dataset"].startswith("imagenet"):
        t = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]
    else:
        t = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0)),
        ]

    return t

def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t

class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True

        if args["model_name"] == "coda_prompt":
            self.train_trsf = build_transform_coda_prompt(True, args)
            self.test_trsf = build_transform_coda_prompt(False, args)
        else:
            self.train_trsf = build_transform(True, args)
            self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            # transforms.ToTensor(),
        ]

        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-r/train/"
        test_dir = "./data/imagenet-r/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/imagenet-a/train/"
        test_dir = "./data/imagenet-a/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class CUB(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/cub/train/"
        test_dir = "./data/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/objectnet/train/"
        test_dir = "./data/objectnet/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True
    
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(300).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/omnibenchmark/train/"
        test_dir = "./data/omnibenchmark/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class vtab(iData):
    use_path = True

    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = [    ]

    class_order = np.arange(50).tolist()

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "./data/vtab-cil/vtab/train/"
        test_dir = "./data/vtab-cil/vtab/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        print(train_dset.class_to_idx)
        print(test_dset.class_to_idx)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class NWPU_RESISC45(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.current_task = 0  # 跟踪当前任务

        # 初始化遥感数据增强器
        self.rs_augmentation = None
        self._init_rs_augmentation()

        # 初始化变换（默认为Task 0）
        self._update_transforms()
        self.class_order = np.arange(45).tolist()

    def _init_rs_augmentation(self):
        """初始化遥感数据增强器"""
        if self.args.get('enable_rs_augmentation', False):  # 使用统一的参数名
            difficulty_level = self.args.get('rs_augmentation_difficulty', 'medium')
            performance_mode = self.args.get('rs_augmentation_performance_mode', 'balanced')
            self.rs_augmentation = RemoteSensingAugmentation(
                task_id=self.current_task,
                difficulty_level=difficulty_level,
                performance_mode=performance_mode
            )
            print(f"[NWPU_RESISC45] Initialized RemoteSensingAugmentation with difficulty: {difficulty_level}, performance: {performance_mode}")
        else:
            print("[NWPU_RESISC45] Using basic augmentation (RemoteSensingAugmentation disabled)")

    def _update_transforms(self):
        """更新数据变换"""
        if self.rs_augmentation is not None:
            # 使用遥感专用增强策略
            self.train_trsf = self.rs_augmentation.get_train_transforms()
            self.test_trsf = self.rs_augmentation.get_test_transforms()
            print(f"[NWPU_RESISC45] Using RemoteSensingAugmentation for task {self.current_task}")
        else:
            # 使用基础变换（向后兼容）
            self.train_trsf = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
            self.test_trsf = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
            print(f"[NWPU_RESISC45] Using basic augmentation for task {self.current_task}")

        # 通用变换保持不变
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    def set_current_task(self, task_id):
        """设置当前任务ID并更新变换"""
        if self.current_task != task_id:
            self.current_task = task_id

            # 更新遥感增强器的任务ID
            if self.rs_augmentation is not None:
                self.rs_augmentation.task_id = task_id
                # 重新计算增强强度
                self.rs_augmentation.strength_multiplier = self.rs_augmentation._get_strength_multiplier()

            self._update_transforms()

            rs_status = "enabled" if self.rs_augmentation is not None else "disabled"
            print(f"[NWPU_RESISC45] Updated transforms for task {task_id}, RemoteSensingAugmentation: {rs_status}")

    def get_mixup_cutmix_transforms(self):
        """获取MixUp/CutMix增强器（用于训练循环中）"""
        if self.rs_augmentation is not None:
            return self.rs_augmentation
        return None

    def download_data(self):
        train_dir = "./data/NWPU-RESISC45/train/"
        test_dir = "./data/NWPU-RESISC45/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class UCMerced(iData):
    """UCMerced LandUse ImageFolder wrapper for FSCIL experiments."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.class_order = np.arange(21).tolist()

    def download_data(self):
        train_dir = "./data/UCMerced_LandUse_processed/train/"
        test_dir = "./data/UCMerced_LandUse_processed/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class SIRIWHU(iData):
    """SIRI-WHU ImageFolder wrapper for auxiliary ablations."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.class_order = np.arange(12).tolist()

    def download_data(self):
        train_dir = "./data/SIRI-WHU/train/"
        test_dir = "./data/SIRI-WHU/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class So2SatLCZ42(iData):
    """So2Sat-LCZ42 PCA/RGB ImageFolder wrapper for auxiliary experiments."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.class_order = np.arange(17).tolist()

    def download_data(self):
        train_dir = "./data/So2Sat-LCZ42_ImageFolder/train/"
        test_dir = "./data/So2Sat-LCZ42_ImageFolder/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class PaviaUniversity(iData):
    """Pavia University PCA-patch ImageFolder wrapper."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        # Base5 + four one-class incremental sessions.
        self.class_order = [1, 2, 3, 4, 5, 0, 6, 7, 8]

    def download_data(self):
        train_dir = "./data/PaviaU_ImageFolder/train/"
        test_dir = "./data/PaviaU_ImageFolder/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class Houston2013(iData):
    """Houston 2013 GRSS Data Fusion PCA-patch ImageFolder wrapper."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        # Base7 + four two-class incremental sessions.
        self.class_order = [0, 1, 2, 3, 4, 5, 14, 6, 7, 8, 9, 10, 11, 12, 13]

    def download_data(self):
        train_dir = "./data/GRSS2013_ImageFolder/train/"
        test_dir = "./data/GRSS2013_ImageFolder/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class MSTAR(iData):
    """
    MSTAR SAR图像数据集 - 完全符合FSCIL标准配置

    符合论文描述的设置：
    - 训练集使用17°俯仰角图像（高分辨率合成孔径雷达，X波段，0.3m×0.3m分辨率，HH极化）
    - 测试集使用15°俯仰角图像
    - 包含10种地面移动目标，跨越不同方位角观察
    - 数据来源：DARPA与空军研究实验室合作的移动和静止目标获取识别项目

    Training set (17°)		Testing set (15°)
    Class	Name	Serial	Depression	Number	Depression	Number
    1	BTR70	c71	17°	233	15°	196
    2	2S1	b01	17°	299	15°	274
    3	BRDM2	E-71	17°	298	15°	274
    4	BMP2	9563	17°	233	15°	196
    5	ZIL131	E12	17°	5	15°	274
    6	T62	A51	17°	5	15°	273
    7	D7	92v13015	17°	5	15°	274
    8	BTR60	k10yt7532	17°	5	15°	195
    9	T72	132	17°	5	15°	196
    10	ZSU234	d08	17°	5	15°	274

    完全符合论文要求：17°elevation for training, 15°elevation for testing
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.current_task = 0

        print("[MSTAR] 使用新的MSTAR数据集配置")
        print("   T0: 4个类别 (BTR70, 2S1, BRDM2, BMP2) - 大量训练样本")
        print("   T1-T6: 每个任务1个类别 (ZIL131, T62, D7, BTR60, T72, ZSU234) - 5-shot")

        # 初始化SAR图像专用增强器
        self.sar_augmentation = None
        self._init_sar_augmentation()

        # 初始化变换
        self._update_transforms()

        # 标准FSCIL类别顺序：T0(基础类) + T1-T6(新类)
        # 按照表格要求的顺序：BTR70, 2S1, BRDM2, BMP2, ZIL131, T62, D7, BTR60, T72, ZSU234
        self.standard_class_order = ['BTR70', '2S1', 'BRDM2', 'BMP2', 'ZIL131', 'T62', 'D7', 'BTR60', 'T72', 'ZSU234']

    def _init_sar_augmentation(self):
        """初始化SAR图像专用增强器"""
        if self.args.get('enable_sar_augmentation', False):
            # SAR图像需要特殊的增强策略
            print("[MSTAR] SAR专用增强已启用")
            self.sar_augmentation = True
        else:
            print("[MSTAR] 使用基础增强策略")
            self.sar_augmentation = False

    def _update_transforms(self):
        """更新数据变换 - 针对SAR图像优化"""
        if self.sar_augmentation:
            # SAR图像专用增强策略
            self.train_trsf = [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomRotation(degrees=5),  # 小角度旋转，保持SAR散射特性
                # 避免水平翻转，会改变SAR散射模式
            ]
            print(f"[MSTAR] 使用SAR专用增强策略 (任务 {self.current_task})")
        else:
            # 基础变换策略
            self.train_trsf = [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.3),  # 降低翻转概率
            ]
            print(f"[MSTAR] 使用基础增强策略 (任务 {self.current_task})")

        # 测试变换保持一致
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]

        # SAR图像归一化 - 使用ImageNet预训练参数但适配灰度特性
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    def set_current_task(self, task_id):
        """设置当前任务ID并更新变换"""
        if self.current_task != task_id:
            self.current_task = task_id
            self._update_transforms()

            sar_status = "enabled" if self.sar_augmentation else "disabled"
            print(f"[MSTAR] 更新任务 {task_id} 的变换策略, SAR增强: {sar_status}")

    def download_data(self):
        """加载MSTAR数据集"""
        print("[MSTAR] 加载标准数据集")

        train_dir = "./data/MSTAR/train/"
        test_dir = "./data/MSTAR/test/"

        print(f"   训练目录: {train_dir}")
        print(f"   测试目录: {test_dir}")

        # 使用ImageFolder加载数据
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        # 获取原始数据和标签
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        print(f"[MSTAR] 数据加载完成:")
        print(f"   训练样本: {len(self.train_data)}")
        print(f"   测试样本: {len(self.test_data)}")
        print(f"   类别数量: {len(set(self.train_targets))}")

        # 显示ImageFolder的类别映射
        print(f"[MSTAR] 原始类别映射: {train_dset.class_to_idx}")

        # 重新排序以符合FSCIL标准
        self._reorder_classes_for_fscil(train_dset.class_to_idx)

    def _reorder_classes_for_fscil(self, original_class_to_idx):
        """重新排序类别以符合FSCIL标准配置"""

        # 创建新的类别顺序映射
        new_class_order = []
        for std_class in self.standard_class_order:
            if std_class in original_class_to_idx:
                new_class_order.append(original_class_to_idx[std_class])
            else:
                print(f"[MSTAR] ⚠️  警告: 未找到标准类别 {std_class}")

        print(f"[MSTAR] 标准类别顺序: {new_class_order}")

        # 创建标签映射：原始标签 -> 新标签
        old_to_new_label = {}
        for new_idx, old_idx in enumerate(new_class_order):
            old_to_new_label[old_idx] = new_idx

        # 重新映射训练和测试标签
        self.train_targets = [old_to_new_label[label] for label in self.train_targets]
        self.test_targets = [old_to_new_label[label] for label in self.test_targets]

        # 更新类别顺序
        self.class_order = list(range(len(self.standard_class_order)))

        print(f"[MSTAR] 任务分配:")
        print(f"   T0 (init): 类别 0-3 (BTR70, 2S1, BRDM2, BMP2)")
        print(f"   T1: 类别 4 (ZIL131)")
        print(f"   T2: 类别 5 (T62)")
        print(f"   T3: 类别 6 (D7)")
        print(f"   T4: 类别 7 (BTR60)")
        print(f"   T5: 类别 8 (T72)")
        print(f"   T6: 类别 9 (ZSU234)")

        # 验证样本数量分布
        self._validate_sample_distribution()

    def _validate_sample_distribution(self):
        """验证样本数量是否符合要求"""
        from collections import Counter

        train_counts = Counter(self.train_targets)
        test_counts = Counter(self.test_targets)

        print(f"\n[MSTAR] 📊 样本分布验证:")
        print(f"{'类别':<10} {'类名':<10} {'训练样本':<10} {'测试样本':<10} {'类型':<15}")
        print("-" * 65)

        expected_config = {
            0: {'name': 'BTR70', 'train': 233, 'test': 196, 'type': '基础类'},
            1: {'name': '2S1', 'train': 299, 'test': 274, 'type': '基础类'},
            2: {'name': 'BRDM2', 'train': 298, 'test': 274, 'type': '基础类'},
            3: {'name': 'BMP2', 'train': 233, 'test': 196, 'type': '基础类'},
            4: {'name': 'ZIL131', 'train': 5, 'test': 274, 'type': '新类(5-shot)'},
            5: {'name': 'T62', 'train': 5, 'test': 273, 'type': '新类(5-shot)'},
            6: {'name': 'D7', 'train': 5, 'test': 274, 'type': '新类(5-shot)'},
            7: {'name': 'BTR60', 'train': 5, 'test': 195, 'type': '新类(5-shot)'},
            8: {'name': 'T72', 'train': 5, 'test': 196, 'type': '新类(5-shot)'},
            9: {'name': 'ZSU234', 'train': 5, 'test': 274, 'type': '新类(5-shot)'}
        }

        all_correct = True
        for class_id in range(10):
            actual_train = train_counts.get(class_id, 0)
            actual_test = test_counts.get(class_id, 0)
            expected = expected_config[class_id]

            status = "✓" if (actual_train == expected['train'] and actual_test == expected['test']) else "✗"
            if status == "✗":
                all_correct = False

            print(f"{class_id:<10} {expected['name']:<10} {actual_train:<10} {actual_test:<10} {expected['type']:<15} {status}")

        if all_correct:
            print("\n[MSTAR] ✅ 样本分布完全符合FSCIL标准配置!")
        else:
            print("\n[MSTAR] ⚠️  样本分布与预期不符，请检查数据集构建")

    def get_mixup_cutmix_transforms(self):
        """获取MixUp/CutMix增强器（SAR图像慎用）"""
        # SAR图像的MixUp/CutMix需要特别小心，可能破坏散射特性
        if self.sar_augmentation and self.args.get('enable_sar_mixup', False):
            return self.sar_augmentation
        return None    
