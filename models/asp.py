# watermark version

import logging
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from backbone.asp_backbone import SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
num_workers = 1

def cos_loss(cosine, label):
    loss = 0
    for i, y in enumerate(label):
        loss += 1 - cosine[i, y]
    return loss / len(label)

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self. batch_size= args["batch_size"]
        self. init_lr=args["init_lr"]

        self.weight_decay=args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr=args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args=args
        self._used_ncac_expansion = False

        # 分层学习率配置（优化版）
        self.backbone_lr_ratio = args.get('backbone_lr_ratio', 0.1)  # backbone学习率倍数
        self.prompt_lr_ratio = args.get('prompt_lr_ratio', 1.0)      # 提示编码器学习率倍数
        self.spatial_prompt_lr_ratio = args.get('spatial_prompt_lr_ratio', 1.2)  # 空间提示学习率倍数
        self.classifier_lr_ratio = args.get('classifier_lr_ratio', 0.5)  # 分类器学习率倍数
        self.consistency_lr_ratio = args.get('consistency_lr_ratio', 0.5)  # 一致性模块学习率倍数

        # 多尺度模块学习率配置
        self.multiscale_lr_ratio = args.get('multiscale_lr_ratio', 1.2)  # 多尺度模块学习率倍数
        self.spectral_lr_ratio = args.get('spectral_lr_ratio', 1.0)      # 光谱一致性模块学习率倍数
        self.adjacency_lr_ratio = args.get('adjacency_lr_ratio', 0.8)    # 空间邻接模块学习率倍数

        # 多尺度光谱融合模块学习率配置（方案一）
        self.multiscale_spectral_lr_ratio = args.get('multiscale_spectral_lr_ratio', 1.2)  # 多尺度光谱融合学习率倍数


        # 使用ASP+空间提示的标准架构
        logging.info("使用ASP+空间提示的标准架构")

        # 遥感数据增强配置（由数据加载类处理，这里只记录状态）
        self.enable_rs_augmentation = args.get('enable_rs_augmentation', False)
        if self.enable_rs_augmentation:
            logging.info("🔧 遥感数据增强已启用（由数据加载类处理）")
            logging.info(f"  增强模式: {args.get('rs_augmentation_mode', 'balanced')}")
            logging.info(f"  MixUp: {args.get('use_mixup', True)}")
            logging.info(f"  CutMix: {args.get('use_cutmix', True)}")
        else:
            logging.info("遥感数据增强未启用")

    def _create_layered_optimizer(self):
        """
        创建分层学习率优化器
        为不同模块设置不同的学习率

        优化版本：针对ASP+空间提示的精确参数分组
        """
        # 分组参数
        backbone_params = []
        prompt_params = []
        spatial_prompt_params = []  # 空间感知提示参数
        consistency_params = []
        classifier_params = []  # 分类器参数
        multiscale_params = []  # 多尺度模块参数
        spectral_params = []    # 光谱一致性模块参数
        adjacency_params = []   # 空间邻接模块参数
        multiscale_spectral_params = []  # 多尺度光谱融合模块参数（方案一）
        other_params = []

        # 统计参数分组情况
        param_count = {
            'backbone': 0, 'prompt': 0, 'spatial_prompt': 0,
            'consistency': 0, 'classifier': 0, 'multiscale': 0,
            'spectral': 0, 'adjacency': 0, 'multiscale_spectral': 0, 'other': 0
        }

        for name, param in self._network.named_parameters():
            if not param.requires_grad:
                continue

            # 多尺度光谱融合模块参数（方案一，最高优先级）
            elif any(keyword in name for keyword in ['multiscale_spectral_fusion', 'scale_weights', 'spectral_weights']):
                multiscale_spectral_params.append(param)
                param_count['multiscale_spectral'] += param.numel()

            # 多尺度模块参数（最高学习率）- 优先于空间提示参数
            elif any(keyword in name for keyword in ['multiscale', 'scale_', 'spectral', 'adjacency', 'remote_sensing']):
                if 'multiscale' in name or 'scale_' in name:
                    multiscale_params.append(param)
                    param_count['multiscale'] += param.numel()
                elif 'spectral' in name:
                    spectral_params.append(param)
                    param_count['spectral'] += param.numel()
                elif 'adjacency' in name:
                    adjacency_params.append(param)
                    param_count['adjacency'] += param.numel()
                else:
                    # 其他遥感模块参数
                    spatial_prompt_params.append(param)
                    param_count['spatial_prompt'] += param.numel()

            # 空间感知提示编码器参数（高学习率）- 优先于普通提示参数
            elif any(keyword in name for keyword in ['pos_embedding', 'rel_pos_attention', 'spatial_relation_mlp',
                                                    'base_encoder', 'spatial_projection', '_temp_spatial_proj']):
                spatial_prompt_params.append(param)
                param_count['spatial_prompt'] += param.numel()

            # 普通提示相关参数（较高学习率）
            elif any(keyword in name for keyword in ['TIP', 'Prompt_Encoder', 'prompt', 'Avg_TSP']):
                prompt_params.append(param)
                param_count['prompt'] += param.numel()

            # 分类器参数（中等学习率）
            elif any(keyword in name for keyword in ['fc', 'classifier', 'head']):
                classifier_params.append(param)
                param_count['classifier'] += param.numel()

            # 预训练backbone参数（较低学习率）
            elif (any(keyword in name for keyword in ['backbone.cls_token', 'backbone.pos_embed', 'backbone.patch_embed']) or
                  (name.startswith('backbone.blocks.') and 'Prompt_Encoder' not in name)):
                backbone_params.append(param)
                param_count['backbone'] += param.numel()

            # 其他参数
            else:
                other_params.append(param)
                param_count['other'] += param.numel()

        # 打印参数分组统计
        total_params = sum(param_count.values())
        print(f"\n=== 分层学习率参数分组统计 ===")
        for group_name, count in param_count.items():
            if count > 0:
                percentage = count / total_params * 100
                print(f"{group_name}: {count:,} 参数 ({percentage:.1f}%)")
        print(f"总参数: {total_params:,}")
        print("=" * 40)

        # 计算各模块学习率
        backbone_lr = self.init_lr * self.backbone_lr_ratio
        prompt_lr = self.init_lr * self.prompt_lr_ratio
        consistency_lr = self.init_lr * self.consistency_lr_ratio

        # 空间感知提示学习率（通常比普通提示稍高）
        spatial_prompt_lr_ratio = getattr(self, 'spatial_prompt_lr_ratio', self.prompt_lr_ratio * 1.2)
        spatial_prompt_lr = self.init_lr * spatial_prompt_lr_ratio

        # 分类器学习率（中等）
        classifier_lr_ratio = getattr(self, 'classifier_lr_ratio', 0.5)
        classifier_lr = self.init_lr * classifier_lr_ratio

        # 多尺度模块学习率
        multiscale_lr = self.init_lr * self.multiscale_lr_ratio
        spectral_lr = self.init_lr * self.spectral_lr_ratio
        adjacency_lr = self.init_lr * self.adjacency_lr_ratio

        # 多尺度光谱融合模块学习率（方案一）
        multiscale_spectral_lr = self.init_lr * self.multiscale_spectral_lr_ratio


        # 创建参数组
        param_groups = []

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': backbone_lr,
                'weight_decay': self.weight_decay * 0.1,  # backbone用较小的权重衰减
                'name': 'backbone'
            })

        if prompt_params:
            param_groups.append({
                'params': prompt_params,
                'lr': prompt_lr,
                'weight_decay': self.weight_decay,
                'name': 'prompt'
            })

        if spatial_prompt_params:
            param_groups.append({
                'params': spatial_prompt_params,
                'lr': spatial_prompt_lr,
                'weight_decay': self.weight_decay * 0.5,  # 空间提示用中等权重衰减
                'name': 'spatial_prompt'
            })

        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': classifier_lr,
                'weight_decay': self.weight_decay,
                'name': 'classifier'
            })

        if consistency_params:
            param_groups.append({
                'params': consistency_params,
                'lr': consistency_lr,
                'weight_decay': self.weight_decay,
                'name': 'consistency'
            })

        # 多尺度光谱融合模块参数组（方案一，最高优先级）
        if multiscale_spectral_params:
            param_groups.append({
                'params': multiscale_spectral_params,
                'lr': multiscale_spectral_lr,
                'weight_decay': self.weight_decay * 0.2,  # 多尺度光谱融合用最小权重衰减
                'name': 'multiscale_spectral'
            })

        # 多尺度模块参数组
        if multiscale_params:
            param_groups.append({
                'params': multiscale_params,
                'lr': multiscale_lr,
                'weight_decay': self.weight_decay * 0.3,  # 多尺度模块用较小权重衰减
                'name': 'multiscale'
            })

        if spectral_params:
            param_groups.append({
                'params': spectral_params,
                'lr': spectral_lr,
                'weight_decay': self.weight_decay * 0.5,
                'name': 'spectral'
            })

        if adjacency_params:
            param_groups.append({
                'params': adjacency_params,
                'lr': adjacency_lr,
                'weight_decay': self.weight_decay * 0.4,
                'name': 'adjacency'
            })

        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.init_lr,
                'weight_decay': self.weight_decay,
                'name': 'other'
            })

        # 打印分层学习率信息
        print(f"\n🔧 分层学习率调度 (多尺度光谱融合版):")
        print(f"  Backbone LR: {backbone_lr:.6f} (ratio: {self.backbone_lr_ratio})")
        print(f"  Prompt LR: {prompt_lr:.6f} (ratio: {self.prompt_lr_ratio})")
        if spatial_prompt_params:
            print(f"  Spatial Prompt LR: {spatial_prompt_lr:.6f} (ratio: {spatial_prompt_lr_ratio:.2f})")
        if multiscale_spectral_params:
            print(f"  Multiscale Spectral LR: {multiscale_spectral_lr:.6f} (ratio: {self.multiscale_spectral_lr_ratio})")
        if multiscale_params:
            print(f"  Multiscale LR: {multiscale_lr:.6f} (ratio: {self.multiscale_lr_ratio})")
        if spectral_params:
            print(f"  Spectral LR: {spectral_lr:.6f} (ratio: {self.spectral_lr_ratio})")
        if adjacency_params:
            print(f"  Adjacency LR: {adjacency_lr:.6f} (ratio: {self.adjacency_lr_ratio})")
        if classifier_params:
            print(f"  Classifier LR: {classifier_lr:.6f} (ratio: {classifier_lr_ratio})")
        print(f"  Consistency LR: {consistency_lr:.6f} (ratio: {self.consistency_lr_ratio})")
        
        print(f"  Other LR: {self.init_lr:.6f} (ratio: 1.0)")
        print(f"  参数组数量: {len(param_groups)}")

        # 创建优化器（不在这里设置weight_decay，因为每个参数组已经设置了）
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif self.args['optimizer'] in ['adam', 'adamw']:
            optimizer = optim.AdamW(param_groups)
        else:
            optimizer = optim.SGD(param_groups, momentum=0.9)

        return optimizer

    def after_task(self):
        self._known_classes = self._total_classes

    def _set_adaptive_modulation_task_id(self, task_id):
        """设置自适应调制的任务ID"""
        try:
            backbone = self._network.backbone
            if hasattr(backbone, 'Prompt_Encoder') and hasattr(backbone.Prompt_Encoder, 'set_task_id'):
                backbone.Prompt_Encoder.set_task_id(task_id)
                logging.info(f"🔧 自适应调制任务ID已设置: {task_id}")
        except Exception as e:
            logging.warning(f"⚠️ 设置自适应调制任务ID失败: {e}")

    def replace_fc(self,trainloader, model, args):
        # use class prototype as classifier weights.
        model = model.eval()
        embedding_list = []
        label_list = []
        # 空间特征收集已移除

        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

                # 空间感知特征收集已移除

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # 简化原型更新：使用原始的简单平均策略
        class_list=np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]

            # 使用简单的特征平均作为原型（回退到有效的基础策略）
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto

        return model
    

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._used_ncac_expansion = False

        # 检查是否使用新类感知分类器
        if (hasattr(self._network.fc, 'smart_expand_with_support') and
            self.args.get('enable_new_class_aware_classifier', False)):

            # 创建支持集数据加载器
            support_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train", mode="train", kshot=self.args["kshot"]
            )
            support_loader = DataLoader(support_dataset, batch_size=16, shuffle=False, num_workers=1)

            # 智能扩展分类器
            logging.info("🚀 使用新类感知分类器进行智能扩展")
            self._network.fc.smart_expand_with_support(
                new_classes=data_manager.get_task_size(self._cur_task),
                support_loader=support_loader,
                backbone=self._network,
                device=self._device
            )
            self._used_ncac_expansion = True
        else:
            # 使用原始方法
            self._network.update_fc(self._total_classes)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # 设置自适应调制的任务ID
        self._set_adaptive_modulation_task_id(self._cur_task)

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", kshot=self.args["kshot"] )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        if isinstance(self.args['kshot'], int) and self._known_classes>0:
            train_bs = self.args['fs_batch_size']
        else:
            train_bs = self.batch_size

        # 简化的数据加载器创建
        self.train_loader = DataLoader(
            train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
        )

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        test_curr_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="test", mode="test")
        self.test_curr_loader = DataLoader(
            test_curr_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", kshot=self.args["kshot"])
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        
        logging.info("training set size: {}, fc construct set size: {}".format(len(train_dataset), len(train_dataset_for_protonet)))

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):

        self._network.to(self._device)



        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info('total parameters: {}'.format(total_params))
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info('trainable parameters: {}'.format(total_trainable_params))

        # if some parameters are trainable, print the key name and corresponding parameter number
        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())
        
        if self._cur_task > 0:
            self.update_ema_prompt(train_loader_for_protonet)  
            if self._used_ncac_expansion:
                logging.info("NCAC initialized classifier weights; skip mean-prototype overwrite.")
            else:
                self.replace_fc(train_loader_for_protonet, self._network, None)

        if os.path.exists(self.args["base_model_path"]) and self._cur_task==0:
            logging.info('================= load base model from: {} ================='.format(self.args["base_model_path"]))
            # 安全加载模型，处理不匹配的键和设备映射
            checkpoint = torch.load(self.args["base_model_path"], map_location=self._device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 过滤掉当前模型中不存在的键
            model_state_dict = self._network.state_dict()
            filtered_state_dict = {}

            for key, value in state_dict.items():
                if key in model_state_dict:
                    # 检查形状是否匹配
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        logging.warning(f"⚠️ 跳过形状不匹配的参数: {key} (模型: {model_state_dict[key].shape}, 检查点: {value.shape})")
                else:
                    logging.warning(f"⚠️ 跳过不存在的参数: {key}")

            # 使用strict=False加载过滤后的状态字典
            missing_keys, unexpected_keys = self._network.load_state_dict(filtered_state_dict, strict=False)

            if missing_keys:
                logging.warning(f"⚠️ 缺失的参数键: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"⚠️ 意外的参数键: {unexpected_keys}")

            logging.info(f"✅ 成功加载 {len(filtered_state_dict)} 个参数")

        else:
            # 分层学习率调度优化
            if self.args.get('use_layered_lr', True):  # 默认启用分层学习率
                optimizer = self._create_layered_optimizer()
            else:
                # 原始单一学习率设置
                if self.args['optimizer']=='sgd':
                    optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
                elif self.args['optimizer'] in ['adam', 'adamw']:
                    optimizer=optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)

            self._init_train(train_loader, test_loader, optimizer, scheduler)
            if self._cur_task == 0:
                # 保存最佳模型并删除非最佳模型
                best_model_path = self.args["base_model_path"]
                torch.save(self._network.state_dict(), best_model_path)
                logging.info(f"💾 基础任务最佳模型已保存: {best_model_path}")

                # 删除同一实验的其他种子模型（如果存在）
                self._cleanup_non_best_models(best_model_path)

        if self._cur_task == 0:
            self.update_ema_prompt(train_loader_for_protonet, mode='base')
            self.replace_fc(train_loader_for_protonet, self._network, None)
        else:
            # 增量任务完成，记录日志但不保存模型
            logging.info(f"📝 增量学习任务 {self._cur_task} 完成，未保存模型")

    def eval_task(self):
        y_pred, y_true = self._eval_acc(self.test_loader)
        accy = self._evaluate(y_pred, y_true)
        return accy
    
    def _eval_acc(self, loader):
        import time
        import numpy as np

        self._network.eval()
        y_pred, y_true = [], []
        all_outputs, all_embedding = [], []
        batch_times = []
        total_samples = 0

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            batch_start_time = time.time()

            with torch.no_grad():
                out = self._network(inputs)
                outputs = out["logits"]
                embedding = out["features"]

            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            total_samples += len(targets)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu())
            all_embedding.append(embedding.cpu())

        # 计算推理时间统计
        if batch_times:
            total_inference_time = sum(batch_times)
            avg_batch_time = np.mean(batch_times)
            median_batch_time = np.median(batch_times)
            p90_batch_time = np.percentile(batch_times, 90)
            avg_per_image_time = total_inference_time / total_samples * 1000  # ms

            logging.info("⏱️ 推理时间统计 (批次级别):")
            logging.info(f"   总批次数: {len(batch_times)}, 总样本数: {total_samples}")
            logging.info(f"   总推理时间: {total_inference_time:.3f}s")
            logging.info(f"   平均批次时间: {avg_batch_time:.3f}s")
            logging.info(f"   中位数批次时间: {median_batch_time:.3f}s")
            logging.info(f"   P90批次时间: {p90_batch_time:.3f}s")
            logging.info(f"   平均单图像时间: {avg_per_image_time:.2f}ms")

            # 单张图片推理时间测量
            self._measure_single_image_inference_time(loader)

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        all_outputs = torch.cat(all_outputs)
        all_embedding = torch.cat(all_embedding)

        return y_pred, y_true # [N, topk]

    def _measure_single_image_inference_time(self, loader):
        """测量单张图片推理时间"""
        import time
        import torch
        import numpy as np

        logging.info("🔍 执行单张图片推理时间测量...")

        # 获取一个样本用于测试
        sample_batch = next(iter(loader))
        _, sample_input, _ = sample_batch
        single_input = sample_input[:1].to(self._device)  # 只取一张图片

        # 预热
        warmup_runs = 15
        logging.info(f"🔍 开始单张图片推理时间测量: {warmup_runs}次warm-up + 200次测量")

        self._network.eval()
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self._network(single_input)["logits"]

        # 正式测量
        measurement_runs = 200
        times = []

        torch.cuda.synchronize()  # 确保GPU操作完成

        for _ in range(measurement_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self._network(single_input)["logits"]
            torch.cuda.synchronize()  # 确保GPU操作完成
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒

        # 计算统计信息
        avg_time = np.mean(times)
        median_time = np.median(times)
        p90_time = np.percentile(times, 90)
        std_time = np.std(times)

        logging.info("📊 单张图片推理时间统计:")
        logging.info(f"   测量样本数: {measurement_runs}")
        logging.info(f"   平均时间: {avg_time:.2f}ms")
        logging.info(f"   中位数时间: {median_time:.2f}ms")
        logging.info(f"   P90时间: {p90_time:.2f}ms")
        logging.info(f"   标准差: {std_time:.2f}ms")
    
    # naive train
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if isinstance(self.args['kshot'], int) and self._known_classes>0:
            total_epoch = self.args['fs_epoch']
        else:
            total_epoch = self.args['tuned_epoch']

        # 最佳模型跟踪
        best_acc = 0.0
        best_epoch = 0
        best_model_state = None

        # 添加epoch进度条
        epoch_pbar = tqdm(range(total_epoch), desc=f"Task {self._cur_task} Training", unit="epoch")

        for _, epoch in enumerate(epoch_pbar):

            if self._cur_task == 0:
                anchor_samples = self.find_anchor_sample(self._network, self.train_loader_for_protonet)
                print('anchor samples found')

            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            # 添加batch进度条
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epoch}", leave=False, unit="batch")

            for i, (_, inputs, targets) in enumerate(batch_pbar):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                if self._cur_task == 0:
                    cur_class = set(targets.cpu())
                    for c in cur_class:
                        inputs = torch.cat([inputs, anchor_samples[c].unsqueeze(0).to(self._device)])
                    out = self._network(inputs, self.args["perturb_var"])
                    logits = out["logits"][:-len(cur_class),:]
                    features = out["features"][:-len(cur_class),:]  # 确保features和logits大小一致
                    (mu, std) = out["kl"]

                    # 计算各个损失组件
                    ce_loss = F.cross_entropy(logits, targets)

                    # 锚点相似性损失
                    sim_loss = 0.0
                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    anchor_id = 0
                    all_features = out["features"]  # 获取完整的features（包含锚点）
                    for c in cur_class:
                        fea_c = features[targets==c]  # 使用已裁剪的features
                        fea_anchor = all_features[len(targets):][anchor_id].detach()  # 从完整features中获取锚点
                        fea_anchor = fea_anchor.unsqueeze(0).repeat(len(fea_c), 1)
                        sim_loss += (1-cos(fea_c, fea_anchor)).mean()
                        anchor_id += 1
                    sim_loss = sim_loss / len(cur_class)

                    # KL散度损失
                    KL = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 2*std.log() - 1) / mu.size(0)

                    # 使用原始固定权重（简化策略）
                    loss = ce_loss + self.args["anchor_lambda"] * sim_loss + self.args["kl_weight"] * KL


                else:
                    out = self._network(inputs, self.args["perturb_var"])
                    logits = out["logits"]
                    features = out["features"]
                    logits[:, :self._known_classes] = float('-inf')
                    loss = F.cross_entropy(logits, targets)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 安全地累积损失值
                loss_item = loss.item()
                if not (torch.isnan(torch.tensor(loss_item)) or torch.isinf(torch.tensor(loss_item))):
                    losses += loss_item
                else:
                    logging.warning(f"检测到异常损失值: {loss_item}，跳过累积")

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                # 每个batch都更新进度条
                current_loss = losses / (i + 1) if (i + 1) > 0 else 0.0
                current_acc = tensor2numpy(correct) * 100 / total if total > 0 else 0.0
                batch_pbar.set_postfix({
                    'Loss': f'{current_loss:.3f}',
                    'Acc': f'{current_acc:.2f}%'
                })

                # 内存清理（ASP+空间提示标准清理）
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_cur_acc = self._compute_accuracy(self._network, self.test_curr_loader)
            test_acc = self._compute_accuracy(self._network, test_loader)

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                best_model_state = self._network.state_dict().copy()
                logging.info(f"🎯 发现更优模型! test_acc: {test_acc:.4f} (Epoch {epoch + 1})")

            # 计算平均损失，处理异常值
            avg_loss = losses / len(train_loader) if len(train_loader) > 0 else 0.0
            if torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)):
                avg_loss = 0.0
                logging.warning(f"检测到异常平均损失值，重置为0.0")

            # 更新epoch进度条
            epoch_pbar.set_postfix({
                'Train_Acc': f'{train_acc:.2f}%',
                'Test_Acc': f'{test_acc:.2f}%',
                'Loss': f'{avg_loss:.3f}'
            })

            info = "Task {}, Epoch {} => Loss {:.3f}, Train_accy {:.2f}, Test_curr_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                avg_loss,
                train_acc,
                test_cur_acc,
                test_acc
            )
            logging.info(info)

        # 恢复最佳模型状态
        if best_model_state is not None:
            self._network.load_state_dict(best_model_state)
            logging.info(f"✅ 训练完成: 共{total_epoch}轮，最佳性能在第{best_epoch}轮")
        else:
            logging.info(f"✅ 训练完成: 共{total_epoch}轮，未找到最佳模型")


    def update_ema_prompt(self, train_loader, mode='new'):
        self._network.eval()
        original_prompt_list = []
        modulated_prompt_list = []

        with torch.no_grad():
            for _, batch in enumerate(train_loader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                backbone = self._network.backbone
                prompt_encoder = backbone.Prompt_Encoder

                if hasattr(backbone, 'spatial_aware_prompts') and backbone.spatial_aware_prompts:
                    x_patches = backbone.patch_embed(data)
                    x_features = x_patches.mean(dim=1)
                    tip_features = backbone.TIP.mean(dim=0).mean(dim=0).unsqueeze(0).expand(data.size(0), -1)

                    if hasattr(prompt_encoder, 'forward_without_feature_modulation'):
                        original_prompt = prompt_encoder.forward_without_feature_modulation(tip_features, x_features)
                    else:
                        original_prompt = prompt_encoder(tip_features, x_features)

                    if (hasattr(backbone, 'enable_spatial_context_pipeline') and
                        backbone.enable_spatial_context_pipeline and
                        hasattr(prompt_encoder, 'forward_with_context')):
                        modulated_prompt, _ = prompt_encoder.forward_with_context(tip_features, x_features)
                    else:
                        modulated_prompt = prompt_encoder(tip_features, x_features)
                else:
                    modulated_prompt, _ = prompt_encoder(data, backbone.TIP, 0)
                    original_prompt = modulated_prompt

                original_prompt_list.append(original_prompt.detach().cpu())
                modulated_prompt_list.append(modulated_prompt.detach().cpu())

        original_prompt_mean = torch.mean(torch.cat(original_prompt_list, dim=0), dim=0).to(self._device)
        modulated_prompt_mean = torch.mean(torch.cat(modulated_prompt_list, dim=0), dim=0).to(self._device)

        if hasattr(self._network.backbone, 'enable_dual_path_ema') and self._network.backbone.enable_dual_path_ema:
            if mode == 'new':
                beta = self.args["EMA_beta"]
                self._network.backbone.Avg_TSP_original = (
                    beta * self._network.backbone.Avg_TSP_original.to(self._device)
                    + (1 - beta) * original_prompt_mean
                )
                self._network.backbone.Avg_TSP_modulated = (
                    beta * self._network.backbone.Avg_TSP_modulated.to(self._device)
                    + (1 - beta) * modulated_prompt_mean
                )
            else:
                self._network.backbone.Avg_TSP_original = original_prompt_mean.clone()
                self._network.backbone.Avg_TSP_modulated = modulated_prompt_mean.clone()

            fusion_weight = self.args.get("dual_ema_fusion_weight", 0.5)
            self._network.backbone.Avg_TSP = (
                fusion_weight * self._network.backbone.Avg_TSP_original
                + (1 - fusion_weight) * self._network.backbone.Avg_TSP_modulated
            )
            logging.info(f"Dual-path EMA updated: fusion_weight={fusion_weight}")
        else:
            if mode == 'new':
                beta = self.args["EMA_beta"]
                self._network.backbone.Avg_TSP = (
                    beta * self._network.backbone.Avg_TSP.to(self._device)
                    + (1 - beta) * modulated_prompt_mean
                )
            else:
                self._network.backbone.Avg_TSP = modulated_prompt_mean
        return

        self._network.eval()
        prompt_list = []

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                # 兼容新的空间感知提示编码器
                if hasattr(self._network.backbone, 'spatial_aware_prompts') and self._network.backbone.spatial_aware_prompts:
                    # 对于空间感知提示编码器，需要不同的调用方式
                    x_patches = self._network.backbone.patch_embed(data)
                    x_features = x_patches.mean(dim=1)  # [B, D] 全局特征
                    tip_features = self._network.backbone.TIP.mean(dim=0).mean(dim=0).unsqueeze(0).expand(data.size(0), -1)
                    prompt = self._network.backbone.Prompt_Encoder(tip_features, x_features)
                else:
                    # 原始提示编码器
                    prompt, _ = self._network.backbone.Prompt_Encoder(data, self._network.backbone.TIP, 0)

                prompt_list.append(prompt.detach().cpu())

        # 协同优化：双路径EMA更新（阶段2）
        if hasattr(self._network.backbone, 'enable_dual_path_ema') and self._network.backbone.enable_dual_path_ema:
            # 双路径EMA更新
            prompt_mean = torch.mean(torch.cat(prompt_list, dim=0), dim=0)

            if mode == 'new':
                # 更新原始TSP的EMA（假设prompt_list包含原始TSP）
                self._network.backbone.Avg_TSP_original = self.args["EMA_beta"] * self._network.backbone.Avg_TSP_original + (1 - self.args["EMA_beta"]) * prompt_mean

                # 更新调制TSP的EMA（这里先使用相同的prompt_mean，后续会在特征调制时分离）
                self._network.backbone.Avg_TSP_modulated = self.args["EMA_beta"] * self._network.backbone.Avg_TSP_modulated + (1 - self.args["EMA_beta"]) * prompt_mean

                # 根据融合策略更新主EMA
                fusion_weight = self.args.get("dual_ema_fusion_weight", 0.5)
                self._network.backbone.Avg_TSP = fusion_weight * self._network.backbone.Avg_TSP_original + (1 - fusion_weight) * self._network.backbone.Avg_TSP_modulated
            else:
                # 初始化时同步更新所有EMA
                self._network.backbone.Avg_TSP = prompt_mean
                self._network.backbone.Avg_TSP_original = prompt_mean.clone()
                self._network.backbone.Avg_TSP_modulated = prompt_mean.clone()

            # 确保所有EMA都在正确设备上
            self._network.backbone.Avg_TSP.to(self._device)
            self._network.backbone.Avg_TSP_original.to(self._device)
            self._network.backbone.Avg_TSP_modulated.to(self._device)

            logging.info(f"🔄 双路径EMA更新完成: fusion_weight={self.args.get('dual_ema_fusion_weight', 0.5)}")
        else:
            # 原有的单路径EMA更新
            if mode == 'new':
                self._network.backbone.Avg_TSP = self.args["EMA_beta"]*self._network.backbone.Avg_TSP + (1-self.args["EMA_beta"])*torch.mean(torch.cat(prompt_list, dim=0), dim=0)
            else:
                self._network.backbone.Avg_TSP = torch.mean(torch.cat(prompt_list, dim=0), dim=0)

            self._network.backbone.Avg_TSP.to(self._device)



    def find_anchor_sample(self, model, train_loader):
        # train_loader must be Shuffle == False.

        model.eval()
        embedding_list = []
        label_list = []
        prompt_list = []
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

                # 兼容新的空间感知提示编码器
                if hasattr(self._network.backbone, 'spatial_aware_prompts') and self._network.backbone.spatial_aware_prompts:
                    # 对于空间感知提示编码器，需要不同的调用方式
                    x_patches = self._network.backbone.patch_embed(data)
                    x_features = x_patches.mean(dim=1)  # [B, D] 全局特征
                    tip_features = self._network.backbone.TIP.mean(dim=0).mean(dim=0).unsqueeze(0).expand(data.size(0), -1)
                    prompt = self._network.backbone.Prompt_Encoder(tip_features, x_features)
                else:
                    # 原始提示编码器
                    prompt, _ = self._network.backbone.Prompt_Encoder(data, self._network.backbone.TIP, 0)

                prompt_list.append(prompt.detach().cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        self._network.backbone.Avg_TSP = torch.mean(torch.cat(prompt_list, dim=0), dim=0)   
        self._network.backbone.Avg_TSP.to(self._device)   

        class_list=np.unique(train_loader.dataset.labels)
        anchor_sample = []
        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            class_mean = embedding.mean(0)
            class_mean = class_mean.unsqueeze(0).repeat(len(embedding), 1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_sim = cos(embedding, class_mean)
            anchor_index = torch.argmax(cos_sim)
            anchor_sample.append(train_loader.dataset[data_index[anchor_index]][1])
        return anchor_sample

    def _cleanup_non_best_models(self, best_model_path):
        """删除同一实验的非最佳种子模型"""
        import os
        import glob

        try:
            # 从最佳模型路径中提取实验信息
            model_dir = os.path.dirname(best_model_path)
            best_filename = os.path.basename(best_model_path)

            # 提取实验前缀（去掉种子和backbone信息）
            parts = best_filename.split('_')
            if len(parts) >= 3:
                # 构建模式匹配，查找同一实验的其他种子模型
                prefix_parts = []
                for part in parts:
                    if part.isdigit() and len(part) == 4:  # 种子通常是4位数字
                        break
                    prefix_parts.append(part)

                if prefix_parts:
                    prefix = '_'.join(prefix_parts)
                    pattern = os.path.join(model_dir, f"{prefix}_*_*.pth")

                    # 查找所有匹配的模型文件
                    matching_files = glob.glob(pattern)
                    deleted_count = 0
                    freed_space = 0

                    for file_path in matching_files:
                        if file_path != best_model_path and os.path.exists(file_path):
                            try:
                                file_size = os.path.getsize(file_path)
                                os.remove(file_path)
                                deleted_count += 1
                                freed_space += file_size
                                logging.info(f"🗑️ 删除非最佳模型: {os.path.basename(file_path)}")
                            except Exception as e:
                                logging.warning(f"⚠️ 删除模型文件失败 {file_path}: {e}")

                    if deleted_count > 0:
                        logging.info(f"🧹 清理完成: 删除 {deleted_count} 个非最佳模型，释放 {freed_space/(1024*1024):.1f} MB")
                    else:
                        logging.info("📝 增量学习任务完成，未保存模型" if self._cur_task > 0 else "💾 仅保留最佳模型")

        except Exception as e:
            logging.warning(f"⚠️ 清理非最佳模型时出错: {e}")
