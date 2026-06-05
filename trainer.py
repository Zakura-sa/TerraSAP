import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters

import os
import random
import numpy as np
import pickle


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}_{}".format(args["model_name"],args["dataset"], init_cls, args['increment'], args["kshot"])
    saved_path = "saved_model/{}/{}/{}_{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    # 为每个种子创建独立的日志文件名，避免冲突
    import time
    timestamp = int(time.time() * 1000) % 100000  # 使用时间戳后5位避免冲突

    logfilename = "logs/{}/{}/{}/{}_{}/{}_seed{}_{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["kshot"],
        args["prefix"],
        args["seed"],
        timestamp,
        args["backbone_type"],
        "log"
    )

    # 重新配置日志系统以确保每个种子使用独立的日志文件
    # 首先移除现有的handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 重新配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(sys.stdout),
        ],
        force=True  # 强制重新配置
    )

    args["base_model_path"] = "saved_model/{}/{}/{}_{}/{}_{}_{}_{}.pth".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["model_prefix"],
        args["tuned_epoch"],
        args["seed"],
        args["backbone_type"],
    )
    

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    # 简化后的训练流程：删除综合评估器

    top1_curve = {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        model.incremental_train(data_manager)

        # 标准评估
        top1_accy = model.eval_task()
        model.after_task()

        top1_curve["top1"].append(top1_accy["top1"])

        logging.info("Top1 curve: {}".format(top1_curve["top1"]))

        Hacc, old_acc, new_acc = Harmonic_Accuracy(top1_accy["grouped"], args["init_cls"])
        logging.info("Average Accuracy (Top1): {}   (Harmonic Accuracy): {} (Old Acc): {} (New Acc): {} \n".format(sum(top1_curve["top1"])/len(top1_curve["top1"]),
                                                                            Hacc, old_acc, new_acc))



    logging.info("\n")

    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def Harmonic_Accuracy(grouped_acc, init_cls):
    old_acc, new_acc = [], []
    for key in grouped_acc.keys():
        if '-' in key:
            try:
                # 安全解析键格式，处理可能的异常
                key_parts = key.split('-')
                if len(key_parts) >= 2:
                    end_class_id = int(key_parts[1])
                    if end_class_id < init_cls:
                        old_acc.append(grouped_acc[key])
                    elif end_class_id >= init_cls:  # 修改为>=，包含边界情况
                        new_acc.append(grouped_acc[key])
            except (ValueError, IndexError) as e:
                # 忽略无法解析的键，记录警告
                logging.warning(f"⚠️ 无法解析键格式: {key}, 错误: {e}")
                continue

    # 修复除零错误：检查old_acc是否为空
    if len(old_acc) > 0:
        old_acc = sum(old_acc) / len(old_acc)
    else:
        old_acc = 0.0  # 如果没有旧类别，设置为0
        logging.warning("⚠️ 没有找到旧类别的准确率数据，设置old_acc=0.0")

    if len(new_acc) > 0:
        new_acc = sum(new_acc) / len(new_acc)
        # 只有当old_acc > 0时才计算调和平均数
        if old_acc > 0:
            Hacc = 2 * old_acc * new_acc / (old_acc + new_acc)
        else:
            Hacc = new_acc  # 如果没有旧类别，调和准确率等于新类别准确率
            logging.info("📊 由于没有旧类别数据，调和准确率设置为新类别准确率")
    else:
        new_acc = 0.0  # 如果没有新类别，设置为0
        Hacc = old_acc if old_acc > 0 else 0.0  # 如果没有新类别，调和准确率等于旧类别准确率
        logging.info("📊 由于没有新类别数据，调和准确率设置为旧类别准确率")

    return Hacc, old_acc, new_acc
