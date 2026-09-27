import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils.metrics import harmonic_accuracy

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
    
    os.makedirs(logs_name, exist_ok=True)
    os.makedirs(saved_path, exist_ok=True)

    # 为每个种子创建独立的日志文件名，避免冲突
    import time
    timestamp = time.time_ns()

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
        support_paths = {
            str(label): data_manager._train_data[indices].tolist()
            for label, indices in data_manager.support_sampler.indices.items()
        }
        logging.info('Support files: %s', support_paths)

        # 标准评估
        top1_accy = model.eval_task()
        model.after_task()

        top1_curve["top1"].append(top1_accy["top1"])

        logging.info("Top1 curve: {}".format(top1_curve["top1"]))

        score, old_acc, new_acc = harmonic_accuracy(top1_accy['grouped'], has_old=task > 0)
        logging.info('Average Accuracy (Top1): %s; Harmonic: %s; Old: %s; New: %s',
                     sum(top1_curve['top1']) / len(top1_curve['top1']),
                     'N/A' if score is None else score, old_acc, new_acc)



    logging.info("\n")

    
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if str(device) == '-1':
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
