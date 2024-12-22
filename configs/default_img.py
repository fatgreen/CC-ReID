import os
import yaml
from yacs.config import CfgNode as CN
import time



_C = CN()



# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Root path for dataset directory
_C.DATA.ROOT = 'DATA_ROOT'
# Dataset for evaluation
_C.DATA.DATASET = 'prcc'
# Workers for dataloader
_C.DATA.NUM_WORKERS = 4
# Height of input image
_C.DATA.HEIGHT = 384
# Width of input image
_C.DATA.WIDTH = 192
# Batch size for training
_C.DATA.TRAIN_BATCH = 32
# Batch size for testing
_C.DATA.TEST_BATCH = 64
# The number of instances per identity for training sampler
_C.DATA.NUM_INSTANCES = 8
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Random crop prob
_C.AUG.RC_PROB = 0.5
# Random erase prob
_C.AUG.RE_PROB = 0.5
# Random flip prob
_C.AUG.RF_PROB = 0.5
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'resnet101'
# The stride for laery4 in resnet
_C.MODEL.RES4_STRIDE = 1
# feature dim
_C.MODEL.FEATURE_DIM = 4096
# Model path for resuming
_C.MODEL.RESUME = ''
# Global pooling after the backbone
_C.MODEL.POOLING = CN()
# Choose in ['avg', 'max', 'gem', 'maxavg']
_C.MODEL.POOLING.NAME = 'maxavg'
# Initialized power for GeM pooling
_C.MODEL.POOLING.P = 3
# -----------------------------------------------------------------------------
# Losses for training 
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Classification loss
_C.LOSS.CLA_LOSS = 'crossentropylabelsmooth'
# Clothes classification loss
_C.LOSS.CLOTHES_CLA_LOSS = 'cosface'
# Pairwise loss type
_C.LOSS.PAIR_LOSS = 'triplet'
# Margin for pairwise loss
_C.LOSS.PAIR_M = 0.3
# Scale for pairwise loss
_C.LOSS.PAIR_S = 1.0
# Weight for pairwise loss
_C.LOSS.PAIR_LOSS_WEIGHT = 1.0  # 新增權重配置
# Scale for classification loss
_C.LOSS.CLA_S = 16.0
# Margin for classification loss
_C.LOSS.CLA_M = 0.0
# Clothes-based adversarial loss
_C.LOSS.CAL = 'cal'
# Epsilon for clothes-based adversarial loss
_C.LOSS.EPSILON = 0.1
# Momentum for clothes-based adversarial loss with memory bank
_C.LOSS.MOMENTUM = 0.0


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 4
# Start epoch for clothes classification
_C.TRAIN.START_EPOCH_CC = 2
# Start epoch for adversarial training
_C.TRAIN.START_EPOCH_ADV = 2
# Start epoch for debias
_C.TRAIN.START_EPOCH_GENERAL = 2
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.LR = 0.00035
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# Stepsize to decay learning rate
_C.TRAIN.LR_SCHEDULER.STEPSIZE = [2, 3]
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.AMP = False  # 默認值為 False
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Perform evaluation after every N epochs (set to -1 to test after training)
_C.TEST.EVAL_STEP = 2
# Start to evaluate after specific epoch
_C.TEST.START_EVAL = 0
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 1
# Perform evaluation only
_C.EVAL_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_C.GPU = '0'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = 'OUTPUT_PATH'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'eval'
# -----------------------------------------------------------------------------
# Hyperparameters
_C.k_cal = 1.0
_C.k_kl = 1.0
# -----------------------------------------------------------------------------

def update_config(config, args):
    config.defrost()

    # 使用 yaml.safe_load 解析 YAML 文件
    with open(args.cfg, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.safe_load(f)

    # 將 dict 轉換為 CfgNode
    yaml_cfg = CN(yaml_dict)

    # 將 YAML 配置更新到 yacs 的 config 中
    config.merge_from_other_cfg(yaml_cfg)

    #config.merge_from_file(args.cfg)

    # merge from specific arguments
    if args.root:
        config.DATA.ROOT = args.root
    if hasattr(args, 'output') and args.output:
        config.OUTPUT = args.output
    if hasattr(args, 'resume') and args.resume:
        config.MODEL.RESUME = args.resume
    if hasattr(args, 'eval') and args.eval:
        config.EVAL_MODE = True
    if hasattr(args, 'tag') and args.tag:
        config.TAG = args.tag
    if hasattr(args, 'dataset') and args.dataset:
        config.DATA.DATASET = args.dataset
    if hasattr(args, 'gpu') and args.gpu:
        config.GPU = args.gpu

# def update_config(config, args):
#     config.defrost()
#     config.merge_from_file(args.cfg)

#     # merge from specific arguments
#     if args.root:
#         config.DATA.ROOT = args.root
#     if args.output:
#         config.OUTPUT = args.output
#     if args.resume:
#         config.MODEL.RESUME = args.resume
#     if args.eval:
#         config.EVAL_MODE = True
#     if args.tag:
#         config.TAG = args.tag
#     if args.dataset:
#         config.DATA.DATASET = args.dataset
#     if args.gpu:
#         config.GPU = args.gpu

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.TAG)
    config.freeze()


def get_img_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config
