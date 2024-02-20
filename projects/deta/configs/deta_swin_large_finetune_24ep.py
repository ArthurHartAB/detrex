from detrex.config import get_config
from .deta_r50_5scale_12ep import (
    train,
    optimizer,
)

from .models.deta_swin import model
# from .data.coco_detr_larger import dataloader
from .data.ab_detr_larger_4_cls import dataloader

from .scheduler.coco_scheduler import default_coco_scheduler

# 24ep for finetuning
lr_multiplier = default_coco_scheduler(24, 20, 0)#get_config("common/coco_schedule.py").lr_multiplier_24ep


# modify learning rate
optimizer.lr = 5e-5
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

train.max_iter=30*50000
train.init_checkpoint = "./weights/converted_deta_swin_o365_finetune.pth"
train.output_dir = "/mnt/s3/ab-b2b-dev/arthur/deta/weights_and_logs"

# train.fast_dev_run = dict(enabled=True)

# options for PeriodicCheckpointer, which saves a model checkpoint
# after every `checkpointer.period` iterations,
# and only `checkpointer.max_to_keep` number of checkpoint will be kept.

train.checkpointer = dict(period=25000, max_to_keep=6)
train.eval_period = 1000
train.log_period = 10

train.wandb = dict(
    enabled=True,
    params=dict(
        dir="/mnt/s3/ab-b2b-dev/arthur/deta/wandb_output",
        project="deta",
        name="deta_experiment_resume",
        # config={
        #    "learning_rate": optimizer.lr,
        # }
    )
)
