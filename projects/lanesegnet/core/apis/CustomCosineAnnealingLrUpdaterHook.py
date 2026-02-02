from mmcv.runner.hooks import HOOKS, LrUpdaterHook
from math import cos, pi
from typing import Callable, List, Optional, Union
@HOOKS.register_module()
class CustomCosineAnnealingLrUpdaterHook(LrUpdaterHook):
    def __init__(self, epoch_iter, min_lr: Optional[float] = None, min_lr_ratio: Optional[float] = None, **kwargs):
        super(CustomCosineAnnealingLrUpdaterHook, self).__init__(**kwargs)
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.epoch_iter = epoch_iter

        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio

    def get_lr(self, runner, base_lr):
        # 当前 iter 转化为 epoch 的进度
        current_epoch = runner.iter // self.epoch_iter
        max_epochs = runner.max_iters // self.epoch_iter
        progress = current_epoch / max_epochs
        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr  # type:ignore
        # import pdb; pdb.set_trace()
        # Cosine Annealing公式计算lr
        return target_lr + 0.5 * (base_lr - target_lr) * (1 + cos(pi * progress))