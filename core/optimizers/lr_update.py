from math import cos, pi

class LrUpdater(object):
    """LR Scheduler in MMCV.

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed
        
    # 给optimizer设置新lr
    def _set_lr(self, runner, lr_groups):

        for param_group, lr in zip(runner.get("optimizer").param_groups,
                                    lr_groups):
            param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        raise NotImplementedError
    
    # 获取新lr
    def get_regular_lr(self, runner):

        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    # 获取warmup学习率更新策略的lr
    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)
        
    # 记录初始lr
    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        
        for group in runner.get("optimizer").param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in runner.get("optimizer").param_groups
        ]
        
    # 在周期更新前获取新lr并在optimizer中更新它
    def before_train_epoch(self, runner):
        if self.warmup_iters is None: # 即self.warmup_by_epoch为True，warmup_epochs = warmup_iters
            epoch_len = len(runner.get("train_loader")) # 获取一个epoch迭代多少iter，即datasets//batch size
            self.warmup_iters = self.warmup_epochs * epoch_len # 按周期更新则warmup iters = warmup_epochs * datasets//batch size
        # 不按周期更新则没必要在此进行lr更新，在下一步before_train_iter中更新
        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)
        
     # 首先判断是否按周期更新lr，若按迭代次数更新即by_epoch为False，大于等于warmup_iters使用正常lr更新方式，小于则用warmup方式更新lr
    def before_train_iter(self, runner):
        cur_iter = runner.get("iter")
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            # 按周期更新lr，由于当前迭代已经大于warmup迭代，所以直接返回，使用before_train_epoch中的lr
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters: # 等于则用常规方式
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter) # 小于则用warmup方式
                self._set_lr(runner, warmup_lr)

class StepLrUpdater(LrUpdater):
    """Step LR scheduler with min_lr clipping.

    Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float, optional): Decay LR ratio. Default: 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
    """

    def __init__(self, step, gamma=0.1, min_lr=None, **kwargs):

        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        super(StepLrUpdater, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.get('epoch') if self.by_epoch else runner.get('iter')

        # calculate exponential term
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        lr = base_lr * (self.gamma**exp)
        if self.min_lr is not None:
            # clip to a minimum value
            lr = max(lr, self.min_lr)
        return lr

class PolyLrUpdater(LrUpdater):

    def __init__(self, power=1., min_lr=0., **kwargs):
        self.power = power
        self.min_lr = min_lr
        super(PolyLrUpdater, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner['epoch']
            max_progress = runner['max_epochs']
        else:
            progress = runner['iter']
            max_progress = runner['max_iters']
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr
    
class CosineAnnealingLrUpdater(LrUpdater):

    def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        super(CosineAnnealingLrUpdater, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.get('epoch')
            max_progress = runner.get('max_epochs')
        else:
            progress = runner.get('iter')
            max_progress = runner.get('max_iters')

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr
        return annealing_cos(base_lr, target_lr, progress / max_progress)

class CosineAnnealingCooldownLrUpdater(LrUpdater):
    """Cosine annealing learning rate scheduler with cooldown.

    Args:
        min_lr (float, optional): The minimum learning rate after annealing.
            Defaults to None.
        min_lr_ratio (float, optional): The minimum learning ratio after
            nnealing. Defaults to None.
        cool_down_ratio (float): The cooldown ratio. Defaults to 0.1.
        cool_down_time (int): The cooldown time. Defaults to 10.
        by_epoch (bool): If True, the learning rate changes epoch by epoch. If
            False, the learning rate changes iter by iter. Defaults to True.
        warmup (string, optional): Type of warmup used. It can be None (use no
            warmup), 'constant', 'linear' or 'exp'. Defaults to None.
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts. Defaults to 0.
        warmup_ratio (float): LR used at the beginning of warmup equals to
            ``warmup_ratio * initial_lr``. Defaults to 0.1.
        warmup_by_epoch (bool): If True, the ``warmup_iters``
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts. Defaults to False.

    Note:
        You need to set one and only one of ``min_lr`` and ``min_lr_ratio``.
    """

    def __init__(self,
                 min_lr=None,
                 min_lr_ratio=None,
                 cool_down_ratio=0.1,
                 cool_down_time=10,
                 **kwargs):
        assert (min_lr is None) ^ (min_lr_ratio is None)
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.cool_down_time = cool_down_time
        self.cool_down_ratio = cool_down_ratio
        super(CosineAnnealingCooldownLrUpdater, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.get('epoch')
            max_progress = runner.get('max_epochs')
        else:
            progress = runner.get('iter')
            max_progress = runner.get('max_iters')

        if self.min_lr_ratio is not None:
            target_lr = base_lr * self.min_lr_ratio
        else:
            target_lr = self.min_lr

        if progress > max_progress - self.cool_down_time:
            return target_lr * self.cool_down_ratio
        else:
            max_progress = max_progress - self.cool_down_time

        return annealing_cos(base_lr, target_lr, progress / max_progress)

def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out
