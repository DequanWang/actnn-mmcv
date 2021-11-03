# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from .hook import HOOKS, Hook


@HOOKS.register_module()
class ActNNHook(Hook):
    """.
    This hook will call actnn controller.iterate, which is used to update auto precision
    Args:
        actnn (bool): If actnn is enabled
        interval (int): Update interval (every k iterations)
    """

    # def __init__(self, interval=1, default_bit=4, auto_prec=False):
    #     self.interval = interval
    #     self.controller = actnn.controller.Controller(
    #         default_bit=default_bit, auto_prec=auto_prec)
    # TODO: move with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook): here 

    def __init__(self, interval=1):
        self.interval = interval

    def before_train_epoch(self, runner):
        runner.controller.unrelated_tensors = set()
        runner.controller.filter_tensors(runner.model.named_parameters())

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            model = (
                runner.model.module if is_module_wrapper(
                    runner.model) else runner.model
            )
            runner.controller.iterate(model)
