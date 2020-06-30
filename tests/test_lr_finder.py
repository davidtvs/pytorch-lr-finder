import pytest
from torch_lr_finder import LRFinder

import task as mod_task

import matplotlib.pyplot as plt


def collect_task_classes():
    names = [v for v in dir(mod_task) if v.endswith("Task") and v != "BaseTask"]
    attrs = [getattr(mod_task, v) for v in names]
    classes = [v for v in attrs if issubclass(v, mod_task.BaseTask)]
    return classes


def prepare_lr_finder(task, **kwargs):
    model = task.model
    optimizer = task.optimizer
    criterion = task.criterion
    config = {
        "device": kwargs.get("device", None),
        "memory_cache": kwargs.get("memory_cache", True),
        "cache_dir": kwargs.get("cache_dir", None),
    }
    lr_finder = LRFinder(model, optimizer, criterion, **config)
    return lr_finder


def get_optim_lr(optimizer):
    return [grp["lr"] for grp in optimizer.param_groups]


class TestRangeTest:
    @pytest.mark.parametrize("cls_task", collect_task_classes())
    def test_run(self, cls_task):
        task = cls_task()
        init_lrs = get_optim_lr(task.optimizer)

        lr_finder = prepare_lr_finder(task)
        lr_finder.range_test(task.train_loader, end_lr=0.1)

        # check whether lr is actually changed
        assert max(lr_finder.history["lr"]) >= init_lrs[0]

    @pytest.mark.parametrize("cls_task", collect_task_classes())
    def test_run_with_val_loader(self, cls_task):
        task = cls_task(validate=True)
        init_lrs = get_optim_lr(task.optimizer)

        lr_finder = prepare_lr_finder(task)
        lr_finder.range_test(task.train_loader, val_loader=task.val_loader, end_lr=0.1)

        # check whether lr is actually changed
        assert max(lr_finder.history["lr"]) >= init_lrs[0]


class TestReset:
    @pytest.mark.parametrize(
        "cls_task", [mod_task.XORTask, mod_task.DiscriminativeLearningRateTask],
    )
    def test_reset(self, cls_task):
        task = cls_task()
        init_lrs = get_optim_lr(task.optimizer)

        lr_finder = prepare_lr_finder(task)
        lr_finder.range_test(task.train_loader, val_loader=task.val_loader, end_lr=0.1)
        lr_finder.reset()

        restored_lrs = get_optim_lr(task.optimizer)
        assert init_lrs == restored_lrs


class TestLRHistory:
    def test_linear_lr_history(self):
        task = mod_task.XORTask()
        # prepare_lr_finder sets the starting lr to 1e-5
        lr_finder = prepare_lr_finder(task)
        lr_finder.range_test(
            task.train_loader, num_iter=5, step_mode="linear", end_lr=5e-5
        )

        assert len(lr_finder.history["lr"]) == 5
        assert lr_finder.history["lr"] == pytest.approx([1e-5, 2e-5, 3e-5, 4e-5, 5e-5])

    def test_exponential_lr_history(self):
        task = mod_task.XORTask()
        # prepare_lr_finder sets the starting lr to 1e-5
        lr_finder = prepare_lr_finder(task)
        lr_finder.range_test(task.train_loader, num_iter=5, step_mode="exp", end_lr=0.1)

        assert len(lr_finder.history["lr"]) == 5
        assert lr_finder.history["lr"] == pytest.approx([1e-5, 1e-4, 1e-3, 1e-2, 0.1])


@pytest.mark.parametrize("num_iter", [0, 1])
@pytest.mark.parametrize("scheduler", ["exp", "linear"])
def test_scheduler_and_num_iter(num_iter, scheduler):
    task = mod_task.XORTask()
    # prepare_lr_finder sets the starting lr to 1e-5
    lr_finder = prepare_lr_finder(task)
    with pytest.raises(ValueError, match="num_iter"):
        lr_finder.range_test(
            task.train_loader, num_iter=num_iter, step_mode=scheduler, end_lr=5e-5
        )


@pytest.mark.parametrize("suggest_lr", [False, True])
@pytest.mark.parametrize("skip_start", [0, 5, 10])
@pytest.mark.parametrize("skip_end", [0, 5, 10])
def test_plot_with_skip_and_suggest_lr(suggest_lr, skip_start, skip_end):
    task = mod_task.XORTask()
    num_iter = 11
    # prepare_lr_finder sets the starting lr to 1e-5
    lr_finder = prepare_lr_finder(task)
    lr_finder.range_test(
            task.train_loader, num_iter=num_iter, step_mode="exp", end_lr=0.1
    )

    fig, ax = plt.subplots()
    results = lr_finder.plot(
            skip_start=skip_start, skip_end=skip_end, suggest_lr=suggest_lr, ax=ax
    )

    if num_iter - skip_start - skip_end <= 1:
        # handle data with one or zero lr
        assert len(ax.lines) == 1
        assert results is ax
    else:
        # handle different suggest_lr
        # for 'steepest': the point with steepest gradient (minimal gradient)
        assert len(ax.lines) == 1
        assert len(ax.collections) == int(suggest_lr)
        if results is not ax:
            assert len(results) == 2
