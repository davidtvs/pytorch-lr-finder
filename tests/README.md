## Requirements
- pytest

## Run tests
- normal (use GPU if it's available)
    ```bash
    # in root directory of this package
    $ python -mpytest ./tests
    ```

- forcibly run all tests on CPU
    ```bash
    # in root directory of this package
    $ python -mpytest --cpu_only ./tests
    ```

## How to add new test cases
To make it able to create test cases and re-use settings conveniently, here we package those basic elements for running a training task into objects inheriting `BaseTask` in `task.py`.

A `BaseTask` is formed of these members:
- `batch_size`
- `model`
- `optimizer`
- `criterion` (loss function)
- `device` (`cpu`, `cuda`, etc.)
- `train_loader` (`torch.utils.data.DataLoader` for training set)
- `val_loader` (`torch.utils.data.DataLoader` for validation set)

If you want to create a new task, just write a new class inheriting `BaseTask` and add your configuration in `__init__`.

Note-1: Any task inheriting `BaseTask` in `task.py` will be collected by the function `test_lr_finder.py::collect_task_classes()`.

Note-2: Model and dataset will be instantiated when a task class is **initialized**, so that it is not recommended to collect a lot of task **objects** at once.


### Directly use specific task in a test case
```python
from . import task as mod_task
def test_run():
    task = mod_task.FooTask()
    ...
```

### Use `pytest.mark.parametrize`
- Use specified task in a test case
    ```python
    @pytest.mark.parametrize(
        'cls_task, arg',  # names of parameters (see also the signature of the following function)
        [
            (task.FooTask, 'foo'),
            (task.BarTask, 'bar'),
        ],  # list of parameters
    )
    def test_run(cls_task, arg):
        ...
    ```

- Use all existing tasks in a test case
    ```python
    @pytest.mark.parametrize(
        'cls_task',
        collect_task_classes(),
    )
    def test_run(cls_task):
        ...
    ```
