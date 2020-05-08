# Contributing

All contributions are welcome but make sure you read this first before making a PR.

1. Generally there are two contribution categories:
   - Bug-fixing: please search for the issue [here](https://github.com/davidtvs/pytorch-lr-finder/issues) first (including the closed issues). If you cannot find it feel free to create an issue so we can discuss it before making the PR. If there's already an issue, read through the discussion and propose the bug fix there before making the PR.
   - New features: making the proposal first as an issue is recommended but you can also go directly for the PR.
2. Write the code in your forked repository. Consider the following:
   1. Please follow the instruction in [Development setup](#development-setup).
   2. Make sure any class or function you create or change has proper docstrings. Look at the existing code for examples on how they are formatted.
   3. Write tests.
   4. Run the tests locally and make sure everything is passing.
   5. Try your best not to introduce more flake8 warnings.
3. Use descriptive commit messages.
4. Submit your PR.


## Development setup

A virtual environment or other forms of environment isolation are highly recommended before you continue.

- Installing torch-lr-finder in development mode:
    ```sh
    pip install -e .[dev]
    ```
    Note that the command above will also install all the required dependencies.
- To run the tests, run the following command from the root directory of the repository or within the `tests` folder:
    ```sh
    pytest
    ```
- Use black to automatically format your code - if you need help, check the documentation in [black's github repository](https://github.com/psf/black).
- Use flake8 to lint your code - if you need help, check [flake8's documentation](https://flake8.pycqa.org/en/latest/index.html).
