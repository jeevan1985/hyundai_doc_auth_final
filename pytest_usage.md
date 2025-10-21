
# Pytest Usage Guide

This guide provides a comprehensive overview of how to use the `pytest` test suite for this project to ensure code quality, prevent regressions, and maintain stability.

## 1. Introduction to the Test Suite

This project uses `pytest`, a powerful Python testing framework, to automate the process of verifying code correctness. The test suite is designed to cover critical components of the application through different types of tests:

- **Unit Tests:** These test the smallest pieces of code, like a single function or class, in isolation. Mocks are heavily used to simulate external dependencies (e.g., databases, APIs, libraries).
- **Integration Tests:** These test how different parts of the application work together. For example, verifying that the main `workflow` correctly orchestrates its various helper modules.
- **CLI Tests:** These test the main user entry point of the application (`doc_image_verifier.py`) to ensure it behaves as expected from a user's perspective.

## 2. Setup and Installation

Before running the tests, you must ensure your Conda environment has the necessary testing libraries installed.

1.  **Open your Conda-enabled terminal** (e.g., Anaconda Prompt) in the root directory of this project.
2.  **Update your environment** using the `environment.dev.yml` file. This will install `pytest`, `pytest-mock`, and `pytest-html`.

    ```bash
    conda env update --file environment.dev.yml --prune
    ```

## 3. Running Tests

You can run tests in several ways, from executing the entire suite to targeting a single test function.

### Running the Full Suite with an HTML Report (Recommended)

I have created a helper script, `run_tests.py`, that executes the entire test suite and generates a polished, single-file HTML report.

To use it, simply run:

```bash
python run_tests.py
```

Upon completion, a file named `report.html` will be created in the project root. You can open this file in any web browser to see a detailed, interactive report of the test results.

### Running from the Command Line

To run tests directly from the command line, use the `pytest` command.

- **Run all tests:**

  ```bash
  pytest
  ```

- **Run with verbose output** (shows each test function name and its status):

  ```bash
  pytest -v
  ```

- **Run tests in a specific file:**

  ```bash
  # Example: Run only the tests for QdrantManager
  pytest tests/core_engine/image_similarity_system/test_qdrant_manager.py
  ```

- **Run a specific test function** by name using the `-k` flag:

  ```bash
  # Example: Run only the test for shard rollover
  pytest -k "test_upsert_with_shard_rollover"
  ```

## 4. Understanding the Test Structure

- **Directory:** All tests are located in the top-level `tests/` directory. The structure within `tests/` mirrors the main `hyundai_document_authenticator/` source directory.
- **Fixtures (`@pytest.fixture`):** You will see functions decorated with `@pytest.fixture`. These are special setup functions that prepare reusable components for tests, such as configuration objects or mock services. Tests use them by simply accepting them as function arguments.
- **Mocks (`mocker`):** The `mocker` fixture (from `pytest-mock`) is used to replace parts of your code with mock objects during a test. This is crucial for isolating the code you want to test from its dependencies (e.g., testing database logic without a real database).

## 5. Writing a New Test

Follow these steps to add a new test for a module (e.g., for a new file `my_module.py`):

1.  **Create a Test File:** In the `tests/` directory, create a new file with a matching path and name it `test_my_module.py`.
2.  **Import Necessary Modules:** Import `pytest`, the module/function you want to test, and any other required libraries.
3.  **Write a Test Function:** Define a function whose name starts with `test_`.
4.  **Arrange, Act, Assert:** Structure your test function clearly:
    -   **Arrange:** Set up the initial state. This might involve creating objects or defining input data. Use fixtures for complex setup.
    -   **Act:** Call the function or method you want to test.
    -   **Assert:** Use the `assert` keyword to check if the outcome of the `Act` step is what you expected.

### Simple Test Example

```python
# In tests/test_my_module.py

from hyundai_document_authenticator.my_module import add_numbers

def test_add_numbers():
    # Arrange
    num1 = 5
    num2 = 10

    # Act
    result = add_numbers(num1, num2)

    # Assert
    assert result == 15
```

## 6. Advanced: Customizing Test Discovery

By default, `pytest` only discovers files named `test_*.py` or `*_test.py`. While it is highly recommended to stick to this convention, you can configure `pytest` to recognize other patterns.

To do this, create a `pytest.ini` file in the root of your project and add a `python_files` key. 

For example, to make `pytest` also discover files that start with `tests_`, you would create the following `pytest.ini`:

```ini
[pytest]
python_files = test_*.py *_test.py tests_*.py
```
