# Installation

We recommend installing Caliscope using `uv`, which is a modern, high-performance tool that simplifies and accelerates the installation process.

## 1. Check for `uv`

First, check if you already have `uv` installed by running the following command in your terminal:

```bash
uv --version
```

If a version number is printed (e.g., `uv 0.8.5`), you can skip to step 3. If you see an error that the command is not found, please proceed to the next step.

## 2. Install `uv` (if needed)

If you don't have `uv` installed, you can install it with a single command.

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "macOS & Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

## 3. Create a virtual environment

We strongly advise installing Caliscope within a virtual environment to avoid conflicts with other packages. Caliscope is compatible with Python 3.10 and 3.11.

=== "Windows"

    ```bash
    # Navigate to the directory that will hold your project
    cd path\to\your\project

    # Create a virtual environment named '.venv' using Python 3.10
    uv venv --python 3.11

    # Activate the virtual environment
    .\.venv\Scripts\activate
    ```

=== "macOS"

    ```bash
    # Note that there are some environment variables that must be set on MacOS
    # to ensure everything works:
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    # Navigate to the directory that will hold your project
    cd path/to/your/project

    # Create a virtual environment named '.venv' using Python 3.10
    uv venv --python 3.11

    # Activate the virtual environment
    source .venv/bin/activate
    ```

=== "Linux"

    ```bash
    # Install prerequisite packages for GUI display (Ubuntu)
    sudo apt-get update
    sudo apt-get install --fix-missing libgl1-mesa-dev

    # Navigate to the directory that will hold your project
    cd path/to/your/project

    # Create a virtual environment named '.venv' using Python 3.10
    uv venv --python 3.11

    # Activate the virtual environment
    source .venv/bin/activate
    ```

## 4. Install Caliscope

With your virtual environment activated, you can now install Caliscope using `uv`.

```bash
uv pip install caliscope
```

Installation may take a moment as some dependencies are large, but `uv`'s performance makes this process significantly faster than traditional tools.

## 5. Launch from the command line

With the package installed and the virtual environment activated, the main GUI can be launched by running:

```bash
caliscope
```

*Note on First Launch*: The first time you launch after installation, you might experience a longer startup time. This is normal as the application performs initial setup tasks. Subsequent launches will be significantly faster.
