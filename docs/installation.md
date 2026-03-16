# Installation

We recommend installing Caliscope using `uv`, a fast Python package manager that handles virtual environments and dependency resolution.

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

We strongly advise installing Caliscope within a virtual environment to avoid conflicts with other packages. Caliscope is compatible with Python 3.10 through 3.13.

=== "Windows"

    ```bash
    # Navigate to the directory that will hold your project
    cd path\to\your\project

    # Create a virtual environment using Python 3.12
    uv venv --python 3.12

    # Activate the virtual environment
    .\.venv\Scripts\activate
    ```

=== "macOS"

    ```bash
    # Navigate to the directory that will hold your project
    cd path/to/your/project

    # Create a virtual environment using Python 3.12
    uv venv --python 3.12

    # Activate the virtual environment
    source .venv/bin/activate
    ```

=== "Linux"

    ```bash
    # Install prerequisite packages for GUI display (Ubuntu/Debian)
    sudo apt-get update
    sudo apt-get install --fix-missing libgl1-mesa-dev

    # Navigate to the directory that will hold your project
    cd path/to/your/project

    # Create a virtual environment using Python 3.12
    uv venv --python 3.12

    # Activate the virtual environment
    source .venv/bin/activate
    ```

## 4. Install Caliscope

With your virtual environment activated, you can now install Caliscope using `uv`.

For the **desktop app** (GUI with 3D visualization):

```bash
uv pip install caliscope[gui]
```

For **scripting and library use** (no GUI dependencies):

```bash
uv pip install caliscope
```

The standard install includes the calibration pipeline (intrinsic and extrinsic) and the scripting API (`caliscope.api`) — everything except the PySide6 desktop interface and 3D visualization.

Installation may take a moment as some dependencies are large.

ONNX model inference via `onnxruntime` is included in both install targets. See [Custom ONNX Trackers](onnx_trackers.md) for details.

## 5. Launch from the command line

With the package installed and the virtual environment activated, the main GUI can be launched by running:

```bash
caliscope
```
