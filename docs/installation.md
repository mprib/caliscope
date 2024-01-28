# Installation

Caliscope can be installed easily using pip and launched from the command line.

## 1. Create a virtual environment

We strongly advise installing within a virtual environment. This approach helps in maintaining a clean workspace and avoids conflicts with other packages. Caliscope is compatible with Python versions [3.10](https://www.python.org/downloads/release/python-3100/) and [3.11](https://www.python.org/downloads/release/python-3110/). To avoid complications, we recommend you use the full file path to your Python executable. Here's how to do it for different operating systems:

=== "Windows"

    ```bash
    # Open Command Prompt and navigate to directory that will hold venv
    # this does not need to be the same as where your project workspace is held
    cd path\to\your\project

    # Create a virtual environment named 'env' using Python 3.10
    "C:\Path\To\Python3.10\python.exe" -m venv .venv

    # Activate the virtual environment
    .\env\Scripts\activate

    # Your virtual environment is now active.
    # You can install dependencies using pip
    ```

=== "macOS"

    ```bash
    # Open Command Prompt and navigate to directory that will hold venv
    # this does not need to be the same as where your project workspace is held
    cd path/to/your/project

    # Create a virtual environment named 'venv' using Python 3.10
    /path/to/python3.10 -m venv .venv

    # Activate the virtual environment
    source .venv/bin/activate

    # Your virtual environment is now active.
    # You can install dependencies using pip
    ```

=== "Linux"

    ```bash
    # Open Command Prompt and navigate to directory that will hold venv
    # this does not need to be the same as where your project workspace is held
    cd path/to/your/project

    # Create a virtual environment named 'env' using Python 3.10
    /path/to/python3.10 -m venv .venv

    # Activate the virtual environment
    source .venv/bin/activate

    # Your virtual environment is now active.
    # You can install dependencies using pip
    ```

## 2. Install Caliscope via pip

With your virtual environment activated, the next step is to install Caliscope itself. This is a straightforward process that can be done using pip, Python's package installer. Follow the instructions specific to your operating system below:


=== "Windows"

    ``` bash
    # Install Caliscope via pip
    pip install caliscope
    ```

=== "macOS"

    ``` bash
    # Install Caliscope via pip
    pip3 install caliscope
    ```

=== "Linux"

    ``` bash
    # Install Caliscope via pip
    pip3 install caliscope
    ```

Remember, installation may take a moment as some dependencies, like OpenCV and MediaPipe, are quite large. But don't worry, this is a one-time process, and you'll soon be ready to dive in.


## 3. Launch from the command line

With the package installed and the virtual environment activated, the main GUI can be launched by running the following command:

``` bash
caliscope
```

*Note on First Launch*: The first time you launch after installation, you might experience a longer than usual startup time. This is normal and expected as the application performs initial setup tasks like compiling components. Rest assured, these processes are one-time events, and subsequent launches of the GUI will be significantly faster.