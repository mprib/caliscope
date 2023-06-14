# Installation

---

## Highlights

- Pyxy3D is installable via pip and the GUI can be launched from the command line. 
- It is **strongly** advised that you do so within a virtual environment. 
- The package requires [Python 3.10](https://www.python.org/downloads/release/python-3100/)  or higher. 
- Because the Mediapipe implementation only works on Windows currently, these steps assume you are installing on Windows 10.

---


## 1. Create a virtual environment

Find the path to your python.exe file. You can install Python 3.10 from [here](https://www.python.org/downloads/release/python-3100/). For me the path is `C:\Python310\python.exe`

Create a folder where you would like the virtual environment to live. This can be different from the folder where your motion capture calibration and recording data is stored.
   
Launch a terminal (on Windows, search 'powershell' in the start menu and launch that).
Run the following at the command prompt, substituting in the path to `python.exe` that is true for your machine
```
C:\Python310\python.exe -m venv .venv
```

This will create a fresh version of python within that folder which you will use to manage your project. Activate the environment using the following command (if this exact command doesn't work, then [some other variation will](https://docs.python.org/3/library/venv.html#how-venvs-work))
```
.\.venv\Scripts\activate
```

The terminal should now show the environment is activated with something like this green parenthetical:

![Pasted image 20230608095719](https://github.com/mprib/pyxy3d/assets/31831778/10a91524-9a81-41d1-b27b-0b6ba723cb27)

You can confirm that your python path is set by running

```
python -c "import sys; print(sys.executable)"
```
which should point to the file in the virtual environment you created:

![Pasted image 20230608100059](https://github.com/mprib/pyxy3d/assets/31831778/e214ebae-692c-4b50-b6f4-f34dcb44df43)

---

## 2. Install pyxy3D via pip

You are now ready to install pyxy3D from the Python Package Index (PyPI) via pip:

```
pip install pyxy3d
```

Installation may take a moment...

---

## 3. Launch from the command line
With the package installed and the virtual environment activated, the main GUI can be launched by running the following command to launch the tool:

```
pyxy3d
```

If you experience crashes after initializing the session folder, then you can launch the individual interface components one at a time as needed. **NAVIGATE TO THE FOLDER OF THE SESSION YOU WANT TO LAUNCH** and run one of the following as needed: `charuco`, `cameras`, `calibrate`, `record`, `process`

For example, if you are getting crashes when trying to record, within the terminal navigate to the session folder you previously created and run:

```
pyxy3d record
```

A recording widget will open up that should be more efficient and stable than the complete GUI.