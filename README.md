# mikibox - my library with various tools for data analysis.

For list of modules and their descriptions see documentation at: 
https://mstekiel.github.io/mikibox/build/html/index.html

Installation
1. Create an empty virtual environment with the python version specified in `setup.py: python-version`
2. Install `mikibox` locally with `python setup.py develop` or `pip install -e .`. The `develop`/`-e` option allows to modify the source files and adapt to current needs. If you want to install extra packages for full development utilities do `pip install -e .[dev]`.

Installation FAQ:
- For a lightweight, fast installation miniconda is a great solution. A throwback is, its package manager conda or pip will nominally install the newest version of packages from their channels, which may produce problems. A good solution is to check the list of packages installed by Anaconda for your platform (https://docs.anaconda.com/anaconda/packages/pkg-docs/) and follow the version of packages from there. Otherwise, one might need to carefully test which version of each packages are compatible with each other.
- setup.py file should contain the compatible combination of packages.
- Install with `python setup.py develop` in development mode, so that all changes in the source code will work for future local scripts.
- One can also just download all source files from this Github project to a location `LOC`, and in each python file include lines
  > import sys
  >
  > sys.path.append(LOC)

  Import mikibox and continue without installation.

- numpy==1.23.1 requires python>3.10 I believe. It didnt want to work with python==3.8
- matplotlib==3.6 is not in conda repository as of 19.10.2022, but 3.5.2 is and works well.

General notes:
- I tried to follow the structure of https://github.com/pypa/sampleproject for the development of this project. Following descriptions from https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html
