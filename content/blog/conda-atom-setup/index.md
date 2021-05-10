---
title: 'Setting up a Python + Markdown environment with Atom'
subtitle: 'Step by step guide to set up a Python virtual environment with Anaconda for interactive programming and document creation'

summary: Step by step guide to set up a Python virtual environment with Anaconda for interactive programming and document creation

authors: []

tags:
- Atom
- Anaconda

categories:
- Atom
- Anaconda


date: "2021-02-01"

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Placement options: 1 = Full column width, 2 = Out-set, 3 = Screen-width
# Focal point options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight

image:
  placement: 2
  caption: ''
  focal_point: Smart
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---


:warning: The set up is based on my personal preferences at 04/02/2021.

## 1. Atom Setup from scratch

First, download atom from https://atom.io/.


### Activate Toogle soft wrap:
```console
click View >> Toggle Soft Wrap
```


### List of packages to install:

1. file-icons
2. hydrogen
3. data-explorer
4. platformio-ide-terminal
5. multi-cursor-plus
6. language-markdown and markdown-preview-plus
7. tblr
8. python-indent
9. atom-python-virtualenv

To install all the packages in one batch:
```
apm install hydrogen file-icons data-explorer platformio-ide-terminal multi-cursor-plus language-markdown markdown-preview-plus https://github.com/mfripp/atom-tablr.git python-indent atom-python-virtualenv
```

### What they do:

1. [File-icons](https://atom.io/packages/file-icons)

Display pretty icons to easy file reading.

<p align="center">
<img width="800" height="400" src="https://curtistimson.co.uk/images/post/atom/file-icons-example.png">
</p>

2. [Hydrogen](https://atom.io/packages/hydrogen)

Interactive Python (Python + an interactive shell).

<p align="center">
<img width="800" height="400" src="https://cloud.githubusercontent.com/assets/13285808/20360886/7e03e524-ac03-11e6-9176-37677f226619.gif">
</p>



3. [Data explorer](https://github.com/BenRussert/data-explorer)

It provides interactive data exploration of plots and databases.

<p align="center">
<img width="800" height="400" src="https://user-images.githubusercontent.com/10860657/60221354-93647100-9836-11e9-930e-a18161c31964.gif">
</p>


4. [platformio-ide-terminal](https://github.com/platformio/platformio-atom-ide-terminal)

Terminal/cmd window inside atom.

<p align="center">
<img width="500" height="400" src="https://raw.githubusercontent.com/platformio/platformio-atom-ide-terminal/master/resources/demo.gif">
</p>

5.[multi-cursor-plus](https://atom.io/packages/multi-cursor-plus)

Multicursor that supports multiple selections and removing previous cursors at any time.

<p align="center">
<img width="800" height="400" src="https://raw.githubusercontent.com/kankaristo/atom-multi-cursor-plus/gif/showcase.gif">
</p>


6. [Markdown language and compiler](https://atom.io/packages/language-markdown)

Markdown syntax inside Atom, displayed with a math-friendly complier.

<p align="center">
<img width="800" height="400" src="https://raw.githubusercontent.com/atom-community/markdown-preview-plus/master/imgs/mpp-full-res-invert.png">
</p>

Use:

- Make sure you checked *Use GitHub&#46;com style* in the package settings
- Toggle Preview: <code>ctrl-shift-m</code>
- Toggle Math Rendering: <code>ctrl-shift-x</code>

7. [tblr](https://github.com/mfripp/atom-tablr)

Table visualizer and editor of CSV files.

<p align="center">
<img width="800" height="400" src="https://camo.githubusercontent.com/3d7daee9b21cb283b79a5f30d2661d7591726051f718484c8c3dcd4e190765ac/687474703a2f2f61626533332e6769746875622e696f2f61746f6d2d7461626c722f7461626c722e676966">
</p>

8. [python-indent](https://atom.io/packages/python-indent)

Automatically applies PEP8 indentation.

<p align="center">
<img width="800" height="400" src="https://raw.githubusercontent.com/DSpeckhals/python-indent/master/resources/img/python-indent-demonstration.gif">
</p>

9. [atom-python-virtualenv](https://atom.io/packages/atom-python-virtualenv)

Allows to select the virtual environment connected to Atom.

<p align="center">
<img width="800" height="400" src="https://cloud.githubusercontent.com/assets/1611808/21472334/671a0614-cabb-11e6-9b33-3ba1459ca072.png">
</p>

Use:
- Select virtual environment: <code>Ctrl+Alt+V</code>



## 2. Create a specifict Virtual environment with Anaconda

A virtual environment is necessary for reproducibility reasons & dependency conflicts. Conda allows to control dependencies and conflicts, since it keeps tracks of possible conflicts.

Download Anaconda at https://www.anaconda.com/products/individual or Miniconda at https://docs.conda.io/en/latest/miniconda.html. In essence:

- Miniconda installer = Python + conda
- Anaconda installer = Python + conda + meta package anaconda

What it is important is conda package manager, which is in both options. The main difference is <code>meta package anaconda</code>, a collection of scientific packages automatically installed with the distribution. Any of the the options work, being Miniconda a more minimalist installation. I personally find it more tedious, but it depends on your tastes.

 Once installed, run on terminal:

```console
conda create --name <YourEnvNameHere> python=3.6
conda activate <YourEnvNameHere>
conda install -n <YourEnvNameHere> <whateverpackage_1> <whateverpackage_2>
```

Chose the python version according to the purpose of the project packages used, being 3.6 to 3.8 the most common at the time.

To connect with the interactive Atom setup, install a Python kernel ([iPython](https://ipython.org/)) by running the following in the respective virtual environment ([Source](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)):

```
conda install ipykernel jupyter
python -m ipykernel install --user --name <YourEnvNameHeRE>
```

## 3. Example seting up a XGBoost project

XGBoost is kind of [tricky to install](https://xgboost.readthedocs.io/en/latest/build.html) without using global pip. However, with conda it cannot be easier and quicker to get it running.

```console
# Create virtual environment
conda create --name xgb_test python=3.6

# Activate the virtual environment
conda activate xgb_test

# Install Ipykernel for the environment
conda install ipykernel jupyter
python -m ipykernel install --user --name xgb_test

# Install GPU-enabled XGBoost conda package
conda install -n xgb_test -c nvidia -c rapidsai py-xgboost
```

It can be checked if the environment has the installed Python and the package, by typing *python* and importing the package *xgboost* without errors:

<p align="center">
<img width="800" height="80" src="https://i.imgur.com/b7puJuq.png">
</p>


Run atom from the environment:

<p align="center">
<img src="https://i.imgur.com/D7nS1M3.png">
</p>


It should pop-up Atom in the desktop. Create a file with a py extension (e.g. *test&#46;py*, so the python kernel is activated. Otherwise you will see a *No kernel for Null grammar found*. All the packages installed should work:

<p align="center">
<img src="https://i.imgur.com/TpduDdu.png">
</p>

Ready to go.
