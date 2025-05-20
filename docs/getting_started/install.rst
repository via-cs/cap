.. _install:

.. highlight:: shell

Installation
============

Requirements
------------

Python
~~~~~~
**CAP** has been tested on **GNU/Linux**, **macOS**, and **Windows** systems running `Python 3.8, 3.9, 3.10, or 3.11`_.

Also, although it is not strictly required, the usage of a `virtualenv`_ is highly recommended in
order to avoid having conflicts with other software installed in the system where you are trying to run **CAP**.

Dependencies
~~~~~~~~~~~
CAP requires the following main dependencies:
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- Pandas >= 1.2.0
- Scikit-learn >= 0.24.0
- Matplotlib >= 3.3.0

Install using pip
----------------

The easiest and recommended way to install **CAP** is using `pip`_:

.. code-block:: console

    pip install cap

This will pull and install the latest stable release from `PyPI`_.

Install from source
------------------

The source code of **CAP** can be downloaded from the `Github repository`_.

You can clone the repository and install it from source by running:

.. code-block:: console

    git clone https://github.com/via-cs/cap.git
    cd cap
    pip install -e .

For development installation, including additional dependencies for testing and documentation:

.. code-block:: console

    pip install -e ".[dev]"

.. note:: The ``main`` branch of the CAP repository contains the latest development version. If you want to install the latest stable version, make sure to use the latest release tag.

GPU Support
----------

CAP supports GPU acceleration through PyTorch. To enable GPU support:

1. Install CUDA toolkit (if not already installed)
2. Install the CUDA-enabled version of PyTorch:

.. code-block:: console

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Replace ``cu118`` with your CUDA version if different.

.. _Python 3.8, 3.9, 3.10, or 3.11: https://docs.python-guide.org/starting/installation/
.. _virtualenv: https://virtualenv.pypa.io/en/latest/
.. _pip: https://pip.pypa.io
.. _PyPI: https://pypi.org/
.. _Github repository: https://github.com/via-cs/cap
