.. raw:: html

   <p align="center">
   <img width=30% src="images/cap_logo.png" alt="CAP" />
   </p>

|Development Status| |PyPi Shield| |Run Tests Shield|
|Downloads| |License|

CAP: Time Series Forecasting Framework
====================================

**Date**: |today| **Version**: |version|

-  License: `MIT <https://github.com/via-cs/cap/blob/master/LICENSE>`__
-  Development Status: `Alpha <https://pypi.org/search/?c=Development+Status+%3A%3A+3+-+Alpha>`__
-  Homepage: https://github.com/via-cs/cap
-  Documentation: https://via-cs.github.io/cap

Overview
--------

CAP is a comprehensive Python package for time series forecasting that implements various state-of-the-art models. The framework is designed to provide researchers and practitioners with a unified interface for experimenting with different forecasting approaches.

Key Features
-----------

* **Multiple Model Support**: Implements various state-of-the-art models:
    - Transformer
    - FEDFormer
    - Autoformer
    - TimesNet
    - Informer
    - LSTM

* **Easy-to-Use Interface**: 
    - Simple Python API
    - Command-line interface
    - YAML-based configuration

* **Flexible Configuration**:
    - Customizable model parameters
    - Configurable training settings
    - Extensible architecture

* **Comprehensive Documentation**:
    - Detailed API reference
    - Usage examples
    - Best practices

Getting Started
-------------

To get started with CAP, you can:

1. Install the package:
   .. code-block:: bash

      pip install cap

2. Use it in your Python code:
   .. code-block:: python

      from cap import Transformer
      model = Transformer()
      model.load_data("your_data.csv")
      predictions = model.predict()

3. Or use the command-line interface:
   .. code-block:: bash

      cap transformer --data input.csv --config config.yaml --output predictions.csv

Documentation Sections
--------------------

* `Getting Started <getting_started/index.html>`_
* `User Guides <user_guides/index.html>`_
* `API Reference <api_reference/index.html>`_
* `Developer Guides <developer_guides/index.html>`_
* `Release Notes <history.html>`_

--------------

.. |Development Status| image:: https://img.shields.io/badge/Development%20Status-3%20--%20Alpha-yellow
   :target: https://pypi.org/search/?c=Development+Status+%3A%3A+3+-+Alpha
.. |PyPi Shield| image:: https://img.shields.io/pypi/v/cap.svg
   :target: https://pypi.python.org/pypi/cap
.. |Run Tests Shield| image:: https://github.com/via-cs/cap/workflows/Run%20Tests/badge.svg
   :target: https://github.com/via-cs/cap/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster
.. |Downloads| image:: https://pepy.tech/badge/cap
   :target: https://pepy.tech/project/cap
.. |License| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT

.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    getting_started/index
    user_guides/index
    api_reference/index
    developer_guides/index
    Release Notes <history>
