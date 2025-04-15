Models
======

CAP provides several state-of-the-art time series forecasting models. This section documents the available models and their configurations.

Transformer
----------

.. autoclass:: cap.models.Transformer
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   .. code-block:: python

       from cap import Transformer

       model = Transformer(
           input_size=1,
           output_size=1,
           d_model=512,
           nhead=8,
           num_layers=3,
           dropout=0.1
       )

FEDFormer
---------

.. autoclass:: cap.models.FEDFormer
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   .. code-block:: python

       from cap import FEDFormer

       model = FEDFormer(
           input_size=1,
           output_size=1,
           d_model=512,
           nhead=8,
           num_layers=3,
           dropout=0.1
       )

Autoformer
---------

.. autoclass:: cap.models.Autoformer
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   .. code-block:: python

       from cap import Autoformer

       model = Autoformer(
           input_size=1,
           output_size=1,
           d_model=512,
           nhead=8,
           num_layers=3,
           dropout=0.1
       )

TimesNet
--------

.. autoclass:: cap.models.TimesNet
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   .. code-block:: python

       from cap import TimesNet

       model = TimesNet(
           input_size=1,
           output_size=1,
           d_model=512,
           nhead=8,
           num_layers=3,
           dropout=0.1
       )

Informer
--------

.. autoclass:: cap.models.Informer
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   .. code-block:: python

       from cap import Informer

       model = Informer(
           input_size=1,
           output_size=1,
           d_model=512,
           nhead=8,
           num_layers=3,
           dropout=0.1
       )

LSTM
----

.. autoclass:: cap.models.LSTM
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   .. code-block:: python

       from cap import LSTM

       model = LSTM(
           input_size=1,
           output_size=1,
           hidden_size=128,
           num_layers=2,
           dropout=0.1
       ) 