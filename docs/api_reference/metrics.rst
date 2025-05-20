Evaluation Metrics
================

CAP provides various metrics for evaluating time series forecasting models. This section documents the available metrics and their usage.

Forecast Metrics
--------------

.. autofunction:: cap.metrics.evaluate_forecast
   :noindex:

   .. rubric:: Example

   .. code-block:: python

       from cap.metrics import evaluate_forecast

       # Evaluate model predictions
       metrics = evaluate_forecast(
           actual=actual_data,
           predicted=predicted_data,
           metrics=['mae', 'mse', 'rmse', 'mape']
       )

       print("Model Performance:")
       for metric, value in metrics.items():
           print(f"{metric}: {value:.4f}")

Available Metrics
---------------

Mean Absolute Error (MAE)
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cap.metrics.mae
   :noindex:

Mean Squared Error (MSE)
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cap.metrics.mse
   :noindex:

Root Mean Squared Error (RMSE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cap.metrics.rmse
   :noindex:

Mean Absolute Percentage Error (MAPE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cap.metrics.mape
   :noindex:

Symmetric Mean Absolute Percentage Error (SMAPE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cap.metrics.smape
   :noindex:

R-squared Score
~~~~~~~~~~~~~

.. autofunction:: cap.metrics.r2_score
   :noindex:

Confidence Interval Metrics
-------------------------

Prediction Interval Coverage Probability (PICP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cap.metrics.picp
   :noindex:

Mean Interval Score (MIS)
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cap.metrics.mis
   :noindex:

Visualization
-----------

.. autofunction:: cap.metrics.plot_forecast
   :noindex:

   .. rubric:: Example

   .. code-block:: python

       from cap.metrics import plot_forecast

       # Plot forecast results
       plot_forecast(
           actual=actual_data,
           predicted=predicted_data,
           confidence_intervals=confidence_intervals,
           title='Forecast Results'
       ) 