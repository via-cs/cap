Evaluation
==========

This guide explains how to evaluate your time series forecasting models using CAP's evaluation tools.

Overview
--------

CAP provides comprehensive evaluation capabilities:

1. **Point Forecast Metrics**: MAE, MSE, RMSE, MAPE, etc.
2. **Interval Forecast Metrics**: PICP, MIS, etc.
3. **Visualization Tools**: Forecast plots, error analysis, etc.

Basic Evaluation
--------------

Evaluate your model's predictions:

.. code-block:: python

    from cap.metrics import evaluate_forecast

    # Evaluate predictions
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

Point Forecast Metrics
~~~~~~~~~~~~~~~~~~~~

1. **Mean Absolute Error (MAE)**:
   - Measures average absolute difference between predictions and actual values
   - Less sensitive to outliers than MSE
   - Same unit as the target variable

2. **Mean Squared Error (MSE)**:
   - Measures average squared difference between predictions and actual values
   - More sensitive to outliers than MAE
   - Square of the target variable's unit

3. **Root Mean Squared Error (RMSE)**:
   - Square root of MSE
   - Same unit as the target variable
   - More interpretable than MSE

4. **Mean Absolute Percentage Error (MAPE)**:
   - Measures average absolute percentage difference
   - Scale-independent
   - Can be misleading when actual values are close to zero

5. **Symmetric Mean Absolute Percentage Error (SMAPE)**:
   - Similar to MAPE but symmetric
   - Scale-independent
   - More robust than MAPE

6. **R-squared Score**:
   - Measures proportion of variance explained by the model
   - Scale-independent
   - Range: (-∞, 1], higher is better

Interval Forecast Metrics
~~~~~~~~~~~~~~~~~~~~~~

1. **Prediction Interval Coverage Probability (PICP)**:
   - Measures proportion of actual values within prediction intervals
   - Range: [0, 1], closer to nominal coverage is better

2. **Mean Interval Score (MIS)**:
   - Combines interval width and coverage
   - Lower values are better
   - Penalizes both narrow intervals and poor coverage

Visualization
-----------

Plot Forecast Results
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from cap.metrics import plot_forecast

    # Basic forecast plot
    plot_forecast(
        actual=actual_data,
        predicted=predicted_data,
        title='Forecast Results'
    )

    # Plot with confidence intervals
    plot_forecast(
        actual=actual_data,
        predicted=predicted_data,
        confidence_intervals=confidence_intervals,
        title='Forecast with Confidence Intervals'
    )

Error Analysis
~~~~~~~~~~~~

.. code-block:: python

    from cap.metrics import plot_errors

    # Plot error distribution
    plot_errors(
        actual=actual_data,
        predicted=predicted_data,
        title='Error Distribution'
    )

    # Plot error over time
    plot_errors(
        actual=actual_data,
        predicted=predicted_data,
        plot_type='time',
        title='Errors Over Time'
    )

Advanced Evaluation
-----------------

Cross-Validation
~~~~~~~~~~~~~~

.. code-block:: python

    from cap.metrics import time_series_cv

    # Perform time series cross-validation
    cv_results = time_series_cv(
        model=model,
        data=data,
        n_splits=5,
        test_size=24,
        metrics=['mae', 'rmse']
    )

    print("Cross-Validation Results:")
    for metric, values in cv_results.items():
        print(f"{metric}: {values.mean():.4f} ± {values.std():.4f}")

Statistical Tests
~~~~~~~~~~~~~~~

.. code-block:: python

    from cap.metrics import statistical_tests

    # Perform statistical tests
    test_results = statistical_tests(
        actual=actual_data,
        predicted=predicted_data,
        tests=['stationarity', 'normality', 'autocorrelation']
    )

    print("Statistical Test Results:")
    for test, result in test_results.items():
        print(f"{test}: {result}")

Best Practices
------------

1. **Metric Selection**:
   - Choose metrics appropriate for your use case
   - Consider multiple metrics for comprehensive evaluation
   - Use scale-independent metrics when comparing across datasets

2. **Evaluation Strategy**:
   - Use proper train/validation/test splits
   - Consider time series cross-validation
   - Evaluate on multiple test sets if possible

3. **Visualization**:
   - Always visualize forecasts and errors
   - Look for patterns in errors
   - Consider seasonal and trend components

4. **Statistical Analysis**:
   - Check for stationarity
   - Analyze error distributions
   - Test for autocorrelation

For more detailed information about evaluation functions, see the :ref:`api_reference` section. 