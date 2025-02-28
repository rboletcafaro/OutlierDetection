import dash
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np

from data_processing import parse_contents, preprocess_data
from models import detect_outliers_simple, detect_outliers_balanced, detect_outliers_complex
from visualization import create_scatter_plot, create_time_series_plot

def register_callbacks(app):
    """Register all Dash callbacks for interactive functionality."""
    @app.callback(
        [Output('outlier-graph', 'figure'), Output('outlier-summary', 'children')],
        [Input('run-button', 'n_clicks')],
        [State('upload-data', 'contents'), State('upload-data', 'filename'), State('mode-dropdown', 'value')]
    )
    def update_outlier_output(n_clicks, contents, filename, mode):
        """Callback to process uploaded file and run outlier detection when button is clicked."""
        if n_clicks is None or n_clicks == 0:
            # No button clicks yet, do nothing
            return dash.no_update, dash.no_update
        if contents is None:
            return dash.no_update, "No file uploaded. Please upload a CSV file."
        # Parse the uploaded file contents into a DataFrame
        df = parse_contents(contents, filename)
        if df is None or df.empty:
            return dash.no_update, "Error: Unable to read the uploaded file."
        # Identify a time column if present (for time-series data)
        time_series = None
        time_col = None
        for col in df.columns:
            # Check if column is datetime type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_col = col
                break
            # If column name indicates time/date, attempt to parse it
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    time_col = col
                    break
                except Exception:
                    continue
        if time_col:
            # Sort by time and separate the time series for visualization
            df = df.sort_values(time_col)
            time_series = df[time_col].reset_index(drop=True)
            df = df.drop(columns=[time_col])
        # Preprocess data (handle missing values, scale features, etc.)
        is_time_series = time_series is not None
        df_processed, X = preprocess_data(df, time_series=is_time_series)
        if X is None or X.shape[1] == 0:
            return dash.no_update, "No numeric features available for outlier detection."
        # Run the selected outlier detection mode
        if mode == 'simple':
            outlier_mask = detect_outliers_simple(X)
        elif mode == 'balanced':
            outlier_mask = detect_outliers_balanced(X)
        else:
            outlier_mask = detect_outliers_complex(X)
        outlier_mask = np.asarray(outlier_mask, dtype=bool)  # ensure boolean array
        # Generate appropriate visualization based on data dimensionality
        if X.shape[1] == 1:
            # Single-feature data
            y_values = df_processed.iloc[:, 0]  # the single column of data
            if time_series is not None:
                # Time-series line plot
                fig = create_time_series_plot(time_series, y_values, outlier_mask,
                                              title="Time Series Anomaly Detection")
            else:
                # Single feature with no time axis: use index as x-axis
                x_index = np.arange(len(y_values))
                fig = create_time_series_plot(x_index, y_values, outlier_mask,
                                              title="Outlier Detection (Single Feature)")
        else:
            # Multi-dimensional data: use scatter plot (with PCA if >2 features)
            fig = create_scatter_plot(df_processed, outlier_mask, title="Outlier Detection Scatter Plot")
        # Prepare summary text about detected outliers
        num_outliers = int(np.sum(outlier_mask))
        total_points = X.shape[0]
        summary_text = f"Detected {num_outliers} outliers out of {total_points} data points."
        return fig, summary_text
