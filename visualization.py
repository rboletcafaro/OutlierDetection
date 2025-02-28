import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def create_scatter_plot(df, outlier_mask, title="Outlier Scatter Plot"):
    """
    Create a 2D scatter plot of the data with outliers highlighted.
    - If df has more than 2 features, reduce to 2 principal components for visualization.
    - If df has 2 features, plot them on x and y axes.
    - Points identified as outliers are colored in red, normal points in cyan.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    dims = numeric_df.shape[1]
    if dims == 0:
        # No numeric data to plot
        return go.Figure()
    if dims > 2:
        # Reduce to 2 dimensions using PCA for visualization
        X_scaled = StandardScaler().fit_transform(numeric_df.values)
        components = PCA(n_components=2).fit_transform(X_scaled)
        plot_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
        # Label points as 'Outlier' or 'Normal'
        plot_df['PointType'] = np.where(outlier_mask, 'Outlier', 'Normal')
        fig = px.scatter(plot_df, x='PC1', y='PC2', color='PointType',
                         color_discrete_map={'Normal': 'cyan', 'Outlier': 'red'},
                         title=title)
        fig.update_xaxes(title_text="PC1")
        fig.update_yaxes(title_text="PC2")
    else:
        # Use existing features for 1D or 2D data
        plot_df = numeric_df.copy().reset_index(drop=True)
        if dims == 2:
            x_col, y_col = plot_df.columns[0], plot_df.columns[1]
        else:
            # One-dimensional data: use index as X and value as Y
            x_col = 'Index'
            y_col = numeric_df.columns[0]
            plot_df[x_col] = plot_df.index
        plot_df['PointType'] = np.where(outlier_mask, 'Outlier', 'Normal')
        fig = px.scatter(plot_df, x=x_col, y=y_col, color='PointType',
                         color_discrete_map={'Normal': 'cyan', 'Outlier': 'red'},
                         title=title)
        if dims == 2:
            fig.update_xaxes(title_text=x_col)
            fig.update_yaxes(title_text=y_col)
        else:
            fig.update_xaxes(title_text="Index")
            fig.update_yaxes(title_text=y_col)
    # Apply dark theme layout
    fig.update_layout(template='plotly_dark', legend_title="Point Type", title=title)
    return fig

def create_time_series_plot(x, y, outlier_mask, title="Time Series Anomalies"):
    """
    Create a time-series line plot with outlier points highlighted.
    x: Sequence for x-axis (e.g., time or index), y: Series of values, outlier_mask: boolean array for outliers.
    Outliers are marked with red 'X' markers on top of the line plot.
    """
    x_series = pd.Series(x)
    y_series = pd.Series(y)
    # Ensure mask is boolean numpy array
    mask = np.asarray(outlier_mask, dtype=bool)
    # Split normal and outlier points
    normal_idx = ~mask
    outlier_idx = mask
    fig = go.Figure()
    # Plot the time series line for normal points
    fig.add_trace(go.Scatter(x=x_series[normal_idx], y=y_series[normal_idx],
                             mode='lines+markers', name='Data',
                             line=dict(color='cyan'), marker=dict(color='cyan', size=5)))
    # Overlay outlier points as red markers
    fig.add_trace(go.Scatter(x=x_series[outlier_idx], y=y_series[outlier_idx],
                             mode='markers', name='Outliers',
                             marker=dict(color='red', size=8, symbol='x')))
    fig.update_layout(title=title, template='plotly_dark', legend_title_text='', 
                      xaxis_title="Time" if x_series.dtype.kind == 'M' else "Index",
                      yaxis_title="Value")
    return fig
