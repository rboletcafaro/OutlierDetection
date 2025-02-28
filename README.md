# OutlierDetection

# Outlier Detection Web Application

## Overview
This project is a Dash web application for detecting outliers in a dataset using various unsupervised learning algorithms. It allows users to upload a CSV file, choose an outlier detection mode (Simple, Balanced, Complex), and visualize the detected anomalies in an interactive, dark-themed dashboard. The application is built with [Plotly Dash](https://plotly.com/dash/) and integrates multiple outlier detection techniques (Histogram-Based Outlier Score, DBSCAN, Local Outlier Factor, Isolation Forest) to identify unusual data points.

## Features
- **Interactive Dashboard** – User-friendly interface with a dark theme for easy viewing. Users can upload their own datasets (CSV files) and configure detection settings.
- **Multiple Detection Modes** – Three modes of outlier detection:
  - *Simple*: Fast single-model detection (uses Isolation Forest by default).
  - *Balanced*: Ensemble of methods (Isolation Forest, LOF, HBOS) with majority voting to balance sensitivity and specificity.
  - *Complex*: Comprehensive detection using multiple methods (IF, LOF, HBOS, DBSCAN), flagging any data point identified by any model for maximum coverage.
- **Real-time Analysis** – Clicking the "Run Detection" button processes the data and updates the visuals with a loading spinner for feedback.
- **Data Preprocessing** – Automatic handling of missing values (median imputation for general data or forward-fill for time-series data) and feature scaling for robust analysis.
- **Outlier Visualization** – Results are displayed in interactive Plotly graphs:
  - Scatter plot of the data (or principal components) with outliers highlighted in red.
  - Time-series line plots for temporal data, with anomalies marked for easy identification.
  - Summary text indicating the number of outliers detected.
- **Modular Codebase** – Clear project structure separating the Dash app, callbacks, data processing, model definitions, and visualization logic, making it easy to maintain and extend.

## Installation
1. **Clone the repository** (or download the source code) to your local machine.
2. **Install Python dependencies** by running:  
   ```bash
   pip install -r requirements.txt
