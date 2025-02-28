import base64
import io
import pandas as pd
import numpy as np

def parse_contents(contents, filename):
    """
    Parse the contents of an uploaded file (CSV or Excel) into a pandas DataFrame.
    Supports CSV (.csv) and Excel (.xls, .xlsx) files.
    """
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    try:
        decoded = base64.b64decode(content_string)
    except Exception as e:
        print(f"Error decoding file contents: {e}")
        return None
    # Determine file type and read accordingly
    try:
        if filename.lower().endswith('.csv'):
            # Read CSV text
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.lower().endswith('.xls') or filename.lower().endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            # Unsupported file type
            return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    # Convert any date strings to datetime objects (except the ones handled in callbacks)
    for col in df.select_dtypes(include=['object']).columns:
        # Try to parse text columns to datetime
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            continue
    return df

def preprocess_data(df, time_series=False):
    """
    Clean and preprocess the DataFrame for outlier detection:
      - Handle missing values (forward-fill for time series data, or median imputation for others).
      - Drop non-numeric columns (since outlier detection is on numerical features).
      - Scale numeric features (StandardScaler) for better model performance.
      - (Optional) Perform any feature engineering or segmentation if required.
    Returns:
      df_imputed: DataFrame of numeric features after imputation (original scale values).
      X: numpy array of scaled numeric features for modeling.
    """
    # Work on a copy to avoid modifying original df
    data = df.copy()
    # Drop columns that have no data at all
    data = data.dropna(axis=1, how='all')
    # Impute missing values
    if time_series:
        # Forward-fill for time-series continuity, then back-fill to handle leading NaNs
        data = data.fillna(method='ffill').fillna(method='bfill')
    # For any remaining missing (or if not a time series), fill numeric NaNs with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
    # After imputation, drop any non-numeric columns (categorical text not used in modeling)
    data_numeric = data.select_dtypes(include=[np.number])
    # If no numeric data remains, return accordingly
    if data_numeric.shape[1] == 0:
        return data_numeric, None
    # Feature scaling (Standardization)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_numeric.values)
    # Return the imputed DataFrame (numeric features) and the scaled numpy array for modeling
    return data_numeric, X_scaled
