import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Reshape, Dense

# Set Streamlit Page Configuration
st.set_page_config(layout="wide", page_title="1D-CNN Autoencoder Anomaly Detection")

## --- 1. Model Definition (Must be identical to the training script) ---

def create_1dcnn_autoencoder(input_dim):
    """Defines the 1D-CNN Autoencoder architecture."""
    # Encoder
    input_layer = Input(shape=(input_dim, 1))
    x = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(pool_size=2, padding='same')(x) # Bottleneck

    # Decoder
    x = Conv1D(filters=16, kernel_size=7, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    # The final layer reconstructs the original shape (input_dim, 1)
    decoded = Conv1D(filters=1, kernel_size=7, activation='linear', padding='same')(x)

    # Use a Dense layer as a final reshaping/reconstruction step if 1D layers don't align
    # This logic is necessary because the intermediate shapes depend on the input_dim
    if decoded.shape[1] != input_dim:
        x_flat = tf.keras.layers.Flatten()(decoded)
        x_dense = Dense(input_dim, activation='linear')(x_flat)
        decoded = Reshape((input_dim, 1))(x_dense)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    return autoencoder

## --- 2. Data Loading and Setup (Cached for efficiency) ---

@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads and preprocesses the KDD dataset."""
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'class'
    ]

    try:
        # Assuming the file is named 'KDDTest+.csv' and is accessible
        df = pd.read_csv(file_path, header=None, names=column_names)
        df = df.iloc[1:].copy()
    except Exception as e:
        st.error(f"Error loading file: {e}. Please ensure 'KDDTest+.csv' is in the app directory.")
        return None, None, None, None

    # Convert numeric columns
    numeric_cols = df.columns.drop(['protocol_type', 'service', 'flag', 'class'])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Label encode target
    df['anomaly'] = df['class'].apply(lambda x: 0 if x == 'normal' else 1)
    y_true = df['anomaly'].values
    df.drop('class', axis=1, inplace=True)

    # One-Hot Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Scale all feature columns
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.drop('anomaly', axis=1))
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.drop('anomaly', axis=1).columns)

    return X_scaled_df, y_true, scaler

## --- 3. Model Training and Prediction (Cached) ---

@st.cache_resource
def train_and_predict(X_scaled_df, y_true):
    """Trains the Autoencoder on normal data and predicts on the full dataset."""
    
    # 1. Prepare Training Data
    X_train_normal = X_scaled_df[y_true == 0].values
    n_features = X_train_normal.shape[1]
    X_train_normal_reshaped = X_train_normal.reshape(X_train_normal.shape[0], n_features, 1)

    # 2. Create and Compile Model
    input_dim = n_features
    autoencoder = create_1dcnn_autoencoder(input_dim)
    autoencoder.compile(optimizer='adam', loss='mae')

    # 3. Training (limited epochs for quick demo)
    with st.spinner("Training Autoencoder on 'Normal' Data..."):
        # Use a minimal number of epochs for the Streamlit demo
        # In a real app, you would load a pre-trained model.
        autoencoder.fit(
            X_train_normal_reshaped,
            X_train_normal_reshaped,
            epochs=5, # Reduced for Streamlit performance
            batch_size=128,
            validation_split=0.1,
            verbose=0 # Silence Keras output in Streamlit
        )

    # 4. Prediction
    X_full_reshaped = X_scaled_df.values.reshape(X_scaled_df.shape[0], n_features, 1)
    X_pred = autoencoder.predict(X_full_reshaped, verbose=0)
    mae = np.mean(np.abs(X_full_reshaped - X_pred), axis=(1, 2))

    # 5. Thresholding
    mae_normal = mae[y_true == 0]
    threshold = np.percentile(mae_normal, 95)
    y_pred = (mae > threshold).astype(int)

    return y_pred, mae, threshold, X_scaled_df.index.tolist()

## --- 4. Streamlit App Layout ---

def main():
    st.title("🛡️ 1D-CNN Autoencoder for Network Anomaly Detection")
    st.markdown("""
        This application demonstrates an **Unsupervised Anomaly Detector** using a 1D Convolutional Autoencoder. 
        The model is trained *only* on **normal** network traffic data to learn its typical patterns. 
        Anomalies are detected when the reconstruction error (MAE) of a sample exceeds a learned threshold.
    """)
    
    st.info("The application simulates the entire training and prediction pipeline on the KDDTest+ dataset. This process is cached for fast subsequent runs.")
    st.markdown("---")

    # --- Load Data Section ---
    st.header("1. Data Loading and Preparation")
    
    # Placeholder for file path. In a real deployment, you might use st.file_uploader.
    data_file = 'KDDTest+.csv' 
    st.code(f"Loading data from: {data_file}")

    X_scaled_df, y_true, scaler = load_and_preprocess_data(data_file)
    
    if X_scaled_df is None:
        return

    st.success(f"Data Loaded and Preprocessed successfully. Total samples: {len(y_true)}")
    st.text(f"Feature Vector Length after One-Hot Encoding: {X_scaled_df.shape[1]}")
    st.dataframe(X_scaled_df.head())
    st.markdown("---")


    # --- Training and Prediction Section ---
    st.header("2. Model Training and Prediction")

    # The cached function handles training and prediction
    y_pred, mae, threshold, data_indices = train_and_predict(X_scaled_df, y_true)

    st.markdown(f"**Anomaly Threshold (95th percentile of Normal MAE):** $\\mathbf{{{threshold:.4f}}}$")
    
    st.success("Prediction Complete!")
    st.markdown("---")


    # --- Evaluation Section ---
    st.header("3. Evaluation Results")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        
        # Display key counts
        st.metric("Total Samples", len(y_true))
        st.metric("True Anomalies (Actual)", np.sum(y_true))
        st.metric("Detected Anomalies (Predicted)", np.sum(y_pred))

        # Classification Report Table
        report = classification_report(y_true, y_pred, target_names=['Normal (0)', 'Anomaly (1)'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.loc[['accuracy'], ['precision', 'recall', 'f1-score', 'support']] = ['-', '-', '-', report_df.loc['accuracy', 'support']]
        st.dataframe(report_df.style.format(precision=4), use_container_width=True)

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        st.code(f"[[TN, FP]\n [FN, TP]]\n\n{cm}")
        
        st.markdown("**Matrix Interpretation:**")
        st.text(f"True Negatives (TN): {cm[0, 0]} (Correctly labeled Normal)")
        st.text(f"False Positives (FP): {cm[0, 1]} (Normal mislabeled Anomaly)")
        st.text(f"False Negatives (FN): {cm[1, 0]} (Anomaly mislabeled Normal)")
        st.text(f"True Positives (TP): {cm[1, 1]} (Correctly labeled Anomaly)")
        
    st.markdown("---")

    # --- Anomaly Inspection Section ---
    st.header("4. Inspecting Reconstruction Error")
    
    # Create a DataFrame for inspection
    results_df = pd.DataFrame({
        'Reconstruction_Error_MAE': mae,
        'True_Label': y_true,
        'Predicted_Label': y_pred
    }, index=data_indices)
    
    # Visualize the MAE distribution and the threshold
    st.subheader("Reconstruction Error Distribution")
    st.line_chart(results_df['Reconstruction_Error_MAE'])
    st.markdown(f"***Note:** The line chart shows the MAE for all {len(y_true)} samples. Anomalies are points above the threshold.*")
    
    # Filter and display the top 10 most severe anomalies
    anomalies = results_df[results_df['Predicted_Label'] == 1].sort_values(by='Reconstruction_Error_MAE', ascending=False)
    
    if not anomalies.empty:
        st.subheader(f"Top 10 Detected Anomalies (out of {len(anomalies)})")
        st.dataframe(anomalies.head(10), use_container_width=True)
    else:
        st.info("No anomalies were detected based on the 95th percentile threshold.")


if __name__ == "__main__":
    main()