import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Reshape, Dense
import io

# Set Streamlit Page Configuration
st.set_page_config(layout="wide", page_title="Interactive 1D-CNN Autoencoder Anomaly Detection")

## --- 1. Model Definition (Kept identical for consistency) ---

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
    decoded = Conv1D(filters=1, kernel_size=7, activation='linear', padding='same')(x)

    # Reshaping/reconstruction step
    if decoded.shape[1] != input_dim:
        x_flat = tf.keras.layers.Flatten()(decoded)
        x_dense = Dense(input_dim, activation='linear')(x_flat)
        decoded = Reshape((input_dim, 1))(x_dense)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    return autoencoder

## --- 2. Data Loading and Setup (Cached for efficiency) ---

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Loads and preprocesses the KDD dataset from an uploaded CSV file."""
    
    # Define column names (41 features + 1 class)
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
        data = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(io.StringIO(data), header=None, names=column_names)
        df = df.iloc[1:].copy()
        
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None, None, None, None

    # Preprocessing Steps
    numeric_cols = df.columns.drop(['protocol_type', 'service', 'flag', 'class'])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['anomaly'] = df['class'].apply(lambda x: 0 if x == 'normal' else 1)
    y_true = df['anomaly'].values
    df.drop('class', axis=1, inplace=True)

    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.drop('anomaly', axis=1))
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.drop('anomaly', axis=1).columns)

    return X_scaled_df, y_true, scaler


## --- 3. Model Training and Prediction (Cached) ---

# We pass EPOCHS as an argument to make the function sensitive to changes in the slider
@st.cache_resource
def train_and_predict(X_scaled_df, y_true, epochs):
    """Trains the Autoencoder and calculates MAE."""
    
    X_train_normal = X_scaled_df[y_true == 0].values
    n_features = X_train_normal.shape[1]
    
    if X_train_normal.shape[0] < 100:
        st.error("Not enough 'normal' traffic (anomaly=0) to train.")
        return None, None
    
    X_train_normal_reshaped = X_train_normal.reshape(X_train_normal.shape[0], n_features, 1)

    input_dim = n_features
    autoencoder = create_1dcnn_autoencoder(input_dim)
    autoencoder.compile(optimizer='adam', loss='mae')

    with st.spinner(f"Training Autoencoder on {X_train_normal.shape[0]} samples for {epochs} epochs..."):
        autoencoder.fit(
            X_train_normal_reshaped, X_train_normal_reshaped,
            epochs=epochs,
            batch_size=128,
            validation_split=0.1,
            verbose=0 
        )

    # 4. Prediction
    X_full_reshaped = X_scaled_df.values.reshape(X_scaled_df.shape[0], n_features, 1)
    X_pred = autoencoder.predict(X_full_reshaped, verbose=0)
    mae = np.mean(np.abs(X_full_reshaped - X_pred), axis=(1, 2))

    return mae, X_scaled_df.index.tolist()


## --- 4. Streamlit App Layout ---

def main():
    st.title("🛡️ Interactive 1D-CNN Autoencoder for Network Anomaly Detection")
    st.markdown("""
        Upload a KDD-style dataset to run the analysis. Use the **Model Configuration** sidebar 
        to adjust training parameters (epochs) and the **Threshold Setting** to control sensitivity.
    """)
    st.markdown("---")
    
    # --- Sidebar for Configuration ---
    st.sidebar.header("⚙️ Model Configuration")
    
    # User input for training epochs
    epochs = st.sidebar.slider(
        'Training Epochs (Higher = Longer Training)',
        min_value=5, max_value=50, value=15, step=5
    )
    st.sidebar.info(f"Set to **{epochs}** epochs. Click 'Start Analysis' to train.")

    # User input for Anomaly Threshold calculation percentile
    threshold_percentile = st.sidebar.slider(
        'Anomaly Threshold Percentile (Sensitivity)',
        min_value=90, max_value=99, value=95, step=1,
        help="The Nth percentile of the normal data's reconstruction error. Higher = less sensitive (fewer false positives)."
    )
    st.sidebar.markdown("---")
    
    # --- Main Content - 1. File Uploader ---
    st.header("1. Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file (e.g., KDDTest+.csv)", 
        type="csv"
    )

    if uploaded_file is not None:
        
        # --- 2. Data Processing ---
        with st.spinner('Preprocessing Data...'):
            X_scaled_df, y_true, scaler = load_and_preprocess_data(uploaded_file)
        
        if X_scaled_df is None:
            return

        st.success(f"Data Loaded and Preprocessed successfully. Total samples: {len(y_true)}")
        st.text(f"Feature Vector Length after One-Hot Encoding: {X_scaled_df.shape[1]}")
        
        # Collapse initial data view for cleaner look
        with st.expander("Preview Preprocessed Data"):
            st.dataframe(X_scaled_df.head())
        st.markdown("---")

        # --- 3. Run Analysis Button ---
        st.header("2. Run Anomaly Detection Analysis")
        
        if st.button('🚀 Start Model Training and Prediction', type="primary"):
            
            # --- 4. Model Training and Prediction ---
            mae, data_indices = train_and_predict(X_scaled_df, y_true, epochs)
            
            if mae is None:
                return

            # Calculate threshold and prediction based on user percentile input
            mae_normal = mae[y_true == 0]
            threshold = np.percentile(mae_normal, threshold_percentile)
            y_pred = (mae > threshold).astype(int)

            st.success("Prediction Complete!")
            st.markdown("---")

            # --- 5. Evaluation Section ---
            st.header("3. Evaluation Results")

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                
                st.markdown(f"**Calculated Threshold ({threshold_percentile}th Pct.):** $\\mathbf{{{threshold:.4f}}}$")
                
                # Display key counts
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
                
                st.markdown("**Key Interpretation:**")
                st.text(f"False Positives (FP - False Alarms): {cm[0, 1]}")
                st.text(f"False Negatives (FN - Missed Attacks): {cm[1, 0]}")
                
            st.markdown("---")

            # --- Anomaly Inspection Section ---
            st.header("4. Inspecting Reconstruction Error")
            
            results_df = pd.DataFrame({
                'Reconstruction_Error_MAE': mae,
                'True_Label': y_true,
                'Predicted_Label': y_pred
            }, index=data_indices)
            
            # Create a line chart and add the threshold line
            st.subheader("Reconstruction Error Distribution (MAE)")
            
            # Add threshold line to the chart data for visualization
            results_df['Threshold'] = threshold
            
            st.line_chart(results_df[['Reconstruction_Error_MAE', 'Threshold']])
            st.markdown(f"***Note:*** *Any sample above the red **Threshold** line is classified as an Anomaly.*")
            
            anomalies = results_df[results_df['Predicted_Label'] == 1].sort_values(by='Reconstruction_Error_MAE', ascending=False)
            
            if not anomalies.empty:
                st.subheader(f"Top 10 Detected Anomalies (out of {len(anomalies)})")
                st.dataframe(anomalies.head(10), use_container_width=True)
            else:
                st.info("No anomalies were detected based on the current threshold settings.")
    else:
        st.warning("Please upload a KDD-style dataset CSV file to proceed.")


if __name__ == "__main__":
    main()
