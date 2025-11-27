import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Reshape, Dense, Flatten
import io
import plotly.express as px

# Set Streamlit Page Configuration
st.set_page_config(layout="wide", page_title="1D-CNN Autoencoder Anomaly Detection")

# --- Initialize Session State ---
if 'phase' not in st.session_state:
    st.session_state['phase'] = 1 # 1: Upload, 2: Process/Train

if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None

# --- 1. Model Definition ---

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

    # Use a Dense layer as a final reshaping/reconstruction step
    if decoded.shape[1] != input_dim:
        x_flat = Flatten()(decoded)
        x_dense = Dense(input_dim, activation='linear')(x_flat) 
        decoded = Reshape((input_dim, 1))(x_dense)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    return autoencoder


# --- 2. Data Loading and Setup (Cached for efficiency) ---

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Loads and preprocesses the KDD dataset from an uploaded CSV file."""
    
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
        st.info("Please ensure your uploaded file is the KDDTest+.csv or KDDTrain.csv dataset.")
        return None, None, None

    numeric_cols = df.columns.drop(['protocol_type', 'service', 'flag', 'class'])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['anomaly'] = df['class'].apply(lambda x: 0 if x.strip().lower() == 'normal' else 1)
    y_true = df['anomaly'].values
    df.drop('class', axis=1, inplace=True)

    categorical_cols = ['protocol_type', 'service', 'flag']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df.drop('anomaly', axis=1))
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.drop('anomaly', axis=1).columns)

    return X_scaled_df, y_true, scaler


# --- 3. Model Training and Prediction (Cached) ---

@st.cache_resource
def train_and_predict(X_scaled_df, y_true):
    """Trains the Autoencoder on normal data and predicts reconstruction error."""
    
    X_train_normal = X_scaled_df[y_true == 0].values
    n_features = X_train_normal.shape[1]
    
    if X_train_normal.shape[0] < 100:
        st.error("Not enough 'normal' traffic (anomaly=0) to train the autoencoder.")
        return None, None, None

    X_train_normal_reshaped = X_train_normal.reshape(X_train_normal.shape[0], n_features, 1)

    input_dim = n_features
    autoencoder = create_1dcnn_autoencoder(input_dim)
    autoencoder.compile(optimizer='adam', loss='mae')

    # Training is silent (verbose=0)
    autoencoder.fit(
        X_train_normal_reshaped,
        X_train_normal_reshaped,
        epochs=10, 
        batch_size=128,
        validation_split=0.1,
        verbose=0 
    )

    X_full_reshaped = X_scaled_df.values.reshape(X_scaled_df.shape[0], n_features, 1)
    X_pred = autoencoder.predict(X_full_reshaped, verbose=0)
    mae = np.mean(np.abs(X_full_reshaped - X_pred), axis=(1, 2))

    mae_normal = mae[y_true == 0]
    threshold_95th = np.percentile(mae_normal, 95)
    
    return mae, threshold_95th, X_scaled_df.index.tolist()


# --- 4. Streamlit App Layout (Main Logic) ---

def main():
    st.title("🛡️ 1D-CNN Autoencoder for Network Anomaly Detection")
    st.markdown("""
        Upload a network intrusion dataset (like **KDDTest+.csv**) to train and evaluate the 1D-CNN Autoencoder. 
    """)
    st.markdown("---")

    if st.session_state['phase'] == 1:
        # ------------------------------------
        # PHASE 1: FILE UPLOAD
        # ------------------------------------
        
        st.header("1. Upload Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file (KDD-style dataset) - **Max size: 1000 MB**", 
            type="csv",
            max_uploader_size=1000 
        )

        if uploaded_file is not None:
            
            # Store the uploaded file in session state
            st.session_state['uploaded_data'] = uploaded_file

            st.success("File uploaded successfully. Click the button below to start processing and training.")
            
            # Button to transition to Phase 2
            if st.button('Start Model Training and Prediction'):
                st.session_state['phase'] = 2
                st.experimental_rerun() # Rerun the script to switch phase

    elif st.session_state['phase'] == 2:
        # ------------------------------------
        # PHASE 2: PROCESSING AND EVALUATION
        # ------------------------------------
        
        st.header("2. Data Preprocessing and Model Training")
        uploaded_file = st.session_state['uploaded_data']

        # 1. Data Preprocessing
        with st.spinner('Preprocessing Data...'):
            X_scaled_df, y_true, scaler = load_and_preprocess_data(uploaded_file)
        
        if X_scaled_df is None:
            st.session_state['phase'] = 1
            st.error("Data processing failed. Please try uploading a different file.")
            st.experimental_rerun()
            return

        st.success(f"Data Preprocessing Complete. Total samples: **{len(y_true)}**, Features: **{X_scaled_df.shape[1]}**.")
        st.dataframe(X_scaled_df.head(), use_container_width=True)
        st.markdown("---")
        
        # 2. Model Training and Prediction
        st.subheader("Starting 1D-CNN Autoencoder Training...")
        with st.spinner('Training Autoencoder on "Normal" traffic and calculating reconstruction errors...'):
            mae, threshold_95th, data_indices = train_and_predict(X_scaled_df, y_true)

        if mae is None:
            st.session_state['phase'] = 1
            st.experimental_rerun()
            return

        st.success("Training and Prediction Complete!")
        st.markdown("---")

        # 3. Interactive Thresholding and Prediction
        st.header("3. Anomaly Threshold and Results")
        
        percentile = st.slider(
            'Set Anomaly Threshold Percentile (from Normal Data MAE):',
            min_value=90.0,
            max_value=99.9,
            value=95.0,
            step=0.1,
            format='%.1f %%'
        )

        mae_normal = mae[y_true == 0]
        dynamic_threshold = np.percentile(mae_normal, percentile)
        y_pred = (mae > dynamic_threshold).astype(int)

        st.markdown(f"**Calculated Anomaly Threshold ({percentile:.1f}th percentile):** $\\mathbf{{{dynamic_threshold:.4f}}}$")
        st.markdown("---")

        # 4. Evaluation Section
        st.subheader("Evaluation Metrics")

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Counts**")
            st.metric("Total Samples", len(y_true))
            st.metric("True Anomalies (Actual)", np.sum(y_true))
            st.metric("Detected Anomalies (Predicted)", np.sum(y_pred))

            st.markdown("---")
            st.markdown("**Classification Report**")
            report = classification_report(y_true, y_pred, target_names=['Normal (0)', 'Anomaly (1)'], output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            report_df.loc[['accuracy'], ['precision', 'recall', 'f1-score', 'support']] = ['-', '-', '-', report_df.loc['accuracy', 'support']]
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)

        with col2:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_true, y_pred)
            st.code(f"[[TN, FP]\n [FN, TP]]\n\n{cm}")
            
            st.markdown("**Matrix Interpretation:**")
            st.text(f"True Negatives (TN): {cm[0, 0]}")
            st.text(f"False Positives (FP): {cm[0, 1]}")
            st.text(f"False Negatives (FN): {cm[1, 0]}")
            st.text(f"True Positives (TP): {cm[1, 1]}")
            
        st.markdown("---")

        # 5. Anomaly Inspection & Visualization Section
        st.header("4. Inspecting Reconstruction Error")
        
        results_df = pd.DataFrame({
            'Reconstruction_Error_MAE': mae,
            'True_Label': y_true,
            'Predicted_Label': y_pred
        })
        results_df['True_Label_Name'] = results_df['True_Label'].apply(lambda x: 'Anomaly (1)' if x == 1 else 'Normal (0)')
        
        st.subheader("Reconstruction Error Distribution and Threshold")
        
        fig = px.histogram(
            results_df, 
            x='Reconstruction_Error_MAE', 
            color='True_Label_Name', 
            marginal="box", 
            histnorm='percent', 
            title='MAE Distribution by True Class',
            labels={'Reconstruction_Error_MAE': 'Mean Absolute Error (MAE)', 'True_Label_Name': 'True Class'},
            hover_data=results_df.columns
        )
        
        # Add a vertical line for the dynamic threshold
        fig.add_vline(x=dynamic_threshold, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Threshold: {dynamic_threshold:.4f}")
        fig.update_layout(height=450)
        
        st.plotly_chart(fig, use_container_width=True)
        
        anomalies = results_df[results_df['Predicted_Label'] == 1].sort_values(by='Reconstruction_Error_MAE', ascending=False)
        
        if not anomalies.empty:
            st.subheader(f"Top 10 Detected Anomalies (out of {len(anomalies)})")
            st.dataframe(anomalies.head(10).reset_index(drop=True), use_container_width=True)
        else:
            st.info("No anomalies were detected based on the current threshold.")
            
        # Button to return to Phase 1
        st.markdown("---")
        if st.button('Upload New Dataset'):
            st.session_state['phase'] = 1
            st.session_state['uploaded_data'] = None
            st.cache_data.clear()
            st.cache_resource.clear()
            st.experimental_rerun()


if __name__ == "__main__":
    main()
