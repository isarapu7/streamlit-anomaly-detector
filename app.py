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
st.set_page_config(layout="wide", page_title="IDS Anomaly Detection Dashboard")

## --- 1. Model Definition ---

def create_1dcnn_autoencoder(input_dim):
    """Defines the 1D-CNN Autoencoder architecture."""
    
    FILTERS_1 = 32 
    FILTERS_2 = 16 
    KERNEL_SIZE = 7

    # Encoder
    input_layer = Input(shape=(input_dim, 1))
    x = Conv1D(filters=FILTERS_1, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=FILTERS_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(pool_size=2, padding='same')(x) # Bottleneck

    # Decoder
    x = Conv1D(filters=FILTERS_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=FILTERS_1, kernel_size=KERNEL_SIZE, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(filters=1, kernel_size=KERNEL_SIZE, activation='linear', padding='same')(x)

    # Reshaping/reconstruction step to ensure output matches input dimension
    if decoded.shape[1] != input_dim:
        x_flat = tf.keras.layers.Flatten()(decoded)
        x_dense = Dense(input_dim, activation='linear')(x_flat)
        decoded = Reshape((input_dim, 1))(x_dense)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    return autoencoder


## --- 2. Data Loading and Preprocessing (Cached) ---

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
        st.info("Ensure your uploaded file is a standard KDD dataset (42 columns) and is not corrupted.")
        return None, None, None, None

    # Preprocessing
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

@st.cache_resource
def train_model_and_get_tools(X_scaled_df, y_true, epochs):
    """Trains the Autoencoder and returns the model and necessary tools."""
    
    X_train_normal = X_scaled_df[y_true == 0].values
    n_features = X_train_normal.shape[1]
    
    if X_train_normal.shape[0] < 100:
        st.error("Not enough 'normal' traffic (anomaly=0) to train the autoencoder.")
        return None, None, None
    
    X_train_normal_reshaped = X_train_normal.reshape(X_train_normal.shape[0], n_features, 1)

    input_dim = n_features
    autoencoder = create_1dcnn_autoencoder(input_dim)
    autoencoder.compile(optimizer='adam', loss='mae')

    # Training
    with st.spinner(f"Training Autoencoder for {epochs} epochs..."):
        autoencoder.fit(
            X_train_normal_reshaped, X_train_normal_reshaped,
            epochs=epochs,
            batch_size=128,
            validation_split=0.1,
            verbose=0 
        )

    # Batch Prediction (Calculating Reconstruction Error - MAE)
    X_full_reshaped = X_scaled_df.values.reshape(X_scaled_df.shape[0], n_features, 1)
    X_pred = autoencoder.predict(X_full_reshaped, verbose=0)
    mae = np.mean(np.abs(X_full_reshaped - X_pred), axis=(1, 2))

    return autoencoder, mae, X_scaled_df.columns.tolist()


# --- NEW CACHED FUNCTION: Get Unique Values for User Select Boxes ---
@st.cache_data
def get_unique_categories():
    """Defines known unique values for KDD categorical columns."""
    
    # Common KDD categories for user input
    protocol_types = ['tcp', 'udp', 'icmp']
    flags = ['SF', 'S0', 'REJ', 'RSTO', 'SH', 'RSTR', 'OTH', 'ECHO', 'ESTAB', 'RSE'] 
    services = ['http', 'smtp', 'ftp_data', 'finger', 'domain_u', 'telnet', 'ecr_i', 'private', 'other', 'pop_3', 'discard', 'eco_i', 'Z39_50', 'systat', 'csnet_ns', 'auth', 'time', 'netbios_dgm', 'whois']
    
    return protocol_types, flags, services

# --- NEW FUNCTION: Real-Time Prediction ---
def predict_single_sample(model, scaler, all_feature_names, batch_threshold, protocol, service, flag, duration, serror_rate):
    """
    Creates a single sample DataFrame, preprocesses it, and predicts the MAE.
    
    We use dummy values for the 38 numerical features and only accept a few key inputs.
    """
    
    # 1. Define placeholder for ALL 41 features
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
        'dst_host_srv_rerror_rate'
    ]
    
    # Create the sample row with placeholder values (list of 0s)
    sample_data = [0] * len(column_names)
    sample_df = pd.DataFrame([sample_data], columns=column_names)
    
    # Inject user inputs into the DataFrame
    sample_df['protocol_type'] = protocol
    sample_df['service'] = service
    sample_df['flag'] = flag
    
    # Inject specific numerical inputs
    sample_df['duration'] = duration
    sample_df['serror_rate'] = serror_rate
    
    # 2. One-Hot Encode
    categorical_cols = ['protocol_type', 'service', 'flag']
    sample_df = pd.get_dummies(sample_df, columns=categorical_cols, drop_first=True)

    # 3. Re-align columns (Crucial step to match the shape of the trained model's input)
    missing_cols = set(all_feature_names) - set(sample_df.columns)
    for c in missing_cols:
        sample_df[c] = 0 # Add missing one-hot columns as 0
    
    # Drop any extra columns that weren't in the training set 
    extra_cols = set(sample_df.columns) - set(all_feature_names)
    sample_df.drop(list(extra_cols), axis=1, inplace=True)
    
    # Final column order must match all_feature_names
    sample_df = sample_df[all_feature_names]

    # 4. Scale the data
    X_scaled = scaler.transform(sample_df.values)
    
    # 5. Reshape and Predict
    n_features = X_scaled.shape[1]
    X_reshaped = X_scaled.reshape(1, n_features, 1)
    
    X_pred = model.predict(X_reshaped, verbose=0)
    mae = np.mean(np.abs(X_reshaped - X_pred), axis=(1, 2))[0]
    
    return mae, batch_threshold

# --- 4. Streamlit App Layout ---

def main():
    st.title("🛡️ Interactive 1D-CNN Autoencoder for Network Anomaly Detection")
    st.markdown("""
        Upload a **KDD-style dataset** to train and evaluate the model. Use the **Sidebar Configuration** to tune the analysis, and the **Connection Checker** below for real-time testing.
    """)
    st.markdown("---")
    
    # --- Sidebar for Configuration ---
    st.sidebar.header("⚙️ Model Configuration")
    
    # Training Epochs Control
    epochs = st.sidebar.slider(
        'Training Epochs',
        min_value=5, max_value=50, value=15, step=5,
        key='epochs_slider'
    )
    st.sidebar.info(f"Model will train for **{epochs}** epochs.")

    # Anomaly Threshold Control
    threshold_percentile = st.sidebar.slider(
        'Anomaly Threshold Percentile (Sensitivity)',
        min_value=90, max_value=99, value=95, step=1,
        key='threshold_slider',
        help="Increasing this reduces False Positives (increases Normal Precision) but risks missing subtle attacks."
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
            X_scaled_df, y_true, scaler_obj = load_and_preprocess_data(uploaded_file)
        
        if X_scaled_df is None:
            return

        st.success(f"Data Loaded and Preprocessed successfully. Total samples: {len(y_true)}")
        st.text(f"Feature Vector Length after One-Hot Encoding: {X_scaled_df.shape[1]}")
        
        with st.expander("Preview Preprocessed Data"):
            st.dataframe(X_scaled_df.head())
        st.markdown("---")
        
        # Store essential tools in session state 
        st.session_state['scaler_obj'] = scaler_obj
        st.session_state['all_feature_names'] = X_scaled_df.columns.tolist()

        # --- 3. Run Analysis Button ---
        st.header("2. Run Batch Anomaly Detection Analysis")
        
        if st.button('🚀 Start Model Training and Prediction', type="primary"):
            
            # --- 4. Model Training and Prediction ---
            model, mae, data_indices = train_model_and_get_tools(X_scaled_df, y_true, epochs)
            
            if model is None:
                return

            # Cache the model object for the single prediction interface
            st.session_state['trained_model'] = model

            # Calculate threshold and prediction based on user percentile input
            mae_normal = mae[y_true == 0]
            
            # Recalculate threshold using the current sidebar value
            threshold = np.percentile(mae_normal, threshold_percentile)
            y_pred = (mae > threshold).astype(int)
            
            # Store threshold for the single prediction interface
            st.session_state['batch_threshold'] = threshold

            st.success("Batch Analysis Complete! (Model is now ready for real-time testing)")
            st.markdown("---")

            # --- 5. Evaluation Section ---
            st.header("3. Evaluation Results")

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                
                st.markdown(f"**Calculated Threshold ({threshold_percentile}th Pct.):** $\\mathbf{{{threshold:.4f}}}$")
                
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
                
                st.markdown("**Key Interpretation:**")
                st.text(f"False Positives (FP - False Alarms): {cm[0, 1]}")
                st.text(f"False Negatives (FN - Missed Attacks): {cm[1, 0]}")
                
            st.markdown("---")
            
            # NOTE: Removed Section 4: Inspecting Reconstruction Error

    
    # -------------------------------------------------------------
    # NEW SECTION: Real-Time Prediction Interface
    # -------------------------------------------------------------
    if uploaded_file is not None and 'trained_model' in st.session_state:
        st.markdown("---")
        st.header("4. 🕵️ Real-Time Connection Checker")
        
        # Retrieve necessary components
        model = st.session_state['trained_model']
        scaler_obj = st.session_state['scaler_obj']
        all_feature_names = st.session_state['all_feature_names']
        batch_threshold = st.session_state.get('batch_threshold', 0.0)

        st.info(f"Model is using the latest trained weights and a threshold of **{batch_threshold:.4f}**")
        
        protocol_types, flags, services = get_unique_categories()

        colA, colB, colC = st.columns(3)
        
        # 1. Categorical Inputs 
        with colA:
            st.subheader("Connection Properties")
            input_protocol = st.selectbox("Protocol Type", protocol_types)
            input_flag = st.selectbox("Flag", flags)

        with colB:
            st.subheader("Service & Duration")
            input_service = st.selectbox("Service", services)
            input_duration = st.number_input("Duration (seconds)", min_value=0, value=0)

        with colC:
            st.subheader("Error Rate")
            input_serror_rate = st.number_input("Serror Rate (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            
        # --- Submit Button ---
        if st.button("Analyze Connection Status", key="single_predict", type="secondary"):
            
            with st.spinner("Analyzing connection..."):
                mae_result, threshold_used = predict_single_sample(
                    model, scaler_obj, all_feature_names, batch_threshold, 
                    input_protocol, input_service, input_flag, input_duration, input_serror_rate
                )

            st.markdown("### Prediction Result")
            
            if mae_result > threshold_used:
                st.error(f"🚨 ANOMALY DETECTED! (Reconstruction Error: {mae_result:.4f})")
                st.markdown(f"**Action:** Flagged as high-risk intrusion attempt.")
            else:
                st.success(f"✅ NORMAL TRAFFIC (Reconstruction Error: {mae_result:.4f})")
                st.markdown(f"**Action:** Connection appears to follow normal network patterns.")
                
            st.markdown(f"*(Threshold Used: {threshold_used:.4f})*")
        
    elif uploaded_file is None:
        st.warning("Please upload a KDD-style dataset CSV file to proceed.")
    elif 'trained_model' not in st.session_state:
        st.warning("Please run the Batch Analysis first (Step 2) to train the model before using the Real-Time Checker.")


if __name__ == "__main__":
    main()
