import streamlit as st
import pickle
import numpy as np
import numpy as np
import streamlit as st
import pickle

# Load model and dataset
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")
    st.stop()  # Stop execution if files are missing

st.title("💻 Laptop Price Predictor")

# Check if columns exist in dataset
df.columns = df.columns.str.strip().str.lower()  # Clean column names to avoid issues

# Define input fields
company = st.selectbox('Brand', df['company'].unique() if 'company' in df.columns else [])
laptop_type = st.selectbox('Type', df['typename'].unique() if 'typename' in df.columns else [])
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop (kg)', min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
touchscreen = 1 if touchscreen == 'Yes' else 0
ips = st.selectbox('IPS Display', ['No', 'Yes'])
ips = 1 if ips == 'Yes' else 0
screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
X_res, Y_res = map(int, resolution.split('x'))
ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size  # Calculate PPI
cpu = st.selectbox('CPU', df['cpu brand'].unique() if 'cpu brand' in df.columns else ['Intel', 'AMD', 'Others'])
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['gpu brand'].unique() if 'gpu brand' in df.columns else ['NVIDIA', 'AMD', 'Intel', 'Others'])
os = st.selectbox('Operating System', df['os'].unique() if 'os' in df.columns else ['Windows', 'Linux', 'macOS'])

# Ensure that all the necessary variables are defined
if company and laptop_type and cpu and gpu and os:
    # Button to predict price
    if st.button('💰 Predict Price'):
        query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

        # Reshape query for prediction
        query = query.reshape(1, -1)

        try:
            predicted_price = np.exp(pipe.predict(query)[0])  # Apply exponential to reverse log transformation
            st.success(f"The predicted price of this laptop is ₹{int(predicted_price):,}")
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
else:
    st.warning("Please fill in all the details above to make a prediction.")
