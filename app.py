# app.py
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from linear_regression import *

# Use minimalist white theme for plots
import seaborn as sns
sns.set_theme(style="whitegrid") 

# Page settings
st.set_page_config(page_title="House Price Prediction", layout="centered", initial_sidebar_state="expanded")

# Custom Header
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h4 {
        text-align: center;
    }
    .css-18e3th9 {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: #00A1E4;'>🏡 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color: #FFD23F;'>Multiple Linear Regression with Gradient Descent</h4><br>", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header(" ⚙️ Tinker around")
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
steps = st.sidebar.slider("Training Steps", 100, 5000, 1000, 100)

# Load dataset and train
x, y, x1, x2, x3, df = load_and_preprocess("kc_house_data.csv")
b, residual_list = train_gradient_descent(x, y, lr=lr, steps=steps)

# Tabs for layout
loss_tab, prediction_tab, plot3d_tab = st.tabs(["📉 Loss Curve", "📊 Predictions", "🌐 3D Visualization"])

with loss_tab:
    st.subheader("📉 Training Loss Curve")
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    ax_loss.plot(range(len(residual_list)), residual_list, color='#ff6600', linewidth=2)
    ax_loss.set_xlabel("Iterations", fontsize=11)
    ax_loss.set_ylabel("Mean Squared Error", fontsize=11)
    ax_loss.set_title("Loss over Time", fontsize=13)
    ax_loss.grid(True, linestyle='dotted', alpha=0.5)
    st.pyplot(fig_loss)

with prediction_tab:
    st.subheader("📊 Actual vs Predicted Prices")
    fig_pred = plot_predictions(x, b, df)
    st.pyplot(fig_pred)

    st.markdown("### 🔍 Predict a New House Price")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        sqft = st.number_input("Sqft Living", min_value=100, max_value=10000, value=1500)
    with col2:
        bed = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    with col3:
        bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

    if st.button("✨ Predict Price", use_container_width=True):
        price = predict_price(sqft, bed, bath, b, df)
        st.success(f"💰 Estimated Price: **${price:,.2f}**")

with plot3d_tab:
    st.subheader("🌐 3D Regression Plane & Predictions")
    st.caption("Rotate, Zoom, and Explore the Multivariate Fit")
    fig_3d = plot_3d_regression(x2, x3, y, x, b)
    st.pyplot(fig_3d, use_container_width=True)

# Footer
st.markdown("""
    <hr style="margin-top: 2em;">
    <div style='text-align: center; color: grey;'>
        Built with ❤️ using <b>Streamlit</b> | Machine Learning | <b>Dhyey Savaliya ⚡️</b>
    </div>
""", unsafe_allow_html=True)
