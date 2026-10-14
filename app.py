
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* General page */
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef2ff, #f8faff);
    color: #1e293b !important;
    font-family: 'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* General text and labels */
.stApp p,
.stApp label,
.stMarkdown,
.stMarkdown p,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4,
.stMarkdown h5 {
    color: #1e293b;
}

/* Hero header */
.hero {
    background: linear-gradient(120deg, #111827, #312e81, #6d28d9);
    padding: 36px 38px;
    border-radius: 24px;
    color: #ffffff !important;
    margin-bottom: 28px;
    box-shadow: 0 15px 40px rgba(49, 46, 129, 0.18);
}

.hero-label {
    display: inline-block;
    padding: 7px 12px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #ffffff !important;
}

.hero h1 {
    color: #ffffff !important;
    font-size: clamp(28px, 4vw, 42px);
    font-weight: 800;
    margin: 18px 0 10px 0;
}

.hero p {
    color: #e0e7ff !important;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 0;
}

/* Section headings */
.section-heading {
    font-size: 20px;
    font-weight: 750;
    color: #1e1b4b !important;
    margin: 10px 0 5px 0;
}

.section-subtitle {
    font-size: 13px;
    color: #64748b !important;
    margin-bottom: 18px;
}

/* Input labels */
.stSelectbox label,
.stNumberInput label,
.stRadio label,
.stCheckbox label {
    color: #334155 !important;
    font-weight: 600 !important;
}

/* Selectbox and input backgrounds */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

/* Selected dropdown text */
div[data-baseweb="select"] div,
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* Number input text */
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
.stNumberInput input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background-color: #ffffff !important;
    caret-color: #4f46e5 !important;
}

/* Dropdown arrow */
div[data-baseweb="select"] svg {
    fill: #475569 !important;
    color: #475569 !important;
}

/* Dropdown menu */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[data-testid="stSelectboxVirtualDropdown"] {
    background-color: #ffffff !important;
}

/* Dropdown choices */
div[role="option"],
ul[data-testid="stSelectboxVirtualDropdown"] li {
    color: #111827 !important;
    background-color: #ffffff !important;
}

/* Dropdown hover */
div[role="option"]:hover,
ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
    color: #312e81 !important;
    background-color: #eef2ff !important;
}

/* Placeholder */
.stNumberInput input::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
}

/* Focus effects */
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px #6366f133 !important;
}

/* Form container */
div[data-testid="stForm"] {
    background: rgba(255,255,255,0.88);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px;
}

/* Predict button */
.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    border: none !important;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    font-weight: 700;
    font-size: 16px;
    color: #ffffff !important;
    background: linear-gradient(110deg, #4f46e5, #7c3aed) !important;
    box-shadow: 0 7px 18px rgba(99, 102, 241, 0.22);
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    color: #ffffff !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(99, 102, 241, 0.32);
}

/* Prediction result */
.price-card {
    background: linear-gradient(135deg, #ffffff, #f5f3ff);
    border: 1px solid #ddd6fe;
    border-radius: 22px;
    padding: 28px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0 12px 35px rgba(79, 70, 229, 0.09);
}

.price-label {
    font-size: 13px;
    font-weight: 700;
    color: #6d28d9 !important;
    text-transform: uppercase;
    letter-spacing: 1.4px;
}

.price-value {
    color: #312e81 !important;
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;
    margin: 12px 0;
    overflow-wrap: anywhere;
}

.price-note {
    color: #64748b !important;
    font-size: 13px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b !important;
    font-size: 12px;
    padding-top: 35px;
}

.footer span {
    color: #8b5cf6 !important;
}

/* Smaller screens */
@media (max-width: 600px) {
    .hero {
        padding: 25px 22px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    with (BASE_DIR / "pipe.pkl").open("rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    with (BASE_DIR / "df.pkl").open("rb") as file:
        return pickle.load(file)


try:
    pipe = load_model()
    df = load_data()
except Exception as error:
    st.error(f"Unable to load model or dataset: {error}")
    st.stop()


# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <span class="hero-label">✦ Smart Price Estimator</span>
    <h1>Find Your Perfect Laptop 💻</h1>
    <p>
        Customize your laptop specifications and get an estimated price
        powered by machine learning.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LAPTOP CONFIGURATION FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">⚙️ Configure Your Laptop</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Choose your specifications to estimate the price.</div>',
    unsafe_allow_html=True
)

with st.form("laptop_form"):

    st.markdown("##### 🏷️ Brand & Category")

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox(
            "Laptop Brand",
            sorted(df["Company"].dropna().unique())
        )

    with col2:
        laptop_type = st.selectbox(
            "Laptop Type",
            sorted(df["TypeName"].dropna().unique())
        )

    st.markdown("---")
    st.markdown("##### 🚀 Performance & Hardware")

    col1, col2 = st.columns(2)

    with col1:
        ram = st.selectbox(
            "RAM (GB)",
            [2, 4, 6, 8, 16, 24, 32, 64],
            index=3
        )

        cpu = st.selectbox(
            "Processor",
            sorted(df["Cpu brand"].dropna().unique())
        )

        hdd = st.selectbox(
            "HDD Storage (GB)",
            [0, 128, 256, 512, 1024, 2048],
            index=0
        )

    with col2:
        weight = st.number_input(
            "Laptop Weight (kg)",
            min_value=0.1,
            max_value=10.0,
            value=1.5,
            step=0.1
        )

        gpu = st.selectbox(
            "Graphics Card",
            sorted(df["Gpu Brand"].dropna().unique())
        )

        ssd = st.selectbox(
            "SSD Storage (GB)",
            [0, 8, 128, 256, 512, 1024],
            index=3
        )

    st.markdown("---")
    st.markdown("##### 🖥️ Display & Operating System")

    col1, col2 = st.columns(2)

    with col1:
        screen_size = st.number_input(
            "Screen Size (inches)",
            min_value=8.0,
            max_value=25.0,
            value=15.6,
            step=0.1
        )

        resolution = st.selectbox(
            "Screen Resolution",
            [
                "1920x1080",
                "1366x768",
                "1600x900",
                "3840x2160",
                "3200x1800",
                "2880x1800",
                "2560x1600",
                "2560x1440",
                "2304x1440"
            ]
        )

    with col2:
        touchscreen = st.selectbox(
            "Touchscreen",
            ["No", "Yes"]
        )

        ips = st.selectbox(
            "IPS Display",
            ["Yes", "No"]
        )

        operating_system = st.selectbox(
            "Operating System",
            sorted(df["os"].dropna().unique())
        )

    submitted = st.form_submit_button("✨ Predict Laptop Price")


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:
    try:
        touchscreen_value = 1 if touchscreen == "Yes" else 0
        ips_value = 1 if ips == "Yes" else 0

        x_res, y_res = map(int, resolution.split("x"))
        ppi = np.sqrt(x_res**2 + y_res**2) / screen_size

        query = pd.DataFrame(
            [[
                company,
                laptop_type,
                ram,
                weight,
                touchscreen_value,
                ips_value,
                ppi,
                cpu,
                hdd,
                ssd,
                gpu,
                operating_system
            ]],
            columns=[
                "Company",
                "TypeName",
                "Ram",
                "Weight",
                "Touchscreen",
                "IPS",
                "PPI",
                "Cpu brand",
                "HDD",
                "SSD",
                "Gpu Brand",
                "os"
            ]
        )

        prediction = float(np.exp(pipe.predict(query)[0]))

        if not np.isfinite(prediction) or prediction <= 0:
            st.error("The model returned an invalid price. Try different specifications.")
        else:
            st.markdown(f"""
            <div class="price-card">
                <div class="price-label">✦ Estimated Laptop Price</div>
                <div class="price-value">₹ {prediction:,.0f}</div>
                <div class="price-note">
                    This is a machine-learning estimate, not a guaranteed market price.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.caption(
                f"Configuration: {company} · {ram} GB RAM · "
                f"{screen_size:.1f}-inch display · {cpu}"
            )

    except Exception as error:
        st.error(f"Unable to predict the price: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Built with <span>♥</span> using Streamlit, Python & Machine Learning
</div>
""", unsafe_allow_html=True)
