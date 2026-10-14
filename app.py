
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

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 55%, #f8faff 100%);
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #111827, #312e81, #6d28d9);
    padding: 36px 38px;
    border-radius: 24px;
    color: white;
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
}

.hero h1 {
    color: white;
    font-size: clamp(28px, 4vw, 42px);
    font-weight: 800;
    margin: 18px 0 10px 0;
}

.hero p {
    color: #ddd6fe;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 0;
}

/* Section headings */
.section-heading {
    font-size: 20px;
    font-weight: 750;
    color: #1e1b4b;
    margin: 10px 0 5px 0;
}

.section-subtitle {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 18px;
}

/* Input labels */
label, .stSelectbox label, .stNumberInput label {
    font-weight: 600 !important;
    color: #334155 !important;
}

/* Input fields */
div[data-baseweb="select"] > div {
    border-radius: 10px;
    border-color: #dbe2f0;
    background-color: #ffffff;
}

div[data-baseweb="input"] > div {
    border-radius: 10px;
    border-color: #dbe2f0;
    background-color: #ffffff;
}

/* Predict button */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    font-weight: 700;
    font-size: 16px;
    color: white;
    background: linear-gradient(110deg, #4f46e5, #7c3aed);
    box-shadow: 0 7px 18px rgba(99, 102, 241, 0.22);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    color: white;
    border: none;
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(99, 102, 241, 0.32);
}

/* Result card */
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
    color: #6d28d9;
    text-transform: uppercase;
    letter-spacing: 1.4px;
}

.price-value {
    color: #312e81;
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;
    margin: 12px 0;
    overflow-wrap: anywhere;
}

.price-note {
    color: #64748b;
    font-size: 13px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding-top: 35px;
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.78);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px;
}

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
# LOAD SAVED MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    with (BASE_DIR / "pipe.pkl").open("rb") as file:
        model = pickle.load(file)
    return model


@st.cache_data
def load_data():
    with (BASE_DIR / "df.pkl").open("rb") as file:
        data = pickle.load(file)
    return data


try:
    pipe = load_model()
    df = load_data()
except Exception as error:
    st.error(f"Unable to load the saved model or dataset: {error}")
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
# LAPTOP SPECIFICATIONS FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-heading">⚙️ Configure your laptop</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Choose the specifications below to get your price estimate.</div>',
    unsafe_allow_html=True
)

with st.form("laptop_form"):

    # Brand and laptop category
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

    # Hardware
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

    # Display
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

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "✨ Predict Laptop Price"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:
    try:
        touchscreen_value = 1 if touchscreen == "Yes" else 0
        ips_value = 1 if ips == "Yes" else 0

        x_res, y_res = map(int, resolution.split("x"))

        ppi = np.sqrt(x_res**2 + y_res**2) / screen_size

        # Use named columns because the saved model's
        # ColumnTransformer expects a Pandas DataFrame.
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
            st.error("The model returned an invalid price. Please try different specifications.")
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
    Built with ❤️ using Streamlit, Python & Machine Learning
</div>
""", unsafe_allow_html=True)
