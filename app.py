
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


# Get the folder containing this app.py file
BASE_DIR = Path(__file__).resolve().parent


# Load the saved machine learning model
with (BASE_DIR / "pipe.pkl").open("rb") as file:
    pipe = pickle.load(file)


# Load the laptop dataset
with (BASE_DIR / "df.pkl").open("rb") as file:
    df = pickle.load(file)


# Page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)


# Application title
st.title("💻 Laptop Price Predictor")
st.write("Enter your laptop specifications to estimate its price.")


# User inputs
company = st.selectbox(
    "Brand",
    sorted(df["Company"].dropna().unique())
)

laptop_type = st.selectbox(
    "Laptop Type",
    sorted(df["TypeName"].dropna().unique())
)

ram = st.selectbox(
    "RAM (GB)",
    [2, 4, 6, 8, 16, 24, 32, 64]
)

weight = st.number_input(
    "Weight (kg)",
    min_value=0.1,
    value=1.5,
    step=0.1
)

touchscreen = st.selectbox(
    "Touchscreen",
    ["No", "Yes"]
)

ips = st.selectbox(
    "IPS Display",
    ["Yes", "No"]
)

screen_size = st.number_input(
    "Screen Size (inches)",
    min_value=0.1,
    value=13.3,
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

cpu = st.selectbox(
    "CPU",
    sorted(df["Cpu brand"].dropna().unique())
)

hdd = st.selectbox(
    "HDD (GB)",
    [0, 128, 256, 512, 1024, 2048]
)

ssd = st.selectbox(
    "SSD (GB)",
    [0, 8, 128, 256, 512, 1024]
)

gpu = st.selectbox(
    "GPU",
    sorted(df["Gpu Brand"].dropna().unique())
)

os = st.selectbox(
    "Operating System",
    sorted(df["os"].dropna().unique())
)


# Predict laptop price
if st.button("Predict Price", type="primary"):

    # Convert Yes/No values to integers
    touchscreen_value = 1 if touchscreen == "Yes" else 0
    ips_value = 1 if ips == "Yes" else 0

    # Calculate pixels per inch
    x_res, y_res = map(int, resolution.split("x"))

    ppi = (
        np.sqrt(x_res ** 2 + y_res ** 2)
        / screen_size
    )

    # IMPORTANT:
    # Use a Pandas DataFrame with the exact
    # column names expected by the trained model.
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
            os
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

    try:
        # Model predicts log(price), so convert
        # the result back to the original scale.
        prediction = float(np.exp(pipe.predict(query)[0]))

        st.success(
            f"Estimated Laptop Price: ₹{prediction:,.0f}"
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
