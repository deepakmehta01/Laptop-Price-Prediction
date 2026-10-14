from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent


def parse_resolution(resolution: object) -> float:
    if pd.isna(resolution):
        return np.nan

    text = str(resolution)
    match = re.search(r"(\d+)x(\d+)", text)
    if not match:
        return np.nan

    width, height = map(int, match.groups())
    return float(np.sqrt(width**2 + height**2))


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = df.rename(columns={"OpSys": "os"})

    df["Ram"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).astype(int)
    df["Weight"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).astype(float)
    df["Touchscreen"] = df["ScreenResolution"].astype(str).str.contains("Touch", case=False, na=False).astype(int)
    df["IPS"] = df["ScreenResolution"].astype(str).str.contains("IPS", case=False, na=False).astype(int)

    df["PPI"] = df.apply(
        lambda row: parse_resolution(row["ScreenResolution"]) / float(row["Inches"]),
        axis=1,
    )

    def memory_value(text: str, keyword: str) -> int:
        if pd.isna(text):
            return 0
        text = str(text)
        if keyword in text.upper():
            numbers = re.findall(r"(\d+)", text)
            return int(numbers[0]) if numbers else 0
        return 0

    df["HDD"] = df["Memory"].apply(lambda value: memory_value(value, "HDD"))
    df["SSD"] = df["Memory"].apply(lambda value: memory_value(value, "SSD"))
    df["SSD"] = df.apply(
        lambda row: row["SSD"] if row["SSD"] else memory_value(row["Memory"], "FLASH"),
        axis=1,
    )

    df["Cpu brand"] = df["Cpu"].astype(str).str.split().str[:2].str.join(" ")
    df["Gpu Brand"] = df["Gpu"].astype(str).str.split().str[:2].str.join(" ")
    df["os"] = df["os"].astype(str).str.replace("Windows 10", "Windows", regex=False)
    df["os"] = df["os"].astype(str).str.replace("Mac OS X", "macOS", regex=False)

    return df[
        [
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
            "os",
            "Price",
        ]
    ]


def build_model() -> Pipeline:
    raw_df = pd.read_csv(ROOT / "laptop_data.csv")
    feature_df = clean_data(raw_df)

    cat_cols = ["Company", "TypeName", "Cpu brand", "Gpu Brand", "os"]
    num_cols = ["Ram", "Weight", "Touchscreen", "IPS", "PPI", "HDD", "SSD"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )
    model.fit(feature_df.drop(columns=["Price"]), np.log(feature_df["Price"]))

    with (ROOT / "df.pkl").open("wb") as fh:
        pickle.dump(feature_df.drop(columns=["Price"]), fh)

    with (ROOT / "pipe.pkl").open("wb") as fh:
        pickle.dump(model, fh)

    return model


if __name__ == "__main__":
    build_model()
    print(f"Saved model and feature data to {ROOT}")
