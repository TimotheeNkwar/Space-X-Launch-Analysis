# scripts/main.py
import requests
import pandas as pd
import sqlite3
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def fetch_and_clean_data():
    url = "https://api.spacexdata.com/v4/launches"
    df = pd.DataFrame(requests.get(url).json())
    rockets = pd.DataFrame(requests.get("https://api.spacexdata.com/v4/rockets").json())
    launchpads = pd.DataFrame(requests.get("https://api.spacexdata.com/v4/launchpads").json())
    rocket_map = dict(zip(rockets["id"], rockets["name"]))
    launchpad_map = dict(zip(launchpads["id"], launchpads["name"]))
    df = df[["name", "date_utc", "success", "rocket", "launchpad"]]
    df["rocket_name"] = df["rocket"].map(rocket_map)
    df["launchpad_name"] = df["launchpad"].map(launchpad_map)
    df["year"] = pd.to_datetime(df["date_utc"]).dt.year
    df["month"] = pd.to_datetime(df["date_utc"]).dt.month
    df["success"] = df["success"].fillna(False)
    df.to_csv("data/spacex_cleaned.csv", index=False)
    conn = sqlite3.connect("data/spacex.db")
    df.to_sql("launches", conn, if_exists="replace", index=False)
    conn.close()
    return df

def plot_launches_per_year(df):
    launches_per_year = df.groupby("year").size()
    sns.barplot(x=launches_per_year.index, y=launches_per_year.values)
    plt.title("Lancements par an")
    plt.savefig("presentation/launches_per_year.png")
    plt.close()

def train_model(df):
    X = df[["year", "rocket_name", "launchpad_name"]]
    y = df["success"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(), ["rocket_name", "launchpad_name"])],
        remainder="passthrough"
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    df = fetch_and_clean_data()
    plot_launches_per_year(df)
    model = train_model(df)
    print("Project setup complete.")