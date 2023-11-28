import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import json
import joblib

PATH_UNIQUE_VALUES = "unique_values.json"

df = pd.read_csv("flats.csv")
categorical_features = ["city", "house_wall_type", "renovation"]
numeric_feautures = ["area", "floor", "rooms", "house_floors", "build_year"]
passthrough_feats = ["price"]
euro_renovation = pd.read_csv("euro_renovation.xlsx.csv")
df = pd.concat([df, euro_renovation])
df["area"] = df["area"].str.replace(",", ".")
df["area"] = df["area"].astype(float)
df["price"] = df["price_sq"] * df["area"]
df.dropna(subset=["price"], inplace=True)
df["floor"] = df["floor"].astype(int)
df["rooms"] = df["rooms"].astype(int)
df["house_floors"] = df["house_floors"].astype(int)
df['house_wall_type'].fillna(df['house_wall_type'].mode()[0], inplace=True)
df["renovation"] = df["renovation"].replace(
    "косметический", "Косметический").replace(
    "Косметический ремонт", "Косметический").replace(
    "cosmetic", "Косметический").replace(
    "Евро", "Евроремонт").replace(
    "евроремонт", "Евроремонт").replace(
    "Евростандарт", "Евроремонт").replace(
    "euro", "Евроремонт").replace(
    "хорошее", "Хороший").replace(
    "Хорошее", "Хороший").replace(
    "Требует ремонта", "Отсутствует").replace(
    "no", "Отсутствует").replace(
    "Требуется", "Отсутствует").replace(
    "Требуется ремонт", "Отсутствует").replace(
    "Без ремонта", "Отсутствует").replace(
    "требуется косм.ремонт", "Отсутствует").replace(
    "design", "Дизайнерский").replace(
    "Чистовая", "Чистовая отделка").replace(
    "чистовая", "Чистовая отделка").replace(
    "чистовая отделка", "Чистовая отделка").replace(
    "Под чистовой ремонт", "Предчистовая отделка").replace(
    "предчистовая", "Предчистовая отделка").replace(
    "Предчистовая", "Предчистовая отделка").replace(
    "Подготовка под чистовую отделку", "Предчистовая отделка").replace(
    "под самоотделку", "Предчистовая отделка").replace(
    "отличное", "Отличный").replace(
    "Под ключ", "Отличный").replace(
    "нормальное", "Средний").replace(
    "Произведен", "Средний")
df.dropna(subset=["build_year"], inplace=True)
df["build_year"] = df["build_year"].astype(int)
df["build_year"] = df["build_year"].replace(23, 2023)
df = df[["renovation", "area", "city", "floor", "rooms", "house_floors", "house_wall_type",  "build_year", "price"]]
df = df[df.price.between(df.price.quantile(0.05), df.price.quantile(0.95))]
df = df[df.area.between(df.area.quantile(0.01), df.area.quantile(0.99))]
X = df.drop(columns="price")
y = df["price"]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)
preprocessor = make_column_transformer((StandardScaler(), numeric_feautures), (OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features))
clf = make_pipeline(preprocessor, RandomForestRegressor(max_depth=15, random_state=10))
clf.fit(X_train, y_train)
df.to_csv("flats2.csv", index=False)

PATH_MODEL = "lr_pipeline.sav"
joblib.dump(clf, PATH_MODEL)
dict_unique = {key: X[key].unique().tolist() for key in X.columns}
with open(PATH_UNIQUE_VALUES, "w") as file:
    json.dump(dict_unique, file)
