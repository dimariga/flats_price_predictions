import joblib
import streamlit as st
import pandas as pd
import json
st.header('Цена квартиры')

PATH_DATA = "flats2.csv"
PATH_UNIQUE_VALUES = "unique_values.json"
PATH_MODEL = "lr_pipeline.sav"

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data = data.sample(5000)
    return data

@st.cache_data
def load_model(PATH_MODEL):
    model = joblib.load(PATH_MODEL)
    return model

df = load_data(PATH_DATA)
st.write(df[:4])

with open(PATH_UNIQUE_VALUES) as file:
    dict_unique = json.load(file)
city = st.sidebar.selectbox("Город", (dict_unique["city"]))
house_wall_type = st.sidebar.selectbox("Тип стен", (dict_unique["house_wall_type"]))
renovation = st.sidebar.selectbox("Ремонт", ([
'Отсутствует',
'Частичный ремонт',
'Средний',
'Хороший',
'Отличный',
'Предчистовая отделка',
'Чистовая отделка',
'С отделкой',
'Косметический',
'Дизайнерский',
'Евроремонт']))


area = st.sidebar.slider(
    "Площадь",
    min_value=0.0,
    max_value=max(dict_unique["area"])
)

rooms = st.sidebar.slider(
    "Количество комнат",
    min_value=0,
    max_value=max(dict_unique["rooms"])
)

build_year = st.sidebar.slider(
    "Год постройки",
    min_value=min(dict_unique["build_year"]),
    max_value=max(dict_unique["build_year"])
)

floor = st.sidebar.slider(
    "Этаж",
    min_value=min(dict_unique["floor"]),
    max_value=max(dict_unique["floor"])
)

house_floors = st.sidebar.slider(
    "Этажность дома",
    min_value=floor,
    max_value=max(dict_unique["house_floors"])
)

dict_data = {
    "city": city,
    "house_wall_type": house_wall_type,
    "renovation": renovation,
    "area": area,
    "rooms": rooms,
    "build_year": build_year,
    "floor": floor,
    "house_floors": house_floors,
}
data_predict = pd.DataFrame([dict_data])
model = joblib.load(PATH_MODEL)
button = st.button("Предварительная цена")
if button:
    result = str(round(model.predict(data_predict)[0])) + " руб."
    st.write(result)
