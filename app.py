import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from src.preprocess import process_city, get_current_weather  

st.title('Weather Analysis App')

uploaded_file = st.file_uploader("Загрузите файл с историческими данными", type=["csv", "xlsx"])

if uploaded_file is not None:

    if uploaded_file.name.endswith('.csv'):
        city_df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        city_df = pd.read_excel(uploaded_file)

    st.write("Загруженные данные:")
    st.write(city_df.head())

    cities = city_df['city'].unique()
    city_name = st.selectbox("Выберите город для анализа", cities)

    city_df = city_df[city_df['city'] == city_name]

    api_key = st.text_input("Введите API-ключ OpenWeatherMap", type="password")

    if api_key:
        current_temp = get_current_weather(city_name, api_key)

        if current_temp is None:
            st.error("Неверный API-ключ! Пожалуйста, проверьте правильность ключа.")
        else:
            st.success(f"Текущая температура в {city_name}: {current_temp:.2f}°C")

            current_date = datetime.now()
            current_month = current_date.month
            if current_month in [12, 1, 2]:
                current_season = 'winter'
            elif current_month in [3, 4, 5]:
                current_season = 'spring'
            elif current_month in [6, 7, 8]:
                current_season = 'summer'
            else:
                current_season = 'fall'

            result, _ = process_city(city_df)
            seasonal_stats = result['seasonal_stats']
            seasonal_row = seasonal_stats[seasonal_stats['season'] == current_season]

            if not seasonal_row.empty:
                seasonal_mean = seasonal_row['mean_temp'].values[0]
                seasonal_std = seasonal_row['std_temp'].values[0]
                lower_bound = seasonal_mean - 2 * seasonal_std
                upper_bound = seasonal_mean + 2 * seasonal_std

                if current_temp < lower_bound or current_temp > upper_bound:
                    st.warning(f"Текущая температура ({current_temp:.2f}°C) является аномальной для сезона '{current_season}'!")
                else:
                    st.info(f"Текущая температура ({current_temp:.2f}°C) находится в пределах нормального диапазона для сезона '{current_season}'.")
            else:
                st.warning(f"Нет данных о сезоне '{current_season}' для города {city_name}.")

    result, city_df = process_city(city_df)

    # Вывод статистики
    st.subheader(f"Описательная статистика для города {city_name}")
    st.write(f"Средняя температура: {result['average_temp']:.2f}°C")
    st.write(f"Минимальная температура: {result['min_temp']:.2f}°C")
    st.write(f"Максимальная температура: {result['max_temp']:.2f}°C")
    st.write(f"Направление тренда: {result['trend_direction']} (Наклон: {result['trend_slope']:.4f})")

    st.subheader("Сезонный профиль температуры")
    st.write(result['seasonal_stats'])

    st.subheader("Временной ряд температуры и аномалии")

    plt.figure(figsize=(12, 6))

    # Линия температуры
    plt.plot(city_df['timestamp'], city_df['temperature'], label='Temperature', color='blue')

    # Скользящее среднее
    plt.plot(city_df['timestamp'], city_df['rolling_mean'], label='Rolling Mean (30)', color='orange')

    # Аномалии
    plt.scatter(result['anomalies']['timestamp'], result['anomalies']['temperature'], 
                color='red', label='Anomalies', marker='x')

    # Оформление графика
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Trend and Anomalies for {result["city"]}')
    plt.legend()
    plt.grid(True)

    # Отображение графика в Streamlit
    st.pyplot(plt)
    plt.close()