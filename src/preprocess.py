import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
import requests

pd.options.mode.chained_assignment = None

warnings.simplefilter('ignore', category=UserWarning)

def process_city(city_df):
    city_name = city_df['city'].iloc[0]

    city_df['timestamp'] = pd.to_datetime(city_df['timestamp'])

    city_df = city_df.sort_values(by='timestamp')

    # 1. Скользящее среднее и стандартное отклонение
    city_df['rolling_mean'] = city_df['temperature'].rolling(window=30, min_periods=1).median()
    city_df['rolling_std'] = city_df['temperature'].rolling(window=30, min_periods=1).std()

    # 2. Выявление аномалий
    lower_bound = city_df['rolling_mean'] - 2 * city_df['rolling_std']
    upper_bound = city_df['rolling_mean'] + 2 * city_df['rolling_std']
    city_df['is_anomaly'] = (city_df['temperature'] < lower_bound) | (city_df['temperature'] > upper_bound)
    anomalies = city_df[city_df['is_anomaly']]

    # 3. Сезонный профиль (mean и std для исходной температуры)
    seasonal_stats = city_df.groupby('season').agg(
        mean_temp=('temperature', 'mean'),
        std_temp=('temperature', 'std')
    ).reset_index()

    # 4. Тренд температуры с помощью линейной регрессии
    city_df['timestamp_numeric'] = (city_df['timestamp'] - city_df['timestamp'].min()).dt.days
    X = city_df['timestamp_numeric'].values.reshape(-1, 1)
    y = city_df['temperature'].values

    model = LinearRegression()
    model.fit(X, y)
    trend_slope = model.coef_[0]
    trend_direction = 'positive' if trend_slope > 0 else 'negative'

    # 5. Средняя, минимальная и максимальная температура
    avg_temp = city_df['temperature'].mean()
    min_temp = city_df['temperature'].min()
    max_temp = city_df['temperature'].max()

    # Возвращаем результаты
    result = {
        'city': city_name,
        'average_temp': avg_temp,
        'min_temp': min_temp,
        'max_temp': max_temp,
        'trend_direction': trend_direction,
        'trend_slope': trend_slope,
        'seasonal_stats': seasonal_stats,
        'anomalies': anomalies[['timestamp', 'temperature']]
    }

    return result, city_df

def get_current_weather(city, api_key):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        return data['main']['temp']  # Возвращаем текущую температуру
    else:
        print(f"Error fetching weather data for {city}: {data.get('message', '')}")
        return None
