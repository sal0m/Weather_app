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
    
def analyze_city(city_df, api_key):

    result, city_df = process_city(city_df)

    city_name = result['city']
    avg_temp = result['average_temp']
    min_temp = result['min_temp']
    max_temp = result['max_temp']
    seasonal_stats = result['seasonal_stats']
    anomalies = result['anomalies']

    print(f"City: {city_name}")
    print(f"Average Temperature: {avg_temp:.2f}°C")
    print(f"Min Temperature: {min_temp:.2f}°C")
    print(f"Max Temperature: {max_temp:.2f}°C")
    print(f"Trend Direction: {result['trend_direction']} (Slope: {result['trend_slope']:.4f})")

    print("\nSeasonal Temperature Profile:")
    print(seasonal_stats)

    current_temp = get_current_weather(city_name, api_key)
    if current_temp is not None:
        print(f"\nCurrent temperature in {city_name}: {current_temp:.2f}°C")

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

        seasonal_row = seasonal_stats[seasonal_stats['season'] == current_season]
        if not seasonal_row.empty:
            seasonal_mean = seasonal_row['mean_temp'].values[0]
            seasonal_std = seasonal_row['std_temp'].values[0]
            lower_bound = seasonal_mean - 2 * seasonal_std
            upper_bound = seasonal_mean + 2 * seasonal_std

            if current_temp < lower_bound or current_temp > upper_bound:
                print(f"Current temperature ({current_temp:.2f}°C) is ANOMALOUS for season '{current_season}'!")
            else:
                print(f"Current temperature ({current_temp:.2f}°C) is within the normal range for season '{current_season}'.")
        else:
            print(f"No seasonal data available for season '{current_season}'.")

    plt.figure(figsize=(12, 6))
    plt.plot(city_df['timestamp'], city_df['temperature'], label='Temperature', color='blue')
    plt.plot(city_df['timestamp'], city_df['rolling_mean'], label='Rolling Mean (30)', color='orange')

    plt.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', label='Anomalies', marker='x')

    plt.xlabel('Timestamp')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Trend and Anomalies for {city_name}')
    plt.legend()
    plt.grid(True)
    plt.show()