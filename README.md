# Weather_app

# Анализ температурных данных и мониторинг текущей температуры

## Описание задания

В рамках данного задания необходимо провести анализ температурных данных и мониторинг текущей температуры через OpenWeatherMap API. Задача состоит в обработке исторических данных о температуре, выявлении аномалий и определении трендов, а также в мониторинге текущей температуры для выбранных городов.

## Цели задания

1. Провести анализ временных рядов, включая:
   - Вычисление скользящего среднего и стандартного отклонения для сглаживания температурных колебаний.
   - Определение аномалий на основе отклонений температуры от $ \text{скользящее среднее} \pm 2\sigma $.
   - Построение долгосрочных трендов изменения температуры.
   - Любые дополнительные исследования будут вам в плюс.

2. Осуществить мониторинг текущей температуры:
   - Получить текущую температуру через OpenWeatherMap API.
   - Сравнить её с историческим нормальным диапазоном для текущего сезона.

3. Разработать интерактивное приложение:
   - Дать пользователю возможность выбрать город.
   - Отобразить результаты анализа температур, включая временные ряды, сезонные профили и аномалии.
   - Провести анализ текущей температуры в контексте исторических данных.

## Описание данных

Данные о температуре находятся в файле temperature_data.csv, который содержит следующие колонки:

- city: Название города.
- timestamp: Дата (с шагом в 1 день).
- temperature: Среднесуточная температура (в °C).
- season: Сезон года (зима, весна, лето, осень).

Пример структуры данных:

```csv
city,timestamp,temperature,season
Berlin,2023-01-01,3.5,winter
Cairo,2023-01-01,20.1,winter
...
```

## Приложение доступно по ссылке:

https://weatherapp-rxnpnvjnzgbw44nvdo3d2n.streamlit.app/