from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DB_PATH, RU_HOLIDAYS


def generate_features_for_date(target_date: datetime, session, restaurant_name: str):
    """Генерация признаков для прогноза"""
    # Получение исторических данных
    history = pd.read_sql(f"""
        SELECT 
            s.date,
            d.name AS dish,
            s.amount,
            r.name AS restaurant
        FROM sales s
        JOIN dishes d ON s.dish_id = d.id
        JOIN restaurants r ON s.restaurant_id = r.id
        WHERE r.name = '{restaurant_name}'
        ORDER BY s.date DESC
        LIMIT 1000
    """, session.bind)

    # Генерация признаков
    features = []
    for dish in history['dish'].unique():
        dish_data = history[history['dish'] == dish].copy()
        dish_data = dish_data.sort_values('date').reset_index(drop=True)

        # Последние доступные данные
        last_row = dish_data.iloc[-1]

        features.append({
            'restaurant': restaurant_name,
            'dish': dish,
            'date': target_date,
            'day_of_week': target_date.weekday(),
            'month': target_date.month,
            'is_weekend': int(target_date.weekday() in [5, 6]),
            'is_holiday': int(target_date.date() in RU_HOLIDAYS),
            'lag_1': last_row['amount'],
            'lag_2': dish_data.iloc[-2]['amount'] if len(dish_data) > 1 else last_row['amount'],
            'lag_3': dish_data.iloc[-3]['amount'] if len(dish_data) > 2 else last_row['amount'],
            'lag_7': dish_data.iloc[-7]['amount'] if len(dish_data) > 6 else last_row['amount'],
            'rolling_7_mean': dish_data['amount'].tail(7).mean()
        })

    return pd.DataFrame(features)


def save_forecast(predictions: pd.DataFrame, session):
    records = predictions[['date', 'dish', 'restaurant_id', 'prediction']].to_dict('records')
    session.bulk_insert_mappings(Forecast, records)
    session.commit()