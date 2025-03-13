from pathlib import Path
import pandas as pd
from sqlalchemy.orm import Session
from database.crud import get_or_create_restaurant, get_or_create_dish, bulk_create_sales


def parse_excel(file_path: Path) -> pd.DataFrame:
    """Парсинг Excel-файла с валидацией"""
    try:
        # Чтение файла
        df = pd.read_excel(
            file_path,
            parse_dates=['OpenDate.Typed'],
            usecols=['OpenDate.Typed', 'DishName', 'DishAmountInt', 'DishDiscountSumInt', 'CloseTime']
        )

        # Проверка наличия обязательных колонок
        required_columns = ['OpenDate.Typed', 'DishName', 'DishAmountInt']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Файл {file_path.name}: отсутствуют колонки {missing}")

        # Переименование
        df.rename(columns={
            'OpenDate.Typed': 'date',
            'DishName': 'dish',
            'DishAmountInt': 'amount',
            'DishDiscountSumInt': 'discount',
            'CloseTime': 'close_time'
        }, inplace=True)

        # Обработка даты
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
        df = df.dropna(subset=['date'])  # Удаление строк с некорректными датами

        return df

    except Exception as e:
        raise ValueError(f"Ошибка парсинга {file_path.name}: {str(e)}")


def load_restaurant_data(session: Session, file_path: Path, restaurant_name: str):
    try:
        df = parse_excel(file_path)
        restaurant = get_or_create_restaurant(session, restaurant_name)

        # Создаем блюда и сохраняем их ID
        dish_ids = {}
        for dish_name in df['dish'].unique():
            if isinstance(dish_name, str):  # Проверка типа
                dish = get_or_create_dish(session, dish_name.strip())  # Удаление пробелов
                dish_ids[dish_name] = dish.id
            else:
                raise ValueError(f"Некорректное название блюда: {dish_name}")

        # Применяем маппинг
        df['restaurant_id'] = restaurant.id
        df['dish_id'] = df['dish'].map(dish_ids)

        # Фильтрация строк с некорректными dish_id
        df = df.dropna(subset=['dish_id'])

        # Запись в БД
        df[['date', 'amount', 'restaurant_id', 'dish_id']].to_sql(
            'sales',
            session.connection(),
            if_exists='append',
            index=False
        )
        session.commit()
        print(f"[SUCCESS] Загружено {len(df)} записей для {restaurant_name}")
        return True, restaurant_name
    except Exception as e:
        session.rollback()
        print(f"[ERROR] Ошибка загрузки {restaurant_name}: {str(e)}")
        return False, f"{restaurant_name}: {str(e)}"