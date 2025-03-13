import pandas as pd
from config import RU_HOLIDAYS

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Оптимизация типов данных
    df = df.astype({
        'restaurant': 'category',
        'dish': 'category',
        'amount': 'int32'
    })

    # Проверка наличия колонок
    required_columns = ['restaurant', 'dish', 'date', 'amount']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Отсутствуют колонки: {required_columns}")

    # Преобразование даты
    df['date'] = pd.to_datetime(df['date'])

    # Временные признаки
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_holiday'] = df['date'].isin(RU_HOLIDAYS).astype(int)

    # Лаговые признаки
    for lag in [1, 2, 3, 7]:
        df[f'lag_{lag}'] = df.groupby(['restaurant', 'dish'], observed=True)['amount'].shift(lag)

    # Скользящее среднее
    df['rolling_7_mean'] = (
        df.groupby(['restaurant', 'dish'], observed=True)['amount']
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    df = df.dropna()
    if df.empty:
        raise ValueError("DataFrame пуст после создания признаков. Проверьте входные данные.")
    return df