from pathlib import Path
import holidays

# Пути к данным
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "input_data"
DB_PATH = BASE_DIR / "sushi_forecast.db"

# Настройки ресторанов
RESTAURANTS = {
}

# Праздники для России
RU_HOLIDAYS = holidays.Russia()

# Настройки модели
MODEL_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1,  # Добавляем использование всех ядер
    'verbose': 1  # Включаем вывод прогресса
}


DB_ENGINE_CONFIG = {
    'pool_size': 10,
    'max_overflow': 5,
    'pool_pre_ping': True
}