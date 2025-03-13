import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm
import sys
from database.models import Base
from data.loaders import load_restaurant_data
from features.engineering import create_features
from models.train import train_model
from models.predict import generate_features_for_date
from config import DB_PATH, DATA_DIR
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


class ConsoleUI:
    @staticmethod
    def print_header():
        print("\n🔮 Sushi Sales Forecast Tool")
        print("=" * 40)

    @staticmethod
    def progress(iterable, desc="Processing"):
        return tqdm(iterable, desc=desc.ljust(25), bar_format="{l_bar}{bar:40}{r_bar}", file=sys.stdout)

    @staticmethod
    def print_success(msg):
        print(f"✅ {msg}")

    @staticmethod
    def print_error(msg):
        print(f"❌ {msg}", file=sys.stderr)

    @staticmethod
    def print_warning(msg):
        print(f"⚠️  {msg}")


def save_report(df, filename="forecast_report"):
    """Сохраняет отчёт в CSV и текстовый файл"""
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = report_dir / f"{filename}_{timestamp}.csv"
    txt_path = report_dir / f"{filename}_{timestamp}.txt"

    # Сохранение в CSV
    df.to_csv(csv_path, index=False)

    # Генерация текстового отчёта
    report = [
        f"Отчёт создан: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        f"Прогноз на дату: {df['date'].max()}",
        "\nПрогноз продаж:",
        tabulate(
            df[['restaurant', 'dish', 'prediction']],
            headers=['Ресторан', 'Блюдо', 'Прогноз, шт'],
            tablefmt='grid',
            numalign="center"
        )
    ]

    # Сохранение текстового отчёта
    with open(txt_path, 'w') as f:
        f.write("\n".join(report))

    return csv_path, txt_path


def process_restaurant(db_path, file_path):
    """Обработка данных для одного ресторана"""
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=engine)
        with Session() as session:
            # Имя ресторана = имя файла без расширения
            restaurant_name = file_path.stem
            load_restaurant_data(session, file_path, restaurant_name)
        return True, restaurant_name
    except Exception as e:
        return False, f"{file_path.stem}: {str(e)}"


def main():
    ConsoleUI.print_header()
    start_time = time.time()

    # Проверка наличия файлов
    report_files = list(DATA_DIR.glob("*.xlsx"))
    if not report_files:
        ConsoleUI.print_error("Нет отчётов для анализа. Пожалуйста, загрузите отчёты в папку input_data.")
        return

    # Инициализация БД
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)

    # Создание сессии
    Session = sessionmaker(bind=engine)
    session = Session()

    # Загрузка данных
    ConsoleUI.print_success(f"Найдено {len(report_files)} файлов для анализа. Начало загрузки данных...")
    results = []
    for file_path in ConsoleUI.progress(report_files, "Загрузка отчётов"):
        result = process_restaurant(DB_PATH, file_path)
        results.append(result)

    # Обработка результатов загрузки
    errors = [msg for status, msg in results if not status]
    if errors:
        ConsoleUI.print_error("Ошибки при загрузке:")
        for error in errors:
            print(f"  - {error}")

    # Подготовка данных для обучения
    ConsoleUI.print_success("Анализ исторических данных")
    with ConsoleUI.progress([], "Обработка временных рядов"):
        df = pd.read_sql(
            "SELECT s.date, d.name AS dish, s.amount, r.name AS restaurant "
            "FROM sales s "
            "JOIN dishes d ON s.dish_id = d.id "
            "JOIN restaurants r ON s.restaurant_id = r.id",
            session.bind,
            parse_dates=['date']
        )
        df['date'] = pd.to_datetime(df['date']).dt.normalize()


    # Инжиниринг признаков
    with ConsoleUI.progress([], "Генерация признаков"):
        model_data = create_features(df)

    # Обучение модели
    ConsoleUI.print_success("Обучение модели прогнозирования")
    with ConsoleUI.progress(range(100), "Оптимизация параметров") as pbar:
        features = ['day_of_week', 'month', 'is_weekend', 'is_holiday',
                    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'rolling_7_mean']
        model = train_model(model_data[features], model_data['amount'])
        pbar.update(100)

    # Прогнозирование
    ConsoleUI.print_success("Генерация прогноза")
    tomorrow = datetime.now() + timedelta(days=1)
    predictions = []

    with ConsoleUI.progress(report_files, "Обработка ресторанов") as pbar:
        for file_path in pbar:
            restaurant_name = file_path.stem
            features_df = generate_features_for_date(tomorrow, session, restaurant_name)
            if not features_df.empty:
                preds = model.predict(features_df[features])
                features_df['prediction'] = preds.round().astype(int)
                predictions.append(features_df)

    # Сохранение и вывод
    if predictions:
        full_predictions = pd.concat(predictions)
        csv_path, txt_path = save_report(full_predictions)

        ConsoleUI.print_success("Результаты сохранены:")
        print(f"  - Машиночитаемый отчёт: {csv_path}")
        print(f"  - Текстовый отчёт: {txt_path}")
    else:
        ConsoleUI.print_warning("Нет данных для прогнозирования")

    # Статистика выполнения
    total_time = time.time() - start_time
    ConsoleUI.print_success(f"Завершено за {total_time:.1f} секунд")

    # Закрытие сессии
    session.close()


if __name__ == "__main__":
    main()