import pandas as pd
import os
from typing import Optional
import sys

sys.path.append('/home/viv232/breast_cancer_etl/utils')

from utils.logger import setup_logger
from utils.validators import DataValidator

logger = setup_logger("load_data")

def load_data(file_path: str) -> pd.DataFrame:
    """Загрузка данных из CSV файла с валидацией"""
    try:
        logger.info(f"Начало загрузки данных из {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        df = pd.read_csv(file_path)

        if df.empty:
            raise pd.errors.EmptyDataError("Загруженный файл пустой")

        logger.info(f"Данные успешно загружены. Размер: {df.shape}")
        logger.info(f"Колонки: {list(df.columns)}")

        # Валидация качества данных
        validator = DataValidator()
        quality_report = validator.validate_data_quality(df)

        if not quality_report["is_valid"]:
            logger.warning(f"Проблемы с качеством данных: {quality_report['critical_issues']}")

        return df

    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Пустой файл: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        raise

def validate_breast_cancer_schema(df: pd.DataFrame) -> bool:
    """Валидация схемы для датасета"""
    expected_columns = ['diagnosis']
    validator = DataValidator()
    return validator.validate_schema(df, expected_columns)
