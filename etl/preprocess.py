import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
import sys

sys.path.append("/home/viv232/breast_cancer_etl/utils")

from utils.logger import setup_logger
from utils.validators import DataValidator

logger = setup_logger("preprocess")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Предобработка данных"""
    try:
        logger.info("Начало предобработки данных")
        
        # Создание копии для безопасности
        df_processed = df.copy()
        
        # Удаление ненужных колонок
        columns_to_drop = ["Unnamed: 32", "id"]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        
        if existing_columns_to_drop:
            df_processed = df_processed.drop(existing_columns_to_drop, axis=1)
            logger.info(f"Удалены колонки: {existing_columns_to_drop}")
        
        # Обработка целевой переменной
        if 'diagnosis' in df_processed.columns:
            # Проверка уникальных значений
            unique_values = df_processed['diagnosis'].unique()
            logger.info(f"Уникальные значения в diagnosis: {unique_values}")
            
            # Маппинг диагноза
            diagnosis_mapping = {'M': 1, 'B': 0}
            df_processed['diagnosis'] = df_processed['diagnosis'].map(diagnosis_mapping)
            
            # Проверка успешности маппинга
            if df_processed['diagnosis'].isnull().any():
                logger.warning("Обнаружены неизвестные значения в diagnosis после маппинга")
                df_processed = df_processed.dropna(subset=['diagnosis'])
            
            logger.info("Диагноз успешно закодирован (M=1, B=0)")
        else:
            logger.warning("Колонка 'diagnosis' не найдена")
        
        # Обработка пропущенных значений
        missing_values = df_processed.isnull().sum()
        if missing_values.any():
            logger.info(f"Пропущенные значения: {missing_values[missing_values > 0].to_dict()}")
            
            # Заполнение числовых колонок медианой
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df_processed[col].isnull().any():
                    median_value = df_processed[col].median()
                    df_processed[col].fillna(median_value, inplace=True)
                    logger.info(f"Заполнены пропуски в {col} медианой: {median_value}")
        
        # Удаление дубликатов
        initial_shape = df_processed.shape
        df_processed = df_processed.drop_duplicates()
        final_shape = df_processed.shape
        
        if initial_shape[0] != final_shape[0]:
            logger.info(f"Удалено {initial_shape[0] - final_shape[0]} дубликатов")
        
        # Финальная валидация
        validator = DataValidator()
        quality_report = validator.validate_data_quality(df_processed)
        
        logger.info(f"Предобработка завершена. Итоговый размер: {df_processed.shape}")
        return df_processed
        
    except Exception as e:
        logger.error(f"Ошибка предобработки: {e}")
        raise

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Дополнительная инженерия признаков"""
    try:
        logger.info("Feature engineering")
        
        df_features = df.copy()
        
        # Получение числовых колонок (исключая target)
        numeric_columns = df_features.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col != 'diagnosis']
        
        if len(feature_columns) > 0:
            # Нормализация признаков (min-max scaling)
            scaler = MinMaxScaler()
            df_features[feature_columns] = scaler.fit_transform(df_features[feature_columns])
            logger.info("Применена min-max нормализация к признакам")
        
        return df_features
        
    except Exception as e:
        logger.error(f"Ошибка инженерии признаков: {e}")
        raise
