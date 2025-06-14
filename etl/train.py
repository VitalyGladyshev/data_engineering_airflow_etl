import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
from typing import Tuple, Dict, Any
import sys

sys.path.append("/home/viv232/breast_cancer_etl/utils")

from utils.logger import setup_logger

logger = setup_logger("train")

def train_model(df: pd.DataFrame, model_path: str, config: Dict[str, Any] = None) -> pd.DataFrame:
    """Обучение модели логистической регрессии"""
    try:
        logger.info("Начало обучения модели")

        # Значения по умолчанию
        if config is None:
            config = {
                'test_size': 0.3,
                'random_state': 42,
                'max_iter': 10000
            }

        # Проверка наличия целевой переменной
        if 'diagnosis' not in df.columns:
            raise ValueError("Целевая переменная 'diagnosis' не найдена")

        # Разделение на признаки и целевую переменную
        features = [col for col in df.columns if col != 'diagnosis']
        X = df[features]
        y = df['diagnosis']

        logger.info(f"Количество признаков: {len(features)}")
        logger.info(f"Распределение классов: {y.value_counts().to_dict()}")

        # Проверка на достаточность данных
        if len(df) < 50:
            logger.warning(f"Мало данных для обучения: {len(df)} образцов")

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=config['random_state'],
            stratify=y  # Стратифицированное разделение
        )

        logger.info(f"Размер обучающей выборки: {X_train.shape}")
        logger.info(f"Размер тестовой выборки: {X_test.shape}")

        # Масштабирование признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Создание и обучение модели
        model = LogisticRegression(
            max_iter=config['max_iter'],
            random_state=config['random_state'],
            class_weight='balanced'  # Балансировка классов
        )

        model.fit(X_train_scaled, y_train)
        logger.info("Модель успешно обучена")

        # Проверка сходимости
        if hasattr(model, 'n_iter_'):
            logger.info(f"Количество итераций до сходимости: {model.n_iter_[0]}")
            if model.n_iter_[0] >= config['max_iter']:
                logger.warning("Модель может не сходиться. Увеличьте max_iter")

        # Создание директории для модели
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Сохранение модели и скейлера
        model_data = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'config': config
        }

        joblib.dump(model_data, model_path)
        logger.info(f"Модель сохранена в {model_path}")

        # Подготовка тестовых данных
        test_df = pd.concat([
            pd.DataFrame(X_test_scaled, columns=features, index=X_test.index),
            y_test
        ], axis=1)

        # Базовая оценка на обучающих данных
        train_accuracy = model.score(X_train_scaled, y_train)
        logger.info(f"Точность на обучающих данных: {train_accuracy:.4f}")

        return test_df

    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        raise
