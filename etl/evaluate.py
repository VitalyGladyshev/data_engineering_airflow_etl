import pandas as pd
import numpy as np
from sklearn import metrics
import joblib
import json
import os
from typing import Dict, Any
import sys

sys.path.append("/home/viv232/breast_cancer_etl/utils")

from utils.logger import setup_logger

logger = setup_logger("evaluate")

def evaluate_model(model_path: str, test_data: pd.DataFrame, metrics_path: str) -> Dict[str, float]:
    """Оценка качества модели"""
    try:
        logger.info("Начало оценки модели")

        # Загрузка модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']

        # Подготовка данных
        if 'diagnosis' not in test_data.columns:
            raise ValueError("Целевая переменная 'diagnosis' не найдена в тестовых данных")

        X_test = test_data[features]
        y_test = test_data['diagnosis']

        logger.info(f"Размер тестовой выборки: {X_test.shape}")
        logger.info(f"Распределение классов в тесте: {y_test.value_counts().to_dict()}")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Вероятности для класса 1

        # Расчет метрик
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='binary')
        recall = metrics.recall_score(y_test, y_pred, average='binary')
        f1 = metrics.f1_score(y_test, y_pred, average='binary')
        roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)

        specificity = metrics.recall_score(y_test, y_pred, pos_label=0, average='binary')

        cm = metrics.confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics_dict = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "total_samples": int(len(y_test))
        }

        class_report = metrics.classification_report(y_test, y_pred, output_dict=True)

        extended_metrics = {
            **metrics_dict,
            "classification_report": class_report,
            "confusion_matrix": cm.tolist()
        }

        logger.info(f"Метрики модели:")
        for metric, value in metrics_dict.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

        # Сохранение метрик
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        with open(metrics_path, 'w') as f:
            json.dump(extended_metrics, f, indent=2)

        logger.info(f"Метрики сохранены в {metrics_path}")

        return metrics_dict

    except Exception as e:
        logger.error(f"Ошибка оценки модели: {e}")
        raise

def generate_model_report(metrics_path: str, report_path: str) -> None:
    """Генерация отчета о модели в человекочитаемом формате"""
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        report = f"""
# Отчет о качестве модели

## Основные метрики
- Точность (Accuracy): {metrics['accuracy']:.4f}
- Точность класса 1 (Precision): {metrics['precision']:.4f}
- Полнота (Recall): {metrics['recall']:.4f}
- Специфичность: {metrics['specificity']:.4f}
- F1-мера: {metrics['f1_score']:.4f}
- ROC AUC: {metrics['roc_auc']:.4f}

## Матрица ошибок
- Истинно положительные: {metrics['true_positives']}
- Истинно отрицательные: {metrics['true_negatives']}
- Ложно положительные: {metrics['false_positives']}
- Ложно отрицательные: {metrics['false_negatives']}

Общее количество тестовых образцов: {metrics['total_samples']}
        """

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Отчет сохранен в {report_path}")

    except Exception as e:
        logger.error(f"Ошибка генерации отчета: {e}")
        raise
