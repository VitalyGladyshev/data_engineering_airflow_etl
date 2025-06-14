import pandas as pd
import numpy as np
from typing import Dict, Any, List

from .logger import setup_logger

logger = setup_logger("validators")

class DataValidator:
    """Валидатор для проверки качества данных"""

    @staticmethod
    def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Проверка схемы данных"""
        try:
            missing_columns = set(expected_columns) - set(df.columns)
            if missing_columns:
                logger.error(f"Отсутствуют колонки: {missing_columns}")
                return False
            logger.info("Схема данных валидна")
            return True
        except Exception as e:
            logger.error(f"Ошибка валидации схемы: {e}")
            return False

    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Проверка качества данных"""
        quality_report = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict()
        }

        # Проверка на критические проблемы
        critical_issues = []

        if quality_report["duplicates"] > 0:
            critical_issues.append(f"Найдено {quality_report['duplicates']} дубликатов")

        missing_threshold = 0.5  # 50% пропущенных значений
        for col, missing_count in quality_report["missing_values"].items():
            if missing_count / len(df) > missing_threshold:
                critical_issues.append(f"Колонка {col} имеет {missing_count/len(df)*100:.1f}% пропущенных значений")

        quality_report["critical_issues"] = critical_issues
        quality_report["is_valid"] = len(critical_issues) == 0

        logger.info(f"Отчет о качестве данных: {quality_report}")
        return quality_report
