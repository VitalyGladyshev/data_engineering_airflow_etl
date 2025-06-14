import logging
import os
from datetime import datetime

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Настройка логгера для компонентов ETL"""

    log_dir = "/home/viv232/breast_cancer_etl/logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Обработчик для файла
    file_handler = logging.FileHandler(
        f"{log_dir}/{name}_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setFormatter(formatter)

    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
