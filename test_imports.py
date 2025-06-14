from utils.validators import DataValidator
from utils.logger import setup_logger

from etl.load_data import load_data
from etl.preprocess import preprocess, feature_engineering
from etl.train import train_model
from etl.evaluate import evaluate_model, generate_model_report
from etl.save_results import save_results_batch

print("Импорт utils работает корректно!")
logger = setup_logger("test_logger")
logger.info("Тестовое сообщение в лог")

# Проверка валидатора
validator = DataValidator()
print("DataValidator создан успешно")
