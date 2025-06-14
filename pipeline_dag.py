from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
import yaml
import os
import sys
import glob
import pandas as pd

# Добавление путей для импорта модулей ETL
#sys.path.append('/home/viv232/breast_cancer_etl/etl')
#sys.path.append('/home/viv232/breast_cancer_etl/utils')
sys.path.append('/home/viv232/breast_cancer_etl')

from etl.load_data import load_data
from etl.preprocess import preprocess, feature_engineering
from etl.train import train_model
from etl.evaluate import evaluate_model, generate_model_report
from etl.save_results import save_results_batch
from utils.logger import setup_logger

logger = setup_logger("pipeline_dag")

# Загрузка конфигурации
def load_config():
    try:
        with open('/home/viv232/breast_cancer_etl/config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        return {}

config = load_config()

# Конфигурация DAG
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 14),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': config.get('airflow', {}).get('retries', 3),
    'retry_delay': timedelta(minutes=config.get('airflow', {}).get('retry_delay_minutes', 5)),
    'execution_timeout': timedelta(seconds=config.get('airflow', {}).get('timeout_seconds', 300))
}

def task_failure_callback(context):
    """Callback при ошибке задачи"""
    task_instance = context['task_instance']
    logger.error(f"Задача {task_instance.task_id} завершилась с ошибкой")

def dag_success_callback(context):
    """Callback при успешном завершении DAG"""
    logger.info("ETL пайплайн успешно завершен")

def dag_failure_callback(context):
    """Callback при ошибке DAG"""
    logger.error("ETL пайплайн завершился с ошибкой")

# Определение DAG
dag = DAG(
    'breast_cancer_etl_pipeline',
    default_args=default_args,
    description='Автоматизированный ETL пайплайн для анализа рака молочной железы',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    on_success_callback=dag_success_callback,
    on_failure_callback=dag_failure_callback,
    tags=['etl', 'ml', 'healthcare']
)

# Wrapper функции для задач
def load_data_task(**context):
    """Задача загрузки данных"""
    try:
        file_path = config.get('data', {}).get('raw_path', '/home/viv232/breast_cancer_etl/data/raw/data.csv')
        df = load_data(file_path)

        # Сохранение промежуточного результата
        temp_path = '/home/viv232/breast_cancer_etl/tmp/loaded_data.pkl'
        df.to_pickle(temp_path)

        logger.info(f"Данные загружены и сохранены в {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Ошибка в задаче загрузки данных: {e}")
        raise

def preprocess_data_task(**context):
    """Задача предобработки данных"""
    try:
        # Получение данных от предыдущей задачи
        loaded_data_path = context['task_instance'].xcom_pull(task_ids='load_data')

        df = pd.read_pickle(loaded_data_path)

        # Предобработка
        df_processed = preprocess(df)
        df_featured = feature_engineering(df_processed)

        # Сохранение результата
        temp_path = '/home/viv232/breast_cancer_etl/tmp/processed_data.pkl'
        df_featured.to_pickle(temp_path)

        logger.info(f"Данные обработаны и сохранены в {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Ошибка в задаче предобработки: {e}")
        raise

def train_model_task(**context):
    """Задача обучения модели"""
    try:
        # Получение обработанных данных
        processed_data_path = context['task_instance'].xcom_pull(task_ids='preprocess_data')

        df = pd.read_pickle(processed_data_path)

        # Обучение модели
        model_path = config.get('model', {}).get('path', '/home/viv232/breast_cancer_etl/data/models/model.joblib')
        model_config = config.get('model', {})

        test_data = train_model(df, model_path, model_config)

        # Сохранение тестовых данных
        test_path = '/home/viv232/breast_cancer_etl/tmp/test_data.pkl'
        test_data.to_pickle(test_path)

        logger.info(f"Модель обучена и сохранена в {model_path}")
        return {'model_path': model_path, 'test_path': test_path}
    except Exception as e:
        logger.error(f"Ошибка в задаче обучения модели: {e}")
        raise

def evaluate_model_task(**context):
    """Задача оценки модели"""
    try:
        # Получение данных от предыдущей задачи
        train_result = context['task_instance'].xcom_pull(task_ids='train_model')
        model_path = train_result['model_path']
        test_path = train_result['test_path']

        test_data = pd.read_pickle(test_path)

        # Оценка модели
        metrics_path = config.get('metrics', {}).get('path', '/home/viv232/breast_cancer_etl/data/metrics/metrics.json')
        metrics = evaluate_model(model_path, test_data, metrics_path)

        # Генерация отчета
        report_path = '/home/viv232/breast_cancer_etl/data/metrics/model_report.md'
        generate_model_report(metrics_path, report_path)

        logger.info(f"Модель оценена, метрики сохранены в {metrics_path}")
        return {'metrics_path': metrics_path, 'report_path': report_path}
    except Exception as e:
        logger.error(f"Ошибка в задаче оценки модели: {e}")
        raise

def save_results_task(**context):
    """Задача сохранения результатов"""
    try:
        # Получение путей к файлам
        train_result = context['task_instance'].xcom_pull(task_ids='train_model')
        eval_result = context['task_instance'].xcom_pull(task_ids='evaluate_model')

        files_to_save = [
            train_result['model_path'],
            eval_result['metrics_path'],
            eval_result['report_path']
        ]

        # Сохранение в облако
        cloud_config = config.get('cloud', {})
        results = save_results_batch(files_to_save, cloud_config)

        success_count = sum(results.values())
        logger.info(f"Сохранено {success_count} из {len(files_to_save)} файлов")

        if success_count == 0:
            raise Exception("Не удалось сохранить ни один файл")

        return results
    except Exception as e:
        logger.error(f"Ошибка в задаче сохранения результатов: {e}")
        raise

def cleanup_task(**context):
    """Задача очистки временных файлов"""
    try:
        temp_files = glob.glob('/home/viv232/breast_cancer_etl/tmp/*.pkl')

        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.info(f"Удален временный файл: {temp_file}")
            except Exception as e:
                logger.warning(f"Не удалось удалить {temp_file}: {e}")

        logger.info("Очистка временных файлов завершена")
    except Exception as e:
        logger.error(f"Ошибка в задаче очистки: {e}")

# Определение задач
load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data_task,
    on_failure_callback=task_failure_callback,
    dag=dag
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data_task,
    on_failure_callback=task_failure_callback,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    on_failure_callback=task_failure_callback,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    on_failure_callback=task_failure_callback,
    dag=dag
)

save_task = PythonOperator(
    task_id='save_results',
    python_callable=save_results_task,
    on_failure_callback=task_failure_callback,
    dag=dag
)

cleanup_task_op = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_task,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag
)

# Задача проверки состояния системы
health_check = BashOperator(
    task_id='health_check',
    bash_command='echo "Система готова к работе" && df -h',
    dag=dag
)

# Построение графа
health_check >> load_task >> preprocess_task >> train_task >> evaluate_task >> save_task >> cleanup_task_op