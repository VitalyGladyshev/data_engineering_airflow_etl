import os
import shutil
from typing import Optional, Dict, List
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import sys

sys.path.append("/home/viv232/breast_cancer_etl/utils")

from utils.logger import setup_logger

logger = setup_logger("save_results")

def authenticate_google_drive(credentials_path: str) -> Optional[GoogleDrive]:
    """Аутентификация в Google Drive"""
    try:
        gauth = GoogleAuth()
        
        # Загрузка существующих учетных данных
        if os.path.exists(credentials_path):
            gauth.LoadCredentialsFile(credentials_path)
        
        # Аутентификация
        if gauth.credentials is None:
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
        
        # Сохранение учетных данных
        gauth.SaveCredentialsFile(credentials_path)
        
        drive = GoogleDrive(gauth)
        logger.info("Успешная аутентификация в Google Drive")
        return drive
        
    except Exception as e:
        logger.error(f"Ошибка аутентификации Google Drive: {e}")
        return None

def upload_to_google_drive(file_path: str, folder_id: str, credentials_path: str = "/home/viv232/breast_cancer_etl/config/credentials.json") -> bool:
    """Загрузка файла в Google Drive"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Файл не найден: {file_path}")
            return False
        
        # Аутентификация
        drive = authenticate_google_drive(credentials_path)
        if not drive:
            return False
        
        # Создание файла в Google Drive
        filename = os.path.basename(file_path)
        gfile = drive.CreateFile({
            'title': filename,
            'parents': [{'id': folder_id}]
        })
        
        # Загрузка файла
        gfile.SetContentFile(file_path)
        gfile.Upload()
        
        logger.info(f"Файл '{filename}' успешно загружен в Google Drive")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка загрузки в Google Drive: {e}")
        return False

def save_to_local(file_path: str, backup_dir: str = "/home/viv232/breast_cancer_etl/results") -> bool:
    """Сохранение файла в локальную директорию"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Исходный файл не найден: {file_path}")
            return False
        
        os.makedirs(backup_dir, exist_ok=True)
        
        # Копирование файла
        filename = os.path.basename(file_path)
        destination = os.path.join(backup_dir, filename)
        shutil.copy2(file_path, destination)
        
        logger.info(f"Файл сохранен локально: {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка локального сохранения: {e}")
        return False

def save_file(file_path: str, gdrive_folder_id: str, backup_dir: str = "/home/viv232/breast_cancer_etl/results", 
              credentials_path: str = "/home/viv232/breast_cancer_etl/config/credentials.json") -> bool:
    """Основная функция сохранения файла (сначала в Google Drive, потом локально)"""
    logger.info(f"Начало сохранения файла: {os.path.basename(file_path)}")
    
    # Попытка сохранить в Google Drive
    gdrive_success = upload_to_google_drive(file_path, gdrive_folder_id, credentials_path)
    
    # Резервное сохранение локально
    local_success = save_to_local(file_path, backup_dir)
    
    # Анализ результатов
    if gdrive_success and local_success:
        logger.info("Файл успешно сохранен в Google Drive и локально")
        return True
    elif gdrive_success:
        logger.info("Файл сохранен в Google Drive (локальное сохранение не удалось)")
        return True
    elif local_success:
        logger.warning("Файл сохранен только локально (Google Drive недоступен)")
        return True
    else:
        logger.error("Не удалось сохранить файл ни в Google Drive, ни локально")
        return False

def save_multiple_files(file_paths: List[str], 
                        gdrive_folder_id: str, 
                        backup_dir: str = "/home/viv232/breast_cancer_etl/results", 
                        credentials_path: str = "/home/viv232/breast_cancer_etl/config/credentials.json") -> Dict[str, bool]:
    """Сохранение нескольких файлов"""
    results = {}
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        results[filename] = save_file(file_path, gdrive_folder_id, backup_dir, credentials_path)
    
    # Статистика
    successful = sum(results.values())
    total = len(file_paths)
    logger.info(f"Сохранено {successful} из {total} файлов")
    
    return results

# Функции для обратной совместимости с исходным кодом
def save_to_drive(local_path: str, gdrive_folder_id: str):
    """Функция для обратной совместимости с исходным API"""
    return save_file(local_path, gdrive_folder_id)

def save_results_batch(file_paths: List[str], config: Dict) -> Dict[str, bool]:
    """Функция для совместимости с DAG"""
    gdrive_folder_id = config.get('folder_id', '')
    backup_dir = config.get('backup_dir', '/home/viv232/breast_cancer_etl/results')
    credentials_path = config.get('credentials_path', '/home/viv232/breast_cancer_etl/config/credentials.json')
    
    return save_multiple_files(file_paths, gdrive_folder_id, backup_dir, credentials_path)