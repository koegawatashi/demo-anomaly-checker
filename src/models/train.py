import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

class AnomalyDetector:
    def __init__(self, model_type='isolation_forest'):
        self.model_type = model_type
        self.model = None
        
    def train(self, X):
        """
        Обучение модели детекции аномалий
        
        Args:
            X (pd.DataFrame): Обучающие данные
        """
        if self.model_type == 'isolation_forest':
            self.model = IsolationForest(contamination=0.1, random_state=42)
        elif self.model_type == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        elif self.model_type == 'lof':
            self.model = LocalOutlierFactor(contamination=0.1, novelty=True)
            
        self.model.fit(X)
        
    def save_model(self, path):
        """
        Сохранение модели
        
        Args:
            path (str): Путь для сохранения модели
        """
        joblib.dump(self.model, path)
        
    def load_model(self, path):
        """
        Загрузка модели
        
        Args:
            path (str): Путь к сохраненной модели
        """
        self.model = joblib.load(path)

class AnomalyDetectionModel:
    def __init__(self, model_type='isolation_forest', contamination=0.1, random_state=42):
        """
        Инициализация модели обнаружения аномалий.
        
        Args:
            model_type (str): Тип модели ('isolation_forest' по умолчанию)
            contamination (float): Ожидаемая доля аномалий в данных (от 0 до 0.5)
            random_state (int): Seed для воспроизводимости результатов
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100,
                max_samples='auto'
            )
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
    
    def train(self, X):
        """
        Обучение модели.
        
        Args:
            X (pd.DataFrame): Данные для обучения
        """
        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)
        
        # Обучение модели
        self.model.fit(X_scaled)
        
        print("Модель успешно обучена")
    
    def predict(self, X):
        """
        Предсказание аномалий.
        
        Args:
            X (pd.DataFrame): Данные для предсказания
            
        Returns:
            np.array: Массив предсказаний (1 - нормально, -1 - аномалия)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, path='models'):
        """
        Сохранение модели.
        
        Args:
            path (str): Путь для сохранения модели
            
        Returns:
            str: Путь к сохраненной модели
        """
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f'{self.model_type}_model.joblib')
        scaler_path = os.path.join(path, f'{self.model_type}_scaler.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        return model_path
    
    def load_model(self, path='models'):
        """
        Загрузка модели.
        
        Args:
            path (str): Путь к сохраненной модели
        """
        model_path = os.path.join(path, f'{self.model_type}_model.joblib')
        scaler_path = os.path.join(path, f'{self.model_type}_scaler.joblib')
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_score(self, X):
        """
        Получение оценки аномальности.
        
        Args:
            X (pd.DataFrame): Данные для оценки
        
        Returns:
            np.ndarray: Оценки аномальности
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        # Масштабирование данных
        X_scaled = self.scaler.transform(X)
        
        # Получение оценок
        if self.model_type == 'isolation_forest':
            # Отрицательные значения - более аномальные
            return -self.model.score_samples(X_scaled)
        elif self.model_type == 'dbscan':
            # Для DBSCAN используем расстояние до ближайшего соседа
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=2).fit(X_scaled)
            distances, _ = nbrs.kneighbors(X_scaled)
            return distances[:, 1]  # Расстояние до ближайшего соседа
        elif self.model_type == 'lof':
            return self.model.negative_outlier_factor_ 