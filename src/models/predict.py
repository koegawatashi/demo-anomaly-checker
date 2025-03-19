import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.neighbors import LocalOutlierFactor

class AnomalyPredictor:
    def __init__(self, model_path, scaler_path=None, threshold=None):
        """
        Инициализация предиктора аномалий.
        
        Args:
            model_path (str): Путь к сохраненной модели
            scaler_path (str): Путь к сохраненному скейлеру
            threshold (float): Порог для определения аномалий (для оценок)
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.threshold = threshold
        
        # Определяем тип модели
        if hasattr(self.model, 'score_samples'):
            self.model_type = 'isolation_forest'
        else:
            self.model_type = 'dbscan'
    
    def predict(self, X):
        """
        Предсказание аномалий.
        
        Args:
            X (pd.DataFrame): Данные для предсказания
        
        Returns:
            pd.DataFrame: Данные с предсказаниями
        """
        # Масштабирование данных, если есть скейлер
        X_scaled = self.scaler.transform(X) if self.scaler else X
        
        # Предсказание
        if self.model_type == 'isolation_forest':
            # Получаем метки (-1 для аномалий, 1 для нормальных)
            labels = self.model.predict(X_scaled)
            # Получаем оценки аномальности
            scores = -self.model.score_samples(X_scaled)
        elif self.model_type == 'dbscan':
            # Для DBSCAN: -1 - выбросы, остальные - кластеры
            labels = self.model.fit_predict(X_scaled)
            
            # Для DBSCAN используем расстояние до ближайшего соседа как оценку
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=2).fit(X_scaled)
            distances, _ = nbrs.kneighbors(X_scaled)
            scores = distances[:, 1]
        
        # Создаем копию данных с предсказаниями
        results = X.copy()
        results['anomaly'] = labels == -1  # True для аномалий
        results['anomaly_score'] = scores
        
        # Если задан порог, используем его для определения аномалий
        if self.threshold is not None:
            results['anomaly'] = scores > self.threshold
        
        return results
    
    def detect_anomalies_realtime(self, data_stream, feature_extractor, window_size=60, callback=None):
        """
        Обнаружение аномалий в реальном времени.
        
        Args:
            data_stream: Источник данных
            feature_extractor: Экстрактор признаков
            window_size (int): Размер окна в секундах
            callback (callable): Функция обратного вызова для обработки аномалий
        """
        buffer = []
        last_process_time = None
        
        for packet in data_stream:
            # Добавляем пакет в буфер
            buffer.append(packet)
            current_time = datetime.now()
            
            # Если прошло достаточно времени, обрабатываем буфер
            if last_process_time is None or (current_time - last_process_time).total_seconds() >= window_size:
                # Преобразуем буфер в DataFrame
                df = pd.DataFrame(buffer)
                
                # Извлекаем признаки
                features = feature_extractor.extract_time_window_features(df)
                
                # Предсказываем аномалии
                results = self.predict(features)
                
                # Если есть аномалии и задан callback, вызываем его
                anomalies = results[results['anomaly']]
                if not anomalies.empty and callback:
                    callback(anomalies)
                
                # Очищаем буфер и обновляем время
                buffer = []
                last_process_time = current_time 

    def get_anomaly_scores(self, X):
        """
        Получение оценок аномальности
        
        Args:
            X (pd.DataFrame): Данные для оценки
        """
        if hasattr(self.model, 'score_samples'):
            return self.model.score_samples(X)
        elif hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X)
        else:
            return None 