import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class TrafficPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df):
        """
        Предварительная обработка данных о трафике
        
        Args:
            df (pd.DataFrame): Исходный датафрейм
        """
        # Обработка категориальных признаков
        categorical_features = ['protocol', 'src_ip', 'dst_ip', 'src_port', 'dst_port']
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
            df[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
        
        # Преобразование временной метки
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Нормализация числовых признаков
        numeric_features = ['length', 'hour', 'minute']
        df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        
        return df
    
    def prepare_features(self, df):
        """
        Подготовка признаков для модели
        
        Args:
            df (pd.DataFrame): Предобработанный датафрейм
        """
        features = ['protocol', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 
                   'length', 'hour', 'minute']
        return df[features] 