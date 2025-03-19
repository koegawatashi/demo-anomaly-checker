import pandas as pd
import numpy as np
from collections import defaultdict

class NetworkFeatureExtractor:
    def __init__(self, window_size=60):
        """
        Инициализация экстрактора признаков.
        
        Args:
            window_size (int): Размер временного окна в секундах
        """
        self.window_size = window_size
    
    def extract_time_window_features(self, df, window_size=10):
        """
        Извлечение признаков из временного окна
        
        Args:
            df (pd.DataFrame): Входные данные
            window_size (int): Размер окна в секундах
        """
        if len(df) == 0:
            print("Предупреждение: Входной DataFrame пуст")
            return pd.DataFrame()
        
        features_list = []
        
        # Группируем данные по временным окнам
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Используем resample вместо date_range для более надежной группировки
        df.set_index('timestamp', inplace=True)
        grouped = df.resample(f'{window_size}s')
        
        for name, window_data in grouped:
            if len(window_data) > 0:
                try:
                    # Базовые статистики
                    packet_count = len(window_data)
                    avg_packet_size = window_data['length'].mean()
                    std_packet_size = window_data['length'].std()
                    max_packet_size = window_data['length'].max()
                    min_packet_size = window_data['length'].min()
                    
                    # Статистики по протоколам
                    protocol_counts = window_data['protocol'].value_counts()
                    unique_protocols = len(protocol_counts)
                    
                    # Статистики по IP-адресам
                    unique_src_ips = window_data['src_ip'].nunique()
                    unique_dst_ips = window_data['dst_ip'].nunique()
                    
                    # Статистики по портам
                    unique_src_ports = window_data['src_port'].nunique()
                    unique_dst_ports = window_data['dst_port'].nunique()
                    
                    # Интенсивность трафика
                    bytes_per_second = window_data['length'].sum() / window_size
                    packets_per_second = packet_count / window_size
                    
                    # Дополнительные признаки для обнаружения аномалий
                    total_bytes = window_data['length'].sum()
                    avg_bytes_per_packet = total_bytes / packet_count if packet_count > 0 else 0
                    
                    feature_dict = {
                        'packet_count': packet_count,
                        'avg_packet_size': avg_packet_size,
                        'std_packet_size': std_packet_size,
                        'max_packet_size': max_packet_size,
                        'min_packet_size': min_packet_size,
                        'unique_protocols': unique_protocols,
                        'unique_src_ips': unique_src_ips,
                        'unique_dst_ips': unique_dst_ips,
                        'unique_src_ports': unique_src_ports,
                        'unique_dst_ports': unique_dst_ports,
                        'bytes_per_second': bytes_per_second,
                        'packets_per_second': packets_per_second,
                        'total_bytes': total_bytes,
                        'avg_bytes_per_packet': avg_bytes_per_packet
                    }
                    
                    features_list.append(feature_dict)
                except Exception as e:
                    print(f"Ошибка при обработке окна: {e}")
                    continue
        
        if not features_list:
            print("Предупреждение: Не удалось извлечь признаки")
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)
        
        # Добавляем проверку на пустые или невалидные значения
        if features_df.isnull().any().any() or features_df.empty:
            print("Предупреждение: В извлеченных признаках есть пустые значения")
            features_df = features_df.fillna(0)
        
        print(f"Извлечено признаков: {len(features_df)} записей, {len(features_df.columns)} признаков")
        return features_df
    
    def extract_flow_features(self, df):
        """
        Извлечение признаков на основе потоков (flow).
        
        Args:
            df (pd.DataFrame): DataFrame с данными пакетами
        
        Returns:
            pd.DataFrame: DataFrame с извлеченными признаками
        """
        # Создаем идентификатор потока (src_ip, dst_ip, src_port, dst_port, protocol)
        df['flow_id'] = df.apply(
            lambda row: f"{row['src_ip']}:{row['src_port']}-{row['dst_ip']}:{row['dst_port']}-{row['protocol']}",
            axis=1
        )
        
        # Группируем по потокам
        grouped = df.groupby('flow_id')
        
        # Извлекаем признаки для каждого потока
        flow_features = []
        
        for flow_id, group in grouped:
            # Базовые статистики
            packet_count = len(group)
            flow_duration = (group['timestamp'].max() - group['timestamp'].min()).total_seconds()
            flow_duration = max(flow_duration, 0.001)  # Избегаем деления на ноль
            
            avg_packet_size = group['length'].mean()
            max_packet_size = group['length'].max()
            std_packet_size = group['length'].std()
            
            # Скорость передачи данных
            bytes_per_second = group['length'].sum() / flow_duration
            packets_per_second = packet_count / flow_duration
            
            # Создаем запись признаков для потока
            feature_dict = {
                'flow_id': flow_id,
                'packet_count': packet_count,
                'flow_duration': flow_duration,
                'avg_packet_size': avg_packet_size,
                'max_packet_size': max_packet_size,
                'std_packet_size': std_packet_size,
                'bytes_per_second': bytes_per_second,
                'packets_per_second': packets_per_second
            }
            
            flow_features.append(feature_dict)
        
        # Создаем DataFrame с признаками
        features_df = pd.DataFrame(flow_features)
        
        return features_df 

class FeatureEngineer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        
    def create_statistical_features(self, df):
        """
        Создание статистических признаков на основе временного окна
        
        Args:
            df (pd.DataFrame): Предобработанный датафрейм
        """
        # Статистические признаки по длине пакетов
        df['packet_length_mean'] = df['length'].rolling(window=self.window_size).mean()
        df['packet_length_std'] = df['length'].rolling(window=self.window_size).std()
        df['packet_length_max'] = df['length'].rolling(window=self.window_size).max()
        
        # Подсчет уникальных IP-адресов в окне
        df['unique_ips'] = df.groupby(df.index // self.window_size)['src_ip'].transform('nunique')
        
        # Частота протоколов
        df['protocol_freq'] = df.groupby(df.index // self.window_size)['protocol'].transform('value_counts')
        
        # Заполнение пропущенных значений
        df = df.fillna(0)
        
        return df
    
    def create_time_features(self, df):
        """
        Создание временных признаков
        
        Args:
            df (pd.DataFrame): Датафрейм с временной меткой
        """
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = (df['hour'] >= 22) | (df['hour'] <= 6).astype(int)
        
        return df 