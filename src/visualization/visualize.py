import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class TrafficVisualizer:
    def __init__(self):
        plt.style.use('default')
        
    def plot_traffic_distribution(self, df, feature, title):
        """
        Построение распределения признака
        
        Args:
            df (pd.DataFrame): Данные
            feature (str): Название признака
            title (str): Заголовок графика
        """
        plt.figure(figsize=(10, 6))
        plt.hist(df[feature], bins=50, density=True, alpha=0.7)
        plt.title(title)
        plt.grid(True)
        plt.show()
        
    def plot_anomalies_2d(self, X, predictions, title):
        """
        Визуализация аномалий в 2D
        
        Args:
            X (pd.DataFrame): Данные
            predictions (np.array): Предсказания аномалий
            title (str): Заголовок графика
        """
        plt.figure(figsize=(12, 8))
        
        # Проверяем размерность данных
        if X.shape[1] >= 2:
            # Если есть хотя бы 2 признака, используем PCA
            pca = PCA(n_components=min(2, X.shape[1], X.shape[0]))
            X_2d = pca.fit_transform(X)
        else:
            # Если признак только один, добавляем случайный шум как второй признак
            X_2d = np.column_stack([X, np.random.normal(0, 0.1, size=X.shape[0])])

        # Отображаем точки
        normal_mask = predictions == 1
        anomaly_mask = predictions == -1
        
        plt.scatter(X_2d[normal_mask, 0], 
                   X_2d[normal_mask, 1] if X_2d.shape[1] > 1 else np.zeros_like(X_2d[normal_mask]),
                   c='blue', label='Normal', alpha=0.6)
        plt.scatter(X_2d[anomaly_mask, 0], 
                   X_2d[anomaly_mask, 1] if X_2d.shape[1] > 1 else np.zeros_like(X_2d[anomaly_mask]),
                   c='red', label='Anomaly', alpha=0.6)
        
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_time_series_anomalies(self, df, predictions, feature='length'):
        """
        Визуализация аномалий во временном ряду
        
        Args:
            df (pd.DataFrame): Данные
            predictions (np.array): Предсказания аномалий
            feature (str): Признак для визуализации
        """
        plt.figure(figsize=(15, 7))
        
        # Проверяем, есть ли данные для построения
        if len(df) > 0:
            # Создаем массив индексов той же длины, что и predictions
            if isinstance(df.index, pd.DatetimeIndex):
                # Если индекс временной, берем уникальные значения
                unique_times = df.index.unique()
                if len(unique_times) == len(predictions):
                    plot_times = unique_times
                else:
                    # Если длины не совпадают, создаем равномерный временной ряд
                    start_time = df.index.min()
                    end_time = df.index.max()
                    plot_times = pd.date_range(start_time, end_time, periods=len(predictions))
            else:
                # Если индекс не временной, используем числовой ряд
                plot_times = np.arange(len(predictions))
            
            # Агрегируем значения признака по временным окнам
            feature_values = df.groupby(df.index)[feature].mean()
            
            # Приводим длину значений признака к длине predictions
            if len(feature_values) != len(predictions):
                # Пересэмплируем данные, чтобы длины совпадали
                feature_values = feature_values.reindex(plot_times, method='nearest')
            
            normal_mask = predictions == 1
            anomaly_mask = predictions == -1
            
            # Построение графика
            plt.plot(plot_times[normal_mask], feature_values[normal_mask], 
                    'b.', label='Normal', alpha=0.5)
            if any(anomaly_mask):
                plt.plot(plot_times[anomaly_mask], feature_values[anomaly_mask], 
                        'r.', label='Anomaly', alpha=0.7)
            
            plt.title(f'Anomalies in {feature} over time')
            plt.legend()
            plt.grid(True)
            
            # Поворот меток времени для лучшей читаемости
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No data available', 
                    horizontalalignment='center',
                    verticalalignment='center')
        
        plt.tight_layout()  # Автоматическая подгонка графика
        plt.show() 

    def plot_anomaly_summary(self, df, predictions, window_features):
        """
        Сводная визуализация аномалий
        
        Args:
            df (pd.DataFrame): Исходные данные
            predictions (np.array): Предсказания аномалий
            window_features (pd.DataFrame): Признаки окон
        """
        plt.figure(figsize=(15, 10))
        
        # 1. Круговая диаграмма распределения нормальных/аномальных пакетов
        plt.subplot(2, 2, 1)
        labels = ['Нормальный трафик', 'Аномальный трафик']
        sizes = [sum(predictions == 1), sum(predictions == -1)]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title('Распределение нормального и аномального трафика')

        # 2. График распределения размеров пакетов
        plt.subplot(2, 2, 2)
        # Приводим размеры массивов в соответствие
        predictions_expanded = np.repeat(predictions, len(df) // len(predictions))
        if len(predictions_expanded) < len(df):
            predictions_expanded = np.append(predictions_expanded, predictions_expanded[-1])
        elif len(predictions_expanded) > len(df):
            predictions_expanded = predictions_expanded[:len(df)]
        
        normal_mask = predictions_expanded == 1
        anomaly_mask = predictions_expanded == -1
        
        plt.hist(df.loc[normal_mask, 'length'], bins=30, alpha=0.5, 
                 label='Нормальный', color='blue', density=True)
        plt.hist(df.loc[anomaly_mask, 'length'], bins=30, alpha=0.5,
                 label='Аномальный', color='red', density=True)
        plt.title('Распределение размеров пакетов')
        plt.legend()

        # 3. Временной ряд интенсивности трафика
        plt.subplot(2, 2, (3, 4))
        
        # Создаем временной индекс, если его нет
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.date_range(start='now', periods=len(df), freq='S')
        
        # Строим скользящее среднее
        rolling_mean = df['length'].rolling(window=10, min_periods=1).mean()
        plt.plot(df.index, rolling_mean, label='Средний размер пакета', color='blue')
        
        # Добавляем аномальные точки
        if sum(anomaly_mask) > 0:
            plt.scatter(df.index[anomaly_mask], df.loc[anomaly_mask, 'length'], 
                       color='red', label='Аномалии', zorder=5)
        
        plt.title('Временной ряд с отмеченными аномалиями')
        plt.legend()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        # Выводим статистику
        print("\nСТАТИСТИКА АНОМАЛИЙ:")
        print(f"Всего пакетов: {len(df)}")
        print(f"Всего окон: {len(predictions)}")
        print(f"Нормальных окон: {sum(predictions == 1)}")
        print(f"Аномальных окон: {sum(predictions == -1)}")
        print(f"Процент аномальных окон: {(sum(predictions == -1) / len(predictions) * 100):.2f}%") 