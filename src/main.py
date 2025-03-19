import os
from data.data_collection import NetworkTrafficCollector
from data.preprocessing import TrafficPreprocessor
from features.feature_engineering import NetworkFeatureExtractor
from models.train import AnomalyDetectionModel
from visualization.visualize import TrafficVisualizer

def main():
    # 1. Создаем необходимые директории
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 2. Инициализируем коллектор трафика
    # Используйте один из вариантов:
    collector = NetworkTrafficCollector(interface='Беспроводная сеть')  # для проводного подключения
    # collector = NetworkTrafficCollector(interface='Беспроводная сеть')  # для беспроводного подключения
    
    try:
        # 3. Собираем данные (60 секунд)
        print("Начинаем сбор сетевого трафика...")
        print("Для остановки нажмите Ctrl+C")
        pcap_file = collector.capture_live_traffic(duration=10)
        
        if not os.path.exists(pcap_file) or os.path.getsize(pcap_file) == 0:
            print("Захват трафика был прерван или не удался")
            return
        
        # 4. Преобразуем pcap в DataFrame
        raw_data = collector.read_pcap_file(pcap_file)
        print(f"Собрано {len(raw_data)} пакетов")

        # 5. Предобработка данных
        preprocessor = TrafficPreprocessor()
        processed_data = preprocessor.preprocess_data(raw_data)
        features = preprocessor.prepare_features(processed_data)

        # 6. Извлечение признаков
        print("\nИзвлечение признаков...")
        feature_extractor = NetworkFeatureExtractor(window_size=3)  # Уменьшаем размер окна до 3 секунд
        window_features = feature_extractor.extract_time_window_features(processed_data)

        if window_features.empty:
            print("Не удалось извлечь признаки из данных")
            return

        # 7. Обучение модели
        print("\nОбучение модели...")
        model = AnomalyDetectionModel(
            model_type='isolation_forest',
            contamination=0.01,  # Делаем модель более чувствительной
            random_state=42
        )

        # Добавляем проверку данных перед обучением
        if len(window_features) < 2:
            print("Недостаточно данных для обучения модели")
            return

        try:
            model.train(window_features)
            print("Модель успешно обучена")
        except Exception as e:
            print(f"Ошибка при обучении модели: {e}")
            return

        # 8. Сохранение модели
        model_path = model.save_model()
        print(f"Модель сохранена в {model_path}")

        # 9. Предсказание аномалий
        predictions = model.predict(window_features)

        # Перед визуализацией добавьте отладочную информацию
        print("Размерность признаков:", window_features.shape)
        print("Количество предсказаний:", len(predictions))
        print("Количество нормальных точек:", sum(predictions == 1))
        print("Количество аномалий:", sum(predictions == -1))
        print("Размер processed_data:", len(processed_data))
        print("Уникальные временные метки:", len(processed_data.index.unique()))

        # Добавим более подробный вывод информации об аномалиях
        print("\nАНАЛИЗ АНОМАЛИЙ:")
        print(f"Всего проанализировано пакетов: {len(processed_data)}")
        print(f"Количество обнаруженных аномалий: {sum(predictions == -1)}")
        print(f"Процент аномальных пакетов: {(sum(predictions == -1) / len(predictions) * 100):.2f}%")

        # Если есть аномалии, покажем их характеристики
        if sum(predictions == -1) > 0:
            print("\nХарактеристики аномальных пакетов:")
            anomaly_data = window_features[predictions == -1]
            print("\nСредние значения признаков аномальных пакетов:")
            print(anomaly_data.mean())
            
            # Показать временные метки аномалий
            anomaly_times = processed_data.index[predictions == -1]
            print("\nВременные метки аномалий:")
            for time in anomaly_times:
                print(f"- {time}")

        # 10. Визуализация результатов
        visualizer = TrafficVisualizer()
        
        # Распределение длины пакетов
        visualizer.plot_traffic_distribution(
            processed_data, 
            'length', 
            'Распределение длины пакетов'
        )
        
        # Визуализация аномалий
        visualizer.plot_anomalies_2d(
            window_features, 
            predictions, 
            'Аномалии в сетевом трафике'
        )
        
        # Временной ряд аномалий
        visualizer.plot_time_series_anomalies(
            processed_data,
            predictions,
            'length'
        )

        # После обнаружения аномалий
        print("\nВизуализация результатов анализа...")
        visualizer.plot_anomaly_summary(processed_data, predictions, window_features)

        # Выводим детальную информацию об аномалиях
        if sum(predictions == -1) > 0:
            print("\nДЕТАЛИ АНОМАЛИЙ:")
            anomaly_data = processed_data[predictions == -1]
            
            print("\nРаспределение по протоколам:")
            print(anomaly_data['protocol'].value_counts())
            
            print("\nIP-адреса источников аномального трафика:")
            print(anomaly_data['src_ip'].value_counts())
            
            print("\nIP-адреса назначения аномального трафика:")
            print(anomaly_data['dst_ip'].value_counts())
        else:
            print("\nАномалий не обнаружено в данном периоде наблюдения.")

    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nЗавершение работы...")

if __name__ == "__main__":
    main() 