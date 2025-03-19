import pyshark
import pandas as pd
import os
from datetime import datetime
import signal
import sys

class NetworkTrafficCollector:
    def __init__(self, interface, output_dir='data/raw'):
        """
        Инициализация коллектора сетевого трафика.
        
        Args:
            interface (str): Сетевой интерфейс для захвата трафика
            output_dir (str): Директория для сохранения данных
        """
        self.interface = interface
        self.output_dir = output_dir
        self.capture = None
        self.is_running = False
        os.makedirs(output_dir, exist_ok=True)
        
        # Регистрируем обработчик сигнала
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигнала для корректного завершения"""
        print("\nПолучен сигнал остановки. Завершаем захват...")
        self.is_running = False
        if self.capture:
            self.capture.close()
    
    def capture_live_traffic(self, duration=60, packet_count=None):
        """
        Захват живого сетевого трафика.
        
        Args:
            duration (int): Продолжительность захвата в секундах
            packet_count (int): Количество пакетов для захвата
        
        Returns:
            str: Путь к сохраненному файлу
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"capture_{timestamp}.pcap")
        
        self.capture = pyshark.LiveCapture(interface=self.interface, output_file=output_file)
        self.is_running = True
        
        print(f"Начинаем захват трафика на интерфейсе {self.interface}...")
        print("Для остановки нажмите Ctrl+C")
        
        try:
            if packet_count:
                self.capture.sniff(packet_count=packet_count)
            else:
                self.capture.sniff(timeout=duration)
        except KeyboardInterrupt:
            print("\nЗахват остановлен пользователем")
        finally:
            if self.capture:
                self.capture.close()
            self.is_running = False
            
        print(f"Захват завершен. Данные сохранены в {output_file}")
        return output_file
    
    def read_pcap_file(self, pcap_file):
        """
        Чтение PCAP файла и преобразование в DataFrame.
        """
        try:
            # Включаем режим отладки
            capture = pyshark.FileCapture(pcap_file, debug=True)
            packets = []
            
            print("\nНачинаем чтение пакетов...")
            try:
                # Сначала проверим, что файл существует и не пустой
                if not os.path.exists(pcap_file):
                    print(f"Файл {pcap_file} не найден")
                    return pd.DataFrame()
                    
                if os.path.getsize(pcap_file) == 0:
                    print(f"Файл {pcap_file} пуст")
                    return pd.DataFrame()
                
                # Добавляем параметры для tshark
                capture = pyshark.FileCapture(
                    pcap_file,
                    debug=True,
                    use_json=True,  # Используем JSON вместо XML
                    include_raw=True,  # Включаем сырые данные
                    custom_parameters=[
                        '-o', 'tcp.desegment_tcp_streams:FALSE',  # Отключаем десегментацию TCP
                        '-n'  # Не преобразовывать адреса
                    ]
                )
                
                for i, packet in enumerate(capture):
                    try:
                        # Базовая информация о пакете
                        packet_dict = {
                            'timestamp': packet.sniff_time,
                            'protocol': packet.highest_layer,
                            'length': int(packet.length)
                        }
                        
                        # IP информация
                        if hasattr(packet, 'ip'):
                            packet_dict.update({
                                'src_ip': packet.ip.src,
                                'dst_ip': packet.ip.dst
                            })
                        else:
                            packet_dict.update({
                                'src_ip': 'Unknown',
                                'dst_ip': 'Unknown'
                            })
                        
                        # Информация о портах
                        if hasattr(packet, 'transport_layer') and packet.transport_layer:
                            try:
                                transport = packet[packet.transport_layer]
                                packet_dict.update({
                                    'src_port': transport.srcport,
                                    'dst_port': transport.dstport
                                })
                            except:
                                packet_dict.update({
                                    'src_port': 'Unknown',
                                    'dst_port': 'Unknown'
                                })
                        else:
                            packet_dict.update({
                                'src_port': 'Unknown',
                                'dst_port': 'Unknown'
                            })
                        
                        packets.append(packet_dict)
                        
                        if i % 100 == 0:
                            print(f"Обработано пакетов: {i}")
                            
                    except Exception as e:
                        print(f"Пропущен пакет {i}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Ошибка при чтении пакетов: {str(e)}")
            finally:
                try:
                    capture.close()
                except Exception as e:
                    print(f"Ошибка при закрытии capture: {str(e)}")
            
            if not packets:
                print("Предупреждение: Не удалось прочитать пакеты из файла")
                return pd.DataFrame()
            
            df = pd.DataFrame(packets)
            print(f"\nУспешно прочитано {len(df)} пакетов")
            print(f"Типы данных колонок:\n{df.dtypes}")
            return df
            
        except Exception as e:
            print(f"Критическая ошибка при работе с файлом: {str(e)}")
            return pd.DataFrame()

def capture_network_traffic(interface, duration=60, output_file='raw_traffic.csv'):
    """
    Захват сетевого трафика с использованием pyshark
    
    Args:
        interface (str): Сетевой интерфейс для захвата
        duration (int): Продолжительность захвата в секундах
        output_file (str): Путь к файлу для сохранения данных
    """
    capture = pyshark.LiveCapture(interface=interface)
    packets_data = []
    
    try:
        capture.sniff(timeout=duration)
        
        for packet in capture:
            packet_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'protocol': packet.transport_layer if hasattr(packet, 'transport_layer') else 'Unknown',
                'src_ip': packet.ip.src if hasattr(packet, 'ip') else 'Unknown',
                'dst_ip': packet.ip.dst if hasattr(packet, 'ip') else 'Unknown',
                'length': packet.length,
                'src_port': packet[packet.transport_layer].srcport if hasattr(packet, 'transport_layer') else 'Unknown',
                'dst_port': packet[packet.transport_layer].dstport if hasattr(packet, 'transport_layer') else 'Unknown'
            }
            packets_data.append(packet_data)
            
    except Exception as e:
        print(f"Ошибка при захвате пакетов: {e}")
    
    finally:
        df = pd.DataFrame(packets_data)
        df.to_csv(output_file, index=False)
        return df

def load_traffic_data(file_path):
    """
    Загрузка данных о трафике из CSV файла
    
    Args:
        file_path (str): Путь к файлу с данными
    """
    return pd.read_csv(file_path) 