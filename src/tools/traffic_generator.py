import socket
import time
import random

def generate_anomaly():
    # TCP флуд
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Попытка подключения к случайным портам
        for _ in range(100):
            port = random.randint(1000, 65535)
            try:
                sock.connect(('127.0.0.1', port))
            except:
                pass
            time.sleep(0.1)
    finally:
        sock.close()

    # UDP флуд
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # Отправка большого количества UDP пакетов
        message = b'X' * 1000
        for _ in range(100):
            port = random.randint(1000, 65535)
            sock.sendto(message, ('127.0.0.1', port))
            time.sleep(0.1)
    finally:
        sock.close()

if __name__ == "__main__":
    print("Генерация аномального трафика...")
    generate_anomaly()
    print("Готово!") 