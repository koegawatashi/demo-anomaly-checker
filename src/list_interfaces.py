import pyshark

def list_interfaces():
    try:
        # Получаем список интерфейсов
        interfaces = pyshark.LiveCapture.get_interfaces()
        print("Доступные сетевые интерфейсы:")
        for interface in interfaces:
            print(f"- {interface}")
    except Exception as e:
        print(f"Ошибка при получении списка интерфейсов: {e}")

if __name__ == "__main__":
    list_interfaces() 