# utils/atomic_write.py
import os
import tempfile
import shutil

def atomic_write(filepath: str, data: bytes, encoding: str = None):
    """
    Escribe datos a un archivo de forma atómica para prevenir corrupción.
    
    Args:
        filepath (str): Ruta del archivo destino
        data (bytes): Datos a escribir
        encoding (str): Codificación si se pasan datos como string
    """
    if isinstance(data, str):
        if encoding is None:
            encoding = 'utf-8'
        data = data.encode(encoding)
    
    # Crear directorio padre si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Escribir a archivo temporal primero
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, 
                                     dir=os.path.dirname(filepath)) as tmp_file:
        tmp_file.write(data)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_path = tmp_file.name
    
    # Mover atómicamente al destino final
    shutil.move(temp_path, filepath)

if __name__ == "__main__":
    # Test básico
    test_data = "Test data for atomic write"
    atomic_write("test_atomic.txt", test_data.encode('utf-8'))
    print("✅ Test de atomic_write completado")
    
    # Limpiar
    if os.path.exists("test_atomic.txt"):
        os.remove("test_atomic.txt")