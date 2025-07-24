#!/usr/bin/env python3
"""
Script para diagnosticar qué archivos están cambiando constantemente
Ejecutar en otra terminal mientras el server está corriendo
"""

import time
import os
from datetime import datetime

def monitor_files():
    print("🔍 Monitoring file changes in ai-agent-lab/...")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Directorios a monitorear
    dirs_to_check = ["logs", "outputs", "metrics", "__pycache__"]
    
    file_timestamps = {}
    
    try:
        while True:
            for dir_name in dirs_to_check:
                if os.path.exists(dir_name):
                    for root, dirs, files in os.walk(dir_name):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                current_mtime = os.path.getmtime(file_path)
                                
                                if file_path in file_timestamps:
                                    time_diff = current_mtime - file_timestamps[file_path]
                                    if time_diff > 0 and time_diff < 2.0:  # Cambio en últimos 2s
                                        now = datetime.now().strftime('%H:%M:%S')
                                        print(f"🔥 {now} - FREQUENT CHANGE: {file_path} (changed {time_diff:.1f}s ago)")
                                
                                file_timestamps[file_path] = current_mtime
                                
                            except OSError:
                                continue  # Archivo eliminado mientras checkeábamos
            
            time.sleep(0.5)  # Check every 500ms
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

if __name__ == "__main__":
    monitor_files()