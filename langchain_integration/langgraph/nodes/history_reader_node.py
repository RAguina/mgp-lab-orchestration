# ========================================
# 1. HISTORY READER NODE - EVOLVED VERSION
# ========================================

# langchain_integration/langgraph/nodes/history_reader_node.py
import os
import time
import logging
from typing import Dict, Any
from langchain_integration.langgraph.agent_state import AgentState

# Setup logger para este nodo
logger = logging.getLogger("history_reader")

def history_reader_node(state: AgentState) -> AgentState:
    """
    Worker especializado en lectura de historial con handling robusto
    """
    start_time = time.time()
    node_id = f"history_reader_{int(start_time)}"
    
    logger.info(f"[{node_id}] === HISTORY READER WORKER STARTED ===")
    print("[HISTORY] Leyendo historial de output anterior...")
    
    messages = state.get("messages", [])
    last_output = ""
    history_metadata = {}
    
    try:
        # Múltiples directorios de output
        output_dirs = [
            "outputs",
            "outputs/deepseek7b/runs",
            "outputs/mistral7b/runs", 
            "outputs/llama3/runs"
        ]
        
        all_files = []
        
        # Buscar en todos los directorios
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                logger.info(f"[{node_id}] Scanning directory: {output_dir}")
                
                files = [
                    os.path.join(output_dir, f) 
                    for f in os.listdir(output_dir) 
                    if f.endswith(('.txt', '.json'))
                ]
                all_files.extend(files)
        
        if all_files:
            # Ordenar por fecha de modificación (más reciente primero)
            all_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            last_file = all_files[0]
            
            logger.info(f"[{node_id}] Found {len(all_files)} output files")
            logger.info(f"[{node_id}] Reading latest: {os.path.basename(last_file)}")
            
            # Leer con encoding seguro
            try:
                with open(last_file, "r", encoding="utf-8", errors="ignore") as f:
                    last_output = f.read()
                
                # Truncar si es muy largo
                if len(last_output) > 5000:
                    last_output = last_output[:5000] + "\n[... truncated ...]"
                
                # Metadata del archivo
                file_stats = os.stat(last_file)
                history_metadata = {
                    "file_path": last_file,
                    "file_size": file_stats.st_size,
                    "modified_time": file_stats.st_mtime,
                    "total_files_found": len(all_files),
                    "content_length": len(last_output)
                }
                
                messages.append(f"[HISTORY] Último output cargado: {os.path.basename(last_file)} ({len(last_output)} chars)")
                logger.info(f"[{node_id}] Successfully loaded {len(last_output)} characters")
                
            except UnicodeDecodeError as e:
                logger.warning(f"[{node_id}] Unicode error reading {last_file}: {e}")
                # Fallback con encoding latin-1
                with open(last_file, "r", encoding="latin-1") as f:
                    last_output = f.read()[:5000]  # Limitar tamaño
                messages.append(f"[HISTORY] Archivo leído con encoding alternativo (posibles caracteres corruptos)")
                
        else:
            logger.info(f"[{node_id}] No output files found in any directory")
            messages.append("[HISTORY] No se encontró output previo en ningún directorio")
            
    except Exception as e:
        logger.error(f"[{node_id}] Error reading history: {str(e)}")
        messages.append(f"[HISTORY] Error al leer historial: {str(e)}")
        last_output = ""
        history_metadata = {"error": str(e)}
    
    # Resultado
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"[{node_id}] === HISTORY READER COMPLETED ===")
    logger.info(f"[{node_id}] Total processing time: {total_time:.3f}s")
    logger.info(f"[{node_id}] Content loaded: {len(last_output)} chars")
    
    print(f"[HISTORY] Completado: {len(last_output)} caracteres en {total_time:.3f}s")
    
    return {
        **state,
        "last_output": last_output,
        "history_metadata": history_metadata,
        "messages": messages
    }