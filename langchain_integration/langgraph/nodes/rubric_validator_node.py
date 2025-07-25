# langchain_integration/langgraph/nodes/rubric_validator_node.py - EVOLVED VERSION
import time
import logging
import re
from typing import Dict, Any, List, Tuple
from langchain_integration.langgraph.agent_state import AgentState
from langchain_integration.langgraph.local_llm_node import build_local_llm_tool_node

# Setup logger para este nodo
logger = logging.getLogger("rubric_validator")

def rubric_validator_node(state: AgentState) -> AgentState:
    """
    Worker especializado en validación de cumplimiento de rúbricas con scoring detallado
    """
    start_time = time.time()
    node_id = f"rubric_validator_{int(start_time)}"
    
    logger.info(f"[{node_id}] === RUBRIC VALIDATOR WORKER STARTED ===")
    print("[VALIDATE-RUBRIC] Validando cumplimiento de rúbricas...")
    
    messages = state.get("messages", [])
    validation_result = ""
    validation_metadata = {}
    
    try:
        # Obtener datos para validación
        rubrics = state.get("analysis_result", "").strip()
        model_output = state.get("output", "").strip()
        task_type = state.get("task_type", "unknown")
        
        if not rubrics:
            logger.warning(f"[{node_id}] No rubrics available for validation")
            messages.append("[VALIDATE-RUBRIC] No hay rúbricas disponibles para validar")
            validation_result = "Validación no posible: rúbricas no encontradas"
            
        elif not model_output:
            logger.warning(f"[{node_id}] No model output available for validation")
            messages.append("[VALIDATE-RUBRIC] No hay output del modelo para validar")
            validation_result = "Validación no posible: output del modelo no encontrado"
            
        else:
            logger.info(f"[{node_id}] Validating output against rubrics")
            logger.info(f"[{node_id}] Rubrics: {len(rubrics)} chars, Output: {len(model_output)} chars")
            
            # Extraer criterios de las rúbricas (parsing inteligente)
            criteria_count = _extract_criteria_count(rubrics)
            
            # Construir LLM tool
            llm_start = time.time()
            llm = build_local_llm_tool_node(
                model_key=state.get("selected_model", "mistral7b"),
                strategy=state.get("strategy", "optimized"),
                max_tokens=768  # Más tokens para validación detallada
            )
            llm_build_time = time.time() - llm_start
            
            # Prompt estructurado para validación
            prompt = f"""Evalúa sistemáticamente si la respuesta del modelo cumple con cada criterio de las rúbricas proporcionadas.

CONTEXTO DE EVALUACIÓN:
- Tipo de tarea: {task_type}
- Criterios a evaluar: {criteria_count} detectados
- Longitud de respuesta: {len(model_output)} caracteres

🎯 RÚBRICAS A APLICAR:
{rubrics[:2000]}{"..." if len(rubrics) > 2000 else ""}

💬 RESPUESTA DEL MODELO A EVALUAR:
{model_output[:2000]}{"..." if len(model_output) > 2000 else ""}

INSTRUCCIONES DE EVALUACIÓN:
1. Evalúa cada criterio identificado en las rúbricas
2. Asigna una puntuación 1-10 para cada criterio
3. Justifica brevemente cada puntuación
4. Calcula un promedio ponderado final
5. Identifica fortalezas y áreas de mejora

FORMATO DE RESPUESTA:
## EVALUACIÓN DE RÚBRICAS

### CRITERIO 1: [NOMBRE DEL CRITERIO]
- **Puntuación**: [1-10]/10
- **Cumplimiento**: [SÍ/PARCIAL/NO]
- **Justificación**: [razón breve]

### CRITERIO 2: [NOMBRE DEL CRITERIO]
- **Puntuación**: [1-10]/10
- **Cumplimiento**: [SÍ/PARCIAL/NO]
- **Justificación**: [razón breve]

[continuar con otros criterios...]

## RESUMEN EJECUTIVO
- **Puntuación Total**: [X.X]/10
- **Criterios Cumplidos**: [X] de [Y]
- **Fortalezas Principales**: [lista breve]
- **Áreas de Mejora**: [lista breve]
- **Recomendación General**: [APROBADO/NECESITA MEJORAS/RECHAZADO]"""

            # Ejecutar validación
            inference_start = time.time()
            validation_result = llm.invoke(prompt)
            inference_time = time.time() - inference_start
            
            # Extraer métricas de la validación
            scores_extracted = _extract_scores_from_validation(validation_result)
            
            # Metadata de la validación
            validation_metadata = {
                "llm_build_time": llm_build_time,
                "inference_time": inference_time,
                "task_type": task_type,
                "rubrics_length": len(rubrics),
                "output_evaluated_length": len(model_output),
                "validation_length": len(validation_result),
                "criteria_detected": criteria_count,
                "scores_extracted": scores_extracted,
                "model_used": state.get("selected_model", "unknown"),
                "strategy_used": state.get("strategy", "unknown")
            }
            
            messages.append(f"[VALIDATE-RUBRIC] Evaluación completada: {criteria_count} criterios analizados")
            logger.info(f"[{node_id}] Validation completed in {inference_time:.2f}s")
            logger.info(f"[{node_id}] Scores extracted: {scores_extracted}")
            
    except Exception as e:
        logger.error(f"[{node_id}] Error in rubric validation: {str(e)}")
        messages.append(f"[VALIDATE-RUBRIC] Error en validación: {str(e)}")
        validation_result = f"Error en validación de rúbricas: {str(e)}"
        validation_metadata = {"error": str(e)}
    
    # Resultado
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"[{node_id}] === RUBRIC VALIDATOR COMPLETED ===")
    logger.info(f"[{node_id}] Total processing time: {total_time:.3f}s")
    logger.info(f"[{node_id}] Validation result length: {len(validation_result)} chars")
    
    print(f"[VALIDATE-RUBRIC] Completado: {len(validation_result)} caracteres en {total_time:.3f}s")
    
    return {
        **state,
        "rubric_validation_result": validation_result,
        "rubric_validation_metadata": validation_metadata,
        "messages": messages
    }


def _extract_criteria_count(rubrics_text: str) -> int:
    """
    Extrae el número de criterios de evaluación del texto de rúbricas
    """
    try:
        # Buscar patrones comunes de criterios
        patterns = [
            r"### CRITERIO \d+:",
            r"\d+\.\s*[A-Z][A-Z\s]+:",
            r"CRITERIO \d+:",
            r"\*\*CRITERIO \d+\*\*",
            r"## \d+\."
        ]
        
        max_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, rubrics_text, re.IGNORECASE)
            max_count = max(max_count, len(matches))
        
        # Si no encuentra patrones específicos, estimar por estructura
        if max_count == 0:
            # Contar líneas que parecen títulos de criterios
            lines = rubrics_text.split('\n')
            title_lines = [line for line in lines if ':' in line and len(line) < 100 and line.strip().isupper()]
            max_count = len(title_lines)
        
        # Fallback: estimar por longitud
        if max_count == 0:
            estimated = min(max(len(rubrics_text) // 200, 3), 10)  # Entre 3 y 10 criterios
            max_count = estimated
        
        return max(max_count, 1)  # Mínimo 1 criterio
        
    except Exception:
        return 5  # Fallback por defecto


def _extract_scores_from_validation(validation_text: str) -> Dict[str, Any]:
    """
    Extrae puntuaciones y métricas del resultado de validación
    """
    try:
        scores = []
        
        # Buscar puntuaciones individuales
        score_patterns = [
            r"Puntuación[:\s]*(\d+(?:\.\d+)?)[/\s]*10",
            r"Score[:\s]*(\d+(?:\.\d+)?)[/\s]*10",
            r"(\d+(?:\.\d+)?)/10",
        ]
        
        for pattern in score_patterns:
            matches = re.findall(pattern, validation_text, re.IGNORECASE)
            for match in matches:
                try:
                    score = float(match)
                    if 0 <= score <= 10:
                        scores.append(score)
                except:
                    continue
        
        # Buscar puntuación total
        total_patterns = [
            r"Puntuación Total[:\s]*(\d+(?:\.\d+)?)",
            r"Total Score[:\s]*(\d+(?:\.\d+)?)",
            r"Promedio[:\s]*(\d+(?:\.\d+)?)"
        ]
        
        total_score = None
        for pattern in total_patterns:
            match = re.search(pattern, validation_text, re.IGNORECASE)
            if match:
                try:
                    total_score = float(match.group(1))
                    break
                except:
                    continue
        
        # Calcular promedio si no hay total explícito
        if total_score is None and scores:
            total_score = sum(scores) / len(scores)
        
        # Buscar estado de cumplimiento
        compliance_indicators = ["APROBADO", "APPROVED", "CUMPLE", "MEETS"]
        improvement_indicators = ["NECESITA MEJORAS", "NEEDS IMPROVEMENT", "PARCIAL"]
        rejection_indicators = ["RECHAZADO", "REJECTED", "NO CUMPLE", "FAILS"]
        
        status = "unknown"
        validation_upper = validation_text.upper()
        
        if any(indicator in validation_upper for indicator in compliance_indicators):
            status = "approved"
        elif any(indicator in validation_upper for indicator in improvement_indicators):
            status = "needs_improvement"
        elif any(indicator in validation_upper for indicator in rejection_indicators):
            status = "rejected"
        
        return {
            "individual_scores": scores,
            "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "total_score": total_score,
            "criteria_evaluated": len(scores),
            "status": status,
            "has_detailed_breakdown": len(scores) > 0
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "individual_scores": [],
            "average_score": 0,
            "total_score": None,
            "criteria_evaluated": 0,
            "status": "error"
        }


# Función auxiliar para integración con el orchestrator
def should_validate_rubrics(state: AgentState) -> str:
    """
    Determina si se debe ejecutar validación de rúbricas
    """
    has_rubrics = bool(state.get("analysis_result", "").strip())
    has_output = bool(state.get("output", "").strip())
    
    if has_rubrics and has_output:
        return "validate_rubrics"
    else:
        return "skip_rubric_validation"


# Test function para validar el nodo
if __name__ == "__main__":
    # Demo del rubric validator
    print("🧪 Testing Rubric Validator Node...")
    
    test_state = {
        "analysis_result": """
## RÚBRICA DE EVALUACIÓN

### CRITERIO 1: FUNCIONALIDAD
- **Descripción**: El código ejecuta correctamente
- **Escala**: 1-10

### CRITERIO 2: LEGIBILIDAD  
- **Descripción**: Es fácil de entender
- **Escala**: 1-10
        """,
        "output": "def sort_list(lst): return sorted(lst)",
        "task_type": "code",
        "selected_model": "mistral7b",
        "strategy": "optimized",
        "messages": []
    }
    
    result = rubric_validator_node(test_state)
    
    print(f"✅ Validation completed")
    print(f"📊 Result length: {len(result.get('rubric_validation_result', ''))} chars")
    print(f"📈 Metadata: {result.get('rubric_validation_metadata', {})}")
    
    print("🎉 Rubric Validator test completed!")