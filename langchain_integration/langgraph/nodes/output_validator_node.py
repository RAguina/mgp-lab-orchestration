# langchain_integration/langgraph/nodes/output_validator_node.py

import time
import logging
import re
from typing import Dict, Any, List
from langchain_integration.langgraph.agent_state import AgentState

# Configurar logger específico para este nodo
logger = logging.getLogger("output_validator")

# -----------------------
# Utilidades internas
# -----------------------

CODE_BLOCK_RE = re.compile(r"```.*?```", flags=re.DOTALL | re.IGNORECASE)

def strip_code_blocks(text: str) -> str:
    """Elimina bloques de código triple-backtick para validar solo el texto explicativo."""
    return CODE_BLOCK_RE.sub("", text)

def is_short_numeric_answer(text: str) -> bool:
    """
    Permite respuestas numéricas cortas (e.g., '7', '-3.14', '1,234.56').
    Útil para queries tipo 'raiz cuadrada de 49'.
    """
    t = text.strip()
    if len(t) > 12:  # si ya es largo, no lo tratamos como 'short numeric'
        return False
    return bool(re.fullmatch(r"[+\-]?\d[\d,\.]*", t))

def is_repetitive_text(text: str) -> bool:
    """Detecta si el texto es excesivamente repetitivo."""
    words = text.lower().split()
    if len(words) < 10:
        return False
    unique_words = len(set(words))
    total_words = len(words)
    return (unique_words / total_words) < 0.3  # Menos del 30% de palabras únicas

# -----------------------
# Nodo principal
# -----------------------

def output_validator_node(state: AgentState) -> AgentState:
    """
    Worker especializado en validación de outputs con logging detallado.
    """
    start_time = time.time()
    node_id = f"validator_worker_{int(start_time)}"
    
    logger.info(f"[{node_id}] === OUTPUT VALIDATOR WORKER STARTED ===")
    
    messages = state.get("messages", [])
    output = state.get("output", "") or ""
    task_type = state.get("task_type", "chat")
    retry_count = state.get("retry_count", 0)
    execution_metrics = state.get("execution_metrics", {})
    
    logger.info(f"[{node_id}] Task type: {task_type}")
    logger.info(f"[{node_id}] Output length: {len(output)} chars")
    logger.info(f"[{node_id}] Current retry count: {retry_count}")
    logger.info(f"[{node_id}] Output preview: '{output[:100]}...'")
    
    print("[VALIDATE] Validando output del modelo...")

    # Análisis detallado de calidad
    validation_results = perform_detailed_validation(output, task_type, node_id)
    
    # Decidir si hacer retry basado en múltiples factores
    should_retry = decide_retry_logic(
        validation_results, 
        retry_count, 
        execution_metrics, 
        task_type,
        node_id
    )
    
    # Preparar mensajes informativos
    validation_summary = create_validation_summary(validation_results, should_retry)
    messages.extend(validation_summary)
    
    # Calcular tiempo total
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"[{node_id}] === VALIDATION COMPLETED ===")
    logger.info(f"[{node_id}] Validation time: {total_time:.2f}s")
    logger.info(f"[{node_id}] Should retry: {should_retry}")
    logger.info(f"[{node_id}] Validation score: {validation_results['overall_score']}/10")
    
    result_state = {
        **state,
        "retry": should_retry,
        "retry_count": retry_count + (1 if should_retry else 0),
        "messages": messages,
        # Agregar metadata de validación
        "validation_metadata": {
            "node_id": node_id,
            "validation_results": validation_results,
            "should_retry": should_retry,
            "validation_time": total_time,
            "retry_reason": validation_results.get("retry_reason", ""),
        }
    }
    
    print(f"[VALIDATE] {'RETRY' if should_retry else 'APPROVED'} - Score: {validation_results['overall_score']}/10")
    
    return result_state

# -----------------------
# Validaciones
# -----------------------

def perform_detailed_validation(output: str, task_type: str, node_id: str) -> Dict[str, Any]:
    """
    Realiza validación detallada del output según el tipo de tarea.
    """
    logger.info(f"[{node_id}] Starting detailed validation for task type: {task_type}")
    
    results = {
        "overall_score": 5,
        "checks_passed": [],
        "checks_failed": [],
        "warnings": [],
        "retry_reason": ""
    }
    
    # Validaciones básicas universales
    basic_checks = validate_basic_quality(output, node_id)
    merge_validation_results(results, basic_checks)
    
    # Validaciones específicas por tipo de tarea
    if task_type == "code":
        code_checks = validate_code_output(output, node_id)
        merge_validation_results(results, code_checks)
    elif task_type == "technical":
        tech_checks = validate_technical_output(output, node_id)
        merge_validation_results(results, tech_checks)
    elif task_type == "creative":
        creative_checks = validate_creative_output(output, node_id)
        merge_validation_results(results, creative_checks)
    elif task_type == "analysis":
        analysis_checks = validate_analysis_output(output, node_id)
        merge_validation_results(results, analysis_checks)
    
    # Calcular score final
    passed_checks = len(results["checks_passed"])
    failed_checks = len(results["checks_failed"])
    warnings = len(results["warnings"])
    
    results["overall_score"] = max(1, min(10, 5 + passed_checks - failed_checks - (warnings * 0.5)))
    
    logger.info(f"[{node_id}] Validation completed: {passed_checks} passed, {failed_checks} failed, {warnings} warnings")
    
    return results

def validate_basic_quality(output: str, node_id: str) -> Dict[str, Any]:
    """Validaciones básicas que aplican a cualquier tipo de output."""
    results = {"checks_passed": [], "checks_failed": [], "warnings": []}

    # 1) Longitud mínima — pero permite respuestas numéricas cortas
    if len(output.strip()) >= 10 or is_short_numeric_answer(output):
        results["checks_passed"].append("minimum_length_or_short_numeric")
        logger.debug(f"[{node_id}] ✓ Minimum length (or short numeric) check passed")
    else:
        results["checks_failed"].append("minimum_length")
        logger.warning(f"[{node_id}] ✗ Output too short: {len(output)} chars")

    # 2) Errores obvios — ignorar bloques de código
    text_no_code = strip_code_blocks(output).lower()

    # Patrones “críticos” más estrictos
    critical_error_patterns = [
        r"\btraceback\b",
        r"\bexception\b",
        r"\bvalueerror\b",
        r"\btypeerror\b",
        r"\bkeyerror\b",
        r"\bindexerror\b",
        r"\bsegmentation fault\b",
        r"\bmemoryerror\b",
        r"\bassertionerror\b",
        r"\bruntimeerror\b",
        # línea que comienza con "error:" fuera de código
        r"(^|\n)\s*error\s*:",
    ]

    found_critical = False
    for pattern in critical_error_patterns:
        if re.search(pattern, text_no_code):
            results["checks_failed"].append(f"error_pattern_{pattern}")
            logger.warning(f"[{node_id}] ✗ Critical error pattern detected: {pattern}")
            found_critical = True
            break

    if not found_critical:
        results["checks_passed"].append("no_critical_error_patterns")
        logger.debug(f"[{node_id}] ✓ No critical error patterns detected")

    # 3) Detectar texto repetitivo
    if is_repetitive_text(output):
        results["warnings"].append("repetitive_text")
        logger.warning(f"[{node_id}] ⚠ Repetitive text detected")

    # 4) Detectar texto potencialmente incompleto (suavizado)
    end = output.strip().lower()
    if any(end.endswith(suffix) for suffix in (" y", " pero", " sin embargo")):
        results["warnings"].append("potentially_incomplete")
        logger.warning(f"[{node_id}] ⚠ Output might be incomplete")

    return results

def validate_code_output(output: str, node_id: str) -> Dict[str, Any]:
    """Validaciones específicas para código."""
    results = {"checks_passed": [], "checks_failed": [], "warnings": []}
    
    logger.debug(f"[{node_id}] Validating code output")
    
    # Buscar indicadores de código
    code_indicators = ["def ", "function", "class ", "import ", "var ", "const ", "let "]
    if any(indicator in output.lower() for indicator in code_indicators):
        results["checks_passed"].append("code_structure_present")
    else:
        results["warnings"].append("no_code_structure")
    
    # Buscar bloques de código formateados
    if "```" in output or "    " in output:  # Markdown code blocks o indentación
        results["checks_passed"].append("code_formatting")
    else:
        results["warnings"].append("poor_code_formatting")
    
    return results

def validate_technical_output(output: str, node_id: str) -> Dict[str, Any]:
    """Validaciones específicas para contenido técnico."""
    results = {"checks_passed": [], "checks_failed": [], "warnings": []}
    
    logger.debug(f"[{node_id}] Validating technical output")
    
    # Buscar explicaciones técnicas
    tech_indicators = ["porque", "debido a", "significa", "se debe", "funciona"]
    if any(indicator in output.lower() for indicator in tech_indicators):
        results["checks_passed"].append("explanatory_content")
    
    # Longitud apropiada para explicaciones
    if len(output) >= 100:
        results["checks_passed"].append("sufficient_detail")
    else:
        results["warnings"].append("brief_explanation")
    
    return results

def validate_creative_output(output: str, node_id: str) -> Dict[str, Any]:
    """Validaciones específicas para contenido creativo."""
    results = {"checks_passed": [], "checks_failed": [], "warnings": []}
    
    logger.debug(f"[{node_id}] Validating creative output")
    
    # Buscar elementos narrativos
    creative_indicators = ["historia", "personaje", "escena", "había", "era", "vivía"]
    if any(indicator in output.lower() for indicator in creative_indicators):
        results["checks_passed"].append("narrative_elements")
    
    return results

def validate_analysis_output(output: str, node_id: str) -> Dict[str, Any]:
    """Validaciones específicas para análisis."""
    results = {"checks_passed": [], "checks_failed": [], "warnings": []}
    
    logger.debug(f"[{node_id}] Validating analysis output")
    
    # Buscar estructura analítica
    analysis_indicators = ["ventajas", "desventajas", "por un lado", "por otro", "en conclusión"]
    if any(indicator in output.lower() for indicator in analysis_indicators):
        results["checks_passed"].append("analytical_structure")
    
    return results

def decide_retry_logic(validation_results: Dict, retry_count: int, execution_metrics: Dict, 
                      task_type: str, node_id: str) -> bool:
    """Lógica inteligente para decidir si hacer retry."""
    
    # Límite máximo de retries
    MAX_RETRIES = 2
    if retry_count >= MAX_RETRIES:
        logger.info(f"[{node_id}] Max retries reached ({retry_count}), not retrying")
        return False
    
    # Si score es muy bajo, retry
    if validation_results["overall_score"] < 3:
        validation_results["retry_reason"] = f"Low quality score: {validation_results['overall_score']}"
        logger.info(f"[{node_id}] Retry due to low score: {validation_results['overall_score']}")
        return True
    
    # Si hay errores críticos, retry
    critical_failures = [f for f in validation_results["checks_failed"] 
                        if f.startswith("error_pattern_") or f == "minimum_length"]
    if critical_failures:
        validation_results["retry_reason"] = f"Critical failures: {critical_failures}"
        logger.info(f"[{node_id}] Retry due to critical failures: {critical_failures}")
        return True
    
    # Si la ejecución fue muy rápida (posible error), retry
    # Nota: dependiendo de cómo se guarden métricas, puede venir en ms o s.
    infer_time = execution_metrics.get("inference_time", execution_metrics.get("inference_time_sec", 0))
    if infer_time and infer_time < 1.0:  # si está en segundos y <1s, sospechoso
        validation_results["retry_reason"] = "Suspiciously fast execution"
        logger.info(f"[{node_id}] Retry due to fast execution: {infer_time}s")
        return True
    
    logger.info(f"[{node_id}] Validation passed, no retry needed")
    return False

# -----------------------
# Helpers
# -----------------------

def merge_validation_results(main_results: Dict, new_results: Dict):
    """Combina resultados de validación."""
    main_results["checks_passed"].extend(new_results.get("checks_passed", []))
    main_results["checks_failed"].extend(new_results.get("checks_failed", []))
    main_results["warnings"].extend(new_results.get("warnings", []))
    
def create_validation_summary(validation_results: Dict, should_retry: bool) -> List[str]:
    """Crea resumen legible de la validación."""
    summary = []
    
    score = validation_results["overall_score"]
    passed = len(validation_results["checks_passed"])
    failed = len(validation_results["checks_failed"])
    warnings = len(validation_results["warnings"])
    
    summary.append(f"[VALIDATE] Score: {score}/10 | Passed: {passed} | Failed: {failed} | Warnings: {warnings}")
    
    if should_retry:
        reason = validation_results.get("retry_reason", "Quality issues detected")
        summary.append(f"[VALIDATE] ⚠ RETRY necesario: {reason}")
    else:
        summary.append("[VALIDATE] ✅ Output validado correctamente")
    
    # Agregar detalles específicos si hay problemas
    if validation_results["checks_failed"]:
        summary.append(f"[VALIDATE] Checks fallidos: {', '.join(validation_results['checks_failed'][:3])}")
    
    if validation_results["warnings"]:
        summary.append(f"[VALIDATE] Advertencias: {', '.join(validation_results['warnings'][:2])}")
    
    return summary

def route_after_validation(state: AgentState):
    """Función de routing después de la validación."""
    should_retry = state.get("retry", False)
    retry_count = state.get("retry_count", 0)
    
    # Log de la decisión de routing
    validation_metadata = state.get("validation_metadata", {})
    node_id = validation_metadata.get("node_id", "unknown")
    
    logger.info(f"[{node_id}] Routing decision: {'retry_execution' if should_retry else 'continue'}")
    
    if should_retry:
        logger.info(f"[{node_id}] Retry #{retry_count} will be attempted")
    
    return "retry_execution" if should_retry else "continue"
