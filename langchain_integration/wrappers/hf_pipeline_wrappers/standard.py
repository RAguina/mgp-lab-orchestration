from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_model(model_path: str, logger=None, **kwargs):
    if logger:
        logger.info("loading_standard", msg=f"Cargando modelo desde {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=kwargs.get("device_map", "auto"))

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return model, tokenizer, pipe
