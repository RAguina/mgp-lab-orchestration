from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline

def load_model(model_path: str, logger=None, use_quantization=True, **kwargs):
    if logger:
        logger.info("loading_optimized", msg=f"Cargando modelo 4-bit desde {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16"
    ) if use_quantization else None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=kwargs.get("device_map", "auto"),
        quantization_config=quant_config
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return model, tokenizer, pipe
