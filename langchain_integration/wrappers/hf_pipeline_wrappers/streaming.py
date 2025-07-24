from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, pipeline
from threading import Thread

def load_model(model_path: str, logger=None, **kwargs):
    if logger:
        logger.info("loading_streaming", msg=f"Cargando modelo en modo streaming desde {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=kwargs.get("device_map", "auto"))

    def generate(pipeline_obj, tokenizer, prompt, max_tokens, logger=None):
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        gen_args = {
            "text_inputs": prompt,
            "max_new_tokens": max_tokens,
            "streamer": streamer
        }

        thread = Thread(target=pipeline_obj.model.generate, kwargs=gen_args)
        thread.start()

        output = ""
        for token in streamer:
            output += token
            if logger:
                logger.info("stream_token", token=token)
        return output

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipe.generate = lambda *a, **kw: generate(pipe, tokenizer, *a, **kw)

    return model, tokenizer, pipe
