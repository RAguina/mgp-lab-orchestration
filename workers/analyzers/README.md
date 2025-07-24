(venv) PS C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab> python local_models/llm_launcher.py
=== LAUNCHER DE LLMs ===
Modelos disponibles:
  - llama3
  - mistral7b
¿Qué modelo quieres usar? [llama3/mistral7b]: llama3
Prompt a enviar al modelo: ¿Qué es una función lambda en Python?

[INFO] Cargando modelo: llama3 (meta-llama/Meta-Llama-3-8B-Instruct)
Traceback (most recent call last):
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\utils\_http.py", line 409, in hf_raise_for_status
    response.raise_for_status()
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\requests\models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/main/config.json

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\transformers\utils\hub.py", line 470, in cached_files
    hf_hub_download(
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\utils\_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\file_download.py", line 1008, in hf_hub_download
    return _hf_hub_download_to_cache_dir(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\file_download.py", line 1115, in _hf_hub_download_to_cache_dir
    _raise_on_head_call_error(head_call_error, force_download, local_files_only)
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\file_download.py", line 1645, in _raise_on_head_call_error
    raise head_call_error
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\file_download.py", line 1533, in _get_metadata_or_catch_error
    metadata = get_hf_file_metadata(
               ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\utils\_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\file_download.py", line 1450, in get_hf_file_metadata
    r = _request_wrapper(
        ^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\file_download.py", line 286, in _request_wrapper
    response = _request_wrapper(
               ^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\file_download.py", line 310, in _request_wrapper
    hf_raise_for_status(response)
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\huggingface_hub\utils\_http.py", line 426, in hf_raise_for_status
    raise _format(GatedRepoError, message, response) from e
huggingface_hub.errors.GatedRepoError: 401 Client Error. (Request ID: Root=1-68459d94-2dd752c55d56be57524322bf;1f73a5ef-7cba-4d6e-9486-d6aee240a468)

Cannot access gated repo for url https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/main/config.json.   
Access to model meta-llama/Meta-Llama-3-8B-Instruct is restricted. You must have access to it and be authenticated to access it. Please log in.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\local_models\llm_launcher.py", line 120, in <module>
    menu()
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\local_models\llm_launcher.py", line 117, in menu
    launch_model(model_key, prompt)
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\local_models\llm_launcher.py", line 63, in launch_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\transformers\models\auto\tokenization_auto.py", line 970, in from_pretrained
    config = AutoConfig.from_pretrained(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\transformers\models\auto\configuration_auto.py", line 1153, in from_pretrained
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\transformers\configuration_utils.py", line 595, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\transformers\configuration_utils.py", line 654, in _get_config_dict
    resolved_config_file = cached_file(
                           ^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\transformers\utils\hub.py", line 312, in cached_file
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Proyects\Orchestration Models\ai-multi-agent-project\ai-agent-lab\venv\Lib\site-packages\transformers\utils\hub.py", line 533, in cached_files
    raise OSError(
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct.
401 Client Error. (Request ID: Root=1-68459d94-2dd752c55d56be57524322bf;1f73a5ef-7cba-4d6e-9486-d6aee240a468)

Cannot access gated repo for url https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/main/config.json.   
Access to model meta-llama/Meta-Llama-3-8B-Instruct is restricted. You must have access to it and be authenticated to access it. Please log in.