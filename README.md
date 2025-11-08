# practicaIA
Practica # 1:
Enseñando a la IA a contar chistes
Los modelos de lenguaje entienden texto, pero… ¿entienden el humor? 😅

En esta práctica vas a experimentar con modelos de Hugging Face para intentar que un Transformer genere chistes divertidos y coherentes. Tu misión es usar diferentes modelos y técnicas de prompting para obtener el mejor chiste posible.

📋 Requisitos
Instalar las siguiente librerias: pip install "torch>=2.3.0" "transformers==4.44.2" "huggingface_hub==0.25.2" "accelerate==0.33.0" "safetensors==0.4.4"

Cuenta en hugginface

Crear un token de hugginface

🎯 Objetivos
Probar modelos de generación de texto (mrm8488/spanish-gpt2, datificate/gpt2-small-spanish, etc.).

Aplicar las técnicas de prompting para mejorar la calidad y coherencia del chiste.

Analizar cómo afectan los parámetros temperature, top_p y max_new_tokens a la creatividad del modelo.

Documentar cuál fue tu mejor prompt y por qué funcionó.

from huggingface_hub import login

token = "token_huggin_face"
login(token)

# modelo = "datificate/gpt2-small-spanish"
# modelo = "DeepESP/gpt2-spanish"
# modelo = "flax-community/gpt-2-spanish"
modelo = "google/gemma-2b-it"


from transformers import pipeline
modelo_chistes = pipeline(
    "text-generation", 
    model=modelo, 
    device_map=None,     # evita offload
    device="cpu",        # fuerza CPU
    torch_dtype="float16"  # reduce uso de memoria
)
/home/ajramosg/Desarrollo/Demos/PRACTICAS SEMILLERO/PRACTICA 1/.venv/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `huggingface-cli` if you want to set the git credential as well.
Token is valid (permission: fineGrained).
Your token has been saved to /home/ajramosg/.cache/huggingface/token
Login successful
`config.hidden_act` is ignored, you should use `config.hidden_activation` instead.
Gemma's activation function will be set to `gelu_pytorch_tanh`. Please, use
`config.hidden_activation` if you want to override this behaviour.
See https://github.com/huggingface/transformers/pull/29402 for more details.
Loading checkpoint shards: 100%|██████████| 2/2 [00:17<00:00,  8.71s/it]
# Paso 1: Definir un prompt
prompt = """Cuéntame un chiste corto sobre inteligencia artificial."""

# Paso 2: Generar el chiste con el modelo
resultado_fewshot = modelo_chistes(
    prompt,
    max_new_tokens=80,   # longitud máxima del texto generado
    temperature=0.9,     # controla la creatividad
    top_p=0.95           # controla la diversidad
)[0]["generated_text"]

# Paso 3: Mostrar el resultado
print("🎭 Prompt:")
print(prompt)
print("\n🤣 Chiste generado:")
print(resultado_fewshot)
/home/ajramosg/Desarrollo/Demos/PRACTICAS SEMILLERO/PRACTICA 1/.venv/lib/python3.11/site-packages/transformers/generation/configuration_utils.py:567: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/home/ajramosg/Desarrollo/Demos/PRACTICAS SEMILLERO/PRACTICA 1/.venv/lib/python3.11/site-packages/transformers/generation/configuration_utils.py:572: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.95` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
🎭 Prompt:
Cuéntame un chiste corto sobre inteligencia artificial.

🤣 Chiste generado:
Cuéntame un chiste corto sobre inteligencia artificial.

¿Qué es la inteligencia artificial que no puede ser programada?

Una respuesta artificial.
