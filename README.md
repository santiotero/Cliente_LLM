# Cliente_LLM
Cliente LLM

## Estructura del Proyecto

```
Cliente_LLM/
├── .env                    # API keys (no commitear)
├── .env.example            # Template de API keys
├── .gitignore              # Ignora .venv/, .env, __pycache__/
├── configparams.json       # Configuración centralizada
├── requirements.txt        # Dependencias
├── schemas.py              # Excepciones custom + Pydantic schemas
├── client.py               # LLMClient con fallback chain + retry
├── main.py                 # Punto de entrada mínimo
└── providers/
    ├── __init__.py
    ├── base.py             # BaseProvider (clase abstracta)
    ├── gemini_client.py    # Proveedor Google Gemini
    ├── anthropic_client.py # Proveedor Anthropic Claude
    └── openai_client.py    # Proveedor OpenAI GPT
```
# Requisitos:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Para ejecutar (punto de entrada):
    python main.py

# Autor
Santiago Otero
