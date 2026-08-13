"""
Archivo: openai_client.py
Propósito: Implementación del proveedor OpenAI (GPT).

Utiliza la librería openai para comunicarse
con los modelos de GPT (gpt-4o, gpt-4o-mini, etc.).

Soporta:
    - generar(): Síncrono, respuesta completa.
    - generar_async(): Asíncrono, respuesta completa.
    - generar_streaming(): Asíncrono, yielda chunks.
"""

from collections.abc import AsyncIterator
import openai
from .base import BaseProvider
from schemas import LLMParams, AuthError, RateLimitError, NetworkError


class OpenAIClient(BaseProvider):
    """
    Proveedor de IA que utiliza OpenAI GPT.

    Attributes:
        client: Instancia del cliente de OpenAI.
        model: Nombre del modelo a usar.
    """

    def __init__(self, api_key: str, modelo: str = "gpt-4o-mini"):
        """
        Inicializa el proveedor OpenAI.

        Args:
            api_key: Clave API de OpenAI.
            modelo: Nombre del modelo a usar (default: gpt-4o-mini).
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.async_client = openai.AsyncOpenAI(api_key=api_key)
        self.model = modelo

    def _params(self, texto: str, params: LLMParams) -> dict:
        """Crea los parámetros para chat.completions.create()."""
        return {
            "model": self.model,
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "messages": [{"role": "user", "content": texto}]
        }

    def generar(self, texto: str, params: LLMParams) -> str:
        """Envío síncrono. Retorna respuesta completa."""
        try:
            response = self.client.chat.completions.create(**self._params(texto, params))
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            raise RateLimitError(f"OpenAI: Rate limit alcanzado - {e}") from e
        except openai.AuthenticationError as e:
            raise AuthError(f"OpenAI: API key inválida - {e}") from e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise NetworkError(f"OpenAI: Error de conexión - {e}") from e
        except Exception as e:
            raise NetworkError(f"OpenAI: Error desconocido - {e}") from e

    async def generar_async(self, texto: str, params: LLMParams) -> str:
        """Envío asíncrono. Retorna respuesta completa."""
        try:
            response = await self.async_client.chat.completions.create(**self._params(texto, params))
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            raise RateLimitError(f"OpenAI: Rate limit alcanzado - {e}") from e
        except openai.AuthenticationError as e:
            raise AuthError(f"OpenAI: API key inválida - {e}") from e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise NetworkError(f"OpenAI: Error de conexión - {e}") from e
        except Exception as e:
            raise NetworkError(f"OpenAI: Error desconocido - {e}") from e

    async def generar_streaming(self, texto: str, params: LLMParams) -> AsyncIterator[str]:
        """Envío asíncrono con streaming. Yielda chunks."""
        try:
            stream = await self.async_client.chat.completions.create(
                **self._params(texto, params),
                stream=True
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except openai.RateLimitError as e:
            raise RateLimitError(f"OpenAI: Rate limit alcanzado - {e}") from e
        except openai.AuthenticationError as e:
            raise AuthError(f"OpenAI: API key inválida - {e}") from e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise NetworkError(f"OpenAI: Error de conexión - {e}") from e
        except Exception as e:
            raise NetworkError(f"OpenAI: Error desconocido - {e}") from e
