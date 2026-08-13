"""
Archivo: anthropic_client.py
Propósito: Implementación del proveedor Anthropic (Claude).

Utiliza la librería anthropic para comunicarse
con los modelos de Claude (claude-3-haiku, claude-3-sonnet, etc.).

Soporta:
    - generar(): Síncrono, respuesta completa.
    - generar_async(): Asíncrono, respuesta completa.
    - generar_streaming(): Asíncrono, yielda chunks.
"""

from collections.abc import AsyncIterator
import anthropic
from .base import BaseProvider
from schemas import LLMParams, AuthError, RateLimitError, NetworkError


class AnthropicClient(BaseProvider):
    """
    Proveedor de IA que utiliza Anthropic Claude.

    Attributes:
        client: Instancia del cliente de Anthropic.
        model: Nombre del modelo a usar.
    """

    def __init__(self, api_key: str, modelo: str = "claude-3-haiku-20240307"):
        """
        Inicializa el proveedor Anthropic.

        Args:
            api_key: Clave API de Anthropic.
            modelo: Nombre del modelo a usar (default: claude-3-haiku).
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = modelo

    def _params(self, texto: str, params: LLMParams) -> dict:
        """Crea los parámetros para messages.create()."""
        return {
            "model": self.model,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "messages": [{"role": "user", "content": texto}]
        }

    def generar(self, texto: str, params: LLMParams) -> str:
        """Envío síncrono. Retorna respuesta completa."""
        try:
            response = self.client.messages.create(**self._params(texto, params))
            return response.content[0].text
        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic: Rate limit alcanzado - {e}") from e
        except anthropic.AuthenticationError as e:
            raise AuthError(f"Anthropic: API key inválida - {e}") from e
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise NetworkError(f"Anthropic: Error de conexión - {e}") from e
        except Exception as e:
            raise NetworkError(f"Anthropic: Error desconocido - {e}") from e

    async def generar_async(self, texto: str, params: LLMParams) -> str:
        """Envío asíncrono. Retorna respuesta completa."""
        try:
            response = await self.async_client.messages.create(**self._params(texto, params))
            return response.content[0].text
        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic: Rate limit alcanzado - {e}") from e
        except anthropic.AuthenticationError as e:
            raise AuthError(f"Anthropic: API key inválida - {e}") from e
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise NetworkError(f"Anthropic: Error de conexión - {e}") from e
        except Exception as e:
            raise NetworkError(f"Anthropic: Error desconocido - {e}") from e

    async def generar_streaming(self, texto: str, params: LLMParams) -> AsyncIterator[str]:
        """Envío asíncrono con streaming. Yielda chunks."""
        try:
            async with self.async_client.messages.stream(**self._params(texto, params)) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Anthropic: Rate limit alcanzado - {e}") from e
        except anthropic.AuthenticationError as e:
            raise AuthError(f"Anthropic: API key inválida - {e}") from e
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise NetworkError(f"Anthropic: Error de conexión - {e}") from e
        except Exception as e:
            raise NetworkError(f"Anthropic: Error desconocido - {e}") from e
