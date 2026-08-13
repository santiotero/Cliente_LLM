"""
Archivo: gemini_client.py
Propósito: Implementación del proveedor Google Gemini.

Utiliza la librería google-genai para comunicarse
con los modelos de Gemini (gemini-3.1-flash-lite, etc.).

Soporta:
    - generar(): Síncrono, respuesta completa.
    - generar_async(): Asíncrono, respuesta completa.
    - generar_streaming(): Asíncrono, yielda chunks.
"""

from collections.abc import AsyncIterator
from google import genai
from google.genai import types
from .base import BaseProvider
from schemas import LLMParams, AuthError, RateLimitError, NetworkError


class GeminiClient(BaseProvider):
    """
    Proveedor de IA que utiliza Google Gemini.

    Attributes:
        client: Instancia del cliente de Google GenAI.
        model: Nombre del modelo a usar.
    """

    def __init__(self, api_key: str, modelo: str = "gemini-3.1-flash-lite"):
        """
        Inicializa el proveedor Gemini.

        Args:
            api_key: Clave API de Google AI Studio.
            modelo: Nombre del modelo a usar (default: gemini-3.1-flash-lite).
        """
        self.client = genai.Client(api_key=api_key)
        self.model = modelo

    def _config(self, params: LLMParams) -> types.GenerateContentConfig:
        """Crea la configuración de generación."""
        return types.GenerateContentConfig(
            temperature=params.temperature,
            max_output_tokens=params.max_tokens
        )

    def generar(self, texto: str, params: LLMParams) -> str:
        """Envío síncrono. Retorna respuesta completa."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=texto,
                config=self._config(params)
            )
            return response.text
        except Exception as e:
            self._manejar_error(e)

    async def generar_async(self, texto: str, params: LLMParams) -> str:
        """Envío asíncrono. Retorna respuesta completa."""
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=texto,
                config=self._config(params)
            )
            return response.text
        except Exception as e:
            self._manejar_error(e)

    async def generar_streaming(self, texto: str, params: LLMParams) -> AsyncIterator[str]:
        """Envío asíncrono con streaming. Yielda chunks."""
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=texto,
                config=self._config(params)
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            self._manejar_error(e)

    def _manejar_error(self, e: Exception):
        """Captura errores del SDK y re-lanza como excepciones custom."""
        error_str = str(e).lower()
        if "api_key" in error_str or "invalid" in error_str or "401" in error_str:
            raise AuthError(f"Gemini: API key inválida - {e}") from e
        elif "429" in error_str or "rate" in error_str or "quota" in error_str:
            raise RateLimitError(f"Gemini: Rate limit alcanzado - {e}") from e
        elif "connect" in error_str or "timeout" in error_str or "network" in error_str:
            raise NetworkError(f"Gemini: Error de conexión - {e}") from e
        else:
            raise NetworkError(f"Gemini: Error desconocido - {e}") from e
