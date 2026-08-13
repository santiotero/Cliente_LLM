"""
Archivo: base.py
Propósito: Define la clase abstracta BaseProvider que todos los proveedores deben implementar.

Cada proveedor (Gemini, Anthropic, OpenAI) hereda de esta clase y
sobrescribe los métodos con su implementación específica.

Métodos soportados:
    - generar(): Síncrono, retorna respuesta completa.
    - generar_async(): Asíncrono, retorna respuesta completa.
    - generar_streaming(): Asíncrono, yielda chunks a medida que llegan.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from schemas import LLMParams


class BaseProvider(ABC):
    """
    Clase abstracta que define el contrato para todos los proveedores de IA.

    Métodos abstractos:
        generar(texto, params) -> str: Síncrono, respuesta completa.
        generar_async(texto, params) -> str: Asíncrono, respuesta completa.
        generar_streaming(texto, params) -> AsyncIterator[str]: Asíncrono, streaming.
    """

    @abstractmethod
    def generar(self, texto: str, params: LLMParams) -> str:
        """
        Envío síncrono. Retorna la respuesta completa.

        Args:
            texto: El prompt a enviar al modelo.
            params: Parámetros de generación (temperature, max_tokens).

        Returns:
            str: La respuesta generada por el modelo.
        """
        pass

    @abstractmethod
    async def generar_async(self, texto: str, params: LLMParams) -> str:
        """
        Envío asíncrono. Retorna la respuesta completa.

        Args:
            texto: El prompt a enviar al modelo.
            params: Parámetros de generación (temperature, max_tokens).

        Returns:
            str: La respuesta generada por el modelo.
        """
        pass

    @abstractmethod
    async def generar_streaming(self, texto: str, params: LLMParams) -> AsyncIterator[str]:
        """
        Envío asíncrono con streaming. Yielda chunks a medida que llegan.

        Args:
            texto: El prompt a enviar al modelo.
            params: Parámetros de generación (temperature, max_tokens).

        Yields:
            str: Fragmentos de la respuesta.
        """
        pass
