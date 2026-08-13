"""
Archivo: schemas.py
Propósito: Define los esquemas Pydantic para validación de datos.

Este archivo contiene:
    - LLMParams: Parámetros de generación (temperature, max_tokens, modo)
    - ConfigParams: Estructura válida del archivo configparams.json
"""

from enum import Enum
from pydantic import BaseModel, Field


# =====================================================
# EXCEPCIONES CUSTOM PARA ERRORES CONTROLADOS
# =====================================================

class LLMError(Exception):
    """Excepción base para errores del cliente LLM."""
    pass


class AuthError(LLMError):
    """API key inválida o no configurada."""
    pass


class RateLimitError(LLMError):
    """Límite de tasa alcanzado (HTTP 429)."""
    pass


class NetworkError(LLMError):
    """Error de conexión o timeout."""
    pass


class ModoEjecucion(str, Enum):
    """
    Modo de ejecución del cliente LLM.

    SINCRONO: Ejecución síncrona, retorna respuesta completa.
    ASINCRONO: Ejecución asíncrona, puede ser completa o streaming.
    """
    SINCRONO = "sincrono"
    ASINCRONO = "asincrono"


class LLMParams(BaseModel):
    """
    Parámetros de generación para cualquier proveedor de IA.

    Attributes:
        temperature: Creatividad de la respuesta (0.0=fijo, 2.0=muy creativo).
        max_tokens: Cantidad máxima de tokens en la respuesta.
        modo_ejecucion: Si es síncrono (bloquea) o asíncrono (no bloquea).
        modo_streaming: Solo aplica si modo_ejecucion=asincrono.
                       True = yielda chunks, False = retorna todo junto.
    """

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperatura: 0.0=fijo, 2.0=muy creativo"
    )
    max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Máximo de tokens en la respuesta"
    )
    modo_ejecucion: ModoEjecucion = Field(
        default=ModoEjecucion.SINCRONO,
        description="sincrono=bloquea, asincrono=no bloquea"
    )
    modo_streaming: bool = Field(
        default=False,
        description="Solo aplica si modo_ejecucion=asincrono. True=streaming, False=completo"
    )


class ConfigParams(BaseModel):
    """
    Esquema de validación para el archivo configparams.json.

    Contiene:
        - orden_proveedores: lista de strings con nombres de proveedores
        - mensaje_error: mensaje cuando todos los proveedores fallan
        - temperature: parámetro de generación
        - max_tokens: máximo de tokens
        - intentos_retry: reintentos por proveedor antes de fallar
        - modo_ejecucion: sincrono o asincrono
        - modo_streaming: streaming o completo (solo si asincrono)
    """

    orden_proveedores: list[str]
    mensaje_error: str
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0
    )
    max_tokens: int = Field(
        default=1024,
        gt=0
    )
    intentos_retry: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Cantidad de reintentos por proveedor antes de fallar"
    )
    modo_ejecucion: ModoEjecucion = Field(
        default=ModoEjecucion.SINCRONO
    )
    modo_streaming: bool = Field(
        default=False
    )
