"""
Archivo: client.py
Propósito: Implementa el LLMClient con fallback chain.

El cliente:
    1. Lee .env y obtiene API keys.
    2. Lee configparams.json para config y parámetros.
    3. Crea los proveedores según las keys disponibles.
    4. Decide qué método usar según modo_ejecucion y modo_streaming.

Modos de ejecución:
    - Síncrono: Siempre retorna respuesta completa (bloquea).
    - Asíncrono + completo: No bloquea, retorna todo junto.
    - Asíncrono + streaming: No bloquea, yielda chunks.
"""

import json
import os
import time
from collections.abc import AsyncIterator
from dotenv import load_dotenv
from providers.base import BaseProvider
from providers.gemini_client import GeminiClient
from providers.anthropic_client import AnthropicClient
from providers.openai_client import OpenAIClient
from schemas import LLMParams, ConfigParams, ModoEjecucion, AuthError, RateLimitError, NetworkError


class LLMClient:
    """
    Cliente LLM que implementa fallback chain entre proveedores.

    Maneja internamente:
        - Carga de API keys desde .env
        - Carga de configparams.json
        - Creación de proveedores
        - Decisión de modo (sync/async/streaming)

    Attributes:
        proveedores: Lista de proveedores ordenados por prioridad.
        config: Configuración leída de configparams.json.
        params: Parámetros de generación creados desde config.
        mensaje_error: Mensaje a mostrar cuando todos los proveedores fallan.
    """

    def __init__(self):
        """
        Inicializa el cliente LLM.
        No recibe ningún parámetro - todo se lee de archivos.
        """
        # Cargar API keys desde .env
        load_dotenv()
        keys = {
            "gemini": os.getenv("GEMINI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
        }

        # Leer configparams.json
        self.config = self._cargar_config()

        # Crear proveedores según keys disponibles y orden de config
        proveedores_map = {
            "gemini": lambda: GeminiClient(keys["gemini"]),
            "anthropic": lambda: AnthropicClient(keys["anthropic"]),
            "openai": lambda: OpenAIClient(keys["openai"]),
        }

        self.proveedores = []
        for nombre in self.config.orden_proveedores:
            if keys.get(nombre) and keys[nombre] != "tu_api_key_aqui":
                try:
                    proveedor = proveedores_map[nombre]()
                    self.proveedores.append(proveedor)
                    print(f"  [+] {nombre.upper()} instanciado")
                except Exception as e:
                    print(f"  [-] {nombre.upper()} error al crear: {e}")
            else:
                print(f"  [-] {nombre.upper()} omitido (sin API key)")

        # Crear params desde config
        self.params = LLMParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            modo_ejecucion=self.config.modo_ejecucion,
            modo_streaming=self.config.modo_streaming
        )

        self.mensaje_error = self.config.mensaje_error

    def _cargar_config(self) -> ConfigParams:
        """Lee configparams.json y retorna ConfigParams."""
        with open("configparams.json", "r", encoding="utf-8") as f:
            return ConfigParams(**json.load(f))

    # =====================================================
    # MÉTODO PRINCIPAL - Decide según params
    # =====================================================

    def ejecutar(self, texto: str):
        """
        Método principal que ejecuta según modo_ejecucion y modo_streaming.

        Args:
            texto: El prompt a enviar al modelo.

        Returns:
            str: Si modo=sincrono o modo=asincrono+completo.
            AsyncIterator[str]: Si modo=asincrono+streaming.
        """
        if self.params.modo_ejecucion == ModoEjecucion.SINCRONO:
            return self.generar(texto)
        else:
            if self.params.modo_streaming:
                return self.generar_streaming(texto)
            else:
                return self.generar_async(texto)

    # =====================================================
    # MÉTODOS SÍNCRONOS
    # =====================================================

    def generar(self, texto: str) -> str:
        """
        Envío síncrono con fallback chain.
        Retorna la respuesta completa.
        Implementa reintentos en caso de RateLimitError.
        """
        for proveedor in self.proveedores:
            intentos = self.config.intentos_retry
            for intento in range(intentos):
                try:
                    respuesta = proveedor.generar(texto, self.params)
                    print(f"[OK] {proveedor.__class__.__name__} respondió correctamente")
                    return respuesta
                except RateLimitError as e:
                    if intento < intentos - 1:
                        delay = 2 ** intento
                        print(f"[REINTENTO] {proveedor.__class__.__name__}: {type(e).__name__} ({intento + 1}/{intentos}) - esperando {delay}s")
                        time.sleep(delay)
                    else:
                        print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - agotados los reintentos")
                        continue
                except AuthError as e:
                    print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - {e}")
                    break
                except NetworkError as e:
                    print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - {e}")
                    continue
                except Exception as e:
                    print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - {e}")
                    continue
        raise Exception(self.mensaje_error)

    # =====================================================
    # MÉTODOS ASÍNCRONOS
    # =====================================================

    async def generar_async(self, texto: str) -> str:
        """
        Envío asíncrono con fallback chain.
        Retorna la respuesta completa.
        Implementa reintentos en caso de RateLimitError.
        """
        for proveedor in self.proveedores:
            intentos = self.config.intentos_retry
            for intento in range(intentos):
                try:
                    respuesta = await proveedor.generar_async(texto, self.params)
                    print(f"[OK] {proveedor.__class__.__name__} respondió correctamente")
                    return respuesta
                except RateLimitError as e:
                    if intento < intentos - 1:
                        delay = 2 ** intento
                        print(f"[REINTENTO] {proveedor.__class__.__name__}: {type(e).__name__} ({intento + 1}/{intentos}) - esperando {delay}s")
                        time.sleep(delay)
                    else:
                        print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - agotados los reintentos")
                        continue
                except AuthError as e:
                    print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - {e}")
                    break
                except NetworkError as e:
                    print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - {e}")
                    continue
                except Exception as e:
                    print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - {e}")
                    continue
        raise Exception(self.mensaje_error)

    async def generar_streaming(self, texto: str) -> AsyncIterator[str]:
        """
        Envío asíncrono con streaming y fallback chain.
        Yielda chunks a medida que llegan.
        Implementa reintentos en caso de RateLimitError.
        """
        for proveedor in self.proveedores:
            intentos = self.config.intentos_retry
            for intento in range(intentos):
                try:
                    async for chunk in proveedor.generar_streaming(texto, self.params):
                        yield chunk
                    print(f"[OK] {proveedor.__class__.__name__} respondió correctamente")
                    return
                except RateLimitError as e:
                    if intento < intentos - 1:
                        delay = 2 ** intento
                        print(f"[REINTENTO] {proveedor.__class__.__name__}: {type(e).__name__} ({intento + 1}/{intentos}) - esperando {delay}s")
                        time.sleep(delay)
                    else:
                        print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - agotados los reintentos")
                        continue
                except AuthError as e:
                    print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - {e}")
                    break
                except NetworkError as e:
                    print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - {e}")
                    continue
                except Exception as e:
                    print(f"[FALLO] {proveedor.__class__.__name__}: {type(e).__name__} - {e}")
                    continue
        raise Exception(self.mensaje_error)
