"""
Archivo: main.py
Propósito: Punto de entrada del Cliente LLM.

Este archivo SOLO:
    1. Crea el LLMClient (lee todo internamente)
    2. Llama a ejecutar() con un texto de prueba
    3. Muestra la respuesta

"""

import asyncio
from client import LLMClient


async def main():
    """
    Función principal. Mínima lógica.
    """

    print("=" * 50)
    print("Cliente LLM")
    print("=" * 50)

    # Crear cliente (lee .env, config, crea providers internamente)
    client = LLMClient()
    print()

    if not client.proveedores:
        print("ERROR: No hay proveedores configurados.")
        return

    print(f"  Modo: {client.params.modo_ejecucion}")
    print(f"  Streaming: {client.params.modo_streaming}")
    print()

    # Ejecutar
    texto_prueba = "Explica qué es Python en una línea."
    print("=" * 50)
    print(f"Prompt: \"{texto_prueba}\"")
    print("=" * 50)

    try:
        resultado = client.ejecutar(texto_prueba)

        # Streaming (async iterator)
        if hasattr(resultado, '__aiter__'):
            print("RESPUESTA: ", end="")
            async for chunk in resultado:
                print(chunk, end="", flush=True)
            print()
        # Respuesta completa (string)
        elif isinstance(resultado, str):
            print(f"RESPUESTA:\n{resultado}")
        # Corutina (async)
        else:
            respuesta = await resultado
            print(f"RESPUESTA:\n{respuesta}")

    except Exception as e:
        print()
        print("=" * 50)
        print("ERROR:")
        print("=" * 50)
        print(str(e))


if __name__ == "__main__":
    asyncio.run(main())
