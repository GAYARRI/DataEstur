# router_llm_agent.py
from __future__ import annotations

import os
import sys
import json
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI

# Carga variables de entorno (.env)
load_dotenv()

# Importa tus agentes existentes
from frontur_agent import FronturLLMAgent
from egatur_agent import EgaturNL2SQLAgent, df_from_result, plot_df


class RouterLLMAgent:
    """
    Agente LLM que enruta preguntas al agente FRONTUR o EGATUR
    usando un modelo de OpenAI como clasificador.
    """

    def __init__(
        self,
        frontur_db_path: str,
        egatur_db_path: str,
        openai_api_key: str | None = None,
        frontur_config_path: str = "frontur_agent_config.json",
    ) -> None:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: falta OPENAI_API_KEY en el entorno.")
            sys.exit(1)

        self.client = OpenAI(api_key=api_key)

        # Instanciar tus agentes de negocio
        self.frontur_agent = FronturLLMAgent(
            db_path=frontur_db_path,
            openai_api_key=api_key,
            config_path=frontur_config_path,
        )

        self.egatur_agent = EgaturNL2SQLAgent(
            db_path=egatur_db_path,
            table="SURVEY_TABLE",
            default_limit=int(os.getenv("EGATUR_DEFAULT_LIMIT", "200")),
        )

        # Prompt del clasificador
        self.router_system_prompt = """
Eres un clasificador muy estricto. Tu única tarea es decidir si una pregunta
sobre turismo en España debe enviarse a FRONTUR o a EGATUR.

- FRONTUR: preguntas sobre NÚMERO DE VIAJEROS, turistas, visitantes, llegadas,
  salidas, flujos de turistas, procedencias, destinos, volúmenes, etc.
  Ejemplos:
    - "¿Cuántos turistas británicos llegaron a España en 2023?"
    - "Evolución del número de viajeros alemanes en Canarias."
    - "Distribución de turistas por comunidad autónoma."

- EGATUR: preguntas sobre GASTO TURÍSTICO o INGRESOS:
  gasto total, gasto medio, gasto diario, gasto por turista, consumo turístico,
  ingresos por turismo, etc.
  Ejemplos:
    - "¿Cuál fue el gasto total de los turistas británicos en 2023?"
    - "Gasto medio diario de los turistas alemanes en Canarias."
    - "Comparación del gasto medio por turista entre franceses y alemanes."

REGLA PRINCIPAL:
- Si el foco principal de la pregunta es el número de viajeros → responde FRONTUR.
- Si el foco principal es el gasto / ingresos → responde EGATUR.

PREGUNTAS MIXTAS:
- Si mezcla viajeros y gasto, decide según la magnitud principal:
    - "¿Qué gasto medio realizan los turistas que llegan a España?" → EGATUR.
    - "¿Cuántos viajeros gastan más de 1.000 euros?" → si dudas, EGATUR.

IMPORTANTE:
- Devuelve ÚNICAMENTE la palabra FRONTUR o EGATUR, en mayúsculas, sin espacios extra,
  sin explicaciones adicionales.
""".strip()

    # ---------- Clasificación con LLM ----------

    def _decide_target(self, question: str) -> str:
        """
        Usa el LLM para clasificar la pregunta como FRONTUR o EGATUR.
        """
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",  # o el modelo que tengas disponible
            temperature=0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": self.router_system_prompt},
                {"role": "user", "content": question},
            ],
        )

        label = (completion.choices[0].message.content or "").strip().upper()
        if "EGATUR" in label:
            return "EGATUR"
        # Por defecto, FRONTUR
        return "FRONTUR"

    # ---------- Método público ----------

    def answer(self, question: str) -> dict:
        """
        Responde a una pregunta en lenguaje natural.
        Devuelve un dict con:
            - source: FRONTUR o EGATUR
            - success: bool
            - data: lista de dicts (filas) o None
            - sql: str o None
            - message: texto explicativo / error
        """
        target = self._decide_target(question)

        if target == "FRONTUR":
            # FRONTUR ya devuelve un dict estructurado
            res = self.frontur_agent.process_query(question)
            df = res.get("data")
            if isinstance(df, pd.DataFrame):
                data = df.to_dict(orient="records")
            else:
                data = None

            return {
                "source": "FRONTUR",
                "success": bool(res.get("success", False)),
                "data": data,
                "sql": res.get("sql_generated"),
                "message": res.get("explanation") or res.get("error") or "",
            }

        else:  # EGATUR
            try:
                qr = self.egatur_agent.run(question)
                df = df_from_result(qr)

                # 🔹 Generar gráfico igual que hacía el CLI de egatur_agent
                try:
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        print("\n📈 GENERANDO GRÁFICO (EGATUR)...")
                        # No guardamos en fichero, solo mostramos ventana
                        plot_df(df, out_path=None, show=True, title=question)
                    else:
                        print("⚠️  No hay datos para generar gráfico (EGATUR)")
                except Exception as plot_err:
                    print(f"⚠️  Error generando gráfico EGATUR: {plot_err}")

                data = df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else None
                return {
                    "source": "EGATUR",
                    "success": True,
                    "data": data,
                    "sql": qr.sql,
                    "message": "",
                }
            except Exception as e:
                return {
                    "source": "EGATUR",
                    "success": False,
                    "data": None,
                    "sql": None,
                    "message": f"Error en EGATUR: {e}",
                }

            


# ---------- Utilidad para imprimir resultados en consola ----------

def _print_result(result: dict) -> None:
    print(f"\n[Origen interno: {result.get('source', '?')}]")  # si no quieres mostrarlo, borra esta línea

    if not result.get("success"):
        print(f"❌ Error: {result.get('message', 'Error desconocido')}")
        return

    sql = result.get("sql")
    if sql:
        print("\nSQL ejecutado:")
        print(sql)

    data = result.get("data")
    if data:
        df = pd.DataFrame(data)
        print(f"\nResultados ({len(df)} filas):")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        print(df.to_string(index=False))
    else:
        print("\n⚠️ No se devolvieron filas de datos.")


# ---------- CLI sencillo ----------

if __name__ == "__main__":
    FRONTUR_DB = os.getenv("DB_PATH", "frontur.db")
    EGATUR_DB = os.getenv("DB_PATH", "dataestur.db")

    router = RouterLLMAgent(
        frontur_db_path=FRONTUR_DB,
        egatur_db_path=EGATUR_DB,
    )

    print("🔎 ROUTER LLM (FRONTUR + EGATUR)")
    print("Escribe tu consulta o 'salir' para terminar.\n")

    while True:
        try:
            q = input("Pregunta> ").strip()
            if not q:
                continue
            if q.lower() in {"salir", "exit", "quit"}:
                break

            result = router.answer(q)
            _print_result(result)
            print("\n" + "-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\nFin.")
            break
