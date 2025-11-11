# egatur_agent.py (versión mejorada)
from __future__ import annotations

import os, re, json, sqlite3, sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.exit("❌ Falta OPENAI_API_KEY en el entorno (.env o variable de entorno).")
client = OpenAI(api_key=API_KEY)

# --- Configuración ---
SCHEMA_PATH = "egatur_agent_config.json"
EXAMPLES_PATH = "ejemplos_actualizados.json"

# --- Carga de esquema y ejemplos ---
def load_schema(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_few_shots(path: str) -> List[Tuple[str, str]]:
    shots = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    user = None
    for line in lines:
        if line.startswith("Usuario:"):
            user = line.split("Usuario:", 1)[1].strip().strip('"')
        elif line.startswith("SQL:"):
            sql = line.split("SQL:", 1)[1].strip()
            if user:
                shots.append((user, sql))
                user = None
    return shots

try:
    SCHEMA = load_schema(SCHEMA_PATH)
    FEW_SHOTS = load_few_shots(EXAMPLES_PATH)
except Exception as e:
    sys.exit(f"Error cargando archivos de configuración: {e}")

# --- Prompt mejorado ---

SYS_PROMPT = """Eres un asistente experto en SQL para SQLite que trabaja sobre la tabla SURVEY_TABLE de EGATUR.
    Devuelves EXCLUSIVAMENTE una consulta SQL válida, sin explicaciones ni formato de código.

    Restricciones generales:
    - Solo puedes usar sentencias SELECT.
    Están TERMINANTEMENTE prohibidos: UPDATE, DELETE, DROP, ALTER, INSERT, CREATE,
    REPLACE, ATTACH, DETACH, VACUUM y cualquier otra orden que modifique la base de datos.
    - Utiliza los nombres de columnas exactamente como aparecen en el esquema.
    - Cuando exista un par *_min / *_max, por defecto usa la versión *_min salvo que el
    usuario pida explícitamente máximos.
    - Evita SELECT * en consultas con agregaciones. Nombra las columnas de forma explícita.
    - Añade alias explícitos (AS ...) a TODAS las expresiones calculadas:
    * AVG(CAMPO) AS CAMPO_avg
    * SUM(CAMPO) AS CAMPO_total
    * COUNT(*) AS count
    * CASE ... END AS nombre_descriptivo
    - Para porcentajes de indicadores o ratios, puedes usar expresiones como:
    * AVG(campo) * 100 AS campo_pct
    * 100.0 * SUM(num) / NULLIF(SUM(den), 0) AS ratio_pct
    - No inventes nombres de columnas.

    Convención para facilitar la generación de gráficos:
    - Siempre que la consulta sea adecuada para graficar, debe devolver:
        - EXACTAMENTE UNA dimensión principal (por ejemplo YM_min, CCAA_DESTINO_min, PROVINCIA_DESTINO_min,
        CCAA_RESIDENCIA_min, TIPO_ALOJAMIENTO_min, MOTIVO_VIAJE_min, etc.).
        - EXACTAMENTE UNA métrica principal (gasto total, gasto medio, conteo, porcentaje, etc.).
    - La dimensión principal debe tener SIEMPRE el alias:
        dim_value
    - La métrica principal debe tener SIEMPRE el alias:
        metric_value

    Ejemplos de alias esperados:
    - SELECT SUBSTR(YM_min, 1, 4) AS dim_value,
            SUM(NUM_GASTO_TOTAL_min) AS metric_value
    FROM SURVEY_TABLE
    ...
    - SELECT CCAA_DESTINO_min AS dim_value,
            AVG(NUM_GASTO_TOTAL_min) AS metric_value
    FROM SURVEY_TABLE
    ...
    - SELECT CASE
            WHEN NACIONALIDAD_min = 'España' THEN 'Solo española'
            WHEN REGION_ORIGEN_min = 'Europa' THEN 'Española y europea'
            ELSE 'Otros orígenes'
            END AS dim_value,
            SUM(NUM_GASTO_TOTAL_min) AS metric_value
    FROM SURVEY_TABLE
    ...

    Reglas semánticas específicas de EGATUR (según el esquema):

    1) Dimensión temporal:
    - La variable temporal básica es YM_min (año-mes).
    - Si el usuario pregunta por años, puedes usar SUBSTR(YM_min, 1, 4) para obtener el año:
        SUBSTR(YM_min, 1, 4) AS dim_value
    - Si el usuario pregunta por meses dentro de un año concreto, puedes agrupar directamente por YM_min.

    2) Dimensiones de DESTINO (prioritarias cuando se habla de “gasto por región / destino”):
    - Comunidad autónoma de destino: CCAA_DESTINO_min
    - Provincia de destino: PROVINCIA_DESTINO_min
    - País de destino: PAIS_DESTINO_min
    - Región de destino: REGION_DESTINO_min (si existe en el esquema)

    Ejemplos:
    - "Distribución del gasto por comunidad autónoma de destino" →
        CCAA_DESTINO_min AS dim_value
    - "Distribución del gasto por provincia de destino" →
        PROVINCIA_DESTINO_min AS dim_value
    - "España vs resto de Europa como destino" →
        usar un CASE sobre PAIS_DESTINO_min y/o REGION_DESTINO_min, por ejemplo:
        CASE
        WHEN PAIS_DESTINO_min = 'España' THEN 'España'
        WHEN REGION_DESTINO_min = 'Europa' THEN 'Resto de Europa'
        ELSE 'Otros destinos'
        END AS dim_value

    3) Dimensiones de ORIGEN / RESIDENCIA:
    - Comunidad autónoma de residencia: CCAA_RESIDENCIA_min
    - Nacionalidad y región de origen: NACIONALIDAD_min, REGION_ORIGEN_min

    Cuando el usuario pida agregación de nacionalidad en categorías (por ejemplo “solo española / española y europea / otros”),
    usa la siguiente regla:

        CASE
        WHEN NACIONALIDAD_min = 'España' THEN 'Solo española'
        WHEN REGION_ORIGEN_min = 'Europa' THEN 'Española y europea'
        ELSE 'Otros orígenes'
        END AS dim_value

    Ejemplos:
    - "Gasto total según nacionalidad agrupada en solo española, española y europea y otros orígenes" →
        usar el CASE anterior como dim_value y SUM(NUM_GASTO_TOTAL_min) como metric_value.
    - "Distribución del gasto por comunidad de residencia" →
        CCAA_RESIDENCIA_min AS dim_value

    4) Variables de gasto (métricas principales):
    - Gastos parciales típicos:
        NUM_GASTO_ALOJAMIENTO_min
        NUM_GASTO_TRANSPORTE_min
        NUM_GASTO_COMIDA_min
        NUM_GASTO_ACTIVIDADES_min
        NUM_GASTO_PAQUETE_min
        NUM_GASTO_RESTO_min
    - Gasto total del viaje:
        NUM_GASTO_TOTAL_min     (esta suele ser la métrica principal de gasto)

    Reglas:
    - “gasto total”, “importe total”, “gasto acumulado” → usa SUM(NUM_GASTO_TOTAL_min) AS metric_value.
    - “gasto medio”, “gasto medio por viaje”, “gasto medio por persona” → usa AVG(NUM_GASTO_TOTAL_min) AS metric_value.
    - “gasto en alojamiento” → usa NUM_GASTO_ALOJAMIENTO_min.
    - “gasto en transporte” → usa NUM_GASTO_TRANSPORTE_min.
    - “gasto en comida” → usa NUM_GASTO_COMIDA_min.
    - “gasto en actividades” → usa NUM_GASTO_ACTIVIDADES_min.

    5) Recuento de viajes / turistas (cuando se habla de “número de viajes” en EGATUR):
    - Si el usuario habla de “número de viajes”, “cuántos viajes”, “número de registros” dentro del contexto de EGATUR,
    puedes usar:
        COUNT(*) AS metric_value
    o, si existe una variable específica de conteo, SUM(esa_variable).

    6) Agrupaciones:
    - Si el usuario pide "por comunidad autónoma", "por provincia", "por país", "por año", "por motivo del viaje", etc.,
    debes usar GROUP BY sobre esas dimensiones.
    - Si hay varias dimensiones en la pregunta (por ejemplo año y CCAA), agrupa por todas, pero elige UNA como dim_value
    (la más importante para el gráfico) y deja las demás como columnas adicionales no aliasadas con dim_value.

    7) Filtros temporales:
    - "en 2023" → SUBSTR(YM_min, 1, 4) = '2023'
    - "entre 2020 y 2023" → SUBSTR(YM_min, 1, 4) BETWEEN '2020' AND '2023'

    Salida:
    - Tu salida DEBE SER únicamente la sentencia SQL final, sin comentarios, sin texto extra,
    sin bloques ```sql``` ni explicaciones adicionales.
    """
   


# --- Hints heurísticos ---
def derive_aggregation_hints(nl: str) -> str:
    q = nl.lower()
    hints = []

    if any(k in q for k in ["por ", "desglose", "distribución", "segmento"]):
        hints.append("- Usa GROUP BY en las dimensiones mencionadas.")
    if any(k in q for k in ["total", "acum", "importe total", "gasto total", "ingresos"]):
        hints.append("- Usa SUM sobre las variables NUM_GASTO_* pertinentes.")
    if any(k in q for k in ["medio", "promedio", "media"]):
        hints.append("- Usa AVG sobre las variables NUM_GASTO_* pertinentes.")
    if "edad" in q:
        hints.append("- Para edad, usa EDAD_min.")
    if "país" in q or "pais" in q:
        hints.append("- Para país, agrupa por la columna de país adecuada.")
    if "comunidad" in q or "ccaa" in q:
        hints.append("- Para comunidad autónoma, agrupa por la columna CCAA correspondiente.")

    if not hints:
        hints.append("- Devuelve una SELECT sencilla sin agregaciones.")
    return "\n".join(hints)

# --- Prompt de usuario ---
def build_user_prompt(nl: str, schema: List[dict]) -> str:
    schema_text = "\n".join([f"{c.get('name')} ({c.get('type','')})" for c in schema])
    hints = derive_aggregation_hints(nl)
    return (
        f"Esquema de la tabla SURVEY_TABLE (SQLite):\n{schema_text}\n\n"
        f"Petición del usuario: {nl}\n"
        f"Pistas:\n{hints}\n\n"
        f"Devuelve SOLO la sentencia SQL final, sin explicaciones."
    )

# --- LLM call ---
def llm_complete(system: str, user: str, examples: List[Tuple[str, str]]) -> str:
    messages = [{"role": "system", "content": system}]
    for u, s in examples:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": s})
    messages.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

# --- Seguridad SQL ---
FORBIDDEN = re.compile(r"\b(UPDATE|DELETE|DROP|ALTER|INSERT|CREATE|REPLACE|ATTACH|DETACH|VACUUM)\b", re.I)
def sanitize_sql(text: str) -> str:
    return re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.I).strip()

def is_safe_select(sql: str) -> Tuple[bool, Optional[str]]:
    s = sql.strip().upper()
    if not s.startswith("SELECT"):
        return False, "No empieza por SELECT."
    if FORBIDDEN.search(s):
        return False, "Contiene comandos no permitidos."
    return True, None

# --- Agente principal ---
@dataclass
class EgaturNL2SQLAgent:
    db_path: str
    table: str
    default_limit: int = 100

    def generate_sql(self, nl: str) -> str:
        prompt = build_user_prompt(nl, SCHEMA)
        raw_sql = llm_complete(SYS_PROMPT, prompt, FEW_SHOTS)
        sql = sanitize_sql(raw_sql)
        ok, msg = is_safe_select(sql)
        if not ok:
            raise ValueError(f"SQL inseguro: {msg}")
        return sql

    def run(self, nl: str) -> "QueryResult":
        sql = self.generate_sql(nl)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]
            return QueryResult(sql=sql, rows=rows, columns=cols)
        finally:
            conn.close()

@dataclass
class QueryResult:
    sql: str
    rows: list
    columns: list

# --- Visualización ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.figsize": (12, 7),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.size": 12,
})
def _is_time_like(series: pd.Series) -> bool:
    """
    Intenta decidir si dim_value representa tiempo (año, año-mes, etc.).
    - Numérico con 4 dígitos (tipo 2020, 2021, ...).
    - Texto con patrón 'YYYY', 'YYYY-MM', 'YYYYMM', etc.
    """
    if series.empty:
        return False

    # Si ya es numérico
    if pd.api.types.is_numeric_dtype(series):
        # ¿Parece un año de 4 dígitos?
        sample = series.dropna().astype(str).head(5)
        if all(len(s) == 4 and s.isdigit() for s in sample):
            return True
        return False

    # Si es texto: probar patrones típicos
    sample = series.dropna().astype(str).head(5)
    for s in sample:
        s = s.strip()
        # YYYY
        if len(s) == 4 and s.isdigit():
            return True
        # YYYYMM
        if len(s) == 6 and s.isdigit():
            return True
        # YYYY-MM, YYYY/MM
        if len(s) == 7 and (s[4] in "-/") and s[:4].isdigit():
            return True
    return False

TIME_COL_PATTERNS = ["ANIO", "AÑO", "ANO", "YEAR", "PERIODO", "MES", "MONTH"]
DIM_COL_HINTS = ["PAIS", "PAÍS", "CCAA", "COMUNIDAD", "MOTIVO", "TIPO", "REGION", "SEGMENTO"]

def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if any(pat in col.upper() for pat in TIME_COL_PATTERNS):
            return col
    return None

def detect_dim_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if any(pat in col.upper() for pat in DIM_COL_HINTS):
            return col
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 30:
            return col
    return None

def pick_plot(df: pd.DataFrame) -> Tuple[str, str]:
    if df.empty:
        return ("none", "Sin datos para graficar")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    time_col = detect_time_column(df)
    dim_col = detect_dim_column(df)
    if time_col and num_cols:
        return ("line_time", f"Línea temporal {time_col} vs {num_cols[0]}")
    if dim_col and num_cols:
        return ("bar_dim", f"Barras por {dim_col} vs {num_cols[0]}")
    if len(num_cols) >= 2:
        return ("line", f"Líneas para {num_cols}")
    if len(num_cols) == 1:
        return ("hist", f"Histograma de {num_cols[0]}")
    return ("none", "No hay columnas numéricas o categóricas adecuadas")

def plot_df(df: pd.DataFrame, out_path: Optional[str] = None, show: bool = True, title: Optional[str] = None):
    """
    Genera un gráfico a partir de un DataFrame de resultados EGATUR.

    Prioridad:
    1) Si existen las columnas 'metric_value' y/o 'dim_value', se utilizan directamente.
       - Si dim_value es temporal (año / año-mes) → línea temporal.
       - Si dim_value es categórica → barras ordenadas.
    2) Si no existen, se intenta un gráfico sencillo de respaldo.
    """
    if df is None or df.empty:
        print("⚠️  No hay datos para graficar.")
        return

    # 1) Camino principal: dim_value / metric_value
    has_metric = "metric_value" in df.columns
    has_dim = "dim_value" in df.columns

    if has_metric:
        y_col = "metric_value"

        # Determinar eje X
        if has_dim:
            x_series = df["dim_value"]
            x_label = "dim_value"
        else:
            x_series = pd.Series(df.index, index=df.index)
            x_label = "Índice"

        fig, ax = plt.subplots(figsize=(12, 7))
        if title:
            plt.title(title, fontsize=16, fontweight="bold", pad=20)

        # ¿Es temporal o categórico?
        if _is_time_like(x_series):
            # Ordenar por tiempo
            df_plot = df.copy()
            df_plot["_x_"] = x_series
            # Intentar convertir a algo ordenable
            try:
                df_plot["_x_parsed"] = pd.to_datetime(df_plot["_x_"], errors="coerce")
                if df_plot["_x_parsed"].notna().any():
                    df_plot = df_plot.sort_values("_x_parsed")
                    x_vals = df_plot["_x_parsed"]
                else:
                    df_plot = df_plot.sort_values("_x_")
                    x_vals = df_plot["_x_"]
            except Exception:
                df_plot = df.copy()
                df_plot["_x_"] = x_series
                df_plot = df_plot.sort_values("_x_")
                x_vals = df_plot["_x_"]

            ax.plot(x_vals, df_plot[y_col], marker="o", linewidth=2)
            ax.set_xlabel(x_label, fontsize=13)
            ax.set_ylabel(y_col, fontsize=13)
            ax.grid(True, alpha=0.3)

        else:
            # Tratar dim_value como categoría → barras
            df_plot = df.copy()
            if len(df_plot) > 30:
                df_plot = df_plot.head(30)
                print(f"📉 Mostrando solo las primeras 30 filas de {len(df)}")

            x_vals = x_series.loc[df_plot.index]
            indices = range(len(df_plot))
            ax.bar(indices, df_plot[y_col])
            ax.set_xticks(indices)
            ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha="right")
            ax.set_xlabel(x_label, fontsize=13)
            ax.set_ylabel(y_col, fontsize=13)

        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return  # No continuamos con la heurística

    # 2) Camino de respaldo: no hay metric_value → algo muy básico
    print("ℹ️  No se encontraron columnas 'metric_value'; usando heurística simple de respaldo.")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        print("⚠️  No hay columnas numéricas para graficar.")
        return

    y_col = num_cols[0]
    fig, ax = plt.subplots(figsize=(12, 7))
    if title:
        plt.title(title, fontsize=16, fontweight="bold", pad=20)

    if len(df) <= 30:
        # Barras por índice
        indices = range(len(df))
        ax.bar(indices, df[y_col])
        ax.set_xticks(indices)
        ax.set_xticklabels([str(i) for i in df.index], rotation=45, ha="right")
        ax.set_xlabel("Índice", fontsize=13)
        ax.set_ylabel(y_col, fontsize=13)
    else:
        # Línea sobre índice
        ax.plot(df.index, df[y_col], marker="o", linewidth=2)
        ax.set_xlabel("Índice", fontsize=13)
        ax.set_ylabel(y_col, fontsize=13)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()




def df_from_result(qr: QueryResult) -> pd.DataFrame:
    return pd.DataFrame(qr.rows, columns=qr.columns)

# --- CLI ---
if __name__ == "__main__":
    db_path = os.getenv("EGATUR_DB_PATH", "dataestur.db")
    agent = EgaturNL2SQLAgent(db_path=db_path, table="SURVEY_TABLE")
    while True:
        q = input("\nPregunta > ").strip()
        if not q or q.lower() in {"salir", "exit"}:
            break
        qr = agent.run(q)
        df = df_from_result(qr)
        print("\nSQL generado:\n", qr.sql)
        print(df.head())
        plot_df(df, show=True, title=q)
