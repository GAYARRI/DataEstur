# egatur_agent.py (versión mejorada con doble eje para 2 métricas)
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
        - EXACTAMENTE UNA dimensión principal (por ejemplo YM_min, CCAA_DESTINO_min, PROVINCIA_DESTINO_min, PAIS_DESTINO
        CCAA_RESIDENCIA_min, TIPO_ALOJAMIENTO_min, MOTIVO_VIAJE_min, etc.).
        - EXACTAMENTE UNA métrica principal (gasto total, gasto medio, conteo, porcentaje, etc.).

    Ejemplos de alias esperados:
    - SELECT SUBSTR(YM_min, 1, 4) AS periodo,
            SUM(NUM_GASTO_TOTAL_min) AS gasto_total
    FROM SURVEY_TABLE
    ...
    - SELECT CCAA_DESTINO_min AS destino,
            AVG(NUM_GASTO_TOTAL_min) AS gasto_promedio
    FROM SURVEY_TABLE

    - SELECT PAIS_DESTINO_min AS destino,
            AVG(NUM_GASTO_TOTAL_min) AS gasto_promedio
    FROM SURVEY_TABLE
     
    - SELECT CASE
            WHEN NACIONALIDAD_min = 'Sólo española' THEN 'Nacional'
            WHEN NACIONALIDAD_min = 'Española y extranjera' THEN 'Doble_Nac'
            WHEN NACIONALIDAD_min = 'Sólo extranjera' THEN 'Extranjero'
            END AS Nacionalidad,
            SUM(NUM_GASTO_TOTAL_min) AS gasto_total
    FROM SURVEY_TABLE
    ...

    Reglas semánticas específicas de EGATUR (según el esquema):

    1) Dimensión temporal:
    - La variable temporal básica es YM_min (año-mes).
    - Si el usuario pregunta por años, puedes usar SUBSTR(YM_min, 1, 4) para obtener el año:
        SUBSTR(YM_min, 1, 4) AS periodo
    - Si el usuario pregunta por meses dentro de un año concreto, puedes agrupar directamente por YM_min.

    2) Dimensiones de DESTINO (prioritarias cuando se habla de “gasto por región / destino”):
    - Comunidad autónoma de destino: CCAA_DESTINO_min
    - Provincia de destino: PROVINCIA_DESTINO_min
    - Pais de destino : PAIS_DESTINO_min
    

    Ejemplos:
    - "Distribución del gasto por comunidad autónoma de destino" →
        CCAA_DESTINO_min AS ccaa_destino
    - "Distribución del gasto por provincia de destino" →
        PROVINCIA_DESTINO_min AS prov_destino
    - "Evolución de la Distribución del gasto total por nacionalidad de los turistas con destino canarias desde 2020"   
        SELECT SUBSTR(YM_min, 1, 4) AS periodo,
            SUM(CASE WHEN NACIONALIDAD_min='Sólo española' THEN NUM_GASTO_TOTAL_min END) AS solo_espanola,
            SUM(CASE WHEN NACIONALIDAD_min='Española y extranjera' THEN NUM_GASTO_TOTAL_min END) AS esp_y_ext,
            SUM(CASE WHEN NACIONALIDAD_min LIKE 'Sólo extranjera' THEN NUM_GASTO_TOTAL_min END) AS solo_extranjera
        FROM SURVEY_TABLE
        WHERE CCAA_DESTINO_min = 'Canarias'
        AND SUBSTR(YM_min, 1, 4) >= '2020'
        GROUP BY periodo
        ORDER BY periodo;

    3) Dimensiones de ORIGEN / RESIDENCIA / :
    - Comunidad autónoma de residencia: CCAA_RESIDENCIA_min
    - Nacionalidad y región de origen: NACIONALIDAD_min

    Ejemplos:
    - "Gasto total según nacionalidad agrupada en solo española, española y europea y otros orígenes" →
        usar NACIONALIDAD_min como nacionalidad y SUM(NUM_GASTO_TOTAL_min) como gasto_total.
    - "Distribución del gasto por comunidad de residencia" →
        CCAA_RESIDENCIA_min AS region_residencia

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
    (la más importante para el gráfico) y deja las demás como columnas adicionales agrupadas según dim_value.

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
        temperature=0,
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

def _is_time_like(series: pd.Series) -> bool:
    """Heurística sencilla para detectar si una columna parece temporal."""
    if series is None or series.empty:
        return False
    sample = series.dropna().astype(str).head(10)

    # YYYY
    if all(len(s) == 4 and s.isdigit() for s in sample):
        return True
    # YYYYMM
    if all(len(s) == 6 and s.isdigit() for s in sample):
        return True
    # YYYY-MM o YYYY/MM
    if all((len(s) == 7 and s[:4].isdigit() and s[4] in "-/") for s in sample):
        return True

    return False


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Intenta detectar una columna temporal razonable en el DataFrame."""
    candidates = [c for c in df.columns if any(
        token in c.upper() for token in ["YM", "FECHA", "ANIO", "AÑO", "ANO", "PERIODO", "PERIOD", "MES"]
    )]
    if candidates:
        return candidates[0]

    # Si no por nombre, miramos contenido
    for c in df.columns:
        if _is_time_like(df[c]):
            return c
    return None



def plot_df(
    df: pd.DataFrame,
    out_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None,
):
    """
    Genera un gráfico a partir de un DataFrame de resultados (EGATUR/FRONTUR).

    Casos soportados (en orden de prioridad):

    1) Caso multi-serie clásico:
       - Columnas: dim_value, series_value, metric_value
       -> Una línea por serie_value en el MISMO eje Y.

    2) Caso con 1, 2 o 3 métricas numéricas y un eje X temporal/categórico:
       - Se detectan columnas numéricas (métricas) y una columna temporal:
         * 1 métrica: 1 eje Y.
         * 2 métricas: 2 ejes Y (una por eje).
         * 3 métricas: dos en el eje izquierdo, una en el derecho.

    3) Fallback: heurística simple si no hay columnas estándar.
    """
    if df is None or df.empty:
        print("⚠️  No hay datos para graficar.")
        return

    df_plot = df.copy()

    # ─────────────────────────────────────────────
    # Caso 1: multi-serie dim_value + series_value + metric_value
    # ─────────────────────────────────────────────
    if {"dim_value", "series_value", "metric_value"}.issubset(df_plot.columns):
        # Orden temporal si se puede
        x_col = "dim_value"
        x_series = df_plot[x_col]
        try:
            if _is_time_like(x_series):
                df_plot["_x_parsed"] = pd.to_datetime(x_series.astype(str), errors="coerce")
                if df_plot["_x_parsed"].notna().any():
                    df_plot = df_plot.sort_values(["_x_parsed", "series_value"])
                    x_vals_all = df_plot["_x_parsed"]
                else:
                    df_plot = df_plot.sort_values([x_col, "series_value"])
                    x_vals_all = df_plot[x_col]
            else:
                df_plot = df_plot.sort_values([x_col, "series_value"])
                x_vals_all = df_plot[x_col]
        except Exception:
            df_plot = df_plot.sort_values([x_col, "series_value"])
            x_vals_all = df_plot[x_col]

        fig, ax = plt.subplots(figsize=(12, 7))
        if title:
            plt.title(title, fontsize=16, fontweight="bold", pad=20)

        for series, sub in df_plot.groupby("series_value", sort=False):
            x_vals = x_vals_all.loc[sub.index]
            ax.plot(x_vals, sub["metric_value"], marker="o", linewidth=2, label=str(series))

        ax.set_xlabel(x_col, fontsize=13)
        ax.set_ylabel("metric_value", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return

    # ─────────────────────────────────────────────
    # Caso 2: generalización 1–3 series numéricas con hasta 2 ejes Y
    # ─────────────────────────────────────────────

    # Intentar identificar columna de tiempo/dimensión X
    x_col = None
    if "dim_value" in df_plot.columns:
        x_col = "dim_value"
    else:
        x_col = detect_time_column(df_plot)

    # Columnas numéricas candidatas a métricas
    num_cols = [c for c in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[c])]
    # Excluir la columna temporal si también es numérica (p.ej. año como INT)
    if x_col in num_cols:
        num_cols = [c for c in num_cols if c != x_col]

    # Si hay metric_value(s) explícitos, dales prioridad
    priority_metrics = [c for c in num_cols if c.lower() in ["metric_value", "metric_value_total", "metric_value_avg"]]
    if priority_metrics:
        # mover las prioritarias al principio
        remaining = [c for c in num_cols if c not in priority_metrics]
        num_cols = priority_metrics + remaining

    if x_col is None and "dim_value" not in df_plot.columns:
        # Si no hay dimensión clara, creamos un eje X por índice
        df_plot["__index__"] = range(len(df_plot))
        x_col = "__index__"

    # Ordenar por tiempo si el eje X lo parece
    x_series = df_plot[x_col]
    if _is_time_like(x_series):
        try:
            df_plot["_x_parsed"] = pd.to_datetime(x_series.astype(str), errors="coerce")
            if df_plot["_x_parsed"].notna().any():
                df_plot = df_plot.sort_values("_x_parsed")
                x_vals = df_plot["_x_parsed"]
            else:
                df_plot = df_plot.sort_values(x_col)
                x_vals = df_plot[x_col]
        except Exception:
            df_plot = df_plot.sort_values(x_col)
            x_vals = df_plot[x_col]
    else:
        df_plot = df_plot.sort_values(x_col)
        x_vals = df_plot[x_col]

    # Si no hay métricas numéricas → fallback
    if not num_cols:
        print("⚠️  No hay columnas numéricas para graficar.")
        return

    # Limitar a máximo 3 métricas (como pedías)
    metrics = num_cols[:20]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    if title:
        plt.title(title, fontsize=16, fontweight="bold", pad=20)

    # Convertimos X a algo ploteable
    x_vals_series = x_vals

    # Caso 2.a: solo 1 métrica → un eje
    if len(metrics) == 1:
        y1 = metrics[0]
        ax1.plot(x_vals_series, df_plot[y1], marker="o", linewidth=2)
        ax1.set_xlabel(x_col, fontsize=13)
        ax1.set_ylabel(y1, fontsize=13)
        ax1.grid(True, alpha=0.3)

    # Caso 2.b: 2 métricas → 2 ejes (una en cada eje)
    elif len(metrics) == 2:
        y1, y2 = metrics[0], metrics[1]

        # Eje izquierdo
        ax1.plot(x_vals_series, df_plot[y1], marker="o", linewidth=2, label=y1)
        ax1.set_xlabel(x_col, fontsize=13)
        ax1.set_ylabel(y1, fontsize=13)
        ax1.grid(True, alpha=0.3)

        # Eje derecho
        ax2 = ax1.twinx()
        ax2.plot(x_vals_series, df_plot[y2], marker="o", linewidth=2, label=y2)
        ax2.set_ylabel(y2, fontsize=13)

        # Leyenda combinada
        l1, t1 = ax1.get_legend_handles_labels()
        l2, t2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, t1 + t2, loc="best")

    # Caso 2.c: 3 métricas → 2 en eje izquierdo, 1 en derecho
    else:  # len(metrics) == 3
        y1, y2, y3 = metrics[0], metrics[1], metrics[2]

        # Eje izquierdo con dos series
        ax1.plot(x_vals_series, df_plot[y1], marker="o", linewidth=2, label=y1)
        ax1.plot(x_vals_series, df_plot[y2], marker="s", linewidth=2, label=y2)
        ax1.set_xlabel(x_col, fontsize=13)
        ax1.set_ylabel(f"{y1} / {y2}", fontsize=13)
        ax1.grid(True, alpha=0.3)

        # Eje derecho con la tercera serie
        ax2 = ax1.twinx()
        ax2.plot(x_vals_series, df_plot[y3], marker="^", linewidth=2, label=y3)
        ax2.set_ylabel(y3, fontsize=13)

        # Leyenda combinada
        l1, t1 = ax1.get_legend_handles_labels()
        l2, t2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, t1 + t2, loc="best")

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