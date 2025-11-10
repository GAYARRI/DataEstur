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
    - Para porcentajes de indicadores 0/1, usa AVG(campo) * 100 AS nombre_pct.
    - No inventes nombres de columnas.

    Convención para facilitar la generación de gráficos:
    - Siempre que la consulta sea adecuada para graficar, debe devolver:
        - Exactly UNA dimensión principal (por ejemplo YM_min, CCAA_DESTINO_min, NACIONALIDAD_min, EDAD_min, etc.)
        - Exactly UNA métrica principal (un gasto total, un gasto medio, un conteo, etc.)
    - La dimensión principal debe tener SIEMPRE el alias:
        dim_value
    - La métrica principal debe tener SIEMPRE el alias:
        metric_value

    Ejemplos de alias esperados:
    - SELECT YM_min AS dim_value,
            SUM(NUM_GASTO_TOTAL_min) AS metric_value
    FROM SURVEY_TABLE
    ...
    - SELECT CCAA_DESTINO_min AS dim_value,
            AVG(NUM_GASTO_TOTAL_min) AS metric_value
    FROM SURVEY_TABLE
    ...
    - SELECT SUBSTR(YM_min, 1, 4) AS dim_value,
            SUM(NUM_GASTO_TOTAL_min) AS metric_value
    FROM SURVEY_TABLE
    ...

    Reglas semánticas específicas de EGATUR (según el esquema):

    - Tiempo:
    * La variable temporal es YM_min (año-mes).
    * Si el usuario pregunta por años, puedes usar SUBSTR(YM_min, 1, 4) para obtener el año.
        Ejemplo: SUBSTR(YM_min, 1, 4) AS dim_value.

    - Dimensiones geográficas y de segmento:
    * Comunidades autónomas de destino: CCAA_DESTINO_min.
    * País de destino: PAIS_DESTINO_min.
    * Nacionalidad de los turistas: NACIONALIDAD_min.
    * CCAA de residencia: CCAA_RESIDENCIA_min.
    * Edad: EDAD_min.
    * Motivo del viaje: MOTIVO_VIAJE_min.
    * Tipo de viaje: TIPO_VIAJE_min.
    * Tipo de alojamiento: TIPO_ALOJAMIENTO_min.

    - Variables de gasto:
    * Gastos parciales:
        - NUM_GASTO_ALOJAMIENTO_min
        - NUM_GASTO_TRANSPORTE_min
        - NUM_GASTO_COMIDA_min
        - NUM_GASTO_ACTIVIDADES_min
        - NUM_GASTO_BIENES_DUR_min
        - NUM_GASTO_RESTO_min
        - NUM_GASTO_PAQUETE_min
    * Gasto total del viaje:
        - NUM_GASTO_TOTAL_min  (esta suele ser la métrica principal de gasto)

    - Reglas para "gasto":
    * "gasto total", "importe total", "gasto acumulado" → usa SUM(NUM_GASTO_TOTAL_min) AS metric_value.
    * "gasto medio" / "gasto medio por persona" / "gasto medio por viaje" → usa AVG(NUM_GASTO_TOTAL_min) AS metric_value.
    * "gasto en alojamiento" → usa NUM_GASTO_ALOJAMIENTO_min.
    * "gasto en transporte" → usa NUM_GASTO_TRANSPORTE_min.
    * "gasto en comida" → usa NUM_GASTO_COMIDA_min.
    * "gasto en actividades" → usa NUM_GASTO_ACTIVIDADES_min.

    - Reglas para recuentos:
    * Si el usuario habla de "número de viajes", "número de turistas", "cuántos", etc.,
        y la consulta pertenece a EGATUR, puedes usar:
        COUNT(*) AS metric_value
        o, si existe una variable de conteo específica (por ejemplo n_registros), puedes usar
        SUM(n_registros) AS metric_value
        según corresponda al significado.

    - Agrupaciones:
    * Si el usuario pide "por comunidad autónoma", "por país", "por edad", "por año", etc.,
        debes usar GROUP BY sobre esas dimensiones.
    * Si hay varias dimensiones en la pregunta (por ejemplo año y CCAA), agrupa por todas,
        pero elige UNA de ellas como dim_value (la más importante para el gráfico) y deja las demás
        como columnas adicionales no aliasadas con dim_value.

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

def plot_df(df: pd.DataFrame, out_path: Optional[str] = None, show=True, title: Optional[str] = None):

    """
    Genera un gráfico a partir de un DataFrame.
    Prioriza la convención dim_value / metric_value.
    Si no están presentes, usa la heurística pick_plot().
    """
    if df is None or df.empty:
        print("⚠️  No hay datos para graficar.")
        return

    # 1) Camino preferente: el SQL ha seguido la convención dim_value / metric_value
    if "metric_value" in df.columns:
        y_col = "metric_value"

        # Determinar dimensión X
        if "dim_value" in df.columns:
            x_vals = df["dim_value"]
            x_label = "dim_value"
        else:
            # Si no hay dim_value, usamos índice
            x_vals = df.index
            x_label = "index"

        fig, ax = plt.subplots(figsize=(12, 7))
        if title:
            plt.title(title, fontsize=16, fontweight="bold", pad=20)

        # Elegimos tipo de gráfico simple: línea si x es numérico, barras si no
        if pd.api.types.is_numeric_dtype(x_vals):
            # Ordenar por X si tiene sentido
            df_plot = df.copy()
            df_plot["_x_"] = x_vals
            df_plot.sort_values("_x_", inplace=True)
            ax.plot(df_plot["_x_"], df_plot[y_col], marker="o", linewidth=2)
            ax.set_xlabel(x_label, fontsize=13)
            ax.set_ylabel(y_col, fontsize=13)
            ax.grid(True, alpha=0.3)
        else:
            df_plot = df.copy()
            if len(df_plot) > 30:
                df_plot = df_plot.head(30)
                print(f"   📉 Mostrando solo las primeras 30 filas de {len(df)}")
            indices = range(len(df_plot))
            ax.bar(indices, df_plot[y_col])
            ax.set_xticks(indices)
            ax.set_xticklabels([str(v) for v in df_plot["dim_value"]] if "dim_value" in df_plot.columns else [str(i) for i in indices],
                               rotation=45, ha="right")
            ax.set_xlabel(x_label, fontsize=13)
            ax.set_ylabel(y_col, fontsize=13)

        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return  # Muy importante: no seguimos con la heurística

    # 2) Camino de respaldo: heurística clásica basada en pick_plot
    kind, desc = pick_plot(df)
    print(f"📊 {desc}")
    if kind == "none":
        print("No se puede graficar con los datos disponibles.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dim_col = detect_dim_column(df)
    time_col = detect_time_column(df)

    if kind == "line_time" and time_col and num_cols:
        y = num_cols[0]
        df_plot = df.sort_values(time_col).copy()
        ax.plot(df_plot[time_col], df_plot[y], marker='o', linewidth=2)
        ax.set_xlabel(time_col, fontsize=13)
        ax.set_ylabel(y, fontsize=13)
        ax.grid(True, alpha=0.3)

    elif kind == "bar_dim" and dim_col and num_cols:
        y = num_cols[0]
        df_plot = df.copy()
        if len(df_plot) > 30:
            df_plot = df_plot.head(30)
            print(f"   📉 Mostrando solo primeras 30 filas de {len(df)} totales")
        sns.barplot(data=df_plot, x=dim_col, y=y, ax=ax)
        ax.set_xlabel(dim_col, fontsize=13)
        ax.set_ylabel(y, fontsize=13)
        plt.xticks(rotation=45, ha='right')

    elif kind == "line":
        df_plot = df.copy()
        for col in num_cols:
            ax.plot(df_plot.index, df_plot[col], marker='o', linewidth=2, label=col)
        ax.set_xlabel("Índice", fontsize=13)
        ax.set_ylabel("Valor", fontsize=13)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    elif kind == "hist" and num_cols:
        col = num_cols[0]
        ax.hist(df[col].dropna(), bins=15, alpha=0.7, edgecolor='black')
        ax.set_xlabel(col, fontsize=13)
        ax.set_ylabel("Frecuencia", fontsize=13)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
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
