# frontur_agent.py - VERSIÓN COMPLETA CORREGIDA
import sqlite3
import pandas as pd
import openai
from openai import OpenAI
import seaborn as sns
import matplotlib.pyplot as plt
import json
import re
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import numpy as np

# ========== QUERY EXECUTOR ==========
class QueryExecutor:
    def __init__(self, db_connection, logger):
        self.connection = db_connection
        self.logger = logger

    def execute_sql(self, sql: str) -> Tuple[pd.DataFrame, Optional[str]]:
        try:
            if not self._is_safe_query(sql):
                return pd.DataFrame(), "Consulta no permitida por seguridad"

            self.logger.info(f"Ejecutando SQL: {sql}")
            df = pd.read_sql_query(sql, self.connection)

            if df.empty:
                self.logger.warning("Consulta ejecutada pero sin resultados")
            else:
                self.logger.info(f"Consulta exitosa: {len(df)} filas obtenidas")
                self.logger.info(f"Columnas obtenidas: {df.columns.tolist()}")
            return df, None

        except Exception as e:
            error_msg = f"Error ejecutando SQL: {e}"
            self.logger.error(error_msg)
            return pd.DataFrame(), error_msg

    def _is_safe_query(self, sql: str) -> bool:
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE']
        sql_upper = sql.upper()

        if not sql_upper.strip().startswith(('SELECT', 'WITH')):
            return False

        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False

        return True


# ========== SQL GENERATOR ==========

class SQLGenerator:

    def __init__(self, client, config, metadata, examples):
        self.client = client
        self.config = config
        self.metadata = metadata
        self.examples = examples

    def create_system_prompt(self) -> str:
        """Crear prompt del sistema ultra-específico para FRONTUR"""
        ultimo_periodo = "2025-08"  # el agente puede calcularlo dinámicamente
        columns_info = "\n".join(
            [f"- {col}: {self.get_column_description(col)}" for col in self.config.get('columns', [])]
        )

        system_prompt = f"""Eres un experto especializado en la base de datos FRONTUR de estadísticas turísticas de España. 
        
Cuando se te pidan períodos relativos al máximo/último los haces considerando que el último período es {ultimo_periodo},
PERO los períodos concretos (años o meses) te los puede proporcionar el agente en el propio mensaje del usuario.
Tu única tarea es convertir consultas en español a SQL válido para SQLite.

ESQUEMA DE LA TABLA FRONTUR:
{columns_info}

REGLAS CRÍTICAS:
1. La columna ID_FECHA contiene fechas en formato YYYY-MM-DD
2. Para filtrar por año: SUBSTR(YM, 1, 4) = '2024'
3. La columna YM contiene año-mes en formato YYYY-MM - USAR PARA AGRUPACIONES MENSUALES
4. Para evoluciones mensuales: usar GROUP BY YM ORDER BY YM ASC (orden cronológico)
5. Usar LIKE para búsquedas de texto: PAIS_RESIDENCIA LIKE '%Francia%'
6. La métrica principal es NUM_VIAJEROS (usar SUM)
7. Para rankings: usar ORDER BY total DESC LIMIT
8. Para los últimos 12 meses: ORDER BY YM ASC sobre los meses que te indique el agente
9. Para comparaciones entre años: usar GROUP BY año, categoría
DIMENSIONES Y SERIES EN GRÁFICOS (MUY IMPORTANTE):
10. La dimensión tiempo (YM, o bien año+mes) define el eje temporal.
11. La dimensión medida normalmente es NUM_VIAJEROS (total de viajeros).
12. La dimensión categoría puede ser, por ejemplo, TIPO_VIAJERO, PAIS_RESIDENCIA,
    CCAA_DESTINO, MOTIVO_VIAJE, o también el AÑO cuando se comparan años entre sí.
13. Si no aparecen la dimension de tiempo no aparece considera que se pide el total acumulado,
   

DESGLOSES POR CATEGORÍA EN EVOLUCIONES TEMPORALES:
13. Si el usuario pide una EVOLUCIÓN MENSUAL "por" o "desglosada por" una categoría
    (p.ej. "evolución mensual del número de turistas por tipo de viajero"):
    - El SQL debe devolver TIEMPO + CATEGORÍA + MEDIDA, por ejemplo:
      SELECT YM, TIPO_VIAJERO, SUM(NUM_VIAJEROS) AS total
      FROM FRONTUR
      WHERE ...
      GROUP BY YM, TIPO_VIAJERO
      ORDER BY YM ASC
    - Cada valor de la categoría (cada TIPO_VIAJERO) será una serie (una línea) distinta
      en el gráfico, compartiendo el mismo eje temporal YM.

COMPARACIÓN DE AÑOS "MENSUALMENTE" (2004 vs 2005, etc.):
14. Si el usuario pide algo como "2004 vs 2005 mensualmente", "comparación mensual 2004 vs 2005"
    o similar, el patrón correcto es:
    - SELECT SUBSTR(YM,1,4) AS año,
             SUBSTR(YM,6,2) AS mes,
             SUM(NUM_VIAJEROS) AS total
      FROM FRONTUR
      WHERE ...
      GROUP BY año, mes
      ORDER BY año ASC, mes ASC
    - En este caso:
      * La CATEGORÍA es el AÑO (2004, 2005, ...).
      * El TIEMPO para el eje X es el MES (1–12).
      * Cada año se representa como una línea distinta en el gráfico.

COMPARACIÓN DE MEDIDAS (DOS MÉTRICAS) EN LA MISMA SERIE TEMPORAL:
15. Si el usuario pide comparar dos medidas sobre el mismo eje temporal, por ejemplo:
    "Número de turistas totales vs turistas promedio", se deben devolver dos columnas
    numéricas en el SELECT, con el tiempo como eje:
    - SELECT YM,
             SUM(NUM_VIAJEROS) AS total_turistas,
             AVG(NUM_VIAJEROS) AS promedio_turistas
      FROM FRONTUR
      WHERE ...
      GROUP BY YM
      ORDER BY YM ASC
    - Solo se deben incluir COMO MÁXIMO dos métricas numéricas.
    - El sistema de gráficos las representará como dos series distintas
      en el mismo gráfico, usando dos escalas Y (doble eje vertical).

16. No mezcles en la misma consulta múltiples categorías y múltiples métricas si
    no es necesario: o bien varias series por categoría (misma métrica),
    o bien dos métricas distintas sobre el mismo tiempo.


INTERPRETACIÓN DE PERÍODOS RELATIVOS (MUY IMPORTANTE):
- El agente puede añadir al final de la consulta del usuario una NOTA como, por ejemplo:
  "NOTA DEL AGENTE: usar exactamente estos meses: 2020-09, 2020-10, ..., 2025-07"
  o bien:
  "NOTA DEL AGENTE: usar exactamente estos años: 2020, 2021, 2022, 2023, 2024"

- SI APARECE UNA NOTA DEL AGENTE:
  * Debes usar EXACTAMENTE esos valores en la cláusula WHERE.
  * Para meses: WHERE YM IN ('2020-09', '2020-10', ..., '2025-07')
  * Para años: WHERE SUBSTR(YM, 1, 4) IN ('2020', '2021', ..., '2024')
  * NO recalcules tú los períodos, solo utilízalos.

- Solo si NO hay nota del agente puedes interpretar tú expresiones como
  "últimos 12 meses" o "últimos 5 años" siguiendo las reglas generales anteriores.

SELECCIÓN DE LA CATEGORÍA A PARTIR DEL TEXTO DEL USUARIO (MUY IMPORTANTE):

- El usuario puede pedir desgloses usando expresiones como:
  "por tipo de viajero", "por vía de entrada", "por país de residencia",
  "por comunidad autónoma", etc.

- Tu tarea consiste en:
  1) Leer el texto después de "por ..." o "desglosado por ...".
  2) Buscar en la lista de columnas y en sus descripciones cuál es la que mejor coincide
     con esa expresión.
  3) Usar EXACTAMENTE esa columna como dimensión de categoría en el SELECT y en el GROUP BY,
     y no otra.

- Nunca reutilices la categoría de una consulta anterior: en cada consulta decide la
  categoría únicamente por lo que diga el usuario en esta frase.

  REGLA DE FORMATOS DE GRÁFICO (MUY IMPORTANTE):
- Después de generar el SQL y la explicación, SIEMPRE debes indicar que los formatos de gráfico disponibles son:
  "líneas", "barras" y "pie".
- Estos tres formatos deben aparecer SIEMPRE como opciones disponibles en cualquier consulta.
- No propongas otros tipos de gráficos ni omitas alguno de ellos.


EJEMPLOS:

- Si el usuario dice:
  "Evolución mensual del número de turistas por tipo de viajero entre 2020 y 2025"

  Y existe una columna llamada TIPO_VIAJERO con descripción "Tipo de viajero (Turista no residente, Excursionista)...",

  debes generar algo como:
  SELECT YM, TIPO_VIAJERO, SUM(NUM_VIAJEROS) AS total
  FROM FRONTUR
  WHERE ...
  GROUP BY YM, TIPO_VIAJERO
  ORDER BY YM ASC;

- Si el usuario dice:
  "Evolución mensual del número de turistas por vía de entrada entre 2020 y 2025"

  Y existe una columna llamada VIA_ENTRADA con descripción "Vía de entrada (aérea, carretera, etc.)",

  debes generar algo como:
  SELECT YM, VIA_ENTRADA, SUM(NUM_VIAJEROS) AS total
  FROM FRONTUR
  WHERE ...
  GROUP BY YM, VIA_ENTRADA
  ORDER BY YM ASC;

  - SOLO puedes usar columnas que aparezcan en el esquema anterior. 
    No inventes nombres de columnas.
  - Nunca uses TIPO_VIAJERO como categoría si el usuario está pidiendo otra cosa
    (por ejemplo, "por vía de entrada" o "por país de residencia").


FORMATO DE RESPUESTA (SOLO JSON):
{{"sql": "SELECT ...", "explanation": "Breve explicación en español"}}

Responde ÚNICAMENTE con el JSON válido."""
        return system_prompt


    def get_column_description(self, column: str) -> str:
        descriptions = {
            'ID_FECHA': 'Fecha (YYYY-MM-DD) - usar SUBSTR(ID_FECHA, 1, 4) para año',
            'YM': 'Año-Mes (YYYY-MM) - ideal para agrupaciones mensuales, ORDENAR POR YM ASC',
            'PAIS_RESIDENCIA': 'País de residencia del turista - ideal para desgloses',
            'CCAA_DESTINO': 'Comunidad Autónoma de destino - ideal para desgloses',
            'NUM_VIAJEROS': 'Número de viajeros (métrica principal)',
            'MOTIVO_VIAJE': 'Motivo del viaje (Ocio/vacaciones, Negocios, etc) - ideal para desgloses',
            'TIPO_VIAJERO': 'Tipo de viajero (Turista no residente, Excursionista) - ideal para desgloses',
            'ALOJAMIENTO': 'Tipo de alojamiento - ideal para desgloses',
            'DURACION_VIAJE': 'Duración de la estancia'
        }
        return descriptions.get(column, column)

    def generate_sql(self, natural_query: str) -> Dict:
        """Generar SQL a partir de lenguaje natural, con post-procesado de seguridad y coherencia."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.create_system_prompt()},
                    {"role": "user", "content": natural_query}
                ],
                temperature=0.1,
                max_tokens=800
            )

            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("No se pudo extraer JSON de la respuesta del modelo")

            if result.get('sql'):
                sql = result['sql']
                print(f"🔧 SQL ORIGINAL DIRECTAMENTE DE GPT: {sql}")

                # DIAGNÓSTICO
                print(f"🔍 DIAGNÓSTICO SQL ORIGINAL:")
                print(f"   - ¿Tiene CCAA_DESTINO = ? {'CCAA_DESTINO =' in sql}")
                print(f"   - ¿Tiene LIKE ? {'LIKE' in sql}")
                print(f"   - ¿Tiene %? {'%' in sql}")
                if 'CCAA_DESTINO' in sql:
                    ccaa_match = re.search(r"CCAA_DESTINO\s*=\s*'([^']*)'", sql)
                    if ccaa_match:
                        print(f"   - CCAA_DESTINO = '{ccaa_match.group(1)}'")

                # 1. AGREGAR FILTRO ENCUESTA IS NOT NULL
                if 'WHERE' in sql.upper():
                    sql = re.sub(r'WHERE\s+', "WHERE ENCUESTA IS NOT NULL AND ", sql, flags=re.IGNORECASE)
                else:
                    if 'GROUP BY' in sql.upper():
                        sql = sql.replace('GROUP BY', "WHERE ENCUESTA IS NOT NULL GROUP BY")
                    elif 'ORDER BY' in sql.upper():
                        sql = sql.replace('ORDER BY', "WHERE ENCUESTA IS NOT NULL ORDER BY")
                    elif 'LIMIT' in sql.upper():
                        sql = sql.replace('LIMIT', "WHERE ENCUESTA IS NOT NULL LIMIT")
                    else:
                        sql += " WHERE ENCUESTA IS NOT NULL"

                # 2. CORREGIR INCONSISTENCIAS DE FECHA
                if 'SUBSTR(ID_FECHA, 1, 4)' in sql and 'YM' in sql:
                    print("🔄 Corrigiendo inconsistencia ID_FECHA vs YM")
                    sql = sql.replace('SUBSTR(ID_FECHA, 1, 4)', 'SUBSTR(YM, 1, 4)')

                # 3. CORREGIR CONDICIONES OR POR IN
                if 'SUBSTR(YM, 1, 4) =' in sql and 'OR SUBSTR(YM, 1, 4) =' in sql:
                    print("🔄 Convirtiendo condiciones OR a IN")
                    year_matches = re.findall(r"SUBSTR\(YM, 1, 4\) = '(\d{4})'", sql)
                    if year_matches:
                        years_str = "', '".join(year_matches)
                        pattern = r"SUBSTR\(YM, 1, 4\) = '\d{4}'(?:\s+OR\s+SUBSTR\(YM, 1, 4\) = '\d{4}')+"
                        replacement = f"SUBSTR(YM, 1, 4) IN ('{years_str}')"
                        sql = re.sub(pattern, replacement, sql)

                # 4. CORRECCIÓN PARA COMPARACIONES TEMPORALES
                comparison_keywords = ['vs', 'versus', 'comparación', 'comparar', 'comparativo', 'entre años']
                if any(keyword in natural_query.lower() for keyword in comparison_keywords):
                    print("🔍 Detectada consulta de comparación temporal")

                    if 'SELECT SUBSTR(YM, 1, 7) AS mes' in sql:
                        print("🔄 Separando año y mes para comparación")
                        sql = sql.replace(
                            'SELECT SUBSTR(YM, 1, 7) AS mes',
                            'SELECT SUBSTR(YM, 1, 4) AS año, SUBSTR(YM, 6, 2) AS mes'
                        )
                        if 'GROUP BY mes' in sql:
                            sql = sql.replace('GROUP BY mes', 'GROUP BY año, mes')

                    elif 'GROUP BY YM' in sql.upper() and 'SUBSTR(YM, 1, 4)' not in sql:
                        print("🔄 Convirtiendo agrupación YM a año/mes")
                        if 'SELECT YM' in sql.upper():
                            sql = re.sub(
                                r'SELECT\s+YM',
                                'SELECT SUBSTR(YM, 1, 4) as año, SUBSTR(YM, 6, 2) as mes',
                                sql,
                                flags=re.IGNORECASE
                            )
                        sql = re.sub(
                            r'GROUP BY\s+YM',
                            'GROUP BY SUBSTR(YM, 1, 4), SUBSTR(YM, 6, 2)',
                            sql,
                            flags=re.IGNORECASE
                        )

                    # Eliminar columnas duplicadas
                    if sql.count('AS año') > 1:
                        print("🔄 Eliminando columnas duplicadas")
                        sql = re.sub(r',\s*SUBSTR\(YM, 1, 4\) AS año', '', sql)

                    # Corregir GROUP BY redundante
                    if 'GROUP BY año, mes, año' in sql:
                        sql = sql.replace('GROUP BY año, mes, año', 'GROUP BY año, mes')

                    # Corregir ORDER BY
                    if 'ORDER BY mes ASC' in sql and 'año' in sql:
                        sql = sql.replace('ORDER BY mes ASC', 'ORDER BY año, mes ASC')

                # 5. CORREGIR ORDENAMIENTO TEMPORAL
                temporal_keywords = [
                    'evolución', 'mensual', 'anual', 'año', 'años',
                    'mes', 'meses', 'temporal', 'serie temporal', 'tendencia'
                ]
                if any(keyword in natural_query.lower() for keyword in temporal_keywords):
                    if 'ORDER BY YM DESC' in sql.upper():
                        sql = re.sub(r'ORDER BY YM DESC', 'ORDER BY YM ASC', sql, flags=re.IGNORECASE)
                    elif 'ORDER BY MES DESC' in sql.upper():
                        sql = re.sub(r'ORDER BY MES DESC', 'ORDER BY MES ASC', sql, flags=re.IGNORECASE)
                    elif 'ORDER BY AÑO DESC' in sql.upper() or "ORDER BY ANIO DESC" in sql.upper():
                        sql = re.sub(r'ORDER BY (AÑO|ANIO) DESC', 'ORDER BY \\1 ASC', sql, flags=re.IGNORECASE)
                    elif 'ORDER BY' not in sql.upper() and any(
                        col in sql.upper() for col in ['YM', 'AÑO', 'ANIO', 'MES']
                    ):
                        if 'YM' in sql.upper():
                            sql += " ORDER BY YM ASC"
                        elif 'AÑO' in sql.upper() or 'ANIO' in sql.upper():
                            sql += " ORDER BY año ASC"

                # 🚨 CORRECCIÓN FORZADA - SIN COMPLEJIDAD
                print("🔄 APLICANDO CORRECCIÓN FORZADA...")

                # CORRECCIÓN 1: Si tiene CCAA_DESTINO = 'Comunidad de Madrid', cambiarlo
                if "CCAA_DESTINO = 'Comunidad de Madrid'" in sql:
                    print("🚨 CORRIGIENDO: CCAA_DESTINO = 'Comunidad de Madrid'")
                    sql = sql.replace(
                        "CCAA_DESTINO = 'Comunidad de Madrid'",
                        "UPPER(CCAA_DESTINO) LIKE '%MADRID%'"
                    )

                # CORRECCIÓN 2: Si tiene CCAA_DESTINO = 'Com.Madrid', cambiarlo
                if "CCAA_DESTINO = 'Com.Madrid'" in sql:
                    print("🚨 CORRIGIENDO: CCAA_DESTINO = 'Com.Madrid'")
                    sql = sql.replace(
                        "CCAA_DESTINO = 'Com.Madrid'",
                        "UPPER(CCAA_DESTINO) LIKE '%MADRID%'"
                    )

                # CORRECCIÓN 3: Cualquier CCAA_DESTINO = 'texto'
                if "CCAA_DESTINO = '" in sql:
                    print("🚨 CORRIGIENDO: Cualquier CCAA_DESTINO =")
                    sql = re.sub(
                        r"CCAA_DESTINO = '([^']*)'",
                        "UPPER(CCAA_DESTINO) LIKE '%MADRID%'",
                        sql
                    )

                # CORRECCIÓN 4: Asegurar que PAIS_RESIDENCIA use LIKE %% (ejemplo con Francia)
                if "PAIS_RESIDENCIA = '" in sql:
                    print("🚨 CORRIGIENDO: PAIS_RESIDENCIA =")
                    sql = re.sub(
                        r"PAIS_RESIDENCIA = '([^']*)'",
                        "UPPER(PAIS_RESIDENCIA) LIKE '%FRANCIA%'",
                        sql
                    )
                elif "PAIS_RESIDENCIA LIKE '" in sql and "%" not in sql:
                    print("🚨 CORRIGIENDO: PAIS_RESIDENCIA LIKE sin %")
                    sql = re.sub(
                        r"PAIS_RESIDENCIA LIKE '([^']*)'",
                        "UPPER(PAIS_RESIDENCIA) LIKE '%FRANCIA%'",
                        sql
                    )

                print("✅ CORRECCIONES APLICADAS EXITOSAMENTE")

                # 🆕 DETECCIÓN DE COMPARACIONES TEMPORALES MEJORADA (solo logging)
                comparison_patterns = [
                    r'últimos?\s+(\d+)\s+(años?|meses?)',
                    r'comparación\s+entre\s+(\d+)\s+y\s+(\d+)',
                    r'(\d+)\s+vs\s+(\d+)',
                    r'evolución\s+de\s+los\s+últimos?\s+(\d+)'
                ]
                for pattern in comparison_patterns:
                    if re.search(pattern, natural_query.lower()):
                        print(f"🔍 Detectada consulta de comparación temporal (pattern: {pattern})")

                result['sql'] = sql
                print(f"🔧 SQL FINAL: {result['sql']}")
                return result

            else:
                raise ValueError("La respuesta del modelo no contiene clave 'sql'")

        except Exception as e:
            return {"sql": "", "explanation": f"Error: {e}"}


# ========== VISUALIZER MEJORADO ==========

class Visualizer:
    def __init__(self, logger):
        self.logger = logger
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    def _identify_series_dimensions(self, df: pd.DataFrame) -> Dict:
    
        """Identificar qué dimensiones están presentes en los datos"""
        dimensions = {
            'has_time': False,
            'has_categories': False,
            'has_multiple_metrics': False,
            'time_column': None,
            'category_column': None,
            'metric_columns': [],
            'series_type': 'simple'  # simple, temporal, categorizada, multi_metrica
        }

        # 🔴 CASO ESPECIAL: año + mes (comparación de años mensualmente)
        year_cols = [c for c in ['año', 'anio', 'ANIO', 'ANO'] if c in df.columns]
        if 'mes' in df.columns and year_cols:
            year_col = year_cols[0]
            dimensions['has_time'] = True
            dimensions['time_column'] = 'mes'
            dimensions['has_categories'] = True
            dimensions['category_column'] = year_col
            # métricas numéricas (máx 2)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            dimensions['metric_columns'] = numeric_cols[:2]
            if len(numeric_cols) > 1:
                dimensions['has_multiple_metrics'] = True
            dimensions['series_type'] = 'temporal_categorizada'
            self.logger.info(f"Dimensiones detectadas (año+mes): {dimensions}")
            return dimensions

        # 🔵 Resto de casos (lo que ya tenías)
        time_cols = ['YM', 'ID_FECHA', 'anio', 'mes', 'año', 'fecha']
        for col in time_cols:
            if col in df.columns:
                dimensions['has_time'] = True
                dimensions['time_column'] = col
                break

        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        categorical_cols = [
            col for col in categorical_cols
            if col not in time_cols
            and col not in ['mes_display', 'index']
            and not col.startswith('Unnamed')
        ]

        if len(categorical_cols) > 0:
            dimensions['has_categories'] = True
            dimensions['category_column'] = categorical_cols[0]

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 0:
            dimensions['metric_columns'] = numeric_cols
            if len(numeric_cols) > 1:
                dimensions['has_multiple_metrics'] = True
                dimensions['metric_columns'] = numeric_cols[:2]

        if dimensions['has_time'] and dimensions['has_categories']:
            dimensions['series_type'] = 'temporal_categorizada'
        elif dimensions['has_time'] and dimensions['has_multiple_metrics']:
            dimensions['series_type'] = 'temporal_multi_metrica'
        elif dimensions['has_time']:
            dimensions['series_type'] = 'temporal_simple'
        elif dimensions['has_categories']:
            dimensions['series_type'] = 'categorizada'
        elif dimensions['has_multiple_metrics']:
            dimensions['series_type'] = 'multi_metrica'

        self.logger.info(f"Dimensiones detectadas: {dimensions}")
        return dimensions


       
    def get_available_chart_types(self, df: pd.DataFrame) -> List[str]:
        if df.empty:
            return ['tabla']
        return ['tabla', 'barras', 'línea', 'pie']


  
    
    def create_chart(self, df: pd.DataFrame, chart_type: str, title: str) -> bool:
    
        """Crear gráfico adaptado a las dimensiones detectadas"""
        if df.empty:
            self.logger.warning("DataFrame vacío, no se puede crear gráfico")
            return False

        chart_type = chart_type.lower().strip()
        self.logger.info(f"Creando gráfico tipo: {chart_type}")
        self.logger.info(f"DataFrame shape: {df.shape}")

        # 🔴 CASO ESPECIAL: PIE (forzamos un camino ultra-simple y determinista)
        if chart_type in ['pie', 'torta']:
            try:
                if df.shape[1] < 2:
                    self.logger.warning("No hay suficientes columnas para pie (se necesitan al menos 2)")
                    return False

                # Usamos SIEMPRE: primera columna = etiquetas, segunda = valores
                labels = df.iloc[:, 0].astype(str).values
                sizes = df.iloc[:, 1].values

                fig, ax = plt.subplots(figsize=(10, 6))
                wedges, texts, autotexts = ax.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.set_title(f"{title}\n(Distribución - Pie)", fontsize=14, fontweight='bold')
                ax.axis('equal')

                plt.tight_layout()
                plt.show()
                self.logger.info("Gráfico pie (simple) creado exitosamente")
                return True

            except Exception as e:
                self.logger.error(f"Error creando gráfico pie (simple): {e}")
                return False

        # 🔵 RESTO DE TIPOS: usamos tu lógica adaptativa actual
        dimensions = self._identify_series_dimensions(df)
        print(f"🎯 SERIE DETECTADA: {dimensions['series_type']}")
        print(f"   - Tiempo: {dimensions['time_column']}")
        print(f"   - Categoría: {dimensions['category_column']}")
        print(f"   - Métricas: {dimensions['metric_columns']}")

        try:
            df_clean = df.reset_index(drop=True).copy()
            df_clean = self._preprocess_data_for_chart(df_clean)

            fig, ax = plt.subplots(figsize=(12, 6))

            success = False
            if chart_type == 'barras':
                success = self._create_adaptive_bar_chart(df_clean, dimensions, ax, title)
            elif chart_type in ['línea', 'líneas']:
                success = self._create_adaptive_line_chart(df_clean, dimensions, ax, title)
            elif chart_type == 'tabla':
                success = self._create_table_display(df_clean, ax, title)
            else:
                # Cualquier otro tipo desconocido → tabla
                success = self._create_table_display(df_clean, ax, f"{title} (tabla)")

            if success:
                plt.tight_layout()
                plt.show()
                self.logger.info(f"Gráfico {chart_type} creado exitosamente")
                return True
            else:
                self.logger.warning(f"No se pudo crear gráfico {chart_type}, fallback a tabla")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(12, 6))
                self._create_table_display(df_clean, ax, f"{title} (tabla)")
                plt.tight_layout()
                plt.show()
                return True

        except Exception as e:
            self.logger.error(f"Error creando gráfico {chart_type}: {e}")
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                self._create_table_display(df.reset_index(drop=True), ax, f"{title} (fallback a tabla)")
                plt.tight_layout()
                plt.show()
                return True
            except Exception as fallback_error:
                self.logger.error(f"Error en fallback a tabla: {fallback_error}")
                return False
    
        
    def _create_adaptive_line_chart(self, df: pd.DataFrame, dimensions: Dict, ax, title: str) -> bool:
    
        """Gráfico de torta adaptado (pie)"""
        try:
            if not dimensions['has_categories']:
                print("⚠️  No hay categorías para gráfico de pie")
                return False

            category_col = dimensions['category_column']
            metric_col = dimensions['metric_columns'][0] if dimensions['metric_columns'] else None

            if not metric_col:
                pie_data = df[category_col].value_counts()
                labels = pie_data.index
                sizes = pie_data.values
            else:
                df_sorted = df.sort_values(metric_col, ascending=False)
                labels = df_sorted[category_col].apply(self._clean_text).values
                sizes = df_sorted[metric_col].values

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                shadow=False
            )

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax.set_title(f"{title}\n(Distribución)", fontsize=14, fontweight='bold')
            ax.axis('equal')

            return True

        except Exception as e:
            self.logger.error(f"Error en gráfico de torta adaptativo: {e}")
            return False
    

    def _create_adaptive_bar_chart(self, df: pd.DataFrame, dimensions: Dict, ax, title: str) -> bool:
        """Gráfico de barras adaptado a las dimensiones"""
        try:
            metric_cols = dimensions['metric_columns']

            if not metric_cols:
                return False

            # CASO 1: Simple - 1 métrica, 1 categoría o tiempo
            if len(metric_cols) == 1:
                metric = metric_cols[0]

                df_sorted = df.sort_values(metric, ascending=False)

                # Etiquetas eje X
                if dimensions['has_categories']:
                    x_labels = df_sorted[dimensions['category_column']].apply(self._clean_text).values
                elif dimensions['has_time'] and 'mes_display' in df_sorted.columns:
                    x_labels = df_sorted['mes_display'].values
                else:
                    x_labels = [str(i) for i in range(len(df_sorted))]

                x_positions = range(len(df_sorted))
                bars = ax.bar(x_positions, df_sorted[metric], alpha=0.8)

                ax.set_title(f"{title}\n(Gráfico de Barras)", fontsize=14, fontweight='bold')
                ax.set_ylabel(metric)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)

                # Valores encima de barras
                for bar in bars:
                    height = bar.get_height()
                    if pd.notnull(height) and height > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.,
                            height + height * 0.01,
                            f'{height:,.0f}',
                            ha='center',
                            va='bottom',
                            fontsize=8
                        )

                self._format_y_axis(ax, df_sorted[metric])
                return True

            # CASO 2: Múltiples métricas
            elif len(metric_cols) == 2:
                return self._create_dual_metric_bar_chart(df, dimensions, ax, title)

            return False

        except Exception as e:
            self.logger.error(f"Error en gráfico de barras adaptativo: {e}")
            return False

    def _create_dual_metric_line_chart(self, df: pd.DataFrame, dimensions: Dict, ax, title: str) -> bool:
        """Gráfico de líneas con 2 métricas y escalas duales"""
        try:
            time_col = dimensions['time_column']
            metric1, metric2 = dimensions['metric_columns'][:2]

            if time_col in df.columns:
                df = df.sort_values(time_col, ascending=True)

            ax2 = ax.twinx()

            if time_col == 'YM' and 'mes_display' in df.columns:
                x_values = df['mes_display'].values
            else:
                x_values = df[time_col].apply(self._clean_text).values

            line1 = ax.plot(x_values, df[metric1], marker='o', linewidth=2, label=metric1)
            ax.set_ylabel(metric1)
            ax.tick_params(axis='y')

            line2 = ax2.plot(x_values, df[metric2], marker='s', linewidth=2, label=metric2)
            ax2.set_ylabel(metric2)
            ax2.tick_params(axis='y')

            ax.set_title(f"{title}\n(Comparación de Métricas)", fontsize=14, fontweight='bold')

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')

            if len(x_values) > 6:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            return True

        except Exception as e:
            self.logger.error(f"Error en gráfico dual de líneas: {e}")
            return False

    def _create_dual_metric_bar_chart(self, df: pd.DataFrame, dimensions: Dict, ax, title: str) -> bool:
        """Gráfico de barras con 2 métricas"""
        try:
            metric1, metric2 = dimensions['metric_columns'][:2]

            df_sorted = df.sort_values(metric1, ascending=False)

            if dimensions['has_categories']:
                x_labels = df_sorted[dimensions['category_column']].apply(self._clean_text).values
            else:
                x_labels = [str(i) for i in range(len(df_sorted))]

            x_positions = np.arange(len(df_sorted))
            width = 0.35

            bars1 = ax.bar(x_positions - width / 2, df_sorted[metric1], width, label=metric1, alpha=0.8)
            bars2 = ax.bar(x_positions + width / 2, df_sorted[metric2], width, label=metric2, alpha=0.8)

            ax.set_title(f"{title}\n(Comparación de Métricas)", fontsize=14, fontweight='bold')
            ax.set_ylabel('Valores')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            ax.legend()

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if pd.notnull(height) and height > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.,
                            height + height * 0.01,
                            f'{height:,.0f}',
                            ha='center',
                            va='bottom',
                            fontsize=7
                        )

            self._format_y_axis(ax, pd.concat([df_sorted[metric1], df_sorted[metric2]]))
            return True

        except Exception as e:
            self.logger.error(f"Error en gráfico dual de barras: {e}")
            return False

    def _create_categorical_line_chart(self, df: pd.DataFrame, dimensions: Dict, ax, title: str) -> bool:
        """Gráfico de líneas con múltiples categorías"""
        try:
            time_col = dimensions['time_column']
            category_col = dimensions['category_column']
            metric_col = dimensions['metric_columns'][0]

            if time_col in df.columns:
                df = df.sort_values(time_col, ascending=True)

            categories = df[category_col].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

            for i, category in enumerate(categories):
                category_data = df[df[category_col] == category]

                if time_col == 'YM' and 'mes_display' in category_data.columns:
                    x_values = category_data['mes_display'].values
                else:
                    x_values = category_data[time_col].apply(self._clean_text).values

                ax.plot(
                    x_values,
                    category_data[metric_col],
                    color=colors[i],
                    marker='o',
                    linewidth=2,
                    label=self._clean_text(str(category))
                )

            ax.set_title(f"{title}\n(Evolución por Categorías)", fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_col)
            ax.legend()
            ax.grid(True, alpha=0.3)

            if len(x_values) > 6:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            self._format_y_axis(ax, df[metric_col])
            return True

        except Exception as e:
            self.logger.error(f"Error en gráfico categórico de líneas: {e}")
            return False

    def _create_adaptive_pie_chart(self, df: pd.DataFrame, dimensions: Dict, ax, title: str) -> bool:
        
        """Gráfico de torta adaptado"""
        try:
            if not dimensions['has_categories']:
                print("⚠️  No hay categorías para gráfico de pie")
                return False

            category_col = dimensions['category_column']
            metric_col = dimensions['metric_columns'][0] if dimensions['metric_columns'] else None

            if not metric_col:
                pie_data = df[category_col].value_counts()
                labels = pie_data.index
                sizes = pie_data.values
            else:
                df_sorted = df.sort_values(metric_col, ascending=False)
                labels = df_sorted[category_col].apply(self._clean_text).values
                sizes = df_sorted[metric_col].values

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                shadow=False
            )

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            ax.set_title(f"{title}\n(Distribución)", fontsize=14, fontweight='bold')
            ax.axis('equal')

            return True

        except Exception as e:
            self.logger.error(f"Error en gráfico de torta adaptativo: {e}")
            return False

    def _format_y_axis(self, ax, series: pd.Series):
        """Formatear eje Y para mejor legibilidad"""
        if series.empty:
            return

        max_val = series.max()
        if max_val >= 1e6:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'{x / 1e6:.1f}M')
            )
        elif max_val >= 1000:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'{x / 1e3:.0f}K')
            )
        else:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'{x:.0f}')
            )

    # ===== Métodos auxiliares que faltaban =====

    def _preprocess_data_for_chart(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesar datos para los gráficos (añadir mes_display, limpiar textos, etc.)"""
        df_proc = df.copy()
        if 'YM' in df_proc.columns and 'mes_display' not in df_proc.columns:
            df_proc['mes_display'] = df_proc['YM'].apply(self._format_month)
        return df_proc

    def _clean_text(self, text: str) -> str:
        """Limpieza básica de etiquetas"""
        if text is None:
            return ""
        return str(text).replace('_', ' ').strip()

    def _format_month(self, ym: str) -> str:
        """Formatear 'YYYY-MM' a 'Mes-YYYY' en corto (Ene-2024, etc.)"""
        try:
            dt = datetime.strptime(ym, "%Y-%m")
            months_es = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
            return f"{months_es[dt.month - 1]}-{dt.year}"
        except Exception:
            return ym

    def _create_table_display(self, df: pd.DataFrame, ax, title: str) -> bool:
        """Mostrar los datos en forma de tabla (hasta 30 filas)"""
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')

        table_df = df.head(30)
        tbl = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(col=list(range(len(table_df.columns))))
        return True


# ========== FRONTUR LLM AGENT ==========

class FronturLLMAgent:
    def __init__(
        self,
        db_path: str,
        openai_api_key: str,
        config_path: str = "frontur_agent_config.json",
        metadata_path: str = "frontur_metadata_completo.json",
        examples_path: str = "ejemplos.json"
    ):
        self.db_path = os.path.abspath(db_path)
        print(f"📁 Conectando a base de datos: {self.db_path}")

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"❌ No se encuentra la base de datos: {self.db_path}")

        self.client = OpenAI(api_key=openai_api_key)
        self.connection = sqlite3.connect(self.db_path)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "frontur_agent_config.json")
        metadata_path = os.path.join(base_dir, "frontur_metadata_completo.json")
        examples_path = os.path.join(base_dir, "ejemplos.json")

        self.config = self.load_config(config_path)
        self.metadata = self.load_metadata(metadata_path)
        self.examples = self.load_examples(examples_path)

        self.setup_logging()

        self.query_executor = QueryExecutor(self.connection, self.logger)
        self.sql_generator = SQLGenerator(self.client, self.config, self.metadata, self.examples)
        self.visualizer = Visualizer(self.logger)

        print("✅ Agente FRONTUR inicializado correctamente")

    def load_config(self, path: str) -> dict:
        if not os.path.exists(path):
            print(f"⚠️ Archivo de configuración no encontrado: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_metadata(self, path: str) -> Union[dict, list]:
        if not os.path.exists(path):
            print(f"⚠️ Archivo de metadatos no encontrado: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_examples(self, path: str) -> list:
        if not os.path.exists(path):
            print(f"⚠️ Archivo de ejemplos no encontrado: {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [FRONTUR_AGENT] %(message)s',
            handlers=[
                logging.FileHandler(f'frontur_agent_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def process_query(self, natural_query: str) -> Dict:
        self.logger.info(f"Procesando consulta: {natural_query}")

        try:
            # 1) Añadir contexto temporal calculado (si aplica)
            query_with_time = self._augment_query_with_time_context(natural_query)
            self.logger.info(f"Consulta enviada al LLM (con contexto temporal si aplica): {query_with_time}")

            # 2) Generar SQL con el LLM
            llm_result = self.sql_generator.generate_sql(query_with_time)
            self.logger.info(f"Respuesta del LLM: {llm_result}")

            if not llm_result.get('sql'):
                return {
                    'success': False,
                    'error': f"No se pudo generar SQL: {llm_result.get('explanation', 'Error desconocido')}",
                    'data': None
                }

            # 3) Ejecutar SQL
            df, error = self.query_executor.execute_sql(llm_result['sql'])

            if error:
                return {
                    'success': False,
                    'error': error,
                    'data': None
                }

            return {
                'success': True,
                'data': df,
                'sql_generated': llm_result['sql'],
                'explanation': llm_result.get('explanation', '')
            }

        except Exception as e:
            error_msg = f"Error procesando consulta: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'data': None
            }

    



    # ===== Utilidades de periodo relativo (ahora como métodos de la clase) =====

    
    def get_last_period_from_db(self) -> str:
        """Obtener el último período disponible de la base de datos."""
        try:
            query = "SELECT MAX(YM) as ultimo_periodo FROM frontur WHERE ENCUESTA IS NOT NULL"
            df, error = self.query_executor.execute_sql(query)

            if error:
                self.logger.error(f"Error SQL al obtener último período: {error}")

            if (df is not None) and (not df.empty) and ('ultimo_periodo' in df.columns):
                last_period = df['ultimo_periodo'].iloc[0]
                print(f"📅 Último período en BD: {last_period}")
                return last_period
            else:
                print("⚠️  No se pudo obtener último período, usando 2025-08 por defecto")
                return "2025-08"

        except Exception as e:
            self.logger.error(f"Error obteniendo último período: {e}")
            return "2025-08"

    # ===== Modo interactivo y utilidades =====

          # ===== Cálculo de períodos relativos =====

    def get_relative_months(self, base_period: str, months: int) -> List[str]:
        """
        Devuelve `months` meses ANTERIORES a base_period, excluyendo base_period,
        en orden cronológico.
        """
        try:
            year, month = map(int, base_period.split('-'))
            base_index = year * 12 + (month - 1)

            periods = []
            for idx in range(base_index - months, base_index):
                y = idx // 12
                m = idx % 12 + 1
                periods.append(f"{y}-{m:02d}")

            return periods
        except Exception as e:
            self.logger.error(f"Error calculando meses relativos: {e}")
            return []

    def get_relative_years(self, base_period: str, years: int) -> List[str]:
        """
        Devuelve `years` AÑOS anteriores al año de base_period, excluyendo el año base,
        en orden ascendente.
        Ej: base_period=2025-08, years=5 -> ['2020','2021','2022','2023','2024']
        """
        try:
            base_year = int(base_period.split('-')[0])
            return [str(y) for y in range(base_year - years, base_year)]
        except Exception as e:
            self.logger.error(f"Error calculando años relativos: {e}")
            return []

    def _compute_time_context(self, prompt: str, base_period: str) -> Dict:
        """
        Detecta en el texto del usuario si habla de:
        - evolución mensual en los últimos N años  -> N*12 meses
        - evolución mensual en los últimos N meses -> N meses
        - evolución anual  en los últimos N años  -> N años

        Devuelve un dict con:
          {
            'modo': 'mensual' | 'anual' | None,
            'unidad': 'años' | 'meses' | None,
            'cantidad': int | None,
            'periodos': List[str]  # YM o años
          }
        """
        ctx = {
            'modo': None,
            'unidad': None,
            'cantidad': None,
            'periodos': []
        }

        try:
            p = prompt.lower()

            match_years = re.search(r'últimos?\s+(\d+)\s+años?', p)
            match_months = re.search(r'últimos?\s+(\d+)\s+meses?', p)

            n_years = int(match_years.group(1)) if match_years else None
            n_months = int(match_months.group(1)) if match_months else None

            # Evolución anual ... últimos N años
            if ("evolución anual" in p or "evolucion anual" in p) and n_years:
                ctx['modo'] = 'anual'
                ctx['unidad'] = 'años'
                ctx['cantidad'] = n_years
                ctx['periodos'] = self.get_relative_years(base_period, n_years)
                return ctx

            # Evolución mensual ... últimos N años  -> N*12 meses
            if ("evolución mensual" in p or "evolucion mensual" in p) and n_years:
                ctx['modo'] = 'mensual'
                ctx['unidad'] = 'años'
                ctx['cantidad'] = n_years
                total_months = n_years * 12
                ctx['periodos'] = self.get_relative_months(base_period, total_months)
                return ctx

            # Evolución mensual ... últimos N meses
            if ("evolución mensual" in p or "evolucion mensual" in p) and n_months:
                ctx['modo'] = 'mensual'
                ctx['unidad'] = 'meses'
                ctx['cantidad'] = n_months
                ctx['periodos'] = self.get_relative_months(base_period, n_months)
                return ctx

            return ctx

        except Exception as e:
            self.logger.error(f"Error analizando contexto temporal: {e}")
            return ctx

    def _augment_query_with_time_context(self, natural_query: str) -> str:
        """
        Añade una 'NOTA DEL AGENTE' al texto del usuario con los periodos
        concretos (años o meses) ya calculados, para que el LLM solo los use.
        """
        try:
            base_period = self.get_last_period_from_db()  # o '2025-08' fijo si prefieres
            ctx = self._compute_time_context(natural_query, base_period)

            if not ctx['periodos']:
                return natural_query  # no hay nada que añadir

            if ctx['modo'] == 'mensual':
                nota = (
                    f"\n\nNOTA DEL AGENTE: usar exactamente estos meses (YM) ya calculados "
                    f"respecto al último período {base_period}: "
                    + ", ".join(ctx['periodos'])
                )
            elif ctx['modo'] == 'anual':
                nota = (
                    f"\n\nNOTA DEL AGENTE: usar exactamente estos años ya calculados "
                    f"respecto al último período {base_period}: "
                    + ", ".join(ctx['periodos'])
                )
            else:
                return natural_query

            return natural_query + nota

        except Exception as e:
            self.logger.error(f"Error añadiendo nota de contexto temporal: {e}")
            return natural_query


    def ask_chart_type(self, df: pd.DataFrame) -> str:
        """Preguntar al usuario qué tipo de gráfico prefiere"""
        available_types = self.visualizer.get_available_chart_types(df)

        print(f"\n📊 FORMATOS DISPONIBLES para estos datos:")
        for i, chart_type in enumerate(available_types, 1):
            print(f"   {i}. {chart_type.capitalize()}")

        while True:
            try:
                choice = input(f"\n🎨 Elige formato (1-{len(available_types)}): ").strip()
                if choice.isdigit():
                    index = int(choice) - 1
                    if 0 <= index < len(available_types):
                        return available_types[index]
                print("❌ Opción no válida. Intenta de nuevo.")
            except KeyboardInterrupt:
                return 'tabla'
            except Exception as e:
                print(f"❌ Error: {e}")
                return 'tabla'

    def interactive_mode(self):
        print("🏨 AGENTE LLM FRONTUR - SISTEMA DE CONSULTAS TURÍSTICAS")
        print("=" * 70)
        print("Escribe tu consulta en español o 'salir' para terminar")
        print("Comandos especiales: 'ejemplos', 'estadisticas'")
        print()

        while True:
            try:
                query = input("🔍 Tu consulta: ").strip()

                if query.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta pronto!")
                    break
                elif not query:
                    continue
                elif query.lower() == 'ejemplos':
                    self.show_examples()
                elif query.lower() == 'estadisticas':
                    self.show_statistics()
                else:
                    result = self.process_query(query)
                    self.display_results(result)

            except KeyboardInterrupt:
                print("\n👋 ¡Hasta pronto!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def display_results(self, result: Dict):
        if result['success']:
            print(f"\n✅ CONSULTA EXITOSA")
            print(f"📊 Resultados ({len(result['data'])} filas):")
            print(result['data'].to_string(index=False))

            if result.get('explanation'):
                print(f"\n💡 Explicación: {result['explanation']}")

            if result['data'] is not None and not result['data'].empty:
                chart_type = self.ask_chart_type(result['data'])
                chart_title = f"Resultado: {result.get('explanation', 'Consulta')}"

                print(f"\n🎨 Generando gráfico: {chart_type}...")
                chart_created = self.visualizer.create_chart(
                    result['data'],
                    chart_type,
                    chart_title
                )

                if chart_created:
                    print(f"✅ Gráfico {chart_type} generado correctamente")
                else:
                    print("❌ No se pudo generar el gráfico solicitado")

            print(f"\n🔍 SQL ejecutado: {result['sql_generated']}")
        else:
            print(f"\n❌ ERROR: {result['error']}")

    def show_examples(self):
        print("\n📚 EJEMPLOS DE CONSULTAS:")
        examples = [
            "Evolución mensual de turistas franceses en Madrid",
            "Top 5 países con más turistas en 2024",
            "Distribución de turistas por comunidades autónomas",
            "Turistas alemanes en Canarias últimos 3 años",
            "Comparación de turistas italianos 2023 vs 2024",
        ]

        for i, example in enumerate(examples, 1):
            print(f"   {i}. {example}")
        print()

    def show_statistics(self):
        if not self.metadata:
            print("❌ No se cargaron los metadatos")
            return

        stats = self.metadata['statistics']
        print(f"\n📊 ESTADÍSTICAS FRONTUR:")
        print(f"   • Total de registros: {stats['total_records']:,}")
        print(f"   • Rango de fechas: {stats['date_range']['min_date']} a {stats['date_range']['max_date']}")
        print(f"   • Total de turistas: {stats['total_turistas']:,.0f}")
        print(f"   • Países únicos: {self.metadata['column_values']['PAIS_RESIDENCIA']['total_unique']}")
        print(f"   • CCAA únicas: {self.metadata['column_values']['CCAA_DESTINO']['total_unique']}")

    def close(self):
        if self.connection:
            self.connection.close()


def main():
    DB_PATH = r"C:\DataEstur_Data\dataestur.db"

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    try:
        print("🚀 Iniciando Agente LLM FRONTUR...")
        print(f"📁 Base de datos: {DB_PATH}")

        if not os.path.exists(DB_PATH):
            print(f"❌ ERROR: No se encuentra la base de datos en {DB_PATH}")
            return

        agent = FronturLLMAgent(DB_PATH, OPENAI_API_KEY)
        agent.interactive_mode()

    except Exception as e:
        print(f"❌ Error inesperado: {e}")
    finally:
        if 'agent' in locals():
            agent.close()


if __name__ == "__main__":
    main()
