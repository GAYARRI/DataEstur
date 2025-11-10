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
from typing import Dict, List, Optional, Tuple

class SQLGenerator:
    def __init__(self, client, config, metadata, examples):
        self.client = client
        self.config = config
        self.metadata = metadata
        self.examples = examples
    
    def create_system_prompt(self) -> str:
        """Crear prompt del sistema ultra-específico para FRONTUR"""
        
        columns_info = "\n".join([f"- {col}: {self.get_column_description(col)}" for col in self.config['columns']])
        
        paises_info = ", ".join([p['PAIS_RESIDENCIA'] for p in self.metadata['column_values']['PAIS_RESIDENCIA']['top_values'][:8]])
        ccaa_info = ", ".join([c['CCAA_DESTINO'] for c in self.metadata['column_values']['CCAA_DESTINO']['top_values'][:8]])
        motivos_info = ", ".join([m['MOTIVO_VIAJE'] for m in self.metadata['column_values']['MOTIVO_VIAJE']['top_values'][:5]])
        
        examples_text = self.format_few_shot_examples()
        
        system_prompt = f"""Eres un experto especializado en la base de datos FRONTUR de estadísticas turísticas de España. 
Tu única tarea es convertir consultas en español a SQL válido para SQLite . Te paso ejemplos :
    1.
    consulta usuario: Total de turistas franceses en la Comunidad de Madrid en los últimos 5 años,
    consulta sql: SELECT CAST(SUBSTR(YM,1,4) AS INT) AS anio, SUM(NUM_VIAJEROS) AS total FROM FRONTUR WHERE UPPER(TRIM(PAIS_RESIDENCIA)) IN ('FR','FRA','FRANCIA','FRANCE') AND REPLACE(REPLACE(UPPER(TRIM(CCAA_DESTINO)),'.',''),' ','') IN ('COMMADRID','COMUNIDADDEMADRID','MADRID') GROUP BY CAST(SUBSTR(YM,1,4) AS INT) ORDER BY anio";
    2.
    consulta usuario: Evolución mensual de turistas franceses en Madrid durante los últimos 12 meses,
    consulta sql: "SELECT YM AS mes, SUM(NUM_VIAJEROS) AS total FROM FRONTUR WHERE UPPER(TRIM(PAIS_RESIDENCIA)) IN ('FR','FRA','FRANCIA','FRANCE') AND REPLACE(REPLACE(UPPER(TRIM(CCAA_DESTINO)),'.',''),' ','') IN ('COMMADRID','COMUNIDADDEMADRID','MADRID') GROUP BY YM ORDER BY YM DESC LIMIT 12";
    3.
    consulta usuario: Top 10 países emisores de turistas hacia España el último año,
    consulta sql: "WITH max_y AS (SELECT MAX(CAST(SUBSTR(YM,1,4) AS INT)) AS y FROM FRONTUR) SELECT PAIS_RESIDENCIA, SUM(NUM_VIAJEROS) AS total FROM FRONTUR, max_y WHERE CAST(SUBSTR(YM,1,4) AS INT)=y GROUP BY PAIS_RESIDENCIA ORDER BY total DESC LIMIT 10";
    4.
    consulta usuario: Distribución de turistas por comunidad autónoma destino en el año más reciente,
    consulta sql: WITH max_y AS (SELECT MAX(CAST(SUBSTR(YM,1,4) AS INT)) AS y FROM FRONTUR) SELECT CCAA_DESTINO, SUM(NUM_VIAJEROS) AS total FROM FRONTUR, max_y WHERE CAST(SUBSTR(YM,1,4) AS INT)=y GROUP BY CCAA_DESTINO ORDER BY total DESC";
    5.
    consulta usuario: Número de turistas alemanes en Canarias durante los últimos 3 años,
    consulta sql: WITH max_y AS (SELECT MAX(CAST(SUBSTR(YM,1,4) AS INT)) AS y FROM FRONTUR) SELECT CAST(SUBSTR(YM,1,4) AS INT) AS anio, SUM(NUM_VIAJEROS) AS total FROM FRONTUR, max_y WHERE UPPER(TRIM(PAIS_RESIDENCIA)) IN ('DE','DEU','ALEMANIA','GERMANY') AND REPLACE(REPLACE(UPPER(TRIM(CCAA_DESTINO)),'.',''),' ','') IN ('CANARIAS','ISLASCANARIAS') AND CAST(SUBSTR(YM,1,4) AS INT) BETWEEN y-2 AND y GROUP BY anio ORDER BY anio";
    6.  
    consulta usuario: Evolución anual de turistas británicos en Cataluña,
    consulta sql: WITH max_y AS (SELECT MAX(CAST(SUBSTR(YM,1,4) AS INT)) AS y FROM FRONTUR) SELECT CAST(SUBSTR(YM,1,4) AS INT) AS anio, SUM(NUM_VIAJEROS) AS total FROM FRONTUR, max_y WHERE UPPER(TRIM(PAIS_RESIDENCIA)) IN ('UK','GB','REINO UNIDO','UNITED KINGDOM') AND REPLACE(REPLACE(UPPER(TRIM(CCAA_DESTINO)),'.',''),' ','') IN ('CATALUNYA','CATALUNA','CATALONIA') GROUP BY anio ORDER BY anio";
    7.  
    consulta usuario: Top 5 comunidades autónomas con mayor número de turistas internacionales en 2024,
    consulta sql: SELECT CCAA_DESTINO, SUM(NUM_VIAJEROS) AS total FROM FRONTUR WHERE CAST(SUBSTR(YM,1,4) AS INT)=2024 GROUP BY CCAA_DESTINO ORDER BY total DESC LIMIT 5;
    8. consulta usuario: Evolución número de turistas por pais de residencia,
    consulta sql: SELECT CAST(SUBSTR(YM,1,4) AS INT) AS anio,PAIS_RESIDENCIA, SUM(NUM_VIAJEROS) AS total FROM FRONTUR, max_y GROUP BY (anio,PAIS_RESIDENCIA) ORDER BY anio";
    


ESQUEMA DE LA TABLA FRONTUR:
{columns_info}

VALORES CLAVE EN LA BASE DE DATOS:
- PAISES: {paises_info}...
- COMUNIDADES: {ccaa_info}...
- MOTIVOS DE VIAJE: {motivos_info}...
- FECHAS: Desde {self.metadata['statistics']['date_range']['min_date']} hasta {self.metadata['statistics']['date_range']['max_date']}

REGLAS CRÍTICAS:
1. TODAS las consultas DEBEN filtrar por ENCUESTA='FRONTUR'
2. La columna ID_FECHA contiene fechas en formato YYYY-MM-DD
3. Para filtrar por año: SUBSTR(ID_FECHA, 1, 4) = '2024'
4. La columna YM contiene año-mes en formato YYYY-MM - USAR PARA AGRUPACIONES MENSUALES
5. Para evoluciones mensuales: usar GROUP BY YM ORDER BY YM ASC (orden cronológico)
6. Usar LIKE para búsquedas de texto: PAIS_RESIDENCIA LIKE '%Francia%'
7. La métrica principal es NUM_VIAJEROS (usar SUM)
8. Para rankings: usar ORDER BY total DESC LIMIT
9. Para los últimos 12 meses: ORDER BY YM DESC LIMIT 12 pero luego ordenar por YM ASC para gráficos

DETECCIÓN DE TIPO DE GRÁFICO:
- Usar 'line' SOLO para evoluciones temporales (meses, años, series temporales)
- Usar 'bar' para rankings y comparaciones entre categorías (top 5, principales, etc.)
- Usar 'pie' SOLO para desgloses, distribuciones y porcentajes (por países, comunidades, motivos, alojamiento, tipo de viajero)
- Usar 'table' cuando no sea apropiado ningún gráfico

IMPORTANTE: Para consultas que piden distribución, desglose, porcentaje, composición, reparto - SIEMPRE usar 'pie'

EJEMPLOS DE CONSULTAS CORRECTAS:
{"C:\Dataestur_Data\ejemplos.json"}

FORMATO DE RESPUESTA (SOLO JSON):
{{"sql": "SELECT ...", "explanation": "Breve explicación en español", "chart_type": "bar|line|pie|table|None"}}

Responde ÚNICAMENTE con el JSON válido."""
        return system_prompt
    
    def get_column_description(self, column: str) -> str:
        descriptions = {
            'ENCUESTA': 'Tipo de encuesta (SIEMPRE usar ENCUESTA="FRONTUR" en todas las consultas)',
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
    
    def format_few_shot_examples(self) -> str:
        examples_text = ""
        corrected_examples = [
            {
                'question': 'Evolución mensual de turistas franceses en Madrid durante los últimos 12 meses',
                'sql': "SELECT YM AS mes, SUM(NUM_VIAJEROS) AS total FROM FRONTUR WHERE ENCUESTA='FRONTUR' AND PAIS_RESIDENCIA LIKE '%Francia%' AND CCAA_DESTINO LIKE '%Madrid%' GROUP BY YM ORDER BY YM DESC LIMIT 12",
                'chart_type': 'line'
            },
            {
                'question': 'Top 5 países con más turistas en 2024',
                'sql': "SELECT PAIS_RESIDENCIA, SUM(NUM_VIAJEROS) AS total FROM FRONTUR WHERE ENCUESTA='FRONTUR' AND SUBSTR(ID_FECHA, 1, 4) = '2024' GROUP BY PAIS_RESIDENCIA ORDER BY total DESC LIMIT 5",
                'chart_type': 'bar'
            },
            {
                'question': 'Distribución de turistas por comunidades autónomas en 2024',
                'sql': "SELECT CCAA_DESTINO, SUM(NUM_VIAJEROS) AS total FROM FRONTUR WHERE ENCUESTA='FRONTUR' AND SUBSTR(ID_FECHA, 1, 4) = '2024' GROUP BY CCAA_DESTINO ORDER BY total DESC LIMIT 10",
                'chart_type': 'pie'
            },
            {
                'question': 'Desglose de turistas por motivo de viaje',
                'sql': "SELECT MOTIVO_VIAJE, SUM(NUM_VIAJEROS) AS total FROM FRONTUR WHERE ENCUESTA='FRONTUR' GROUP BY MOTIVO_VIAJE ORDER BY total DESC",
                'chart_type': 'pie'
            },
            {
                'question': 'Porcentaje de turistas por tipo de alojamiento en 2024',
                'sql': "SELECT ALOJAMIENTO, SUM(NUM_VIAJEROS) AS total FROM FRONTUR WHERE ENCUESTA='FRONTUR' AND SUBSTR(ID_FECHA, 1, 4) = '2024' GROUP BY ALOJAMIENTO ORDER BY total DESC LIMIT 8",
                'chart_type': 'pie'
            },
            {
                'question': 'Composición de turistas por tipo de viajero',
                'sql': "SELECT TIPO_VIAJERO, SUM(NUM_VIAJEROS) AS total FROM FRONTUR WHERE ENCUESTA='FRONTUR' GROUP BY TIPO_VIAJERO ORDER BY total DESC",
                'chart_type': 'pie'
            }
        ]
        
        for i, example in enumerate(corrected_examples):
            examples_text += f"\n--- EJEMPLO {i+1} ---\n"
            examples_text += f"PREGUNTA: {example['question']}\n"
            examples_text += f"SQL: {example['sql']}\n"
            examples_text += f"CHART_TYPE: {example['chart_type']}\n"
        return examples_text
    
    def generate_sql(self, natural_query: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
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
                # 🔥 FORZAR FILTRO ENCUESTA ='FRONTUR' EN TODAS LAS CONSULTAS
                if result.get('sql'):
                    sql = result['sql']
                    # Verificar si ya tiene cláusula WHERE
                    if 'WHERE' in sql.upper():
                        # Insertar ENCUESTA='FRONTUR' después del WHERE
                        sql = re.sub(r'WHERE\s+', "WHERE ENCUESTA='FRONTUR' AND ", sql, flags=re.IGNORECASE)
                    else:
                        # Agregar WHERE si no existe
                        if 'GROUP BY' in sql.upper():
                            sql = sql.replace('GROUP BY', "WHERE ENCUESTA='FRONTUR' GROUP BY")
                        elif 'ORDER BY' in sql.upper():
                            sql = sql.replace('ORDER BY', "WHERE ENCUESTA='FRONTUR' ORDER BY")
                        elif 'LIMIT' in sql.upper():
                            sql = sql.replace('LIMIT', "WHERE ENCUESTA='FRONTUR' LIMIT")
                        else:
                            sql += " WHERE ENCUESTA='FRONTUR'"
                    
                    # 🔥 ELIMINAR FILTROS DUPLICADOS
                    sql = re.sub(r"ENCUESTA='FRONTUR'\s+AND\s+ENCUESTA='FRONTUR'", "ENCUESTA='FRONTUR'", sql, flags=re.IGNORECASE)
                    sql = re.sub(r"ENCUESTA='FRONTUR'\s+AND\s+ENCUESTA='FRONTUR'", "ENCUESTA='FRONTUR'", sql, flags=re.IGNORECASE)
                    
                    # 🔥 CORREGIR ORDENAMIENTO PARA CONSULTAS TEMPORALES
                    temporal_keywords = ['evolución', 'mensual', 'anual', 'año', 'años', 'mes', 'meses', 'temporal', 'serie temporal', 'tendencia']
                    if any(keyword in natural_query.lower() for keyword in temporal_keywords):
                        if 'ORDER BY YM DESC' in sql.upper():
                            # Reemplazar ORDER BY YM DESC por ORDER BY YM ASC para orden cronológico
                            sql = re.sub(r'ORDER BY YM DESC', 'ORDER BY YM ASC', sql, flags=re.IGNORECASE)
                        elif 'ORDER BY MES DESC' in sql.upper():
                            sql = re.sub(r'ORDER BY MES DESC', 'ORDER BY MES ASC', sql, flags=re.IGNORECASE)
                        elif 'ORDER BY AÑO DESC' in sql.upper() or "ORDER BY ANIO DESC" in sql.upper():
                            sql = re.sub(r'ORDER BY (AÑO|ANIO) DESC', 'ORDER BY \\1 ASC', sql, flags=re.IGNORECASE)
                        elif 'ORDER BY' not in sql.upper() and any(col in sql.upper() for col in ['YM', 'AÑO', 'ANIO', 'MES']):
                            # Agregar ORDER BY ASC si no existe
                            if 'YM' in sql.upper():
                                sql += " ORDER BY YM ASC"
                            elif 'AÑO' in sql.upper() or 'ANIO' in sql.upper():
                                sql += " ORDER BY año ASC"
                    
                    result['sql'] = sql
                
                # 🔥 FORZAR DETECCIÓN DE GRÁFICOS
                current_chart_type = result.get('chart_type', '').lower()
                
                # Palabras clave para gráficos de línea (temporal)
                line_keywords = [
                    'evolución', 'mensual', 'anual', 'a lo largo del tiempo', 'serie temporal',
                    'tendencia', 'durante el año', 'por mes', 'por año', 'cronología',
                    'histórico', 'historico', 'en el tiempo', 'variación', 'variacion',
                    'desde el año', 'a lo largo de', 'progresión', 'progresion'
                ]
                
                # Palabras clave para gráficos de tarta
                pie_keywords = [
                    'distribución', 'distribucion', 'porcentaje', 'desglose', 'reparto', 
                    'composición', 'composicion', 'proporción', 'proporcion', 'participación',
                    'participacion', 'qué porcentaje', 'que porcentaje', 'cómo se distribuye',
                    'como se distribuye', 'repartición', 'reparticion', 'tarta', 'pie chart',
                    'por comunidades', 'por países', 'por pais', 'por motivo', 'por alojamiento',
                    'por tipo', 'por destino', 'composicion', 'reparto', 'qué parte', 'que parte',
                    'cuál es la distribución', 'cual es la distribucion', 'en qué porcentaje',
                    'en que porcentaje', 'división', 'division', 'clasificación', 'clasificacion'
                ]
                
                # Palabras clave para barras (ranking)
                bar_keywords = [
                    'top', 'ranking', 'principales', 'mayores', 'más altos', 'mas altos',
                    'más visitados', 'mas visitados', 'primeros', 'mejores', 'mayor número',
                    'lista de', 'clasificación de', 'clasificacion de', 'ordenados por'
                ]
                
                query_lower = natural_query.lower()
                
                # Prioridad: 1. Línea (temporal), 2. Tarta (distribución), 3. Barras (ranking)
                if any(keyword in query_lower for keyword in line_keywords):
                    result['chart_type'] = 'line'
                    print(f"🔍 DETECTADO: Gráfico de línea por palabras clave temporales")
                elif any(keyword in query_lower for keyword in pie_keywords):
                    result['chart_type'] = 'pie'
                    print(f"🔍 DETECTADO: Gráfico de tarta por palabras clave de distribución")
                elif any(keyword in query_lower for keyword in bar_keywords):
                    result['chart_type'] = 'bar'
                    print(f"🔍 DETECTADO: Gráfico de barras por palabras clave de ranking")
                elif not current_chart_type or current_chart_type == 'none':
                    # Si no se detectó nada, usar tabla por defecto
                    result['chart_type'] = 'table'
                    print(f"🔍 DETECTADO: Tabla por defecto")
                
                print(f"🎯 Gráfico final seleccionado: {result['chart_type']}")
                print(f"🔧 SQL corregido: {result['sql']}")
                    
                return result
            else:
                raise ValueError("No se pudo extraer JSON de la respuesta")
                
        except Exception as e:
            return {"sql": "", "explanation": f"Error: {e}", "chart_type": "table"}

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
                self.logger.info(f"Primeras filas:\n{df.head()}")
            
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

class Visualizer:
    def __init__(self, logger):
        self.logger = logger
        plt.style.use('default')
        sns.set_palette("husl")
        # Configurar fuente para caracteres especiales
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def create_chart(self, df: pd.DataFrame, chart_type: str, title: str) -> bool:
        """Crear gráfico - VERSIÓN CORREGIDA con mejor logging"""
        if df.empty:
            self.logger.warning("DataFrame vacío, no se puede crear gráfico")
            return False
        
        self.logger.info(f"Intentando crear gráfico tipo: {chart_type}")
        self.logger.info(f"DataFrame shape: {df.shape}")
        self.logger.info(f"Columnas: {df.columns.tolist()}")
        self.logger.info(f"Datos para gráfico:\n{df}")
        
        try:
            # RESET INDEX CRÍTICO - Esto evita el error de indexación ambiguo
            df_clean = df.reset_index(drop=True).copy()
            
            # PREPROCESAR DATOS PARA GRÁFICOS - CORREGIDO
            df_clean = self._preprocess_data_for_chart(df_clean)
            
            # Asegurar que las columnas numéricas sean del tipo correcto
            for col in df_clean.select_dtypes(include=['number']).columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            success = False
            if chart_type == 'bar' and len(df_clean) > 1:
                self.logger.info("Creando gráfico de barras")
                success = self._create_bar_chart(df_clean, ax, title)
            elif chart_type == 'line' and len(df_clean) > 1:
                self.logger.info("Creando gráfico de línea")
                success = self._create_line_chart(df_clean, ax, title)
            elif chart_type == 'pie' and len(df_clean) >= 1:
                self.logger.info("Creando gráfico de torta")
                success = self._create_pie_chart(df_clean, ax, title)
            else:
                self.logger.info("Creando tabla de display")
                success = self._create_table_display(df_clean, ax, title)
            
            if success:
                plt.tight_layout()
                plt.show()
                self.logger.info("Gráfico creado exitosamente")
                return True
            else:
                plt.close(fig)
                self.logger.warning("No se pudo crear el gráfico, falló el método específico")
                return False
                    
        except Exception as e:
            self.logger.error(f"Error creando gráfico: {e}")
            # Fallback a tabla - USAR df EN LUGAR DE df_clean
            try:
                self.logger.info("Intentando fallback a tabla")
                fig, ax = plt.subplots(figsize=(12, 6))
                # Usar el DataFrame original (df) en lugar de df_clean que puede no estar definido
                self._create_table_display(df.reset_index(drop=True), ax, f"{title} (fallback a tabla)")
                plt.tight_layout()
                plt.show()
                return True
            except Exception as fallback_error:
                self.logger.error(f"Error en fallback a tabla: {fallback_error}")
                return False
    
    def _preprocess_data_for_chart(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesar datos para gráficos - CORREGIDO para problemas de texto"""
        df_processed = df.copy()
        
        # Limpiar textos de comunidades autónomas
        for col in df_processed.select_dtypes(include=['object']).columns:
            df_processed[col] = df_processed[col].astype(str).apply(self._clean_text)
        
        # Convertir meses YYYY-MM a formato más legible
        if 'mes' in df_processed.columns:
            df_processed['mes_display'] = df_processed['mes'].apply(self._format_month)
        elif 'YM' in df_processed.columns:
            df_processed['mes_display'] = df_processed['YM'].apply(self._format_month)
        
        # Asegurar orden cronológico si hay columna YM
        if 'YM' in df_processed.columns:
            df_processed = df_processed.sort_values('YM', ascending=True)
            self.logger.info(f"DataFrame ordenado por YM: {df_processed['YM'].tolist()}")
        
        return df_processed
    
    def _clean_text(self, text: str) -> str:
        """Limpiar texto para mostrar correctamente en gráficos"""
        if pd.isna(text) or text == 'nan':
            return 'N/A'
        
        # Reemplazar caracteres problemáticos
        text = str(text).strip()
        
        # Abreviar textos muy largos (especialmente importante para gráficos de torta)
        if len(text) > 25:
            words = text.split()
            if len(words) > 3:
                text = ' '.join(words[:3]) + '...'
            else:
                # Si tiene pocas palabras pero es largo, truncar
                text = text[:22] + '...'
        
        return text
    
    def _format_month(self, month_str: str) -> str:
        """Convertir formato YYYY-MM a formato legible"""
        if pd.isna(month_str) or month_str == 'nan':
            return 'N/A'
        
        try:
            # Para formato YYYY-MM
            if len(month_str) == 7 and '-' in month_str:
                year, month = month_str.split('-')
                month_names = {
                    '01': 'Ene', '02': 'Feb', '03': 'Mar', '04': 'Abr',
                    '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Ago',
                    '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dic'
                }
                return f"{month_names.get(month, month)} {year}"
            else:
                return str(month_str)
        except:
            return str(month_str)
    
    def _create_bar_chart(self, df: pd.DataFrame, ax, title: str) -> bool:
        """Crear gráfico de barras - CORREGIDO para problemas de texto"""
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(exclude=['number']).columns
            
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                
                # Usar columna display si existe
                if 'mes_display' in df.columns and x_col == 'mes':
                    x_labels = df['mes_display'].values
                else:
                    x_labels = df[x_col].values
                
                # Ordenar por valor para mejor visualización
                df_sorted = df.sort_values(y_col, ascending=False)
                
                # Usar índices numéricos para evitar problemas de indexación
                x_positions = range(len(df_sorted))
                
                bars = ax.bar(x_positions, df_sorted[y_col], color='skyblue', alpha=0.8)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_ylabel('Número de Turistas')
                
                # Configurar etiquetas del eje X
                ax.set_xticks(x_positions)
                
                # Usar etiquetas procesadas
                if 'mes_display' in df_sorted.columns and x_col == 'mes':
                    display_labels = df_sorted['mes_display'].values
                else:
                    display_labels = df_sorted[x_col].apply(self._clean_text).values
                
                ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=9)
                
                # Añadir valores en las barras
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    if pd.notnull(height) and height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{height:,.0f}', ha='center', va='bottom', fontsize=8)
                
                # Formatear eje Y
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K' if x >= 1000 else f'{x:.0f}'))
                
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error en gráfico de barras: {e}")
            return False
    
    def _create_line_chart(self, df: pd.DataFrame, ax, title: str) -> bool:
        """Crear gráfico de líneas para series temporales - MEJORADO para años"""
        try:
            # RESET INDEX CRÍTICO - Evita el error "truth value of Index is ambiguous"
            df_plot = df.reset_index(drop=True).copy()
            
            # PREPROCESAR DATOS
            df_plot = self._preprocess_data_for_chart(df_plot)
            
            # Identificar columnas de tiempo y valor
            time_col, value_col, category_col = self._identify_chart_columns(df_plot)
            
            if not time_col or not value_col:
                self.logger.warning("No se pudieron identificar columnas para gráfico de líneas")
                return False
            
            # Preparar datos temporales
            df_plot = self._prepare_time_data(df_plot, time_col)
            
            # Si hay columna de categoría, crear múltiples líneas
            if category_col and len(df_plot[category_col].unique()) > 1:
                return self._create_multi_line_chart(df_plot, time_col, value_col, category_col, ax, title)
            else:
                return self._create_single_line_chart(df_plot, time_col, value_col, ax, title)
                
        except Exception as e:
            self.logger.error(f"Error en gráfico de líneas: {e}")
            return False
    
    def _identify_chart_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Identificar automáticamente columnas para el gráfico"""
        time_columns = ['anio', 'mes', 'YM', 'ID_FECHA', 'fecha', 'year', 'month', 'año']
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # Buscar columna de tiempo
        time_col = None
        for col in time_columns:
            if col in df.columns:
                time_col = col
                break
        
        # Buscar columna de valor (excluir posibles índices)
        value_col = None
        for col in numeric_cols:
            if col not in time_columns and col not in ['index', 'level_0']:
                value_col = col
                break
        if not value_col and numeric_cols:
            value_col = numeric_cols[0]
        
        # Buscar columna de categoría
        category_col = None
        for col in categorical_cols:
            if col != time_col and len(df[col].unique()) > 1 and len(df[col].unique()) <= 10:
                category_col = col
                break
        
        return time_col, value_col, category_col
    
    def _prepare_time_data(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """Preparar datos temporales para plotting - MEJORADO para años"""
        df_prepared = df.copy()
        
        # Eliminar filas con time_col nula
        df_prepared = df_prepared.dropna(subset=[time_col])
        
        if df_prepared.empty:
            return df_prepared
        
        if time_col == 'YM':
            # Convertir YM a datetime para ordenar correctamente
            df_prepared['YM_datetime'] = pd.to_datetime(df_prepared[time_col] + '-01', errors='coerce')
            # Eliminar conversiones fallidas
            df_prepared = df_prepared.dropna(subset=['YM_datetime'])
            df_prepared = df_prepared.sort_values('YM_datetime', ascending=True)
        elif time_col in ['anio', 'year', 'año']:
            # Convertir a numérico y ordenar
            df_prepared[time_col] = pd.to_numeric(df_prepared[time_col], errors='coerce')
            df_prepared = df_prepared.dropna(subset=[time_col])
            df_prepared = df_prepared.sort_values(time_col, ascending=True)
        elif time_col == 'mes':
            # Ordenar por YM si existe, sino por el mes directamente
            if 'YM' in df_prepared.columns:
                df_prepared = df_prepared.sort_values('YM', ascending=True)
            else:
                df_prepared = df_prepared.sort_values(time_col, ascending=True)
        
        return df_prepared
    
    def _create_single_line_chart(self, df: pd.DataFrame, time_col: str, value_col: str, ax, title: str) -> bool:
        """Crear gráfico de línea simple - CORREGIDO para orden temporal correcto"""
        try:
            # Hacer una copia y limpiar datos nulos CRÍTICO
            df_clean = df.copy()
            
            # Eliminar filas con valores nulos en las columnas críticas
            df_clean = df_clean.dropna(subset=[time_col, value_col])
            
            if df_clean.empty:
                self.logger.warning("DataFrame vacío después de limpiar NaN")
                return False
            
            # ORDENAR POR TIEMPO CORRECTAMENTE - ORDEN ASCENDENTE para líneas temporales
            if time_col == 'mes' and 'YM' in df_clean.columns:
                # Si tenemos YM, usarlo para ordenar correctamente (orden ascendente)
                df_clean = df_clean.sort_values('YM', ascending=True)
            elif time_col == 'mes':
                # Ordenar meses cronológicamente (del más antiguo al más reciente)
                df_clean = df_clean.sort_values(time_col, ascending=True)
            else:
                # Ordenar por la columna de tiempo (ascendente)
                df_clean = df_clean.sort_values(time_col, ascending=True)
            
            # Reset index después de ordenar
            df_clean = df_clean.reset_index(drop=True)
            
            # Usar índice numérico para el plotting
            x_values = range(len(df_clean))
            y_values = df_clean[value_col].values
            
            # Crear línea principal
            line = ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=6, 
                          color='#2E86AB', alpha=0.8)[0]
            
            # Añadir puntos y valores (solo si no son NaN)
            for i, (x, y) in enumerate(zip(x_values, y_values)):
                if pd.notnull(y) and y > 0:  # Solo añadir texto si el valor no es NaN y es positivo
                    ax.text(x, y, f'{y:,.0f}', ha='center', va='bottom', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
            
            # Configurar ejes
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Número de Turistas')
            
            # Configurar etiquetas del eje X - USAR FORMATO MEJORADO
            if 'mes_display' in df_clean.columns:
                x_labels = df_clean['mes_display'].values
            else:
                x_labels = []
                for val in df_clean[time_col].values:
                    if pd.notnull(val):
                        x_labels.append(self._clean_text(str(val)))
                    else:
                        x_labels.append('N/A')
            
            ax.set_xticks(x_values)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
            ax.set_xlabel('Tiempo')
            
            # Grid y formato
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K' if x >= 1000 else f'{x:.0f}'))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en gráfico de línea simple: {e}")
            return False
    
    def _create_multi_line_chart(self, df: pd.DataFrame, time_col: str, value_col: str, 
                               category_col: str, ax, title: str) -> bool:
        """Crear gráfico de líneas múltiples - CORREGIDO para NaN"""
        try:
            # Limpiar datos nulos primero
            df_clean = df.dropna(subset=[time_col, value_col, category_col]).copy()
            
            if df_clean.empty:
                self.logger.warning("DataFrame vacío después de limpiar NaN")
                return False
                
            categories = df_clean[category_col].unique()
            colors = plt.cm.Set3(range(len(categories)))
            
            # Para cada categoría, trazar una línea
            for i, category in enumerate(categories):
                category_data = df_clean[df_clean[category_col] == category].copy()
                category_data = category_data.sort_values(time_col, ascending=True)  # ORDEN ASCENDENTE
                
                # Usar índice numérico
                x_values = range(len(category_data))
                y_values = category_data[value_col].values
                
                ax.plot(x_values, y_values, marker='o', linewidth=2, markersize=5,
                       color=colors[i], label=str(category), alpha=0.8)
                
                # Añadir etiquetas de valores (solo si no son NaN)
                for x, y in zip(x_values, y_values):
                    if pd.notnull(y) and y > 0:
                        ax.text(x, y, f'{y:,.0f}', ha='center', va='bottom', fontsize=7)
            
            # Configurar el gráfico
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('Número de Turistas')
            
            # Configurar etiquetas del eje X
            ref_data = df_clean[df_clean[category_col] == categories[0]].sort_values(time_col, ascending=True)
            
            if 'mes_display' in ref_data.columns:
                x_labels = ref_data['mes_display'].values
            else:
                x_labels = []
                for val in ref_data[time_col].values:
                    if pd.notnull(val):
                        x_labels.append(self._clean_text(str(val)))
                    else:
                        x_labels.append('N/A')
            
            ax.set_xticks(range(len(ref_data)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
            ax.set_xlabel('Tiempo')
            
            ax.legend(title=category_col, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K' if x >= 1000 else f'{x:.0f}'))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en gráfico de líneas múltiples: {e}")
            return False
    
    def _create_pie_chart(self, df: pd.DataFrame, ax, title: str) -> bool:
        """Crear gráfico de torta - MEJORADO para desgloses"""
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(exclude=['number']).columns
            
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                label_col = categorical_cols[0]
                value_col = numeric_cols[0]
                
                # Limitar a 10 categorías máximo para mejor visualización
                df_plot = df.head(10).copy()
                df_plot = df_plot.reset_index(drop=True)
                
                # Calcular porcentajes
                total = df_plot[value_col].sum()
                if total == 0:
                    self.logger.warning("Suma total es cero, no se puede crear gráfico de torta")
                    return False
                
                df_plot['porcentaje'] = (df_plot[value_col] / total) * 100
                
                # Limpiar etiquetas y preparar datos
                labels = []
                sizes = []
                colors = plt.cm.Pastel2(range(len(df_plot)))
                
                for _, row in df_plot.iterrows():
                    label = self._clean_text(str(row[label_col]))
                    # Añadir porcentaje a la etiqueta
                    porcentaje = row['porcentaje']
                    label_with_pct = f"{label} ({porcentaje:.1f}%)"
                    labels.append(label_with_pct)
                    sizes.append(row[value_col])
                
                # Crear gráfico de torta
                wedges, texts, autotexts = ax.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 9},
                    colors=colors,
                    pctdistance=0.85
                )
                
                # Mejorar legibilidad
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(8)
                
                # Añadir un círculo en el centro para hacer un donut (opcional)
                centre_circle = plt.Circle((0,0),0.70,fc='white')
                ax.add_artist(centre_circle)
                
                # Añadir título y información adicional
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                ax.text(0, -1.2, f'Total: {total:,.0f} turistas', ha='center', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
                
                # Equal aspect ratio asegura que el pie se dibuje como un círculo
                ax.axis('equal')
                
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error en gráfico de torta: {e}")
            return False
    
    def _create_table_display(self, df: pd.DataFrame, ax, title: str) -> bool:
        """Crear visualización de tabla - CORREGIDO"""
        try:
            ax.axis('tight')
            ax.axis('off')
            
            # Reset index para evitar problemas
            df_display = df.reset_index(drop=True).copy()
            
            # Formatear números para mejor legibilidad
            for col in df_display.select_dtypes(include=['number']).columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f'{x:,.0f}' if pd.notnull(x) and abs(x) >= 1000 else 
                             f'{x:.1f}' if pd.notnull(x) else ''
                )
            
            # Limpiar textos
            for col in df_display.select_dtypes(include=['object']).columns:
                df_display[col] = df_display[col].apply(self._clean_text)
            
            # Crear tabla
            table = ax.table(
                cellText=df_display.values,
                colLabels=df_display.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.15] * len(df_display.columns)
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # Estilo de la tabla
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Encabezados
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                elif i % 2 == 0:  # Filas pares
                    cell.set_facecolor('#f0f0f0')
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            return True
            
        except Exception as e:
            self.logger.error(f"Error creando tabla: {e}")
            return False

class FronturLLMAgent:
    def __init__(self, db_path: str, openai_api_key: str, config_path: str = "frontur_agent_config.json"):
        # Convertir a ruta absoluta y verificar
        self.db_path = os.path.abspath(db_path)
        print(f"📁 Conectando a base de datos: {self.db_path}")
        
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"❌ No se encuentra la base de datos: {self.db_path}")
        
        self.client = OpenAI(api_key=openai_api_key)
        self.connection = sqlite3.connect(self.db_path)
        
        self.config = self.load_config(config_path)
        self.metadata = self.load_metadata("frontur_metadata_completo.json")
        self.examples = self.load_examples("frontur_examples.json")
        
        self.setup_logging()
        
        self.sql_generator = SQLGenerator(self.client, self.config, self.metadata, self.examples)
        self.query_executor = QueryExecutor(self.connection, self.logger)
        self.visualizer = Visualizer(self.logger)
        
        print("✅ Agente FRONTUR inicializado correctamente")
    
    def load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Archivo de configuración {config_path} no encontrado")
            return {}
    
    def load_metadata(self, metadata_path: str) -> Dict:
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Archivo de metadatos {metadata_path} no encontrado")
            return {}
    
    def load_examples(self, examples_path: str) -> List[Dict]:
        try:
            with open(examples_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Archivo de ejemplos {ejemplos.json} no encontrado")
            return []
    
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
            llm_result = self.sql_generator.generate_sql(natural_query)
            
            if not llm_result.get('sql'):
                return {
                    'success': False,
                    'error': f"No se pudo generar SQL: {llm_result.get('explanation', 'Error desconocido')}",
                    'data': None,
                    'chart_created': False
                }
            
            df, error = self.query_executor.execute_sql(llm_result['sql'])
            
            if error:
                return {
                    'success': False,
                    'error': error,
                    'data': None,
                    'chart_created': False
                }
            
            chart_created = False
            if not df.empty and llm_result.get('chart_type'):
                chart_title = f"Resultado: {natural_query}"
                chart_created = self.visualizer.create_chart(df, llm_result['chart_type'], chart_title)
            
            return {
                'success': True,
                'data': df,
                'sql_generated': llm_result['sql'],
                'explanation': llm_result.get('explanation', ''),
                'chart_created': chart_created,
                'chart_type': llm_result.get('chart_type')
            }
            
        except Exception as e:
            error_msg = f"Error procesando consulta: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'data': None,
                'chart_created': False
            }
    
    def interactive_mode(self):
        print("🏨 AGENTE LLM FRONTUR - SISTEMA DE CONSULTAS TURÍSTICAS")
        print("=" * 70)
        print("Escribe tu consulta en español o 'salir' para terminar")
        print("Comandos especiales: 'ejemplos', 'estadisticas'")
        print("\n💡 CONSULTAS SUGERIDAS:")
        print("   📈 Línea: 'Evolución anual de turistas italianos desde 2020'")
        print("   🥧 Tarta: 'Distribución de turistas por comunidades autónomas'")
        print("   📊 Barras: 'Top 5 países con más turistas en 2024'")
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
            
            if result['chart_created']:
                print(f"📈 Gráfico {result['chart_type']} generado correctamente")
            else:
                print("📋 Resultados mostrados en tabla (gráfico no generado)")
                
            print(f"\n🔍 SQL ejecutado: {result['sql_generated']}")
        else:
            print(f"\n❌ ERROR: {result['error']}")
    
    def show_examples(self):
        print("\n📚 EJEMPLOS DE CONSULTAS:")
        examples_by_type = {
            "📈 Gráficos de Línea": [
                "Evolución mensual de turistas franceses en Madrid",
                "Turistas italianos por año desde 2020",
                "Serie temporal de turistas en Cataluña"
            ],
            "🥧 Gráficos de Tarta": [
                "Distribución de turistas por comunidades autónomas",
                "Desglose de turistas por motivo de viaje", 
                "Porcentaje de turistas por tipo de alojamiento"
            ],
            "📊 Gráficos de Barras": [
                "Top 5 países con más turistas en 2024",
                "Principales comunidades autónomas por turistas",
                "Ranking de tipos de alojamiento más usados"
            ]
        }
        
        for chart_type, examples in examples_by_type.items():
            print(f"\n{chart_type}:")
            for i, example in enumerate(examples):
                print(f"   {i+1}. {example}")
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
        print(f"   • Motivos de viaje: {self.metadata['column_values']['MOTIVO_VIAJE']['total_unique']}")
    
    def close(self):
        if self.connection:
            self.connection.close()

def main():
    DB_PATH = r"C:\DataEstur_Data\dataestur.db"
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        print("🚀 Iniciando Agente LLM FRONTUR...")
        print(f"📁 Base de datos: {DB_PATH}")
        
        # Verificar que la base de datos existe
        if not os.path.exists(DB_PATH):
            print(f"❌ ERROR: No se encuentra la base de datos en {DB_PATH}")
            print("💡 Verifica la ruta del archivo dataestur.db")
            return
        
        agent = FronturLLMAgent(DB_PATH, OPENAI_API_KEY)
        
        if not agent.config:
            print("⚠️  No se pudo cargar la configuración")
        if not agent.metadata:
            print("⚠️  No se pudo cargar los metadatos")
        if not agent.examples:
            print("⚠️  No se pudo cargar los ejemplos")
        
        agent.interactive_mode()
        
    except sqlite3.OperationalError as e:
        print(f"❌ Error de base de datos: {e}")
        print(f"   Verifica que la ruta {DB_PATH} sea correcta")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
    finally:
        if 'agent' in locals():
            agent.close()

if __name__ == "__main__":
    main()