import sqlite3
import json
import pandas as pd
from datetime import datetime

def get_complete_metadata(db_path, table_name):
    """Obtener esquema y metadatos completos de una tabla"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Esquema de la tabla
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    
    schema = {
        "table": table_name,
        "extraction_date": datetime.now().isoformat(),
        "columns": [],
        "statistics": {}
    }
    
    # Procesar columnas
    for col in columns_info:
        column_data = {
            "name": col[1],
            "type": col[2],
            "not_null": bool(col[3]),
            "default_value": col[4],
            "primary_key": bool(col[5])
        }
        schema["columns"].append(column_data)
    
    # 2. Estadísticas básicas
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_records = cursor.fetchone()[0]
    schema["statistics"]["total_records"] = total_records
    
    # 3. Metadatos por columna
    schema["column_values"] = {}
    
    for col in schema["columns"]:
        col_name = col["name"]
        print(f"📊 Procesando columna: {col_name}")
        
        # Valores únicos y top values
        try:
            cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 20")
            distinct_values = [row[0] for row in cursor.fetchall()]
            
            # Frecuencia de valores (top 10)
            cursor.execute(f"""
                SELECT {col_name}, COUNT(*) as count 
                FROM {table_name} 
                WHERE {col_name} IS NOT NULL 
                GROUP BY {col_name} 
                ORDER BY count DESC 
                LIMIT 10
            """)
            top_values = [{"value": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            schema["column_values"][col_name] = {
                "total_unique": len(distinct_values),
                "sample_values": distinct_values[:10],
                "top_values": top_values
            }
            
        except Exception as e:
            print(f"⚠️ Error procesando columna {col_name}: {e}")
            schema["column_values"][col_name] = {"error": str(e)}
    
    # 4. Información de índices
    cursor.execute(f"PRAGMA index_list({table_name})")
    indexes = cursor.fetchall()
    schema["indexes"] = [index[1] for index in indexes]
    
    conn.close()
    return schema

# Uso
db_path = "C:/DataEstur_Data/dataestur.db"
table_name = "SURVEY_TABLE"

metadata = get_complete_metadata(db_path, table_name)

# Guardar como JSON
with open("metadata_completo.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("✅ Metadata completo guardado en tabla_metadata_completo.json")