import sqlite3
import json

def get_table_schema(db_path, table_name):
    """Obtener esquema completo de una tabla"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Obtener información de columnas
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    
    schema = {
        "table_name": table_name,
        "columns": []
    }
    
    for col in columns:
        column_info = {
            "cid": col[0],
            "name": col[1],
            "type": col[2],
            "notnull": bool(col[3]),
            "default_value": col[4],
            "pk": bool(col[5])
        }
        schema["columns"].append(column_info)
    
    conn.close()
    return schema

# Uso
db_path = "C:/DataEstur_Data/dataestur.db"
table_name = "SURVEY_TABLE"
schema = get_table_schema(db_path, table_name)

# Guardar como JSON
with open("esquema_tabla.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, ensure_ascii=False)

print("✅ Esquema guardado en esquema_agent_tabla.json")