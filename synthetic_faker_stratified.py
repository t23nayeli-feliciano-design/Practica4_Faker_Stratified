from faker import Faker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Configuración de Semillas para Reproducibilidad Experimental
fake = Faker('es_MX')
Faker.seed(42)
np.random.seed(42)

def generar_y_particionar_datos():
    # 2. Generación programática de dataset sintético (1,000 registros)
    registros = []
    for i in range(1000):
        registros.append({
            'id_cliente': i + 1,
            'nombre': fake.name(),
            'email': fake.email(),
            'direccion': fake.address().replace('\n', ', '),  # Campo extendido 1
            'telefono': fake.phone_number(),                  # Campo extendido 2
            'estado_civil': np.random.choice(['Soltero', 'Casado', 'Divorciado'], p=[0.5, 0.4, 0.1]),
            'ingreso_anual': round(np.random.uniform(100000, 1000000), 2),
            'churn': np.random.choice([0, 1], p=[0.8, 0.2])  # Target desbalanceado (20% Churn)
        })

    df_sintetico = pd.DataFrame(registros)

    print("==================================================")
    print("=== 1. MUESTRA DEL DATASET SINTÉTICO GENERADO ===")
    print("==================================================")
    print(df_sintetico.head())

    # 3. Codificación de Variables Categóricas (One-Hot Encoding)
    df_encoded = pd.get_dummies(df_sintetico, columns=['estado_civil'], drop_first=True)

    # 4. Separación de Matriz de Características (X) y Vector Objetivo (y)
    # Eliminamos identificadores personales únicos (PII sintético) y la etiqueta target
    X = df_encoded.drop(columns=['id_cliente', 'nombre', 'email', 'direccion', 'telefono', 'churn'])
    y = df_encoded['churn']

    # 5. Partición Estratificada (Train 80%, Test 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 6. Verificación de Proporciones (Análisis de Desbalance)
    prop_total = y.mean()
    prop_train = y_train.mean()
    prop_test = y_test.mean()

    print("\n==================================================")
    print("=== 2. DIMENSIONES Y PROPORCIÓN DE CHURN ===")
    print("==================================================")
    print(f"Dataset Completo  | Forma: {df_sintetico.shape}  | Proporción Churn: {prop_total:.4f} ({prop_total*100:.2f}%)")
    print(f"Conjunto Train    | Forma: {X_train.shape}       | Proporción Churn: {prop_train:.4f} ({prop_train*100:.2f}%)")
    print(f"Conjunto Test     | Forma: {X_test.shape}        | Proporción Churn: {prop_test:.4f} ({prop_test*100:.2f}%)")

    # Guardar resumen comparativo a CSV para evidencia
    df_resumen = pd.DataFrame({
        'Conjunto': ['Dataset Completo (100%)', 'Entrenamiento (Train 80%)', 'Prueba (Test 20%)'],
        'Total_Registros': [len(y), len(y_train), len(y_test)],
        'Casos_Churn_1': [y.sum(), y_train.sum(), y_test.sum()],
        'Tasa_Churn_Proporcion': [f"{prop_total*100:.2f}%", f"{prop_train*100:.2f}%", f"{prop_test*100:.2f}%"]
    })
    print("\n==================================================")
    print("=== 3. TABLA COMPARATIVA DE ESTRATIFICACIÓN ===")
    print("==================================================")
    print(df_resumen.to_string(index=False))

if __name__ == '__main__':
    generar_y_particionar_datos()
