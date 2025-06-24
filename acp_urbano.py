import geopandas as gpd
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar el GeoDataFrame con variables normalizadas
ruta_original = r"C:\Users\ana_g\Desktop\tfm2\parcelas_disueltas_pct2_normalizado_prueba.gpkg"
df = gpd.read_file(ruta_original)

# Definir las variables urbanas 
variables_urbanas = [
    'antiguedad_zscore', 'usos_cod_zscore', 'planta_baja_vulnerable_zscore',
    'elemento_sensible_zscore', 'centro_educativo_zscore', 'centro_inclusivo_zscore',
    'centro_salud_zscore', 'farmacia_zscore'
]

# Eliminar filas con NaN en las variables seleccionadas
df_no_nan = df.dropna(subset=variables_urbanas)

# Aplicar PCA
X = df_no_nan[variables_urbanas]
pca = PCA()
X_pca = pca.fit_transform(X)

# Varianza explicada
explained_variance = pca.explained_variance_ratio_ * 100
cumulative_variance = np.cumsum(explained_variance)

print(" Porcentaje de varianza explicada por componente:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.2f}%")

# Visualización: Varianza explicada y acumulada
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o', label='Varianza individual')
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='s', linestyle='--', label='Varianza acumulada')
plt.axhline(y=10, color='gray', linestyle=':')
plt.title('Varianza explicada por componente (Variables urbanas)')
plt.xlabel('Componente principal')
plt.ylabel('Varianza (%)')
plt.legend()
plt.grid()
plt.show()

#Cargas factoriales
loadings = pd.DataFrame(
    pca.components_.T,
    index=variables_urbanas,
    columns=[f'PC{i+1}' for i in range(len(variables_urbanas))]
)

# Filtrar componentes con eigenvalue > 1 (criterio de Kaiser)
eigenvalues = pca.explained_variance_
num_components = sum(eigenvalues > 1)
print(f"\n✅ Componentes retenidos según el criterio de Kaiser (>1): {num_components}")
print("Eigenvalues:")
print(eigenvalues)

# Mostrar tabla de cargas factoriales
loadings_filtrados = loadings.iloc[:, :num_components]
print("\nCargas factoriales (loadings):")
print(loadings_filtrados.round(3))

# Añadir scores al GeoDataFrame
for i in range(num_components):
    df_no_nan[f'PC{i+1}_urbano_score'] = X_pca[:, i]

# Crear el índice urbano sumando los componentes retenidos
df_no_nan['indice_urbano'] = df_no_nan[[f'PC{i+1}_urbano_score' for i in range(num_components)]].sum(axis=1)

# Guardar el resultado
ruta_salida = r'C:\Users\ana_g\Desktop\TFM\parcelas_con_scores_urbanos.gpkg'
df_no_nan.to_file(ruta_salida, driver='GPKG')
