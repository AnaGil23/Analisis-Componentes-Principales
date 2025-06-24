import geopandas as gpd
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el GeoDataFrame con variables normalizadas
ruta_original = r'C:\Users\ana_g\Desktop\tfm2\parcelas_disueltas_pct2_normalizado_prueba.gpkg'
df = gpd.read_file(ruta_original)

# Definir las variables normalizadas a incluir en el PCA
variables_sociales = [
    'pob_menor5_pct_zscore', 'pob_mayor65_pct_zscore', 'extran_vulner_pct_zscore',
    'mujeres_pct_zscore', 'ocup_elementales_pct_zscore', 'cuenta_propia_pct_zscore',
    'parados_pct_zscore', 'estudios_primaria_pct_zscore', 'densidad_poblacion_zscore',
    'renta_total_estim_parcela_zscore'
]

# Eliminar filas con NaN en las variables seleccionadas
df_no_nan = df.dropna(subset=variables_sociales)

# Aplicar PCA
X = df_no_nan[variables_sociales]
pca = PCA()
X_pca = pca.fit_transform(X)

# Porcentaje de varianza explicada
explained_variance = pca.explained_variance_ratio_ * 100
print("Porcentaje de varianza explicada por componente:")
for i, var in enumerate(explained_variance):
    print(f"PC{i+1}: {var:.2f}%")

# Visualización de la varianza explicada
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.axhline(y=10, color='gray', linestyle='--', label='Referencia 10%')
plt.title('Porcentaje de varianza explicada')
plt.xlabel('Componente principal')
plt.ylabel('Varianza (%)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Cargas factoriales (loadings)
loadings = pd.DataFrame(
    pca.components_.T,
    index=variables_sociales,
    columns=[f'PC{i+1}' for i in range(len(variables_sociales))]
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
    df_no_nan[f'PC{i+1}_score'] = X_pca[:, i]

# Crear el índice social sumando los tres primeros componentes
df_no_nan['indice_social'] = df_no_nan['PC1_score'] + df_no_nan['PC2_score'] + df_no_nan['PC3_score']

# Guardar el GeoDataFrame con los scores y el índice social
ruta_salida = r'C:\Users\ana_g\Desktop\TFM\parcelas_con_scores_sociales.gpkg'
df_no_nan.to_file(ruta_salida, driver='GPKG')
