import geopandas as gpd
from sklearn.preprocessing import StandardScaler

#Rutas de entrada
ruta_base = r'C:\Users\ana_g\Desktop\tfm2\parcelas_disueltas_pct2_normalizado_prueba.gpkg'
ruta_social = r'C:\Users\ana_g\Desktop\TFM\parcelas_con_scores_sociales.gpkg'
ruta_urbano = r'C:\Users\ana_g\Desktop\TFM\parcelas_con_scores_urbanos.gpkg'

# Leer archivos
gdf_base = gpd.read_file(ruta_base)
gdf_social = gpd.read_file(ruta_social)[['geometry', 'indice_social']]
gdf_urbano = gpd.read_file(ruta_urbano)[['geometry', 'indice_urbano']]

#Unir con la capa base (mantener todas las parcelas)
gdf = gdf_base.merge(gdf_social, on='geometry', how='left')
gdf = gdf.merge(gdf_urbano, on='geometry', how='left')

#Normalizar ambos índices con z-score
scaler = StandardScaler()
gdf[['indice_social_z', 'indice_urbano_z']] = scaler.fit_transform(
    gdf[['indice_social', 'indice_urbano']]
)

# Calcular el índice final
gdf['indice_vulnerabilidad_final'] = gdf['indice_social_z'] + gdf['indice_urbano_z']

#. Guardar el resultado
ruta_salida = r'C:\Users\ana_g\Desktop\TFM\indice_vulnerabilidad_final1.gpkg'
gdf.to_file(ruta_salida, driver='GPKG')

