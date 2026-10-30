import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from kmodes.kprototypes import KPrototypes
import os

# Configuración de página
st.set_page_config(
    page_title="Segmentación de Clientes · Marketing Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo sutil para métricas y encabezados
st.markdown("""
<style>
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Funciones de carga y preprocesamiento con caché
# -------------------------------------------------------------
@st.cache_data
def load_default_data():
    file_path = "segmentos_notebook.csv"
    if not os.path.exists(file_path):
        file_path = "clientes_retail.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        rename_map = {
            'm#id_cliente': 'id_cliente',
            'D#educacion': 'educacion',
            'D#estado_civil': 'estado_civil',
            'mD#respondio_ultima_campana': 'respondio'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df, os.path.basename(file_path)
    return None, None

@st.cache_data
def process_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        rename_map = {
            'm#id_cliente': 'id_cliente',
            'D#educacion': 'educacion',
            'D#estado_civil': 'estado_civil',
            'mD#respondio_ultima_campana': 'respondio'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return None

@st.cache_data
def run_all_clustering(df, n_clusters=3, eps=1.8, min_samples=15):
    num_cols = ['ingreso_anual', 'gasto_total', 'compras_tienda', 'compras_web',
                'compras_catalogo', 'visitas_web_mes', 'recencia_dias', 'hijos_pequenos', 'edad']
    cat_cols = ['educacion', 'estado_civil']
    
    # Validar presencia de columnas
    available_num = [c for c in num_cols if c in df.columns]
    available_cat = [c for c in cat_cols if c in df.columns]
    
    if len(available_num) < 2:
        return None, None, None
        
    scaler = StandardScaler()
    Z = scaler.fit_transform(df[available_num].fillna(df[available_num].median()))
    
    # 1. K-Means
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(Z)
    
    # 2. Jerárquico
    agg = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Z)
    
    # 3. DBSCAN
    dbs = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
    
    # 4. Mezclas Gaussianas (GMM)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=3).fit_predict(Z)
    
    # 5. K-Prototypes
    if len(available_cat) > 0:
        X_kp = pd.concat([
            pd.DataFrame(Z, columns=available_num),
            df[available_cat].astype(str).reset_index(drop=True)
        ], axis=1)
        cat_indices = list(range(len(available_num), len(available_num) + len(available_cat)))
        kp = KPrototypes(n_clusters=n_clusters, init='Cao', random_state=42, n_init=2)
        kp_labels = kp.fit_predict(X_kp.values, categorical=cat_indices)
    else:
        kp_labels = km.copy()

    labels_dict = {
        'K-Means': km,
        'Jerárquico': agg,
        'DBSCAN': dbs,
        'GMM': gmm,
        'K-Prototypes': kp_labels
    }
    
    # Métricas y tamaños
    metrics = []
    for name, labels in labels_dict.items():
        unique_labels = set(labels)
        n_c = len(unique_labels - ({-1} if -1 in unique_labels else set()))
        
        # Silueta sobre variables numéricas
        if n_c > 1 and len(unique_labels) < len(labels):
            # Para DBSCAN, calcular silueta omitiendo ruido si es posible
            if -1 in unique_labels:
                mask = labels != -1
                if len(set(labels[mask])) > 1:
                    sil = round(silhouette_score(Z[mask], labels[mask]), 3)
                else:
                    sil = "N/A"
            else:
                sil = round(silhouette_score(Z, labels), 3)
        else:
            sil = "N/A"
            
        counts = pd.Series(labels).value_counts().sort_index().to_dict()
        sizes_str = ", ".join([f"G{k}: {v}" if k != -1 else f"Ruido: {v}" for k, v in counts.items()])
        
        metrics.append({
            'Técnica': name,
            'N° Grupos': n_c,
            'Silueta': sil,
            'Distribución de Tamaños': sizes_str
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    # Matriz de concordancia (Adjusted Rand Index)
    ari_df = pd.DataFrame(index=labels_dict.keys(), columns=labels_dict.keys())
    for t1 in labels_dict:
        for t2 in labels_dict:
            ari_df.loc[t1, t2] = round(adjusted_rand_score(labels_dict[t1], labels_dict[t2]), 3)
            
    return labels_dict, metrics_df, ari_df


# -------------------------------------------------------------
# Interfaz de Usuario
# -------------------------------------------------------------
st.title("Segmentación de Clientes Retail")
st.caption("Herramienta interactiva para explorar, comparar y validar técnicas de segmentación para Marketing.")

# Sidebar: Carga de Datos y Parámetros
with st.sidebar:
    st.header("Fuente de Datos")
    uploaded_file = st.file_uploader(
        "Subir CSV de clientes",
        type=["csv"],
        help="Sube un archivo con segmentos asignados (desde Orange o notebook)."
    )
    
    df_default, default_name = load_default_data()
    
    if uploaded_file is not None:
        df = process_uploaded_file(uploaded_file)
        data_source_label = f"Archivo subido: `{uploaded_file.name}`"
    elif df_default is not None:
        df = df_default.copy()
        data_source_label = f"Archivo local: `{default_name}`"
    else:
        df = None
        data_source_label = "Sin datos disponibles"
        
    st.info(data_source_label)
    
    st.divider()
    st.header("Ajustes de Comparación")
    k_param = st.slider("Número de clusters (k)", min_value=2, max_value=6, value=3, step=1)
    
    with st.expander("Ajustes avanzados DBSCAN"):
        eps_param = st.slider("Epsilon (eps)", min_value=0.5, max_value=3.5, value=1.8, step=0.1)
        min_samples_param = st.slider("Min Samples", min_value=5, max_value=30, value=15, step=1)

if df is None:
    st.warning("Por favor sube un archivo CSV con datos de clientes o asegúrate de que 'segmentos_notebook.csv' o 'clientes_retail.csv' estén en la carpeta.")
    st.stop()

# Pestañas principales
tab_visor, tab_comparador, tab_recomendacion = st.tabs([
    "Visor de Grupos",
    "Comparador de Técnicas",
    "Recomendación de Marketing"
])

# -------------------------------------------------------------
# PARTE 1: VISOR DE GRUPOS
# -------------------------------------------------------------
with tab_visor:
    st.subheader("Visor de Segmentos Existentes")
    st.markdown("Analiza la composición de los grupos generados en tu archivo CSV (Orange o Notebook).")
    
    # Identificar columnas candidatas a cluster
    potential_clusters = [col for col in df.columns if any(kw in col.lower() for kw in ['cluster', 'seg_', 'grupo'])]
    if not potential_clusters:
        potential_clusters = df.columns.tolist()
        
    # Selección de columna de cluster
    default_idx = 0
    for kw in ['cluster', 'seg_kproto', 'seg_kmeans']:
        matches = [i for i, c in enumerate(potential_clusters) if kw in c.lower()]
        if matches:
            default_idx = matches[0]
            break
            
    col_sel1, col_sel2, col_sel3 = st.columns([1.5, 1, 1])
    with col_sel1:
        cluster_col = st.selectbox(
            "Columna de Segmento / Cluster:",
            potential_clusters,
            index=default_idx,
            help="Selecciona la columna que define el grupo al que pertenece cada cliente."
        )
        
    num_numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != cluster_col]
    default_x = "ingreso_anual" if "ingreso_anual" in num_numeric_cols else num_numeric_cols[0]
    default_y = "gasto_total" if "gasto_total" in num_numeric_cols else (num_numeric_cols[1] if len(num_numeric_cols) > 1 else num_numeric_cols[0])
    
    with col_sel2:
        x_axis = st.selectbox("Eje X (Dispersión):", num_numeric_cols, index=num_numeric_cols.index(default_x) if default_x in num_numeric_cols else 0)
    with col_sel3:
        y_axis = st.selectbox("Eje Y (Dispersión):", num_numeric_cols, index=num_numeric_cols.index(default_y) if default_y in num_numeric_cols else 0)

    # Convertir cluster_col a string para mejor visualización categórica
    df_view = df.copy()
    df_view[cluster_col] = df_view[cluster_col].astype(str)

    # 1. Fila de KPIs
    total_clientes = len(df_view)
    total_gasto = df_view['gasto_total'].sum() if 'gasto_total' in df_view.columns else 0
    n_grupos = df_view[cluster_col].nunique()
    
    with st.container(border=True):
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Clientes Analizados", f"{total_clientes:,}")
        col_m2.metric("Segmentos Detectados", f"{n_grupos}")
        if 'gasto_total' in df_view.columns:
            gasto_medio_global = df_view['gasto_total'].mean()
            col_m3.metric("Gasto Total Acumulado", f"${total_gasto:,.0f}")
            col_m4.metric("Gasto Promedio Global", f"${gasto_medio_global:.1f}")

    # 2. Gráfico de dispersión interactivo
    st.write("")
    fig_scatter = px.scatter(
        df_view,
        x=x_axis,
        y=y_axis,
        color=cluster_col,
        hover_data=[c for c in ['id_cliente', 'edad', 'hijos_pequenos', 'educacion'] if c in df_view.columns],
        title=f"Distribución de Clientes: {x_axis.replace('_', ' ').title()} vs {y_axis.replace('_', ' ').title()}",
        opacity=0.75,
        template="plotly_white",
        height=450
    )
    fig_scatter.update_layout(legend_title_text='Grupo / Cluster', margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_scatter)

    # 3. Tabla de perfil requerida
    st.subheader("Tabla de Perfil por Segmento")
    st.markdown("Resumen de valor y participación por grupo según los requisitos solicitados:")

    if 'gasto_total' in df_view.columns:
        resumen = df_view.groupby(cluster_col).agg(
            clientes=('gasto_total', 'count'),
            gasto_promedio=('gasto_total', 'mean'),
            gasto_total_grupo=('gasto_total', 'sum')
        ).reset_index()

        resumen['%_clientes'] = (resumen['clientes'] / total_clientes * 100).round(1)
        resumen['%_del_gasto'] = (resumen['gasto_total_grupo'] / total_gasto * 100).round(1)
        resumen['gasto_promedio'] = resumen['gasto_promedio'].round(1)

        # Variables adicionales si existen
        extra_aggs = {}
        if 'ingreso_anual' in df_view.columns:
            extra_aggs['ingreso_promedio'] = ('ingreso_anual', 'mean')
        if 'edad' in df_view.columns:
            extra_aggs['edad_promedio'] = ('edad', 'mean')
        if 'hijos_pequenos' in df_view.columns:
            extra_aggs['prom_hijos_pequenos'] = ('hijos_pequenos', 'mean')
        if 'respondio' in df_view.columns:
            extra_aggs['tasa_respuesta'] = ('respondio', lambda x: f"{round(x.mean() * 100, 1)}%")

        if extra_aggs:
            extras_df = df_view.groupby(cluster_col).agg(**extra_aggs).reset_index()
            for col in extras_df.columns:
                if 'promedio' in col:
                    extras_df[col] = extras_df[col].round(1)
            resumen = resumen.merge(extras_df, on=cluster_col)

        # Reordenar y presentar columnas con nombres ejecutivos
        column_order = [cluster_col, 'clientes', '%_clientes', 'gasto_promedio', '%_del_gasto']
        other_cols = [c for c in resumen.columns if c not in column_order and c != 'gasto_total_grupo']
        
        tabla_final = resumen[column_order + other_cols].rename(columns={
            cluster_col: 'Segmento',
            'clientes': 'Clientes',
            '%_clientes': '% Clientes',
            'gasto_promedio': 'Gasto Promedio ($)',
            '%_del_gasto': '% del Gasto Total',
            'ingreso_promedio': 'Ingreso Promedio ($)',
            'edad_promedio': 'Edad Promedio',
            'prom_hijos_pequenos': 'Hijos Pequeños (prom)',
            'tasa_respuesta': 'Tasa Respuesta Campaña'
        })

        st.dataframe(
            tabla_final,
            hide_index=True
        )
    else:
        st.info("La columna 'gasto_total' no está disponible para calcular el perfil de gasto.")


# -------------------------------------------------------------
# PARTE 2: COMPARADOR DE TÉCNICAS
# -------------------------------------------------------------
with tab_comparador:
    st.subheader("Comparador de las 5 Técnicas de Segmentación")
    st.markdown("""
    Ajusta simultáneamente **K-Means**, **Jerárquico (Aglomerativo)**, **DBSCAN**, **Mezclas Gaussianas (GMM)** y **K-Prototypes** 
    utilizando el mismo conjunto de variables para contrastar su comportamiento y concordancia.
    """)

    with st.spinner("Entrenando modelos y evaluando concordancia..."):
        labels_dict, metrics_df, ari_df = run_all_clustering(
            df,
            n_clusters=k_param,
            eps=eps_param,
            min_samples=min_samples_param
        )

    if metrics_df is not None:
        # Tabla comparativa de métricas
        with st.container(border=True):
            st.markdown("**1. Resumen de Modelos: N° de Grupos, Silueta y Tamaños**")
            st.dataframe(metrics_df, hide_index=True)

        st.write("")
        col_c1, col_c2 = st.columns([1.2, 1])

        with col_c1:
            with st.container(border=True):
                st.markdown("**2. Matriz de Concordancia (Adjusted Rand Index)**")
                st.caption("1.0 = idénticos | 0.0 = asignación al azar | Valores altos indican grupos equivalentes.")
                
                # Visualización con Heatmap de Plotly
                z_vals = ari_df.values.astype(float)
                fig_ari = px.imshow(
                    z_vals,
                    x=list(ari_df.columns),
                    y=list(ari_df.index),
                    color_continuous_scale="Viridis",
                    text_auto=".2f",
                    zmin=0, zmax=1
                )
                fig_ari.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_ari)

        with col_c2:
            with st.container(border=True):
                st.markdown("**3. Visualizar Dispersión por Técnica**")
                selected_tech = st.selectbox(
                    "Seleccionar técnica para ver sus clusters:",
                    list(labels_dict.keys()),
                    index=0
                )
                
                df_compare = df.copy()
                df_compare['cluster_temp'] = labels_dict[selected_tech].astype(str)
                
                fig_mini = px.scatter(
                    df_compare,
                    x='ingreso_anual' if 'ingreso_anual' in df.columns else df.columns[1],
                    y='gasto_total' if 'gasto_total' in df.columns else df.columns[2],
                    color='cluster_temp',
                    title=f"Clusters formados por {selected_tech}",
                    height=320,
                    template="plotly_white"
                )
                fig_mini.update_layout(margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                st.plotly_chart(fig_mini)

    else:
        st.error("No se encontraron suficientes variables numéricas requeridas para correr la comparación.")


# -------------------------------------------------------------
# PARTE 3: RECOMENDACIÓN DE MARKETING
# -------------------------------------------------------------
with tab_recomendacion:
    st.subheader("Recomendación de Negocio para Marketing")
    
    with st.container(border=True):
        st.markdown("### Recomendación Ejecutiva (3 Líneas)")
        st.success(r"""
1. **Técnica y Por qué:** Recomiendo **K-Prototypes** porque genera 3 segmentos nítidos y no solapados por nivel de gasto ($85, $670 y $1,298), incorporando de forma equilibrada variables transaccionales y demográficas (educación y estado civil).
2. **Acción Clientes Alto y Medio Valor:** Para el segmento VIP ($1,298 de gasto, 25% de respuesta), implementar programas de fidelización exclusiva y venta cruzada premium; para el segmento Medio ($670 de gasto), incentivar mayor frecuencia y upselling.
3. **Acción Clientes Bajo Valor / Familias:** Para el segmento Bajo ($85 de gasto, 84% con hijos pequeños), enfocar campañas en promociones de volumen, combos familiares y cupones de descuento sensibles al precio para elevar el ticket promedio.
        """)

    st.write("")
    st.subheader("Preguntas Clave del Comentario Final")
    
    with st.expander("1. ¿Las cinco técnicas coinciden? ¿Cuáles se parecen más y por qué?"):
        st.markdown("""
        * **No todas coinciden por igual:**
          * **K-Means y K-Prototypes se parecen casi al 100% (ARI = 0.99):** Ambos optimizan distancias basadas en centroides, y las 9 variables numéricas dominan la estructura respecto a las 2 categóricas.
          * **DBSCAN y GMM se parecen fuertemente entre sí (ARI = 0.85):** Ambas técnicas escapan de la restricción esférica de k-means. DBSCAN agrupa por densidad continua y aísla ruido, mientras GMM modela mezclas de distribuciones con matrices de covarianza flexibles.
          * **Jerárquico** genera una estructura intermedia (ARI ~0.47 a 0.59) al fusionar clusters de forma aglomerativa en cada paso.
        """)

    with st.expander("2. ¿Agregar educación y estado civil (k-prototypes) cambió los grupos?"):
        st.markdown("""
        * **Prácticamente NO:** El índice de concordancia ARI entre K-Means y K-Prototypes es de **0.99**, y más del 99.6% de los clientes se ubicaron en los mismos grupos exactos.
        * **Causa técnica:** Las variables numéricas normalizadas (ingreso, recencia, total de gasto en múltiples canales) poseen una varianza multidimensional mucho mayor, la cual absorbe el impacto de las variables cualitativas cuando se utiliza una ponderación estándar en la función de costo.
        """)

    with st.expander("3. ¿Qué técnica sería más fácil de explicarle a marketing?"):
        st.markdown("""
        * **K-Means / K-Prototypes es la más fácil de explicar:**
          * Se definen clusters claros con centroides arquetípicos ("Cliente Promedio del Segmento A, B y C").
          * No genera ambigüedad de pertenencia probabilística ni clientes aislados como "ruido" (a diferencia de DBSCAN, que excluye clientes que marketing aún necesita contactar).
          * Permite establecer de inmediato reglas de negocio accionables por rangos de ingresos y montos de compra.
        """)
