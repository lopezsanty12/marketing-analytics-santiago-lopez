# Archivo para el despliegue del Agente en Streamlit
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# 1. Configuración de la página (Debe ser el primer comando de Streamlit)
st.set_page_config(
    page_title="Proyección de Ventas | Marketing Analytics",
    page_icon="📈",
    layout="centered"
)

# 2. Encabezado principal
st.title("📈 Proyección de Ventas")
st.markdown("---")
st.markdown("""
Bienvenido a tu herramienta de **Marketing Analytics**. 
Ajusta el nivel de inversión en publicidad para predecir las ventas estimadas utilizando un modelo de Regresión Lineal.
""")

# 3. Definición y ajuste del modelo
variable_x = np.array([[10], [20], [30], [40], [50]])
variable_y = np.array([15, 25, 35, 45, 55])

modelo_lr = LinearRegression()
modelo_lr.fit(variable_x, variable_y)

# 4. Diseño de la interfaz con columnas
col1, col2 = st.columns(2)

with col1:
    st.subheader("💡 Configuración")
    st.info("Desliza para ajustar la inversión en publicidad.")
    gasto = st.slider("Gasto en Publicidad ($)", min_value=10, max_value=200, value=50, step=10)
    predecir = st.button("🔮 Calcular Predicción", use_container_width=True)

with col2:
    st.subheader("📊 Resultados")
    if predecir:
        # Generar predicción
        resultado = modelo_lr.predict([[gasto]])
        
        # Mostrar resultado con un componente visual destacado
        st.metric(
            label="Ventas Proyectadas Estimadas", 
            value=f"Q {resultado[0]:,.2f}", 
            delta=f"Inversión: ${gasto}"
        )
        st.success("¡Cálculo exitoso!")
    else:
        st.write("Haz clic en **Calcular Predicción** para ver los resultados.")

# 5. Visualización del modelo histórico
st.markdown("---")
st.subheader("📉 Datos Históricos")
# Crear un dataframe para graficar
df_datos = pd.DataFrame({
    'Inversión en Publicidad': variable_x.flatten(),
    'Ventas': variable_y
})
# Mostrar el gráfico
st.line_chart(df_datos.set_index('Inversión en Publicidad'))