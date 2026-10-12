"""
Laboratorio Final — Starbucks Rewards
App interactiva de Streamlit para la exploración de resultados de Uplift Modeling.

Permite a perfiles de negocio explorar:
1. El Qini del T-learner comparado contra selección por riesgo de churn y selección al azar.
2. La tabla y curva de políticas de targeting para encontrar el porcentaje óptimo de contacto.

Correr con:  streamlit run streamlit_starter.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Laboratorio Final — Starbucks Rewards",
    page_icon="☕",
    layout="wide"
)

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]
BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_data():
    posibles_rutas = [
        BASE_DIR / "data" / "uplift_campaign.csv",
        Path("Semana_3/Tarea/data/uplift_campaign.csv"),
        Path("data/uplift_campaign.csv"),
    ]
    for ruta in posibles_rutas:
        if ruta.exists():
            return pd.read_csv(ruta)
    raise FileNotFoundError("No se encontró 'uplift_campaign.csv'.")



@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float = 0.30, seed: int = 42, modelo_tipo: str = "Regresión Logística"):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    # 1. Modelo de riesgo/churn (para benchmarking de targeting por riesgo en la curva Qini)
    train_churn = train.copy()
    train_churn["churned"] = 1 - train_churn["responded_60d"]
    modelo_riesgo = LogisticRegression(max_iter=1000, random_state=seed)
    modelo_riesgo.fit(train_churn[FEATURES], train_churn["churned"])

    # 2. T-learner: dos modelos separados según grupo experimental
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]

    if modelo_tipo == "Random Forest":
        modelo_tratado = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=seed)
        modelo_control = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=seed)
    elif modelo_tipo == "Árbol de Decisión":
        modelo_tratado = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=seed)
        modelo_control = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=seed)
    else:
        modelo_tratado = LogisticRegression(max_iter=1000, random_state=seed)
        modelo_control = LogisticRegression(max_iter=1000, random_state=seed)

    modelo_tratado.fit(train_t[FEATURES], train_t["responded_60d"])
    modelo_control.fit(train_c[FEATURES], train_c["responded_60d"])

    test = test.copy()
    X_test = test[FEATURES]
    test["prob_churn"] = modelo_riesgo.predict_proba(X_test)[:, 1]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    """
    Valor NETO: margen bruto menos el costo del cupón ($5, solo si compró) y el envío ($0.05).
    Compara el valor neto obtenido por el grupo tratado contra el contrafactual de control.
    """
    sel = test_df.loc[seleccionados]
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)


def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values

    acumulado_tratados = np.cumsum(np.where(es_tratado, respondio, 0))
    acumulado_control = np.cumsum(np.where(~es_tratado, respondio, 0))
    n_tratados_acum = np.cumsum(es_tratado)
    n_control_acum = np.cumsum(~es_tratado)

    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        nt, nc = n_tratados_acum[i - 1], n_control_acum[i - 1]
        tasa_t = acumulado_tratados[i - 1] / nt if nt > 0 else 0
        tasa_c = acumulado_control[i - 1] / nc if nc > 0 else 0
        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)
    return np.array(xs), np.array(ys)


def coeficiente_qini(xs, ys, n):
    trapezoid_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area_modelo = trapezoid_fn(ys, xs)
    area_azar = trapezoid_fn(np.linspace(0, ys[-1], len(xs)), xs)
    return (area_modelo - area_azar) / n


# --- Título y descripción de negocio ---
st.title("☕ Starbucks Rewards — Optimización de Campaña con Uplift Modeling")
st.caption(
    "Herramienta interactiva de soporte a decisiones para identificar el porcentaje óptimo de clientes a contactar, "
    "maximizando el retorno económico neto y evitando el desperdicio en clientes que compran por su cuenta o no responderán."
)

df = load_data()

# --- Barra lateral de controles interactivos ---
with st.sidebar:
    st.header("Parámetros de Contacto")
    pct_contactar = st.slider(
        "Porcentaje de la base a contactar (%):",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="Proporción de clientes con mayor impacto incremental (uplift) a quienes se enviará la oferta."
    )

    st.markdown("---")
    st.header("Configuración del Modelo")
    modelo_tipo = st.selectbox(
        "Algoritmo base del T-Learner:",
        ["Regresión Logística", "Random Forest", "Árbol de Decisión"],
        index=0,
        help="Permite evaluar cómo diferentes algoritmos impactan la estimación del uplift y la curva Qini."
    )

    st.markdown("---")
    st.markdown("Resumen de la Base")
    st.markdown(f"- **Total clientes:** `{len(df):,}`")
    st.markdown(f"- **Tratados ($5 cupón):** `{(df['treatment_group'] == 'treatment').sum():,}`")
    st.markdown(f"- **Control (sin cupón):** `{(df['treatment_group'] == 'control').sum():,}`")

# --- Entrenamiento y puntuación ---
train, test = entrenar_t_learner(df, test_size=0.30, seed=42, modelo_tipo=modelo_tipo)

# Cálculo de la política para el % seleccionado
n_seleccionados = int(len(test) * pct_contactar / 100)
idx_seleccionados = test.sort_values("uplift_estimado", ascending=False).head(n_seleccionados).index
valor_por_cliente, n_contactados = valor_de_la_politica(test, idx_seleccionados)
valor_total = valor_por_cliente * n_contactados

# Comparación con modelo de riesgo/churn para el mismo %
idx_riesgo_sel = test.sort_values("prob_churn", ascending=False).head(n_seleccionados).index
valor_riesgo_cliente, _ = valor_de_la_politica(test, idx_riesgo_sel)
valor_riesgo_total = valor_riesgo_cliente * n_seleccionados

# --- Métricas Principales (KPI Cards) ---
st.subheader(f"Resultados para el {pct_contactar}% de la base seleccionada")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Clientes Contactados (Test)",
        value=f"{n_contactados:,}",
        delta=f"{pct_contactar}% de la base"
    )

with col2:
    st.metric(
        label="Valor Neto por Cliente",
        value=f"${valor_por_cliente:+.2f}",
        delta=f"${valor_por_cliente - valor_riesgo_cliente:+.2f} vs Churn"
    )

with col3:
    st.metric(
        label="Valor Neto Total Acumulado",
        value=f"${valor_total:+,.2f}",
        delta=f"${valor_total - valor_riesgo_total:+,.2f} vs Churn"
    )

with col4:
    # Comparación económica con no contactar o riesgo
    ganancia_vs_riesgo = valor_total - valor_riesgo_total
    st.metric(
        label="Ventaja sobre Targeting por Riesgo",
        value=f"${ganancia_vs_riesgo:+,.2f}",
        delta="Diferencia neta",
        delta_color="normal" if ganancia_vs_riesgo >= 0 else "inverse"
    )

st.markdown("---")

# --- Curva Qini ---
xs_uplift, ys_uplift = curva_qini(test, "uplift_estimado")
xs_riesgo, ys_riesgo = curva_qini(test, "prob_churn")
qini_uplift = coeficiente_qini(xs_uplift, ys_uplift, len(test))
qini_riesgo = coeficiente_qini(xs_riesgo, ys_riesgo, len(test))

st.subheader("Curva Qini: Evaluación del Poder de Ordenamiento")
col_grafica, col_explicacion = st.columns([2, 1])

with col_grafica:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(xs_uplift * 100, ys_uplift, color=GREEN, linewidth=2.8, label=f"T-learner ({modelo_tipo}, Qini = {qini_uplift:+.3f})")
    ax.plot(xs_riesgo * 100, ys_riesgo, color=ORANGE, linewidth=2.5, label=f"Modelo Riesgo/Churn (Qini = {qini_riesgo:+.3f})")
    ax.plot([0, 100], [0, ys_uplift[-1]], color=GRAY, linestyle="--", linewidth=1.8, label="Selección al Azar")

    # Marcador interactivo del porcentaje actual
    idx_curva = int(np.clip(pct_contactar / 100 * (len(xs_uplift) - 1), 0, len(xs_uplift) - 1))
    ax.axvline(pct_contactar, color="#333333", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.scatter([pct_contactar], [ys_uplift[idx_curva]], color=GREEN, s=90, zorder=5, label=f"Corte actual ({pct_contactar}%)")

    ax.set_xlabel("% de la base contactada", fontsize=11)
    ax.set_ylabel("Respuestas incrementales acumuladas", fontsize=11)
    ax.set_title("Curva Qini — T-learner vs. Riesgo de Churn vs. Azar", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0", loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

with col_explicacion:
    st.markdown("#### ¿Cómo interpretar esta gráfica?")
    st.write(
        """
        - **Curva Verde (T-Learner):** Mide el impacto causal real del cupón. Cuanto más alta sobre la diagonal, más eficiente es contactar primero a quienes realmente cambian su conducta gracias al incentivo.
        - **Curva Naranja (Churn):** Ordena por probabilidad de abandono. Tiende a incentivar a clientes que se irán de todos modos (*Lost Causes*) o genera canibalización.
        - **Línea Gris (Al Azar):** Es el punto base sin personalización.
        """
    )
    if qini_uplift > qini_riesgo:
        st.success(f"El T-learner supera al modelo de churn por **{qini_uplift - qini_riesgo:+.3f}** puntos Qini, confirmando un targeting causal superior.")
    else:
        st.warning("En esta configuración el modelo de riesgo se aproxima al T-learner. Ajusta los hiperparámetros o el algoritmo.")

st.markdown("---")

# --- Tabla de Políticas y Simulación de Escenarios ---
st.subheader("Tabla de Políticas de Decisión (Paso 3)")

# Calcular resultados para deciles 10% a 100%
resultados = []
for pct in range(10, 101, 10):
    n_c = int(len(test) * pct / 100)
    idx_p = test.sort_values("uplift_estimado", ascending=False).head(n_c).index
    val_cli, n_p = valor_de_la_politica(test, idx_p)
    resultados.append({
        "% Contactado": pct,
        "Clientes Contactados": n_p,
        "Valor por Cliente ($)": round(val_cli, 2),
        "Valor Total ($)": round(val_cli * n_p, 2)
    })

tabla_politicas = pd.DataFrame(resultados)
max_valor_total = tabla_politicas["Valor Total ($)"].max()
pct_optimo_total = tabla_politicas.loc[tabla_politicas["Valor Total ($)"].idxmax(), "% Contactado"]
pct_optimo_cliente = tabla_politicas.loc[tabla_politicas["Valor por Cliente ($)"].idxmax(), "% Contactado"]

col_tabla, col_curva_valor = st.columns([1, 1])

with col_tabla:
    st.markdown("##### Rendimiento por decil de la base")
    def highlight_selected(row):
        if row["% Contactado"] == pct_contactar:
            return ["background-color: #E6F4EA; font-weight: bold; color: #00704A"] * len(row)
        return [""] * len(row)

    st.dataframe(
        tabla_politicas.style.apply(highlight_selected, axis=1).format({
            "Clientes Contactados": "{:,}",
            "Valor por Cliente ($)": "${:+.2f}",
            "Valor Total ($)": "${:+,.2f}"
        }),
        use_container_width=True
    )
    st.caption("Fila sombreada en verde: porcentaje seleccionado actualmente en la barra lateral.")

with col_curva_valor:
    st.markdown("##### Curva de Valor Económico Total")
    fig_val, ax_val = plt.subplots(figsize=(7, 4.3))
    ax_val.plot(tabla_politicas["% Contactado"], tabla_politicas["Valor Total ($)"], marker="o", color=GREEN, linewidth=2.5, label="Valor Total ($)")
    ax_val.axvline(pct_optimo_total, color="#D9381E", linestyle="--", linewidth=1.5, label=f"Óptimo Económico ({pct_optimo_total}%)")
    ax_val.scatter([pct_contactar], [valor_total], color="#000000", s=90, zorder=5, label=f"Selección actual ({pct_contactar}%)")

    ax_val.set_xlabel("% de la Base Contactada", fontsize=10)
    ax_val.set_ylabel("Valor Neto Total ($)", fontsize=10)
    ax_val.grid(True, linestyle="--", alpha=0.35)
    ax_val.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")
    plt.tight_layout()
    st.pyplot(fig_val)