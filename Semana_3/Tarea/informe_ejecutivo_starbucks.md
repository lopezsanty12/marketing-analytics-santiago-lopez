# Laboratorio Final: Optimización de Campaña Starbucks Rewards mediante Uplift Modeling

**Fecha:** 11 de Octubre, 2026  
**Para:** Presidente de Marketing, Starbucks Coffee Company  
**De:** Equipo de Marketing Analytics & Data Science  
**Asunto:** Estrategia de Targeting Causal para la Campaña Rewards

---

## Comparación de Modelos de Clasificación en el T-Learner

Evaluación sobre el conjunto de prueba independiente (12,000 clientes, 30% de la base) contra el **Uplift Real** y el **Coeficiente Qini**:

| Modelo dentro del T-learner | Hiperparámetros / Regularización | Correlación con el Uplift Real | Coeficiente Qini | Diagnóstico Técnico |
| :--- | :--- | :---: | :---: | :--- |
| **Logística (la del notebook)** | `max_iter=1000` (L2 regularizada) | **0.619** | **+0.142** | Buen ordenamiento lineal; bajo riesgo de sobreajuste. |
| **Árbol sin límite** | `max_depth=None` (sin poda) | **0.204** | **+0.079** | Severo sobreajuste (*overfitting*); memoriza ruido en cada subgrupo. |
| **Árbol con max_depth=2** | `max_depth=2` | **0.605** | **+0.127** | Poda estricta; recupera señal pero pierde granularidad. |
| **Árbol con max_depth=4** | `max_depth=4`, `min_samples_leaf=30` | **0.781** | **+0.125** | Poda balanceada; captura interacciones sin memorizar ruido. |
| **Random Forest (Ensamble)** | `n_estimators=100`, `max_depth=5`, `leaf=20` | **0.860** | **+0.157** | **Mejor modelo:** Máxima correlación y mayor área bajo la curva Qini. |
| **Gradient Boosting** | `n_estimators=100`, `max_depth=3` | **0.840** | **+0.154** | Excelente consistencia causal y calibración probabilística. |

* **Conclusión técnica:** El T-learner penaliza fuertemente el sobreajuste porque resta dos probabilidades ($P(Y|T) - P(Y|C)$). Los modelos ensamblados regularizados (*Random Forest*) superan a la logística al capturar relaciones no lineales complejas sin memorizar ruido.

---

## Memorando Ejecutivo: Decisión Comercial de Campaña

**A:** Presidente de Marketing  
**DE:** Dirección de Marketing Analytics  
**DECISIÓN:** Restringir el cupón de $5 al **20% – 25%** de los clientes con mayor Uplift estimado.

---

### 1. ¿A qué porcentaje contactar y por qué?
* **Recomendación:** Contactar únicamente al **20% – 25%** de la base.
* **Impacto Económico:** 
  * En este rango se alcanza la **máxima utilidad neta total (+ $8,100)** en la base de 40,000 clientes.
  * Si bien al **10%** se obtiene el mayor retorno unitario (**+$1.62/cliente**), contactar al **25%** maximiza la ganancia total de la empresa antes de que el costo del cupón supere el beneficio incremental.
* **El riesgo del marketing masivo:** Contactar al 100% de la base genera una **pérdida neta de -$67,400** debido a que se regalan cupones a clientes que ya iban a comprar de todos modos.

---

### 2. Validación con la Realidad (Paso 5)
Al comparar las predicciones con los datos reales latentes (correlación de **0.860** con la verdad):

* **Persuadables (22.5%):** El modelo identificó con precisión casi exacta al 22.5% de clientes que **solo compran si reciben el cupón**.
* **Sure Things (48.5%):** Evitó regalar $5 a casi la mitad de la base que compra por hábito sin necesidad de descuento.
* **Lost Causes (21.8%) y Sleeping Dogs (7.1%):** Descartó a quienes no responden o reaccionan negativamente ante el contacto comercial.

---

### 3. Plan de Acción
1. **Lanzar la campaña Rewards** segmentando exclusivamente al **top 25% de la base**.
2. **Ahorro Presupuestario:** Reducir la emisión de cupones de 40,000 a 10,000 unidades, protegiendo $150,000 en margen no canibalizado.
3. **Escalamiento:** Adoptar el modelo T-Learner de *Random Forest* como estándar en el CRM para futuras promociones.
