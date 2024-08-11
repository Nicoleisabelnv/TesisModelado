import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# Cargar el modelo
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')

# Crear una función para hacer predicciones y obtener probabilidades
def make_prediction(features):
    df = pd.DataFrame([features], columns=['NumeroOferentes', 'CategoriaPrincipal', 'Presupuesto', 'MontoOfertado', 'DuracionLicitacionDias', 'DuracionContratoDias'])
    prediction = model.predict(df)
    probabilidad = model.predict_proba(df)[0][1]  # Probabilidad de la clase 1
    return prediction[0], probabilidad

# Aplicación Streamlit
st.title('Predicción con CatBoost')

# Crear entradas de usuario
numero_oferentes = st.number_input('Número de Oferentes', min_value=0)
categoria_principal = st.selectbox('Categoría Principal', options=['goods', 'services', 'works'])  # Categorías actualizadas
presupuesto = st.number_input('Presupuesto')
monto_ofertado = st.number_input('Monto Ofertado')
duracion_licitacion_dias = st.number_input('Duración de la Licitación (Días)')
duracion_contrato_dias = st.number_input('Duración del Contrato (Días)')

# Convertir las entradas a valores numéricos
categoria_map = {'goods': 0, 'services': 1, 'works': 2}
categoria_num = categoria_map[categoria_principal]

if st.button('Hacer Predicción'):
    features = [numero_oferentes, categoria_num, presupuesto, monto_ofertado, duracion_licitacion_dias, duracion_contrato_dias]
    prediccion, probabilidad = make_prediction(features)
    st.write(f'La predicción es: {prediccion}')
    st.write(f'Probabilidad de participación: {probabilidad:.2f}')
