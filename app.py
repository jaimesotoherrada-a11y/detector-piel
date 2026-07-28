import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Detector de Enfermedades de la Piel", page_icon="🩺")

# --------------------------------------------------------------------
# IMPORTANTE: este orden es una referencia (LabelBinarizer ordena
# alfabéticamente). ANTES DE USAR, verifica que coincide exactamente
# con lo que te imprimió Colab en "Clases en orden: [...]" al guardar
# el modelo. Si el orden impreso es distinto, ajusta esta lista.
# --------------------------------------------------------------------
CLASES = [
    "Herpes HPV and other STDs Photos",
    "Nail Fungus and other Nail Disease",
    "Vasculitis Photos",
]

IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
      # compile=False evita incompatibilidades con optimizadores o métricas guardadas
    return tf.keras.models.load_model("modelo_piel.keras", compile=False)
  

model = load_model()

st.title("🩺 Detector de Enfermedades de la Piel")
st.caption("Proyecto educativo — NO reemplaza el diagnóstico de un profesional médico.")

foto = st.camera_input("Toma una foto de la zona afectada")

if foto is not None:
    img = Image.open(foto).convert("RGB")
    st.image(img, caption="Foto capturada", use_container_width=True)

    if st.button("Ejecutar predicción"):
        with st.spinner("Analizando imagen..."):
            img_resized = img.resize(IMG_SIZE)
            img_array = np.array(img_resized).astype("float32")
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)[0]

        st.subheader("Resultados de la predicción")
        resultados = sorted(zip(CLASES, pred), key=lambda x: x[1], reverse=True)
        for clase, prob in resultados:
            st.write(f"**{clase}**: {prob * 100:.2f}%")
            st.progress(float(prob))

        clase_predicha, prob_predicha = resultados[0]
        st.success(f"Predicción principal: **{clase_predicha}** ({prob_predicha * 100:.2f}%)")
        st.warning("⚠️ Este resultado es solo referencial. Consulta a un dermatólogo para un diagnóstico certero.")
