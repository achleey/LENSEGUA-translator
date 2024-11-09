import tensorflow as tf

# Cargar el modelo entrenado
model_path = '/Users/ashley/Desktop/Diseño e Innovación de Ingeniería 1/Tesis/Pruebas/PrimeraPrueba/Dinamicas/Propias/Prueba 1/ModelHolisticP'  # Reemplaza con la ruta a tu modelo

# Configurar el convertidor
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

# Convertir el modelo a formato TensorFlow Lite
tflite_model = converter.convert()

# Guardar el modelo convertido
with open('actionP.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo convertido a TensorFlow Lite y guardado como action.tflite")
