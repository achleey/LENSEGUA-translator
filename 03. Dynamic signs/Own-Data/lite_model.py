import tensorflow as tf  # Importa TensorFlow, necesario para trabajar con modelos y conversiones

# Cargar el modelo entrenado desde la ruta especificada
model_path = '/Users/ashley/Desktop/Diseño e Innovación de Ingeniería 1/Tesis/Pruebas/PrimeraPrueba/Dinamicas/Propias/Prueba 1/ModelHolisticP'  # Reemplaza con la ruta a tu modelo

# Configurar el convertidor para convertir el modelo guardado en formato TensorFlow a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)  # Crea un convertidor a partir del modelo guardado
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]  # Establece los operadores soportados para el modelo Lite
converter._experimental_lower_tensor_list_ops = False  # Desactiva la operación experimental de reducción de tensor
converter.experimental_enable_resource_variables = True  # Habilita el uso de variables de recursos en el modelo

# Convierte el modelo a formato TensorFlow Lite
tflite_model = converter.convert()  # Realiza la conversión

# Guardar el modelo convertido en un archivo .tflite
with open('actionP.tflite', 'wb') as f:  # Abre el archivo 'actionP.tflite' para escritura en modo binario
    f.write(tflite_model)  # Escribe el modelo convertido en el archivo

# Imprime un mensaje confirmando que el modelo ha sido convertido y guardado
print("Modelo convertido a TensorFlow Lite y guardado como action.tflite")
