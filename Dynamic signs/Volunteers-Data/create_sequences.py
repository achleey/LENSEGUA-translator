import cv2
import os

# Ruta de la carpeta principal que contiene las subcarpetas de videos
main_folder = 'videos'
# Ruta de la carpeta donde se guardarán las secuencias de imágenes
output_main_folder = 'Sequences'

# Crear la carpeta principal de secuencias si no existe
os.makedirs(output_main_folder, exist_ok=True)

# Recorremos las subcarpetas (0, 1 y 2)
for subfolder in ['0', '1', '2']:
    subfolder_path = os.path.join(main_folder, subfolder)
    # Crear la subcarpeta correspondiente en "Sequences"
    output_subfolder_path = os.path.join(output_main_folder, subfolder)
    os.makedirs(output_subfolder_path, exist_ok=True)

    # Filtra archivos con extensión .mp4 o .mov y los ordena numéricamente
    video_files = sorted(
        [f for f in os.listdir(subfolder_path) if f.endswith(('.mp4', '.mov', '.MOV', '.MP4'))],
        key=lambda x: int(os.path.splitext(x)[0])  # Convierte el nombre sin extensión a entero para ordenar
    )

    # Recorremos cada archivo de video en la subcarpeta en orden
    for video_index, video_file in enumerate(video_files):
        video_path = os.path.join(subfolder_path, video_file)

        # Crear la carpeta de secuencia para cada video en la estructura de "Sequences"
        output_video_folder = os.path.join(output_subfolder_path, str(video_index))
        os.makedirs(output_video_folder, exist_ok=True)

        # Leer el video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_interval = max(1, total_frames // 30)  # Calcula el intervalo entre frames

        frame_count = 0
        saved_frames = 0

        # Extraer y guardar los 30 fotogramas
        while cap.isOpened() and saved_frames < 30:
            ret, frame = cap.read()
            if not ret:
                break

            # Guardar el frame cada intervalo
            if frame_count % frames_interval == 0:
                image_path = os.path.join(output_video_folder, f"{saved_frames}.jpg")
                cv2.imwrite(image_path, frame)
                saved_frames += 1

            frame_count += 1

        cap.release()

print("Extracción de imágenes completada.")
