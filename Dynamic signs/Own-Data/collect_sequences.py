import os
import cv2
import time

DATA_DIR = './Videos'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3   #Tres clases
no_sequences = 168       # 30 videos
sequence_length = 30    # 30 frames de largo

cap = cv2.VideoCapture(1)
for j in range(number_of_classes):

    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    for sequence in range(no_sequences):
        sequence_dir = os.path.join(class_dir, str(sequence))
        if not os.path.exists(sequence_dir):
            os.makedirs(sequence_dir)

        print('  Collecting sequence {} of class {}'.format(sequence, j))

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('Video Capture', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < sequence_length:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # Aplica el flip aquí también durante la captura
            cv2.putText(frame, 'Collecting sequence {} of class {}'.format(sequence, j), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('Video Capture', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(class_dir, str(sequence), '{}.jpg'.format(counter)), frame)
            counter += 1

        time.sleep(1)
cap.release()
cv2.destroyAllWindows()






