import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime

# Ucitavanje vec treniranog modela
model = DeepFace.build_model("Emotion")

# Definisanje labela
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Ucitavanje haar klasifikatora
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Snimanje kamerom
cap = cv2.VideoCapture(0)

#Kreiranje prazne liste za skladistenje emocija
emotion_logs = []
previous_emotion = None

while True:
    # Hvatanje okvira
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Kovertovanje okvira u sivo
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detekcija lica okviru
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=9, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # Ekstraktovanje "regije interesa"
        face_roi = gray_frame[y:y + h, x:x + w]

        # Menjanje velicine regije interesa da se poklapa sa ulaznim modelom
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalizacija izmenjene slike
        normalized_face = resized_face / 255.0

        # Oblikovanje slike da se poklapa sa ulaznim modelom
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Predikcija emocije 
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]

        # Belezenje emocije
        if emotion != previous_emotion:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            emotion_logs.append({'Timestamp': current_time, 'Emotion': emotion})
            previous_emotion = emotion


        # Iscrtavanja pravugaonika oko lica i ispis emocije
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Prikaz rezultujuceg okvira
    cv2.imshow('Real-time Emotion Detection', frame)

    # Stisni 'Q' za zatvaranje
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Formiranje data okvira na osnovu formirane liste i unosenje u eksel
emotion_log = pd.DataFrame(emotion_logs)

#Kalkulisanje prosecnog procenta emocija
emotion_counts = emotion_log['Emotion'].value_counts(normalize=True) * 100
average_percentages = emotion_counts.reindex(emotion_labels).fillna(0)

#Dodavanje prosecnog procenta u novi data okvir
avg_percent_df = pd.DataFrame(average_percentages).T

#Cuvanje emocija i procenata u Excel fajl
with pd.ExcelWriter('emotion_logs_with_percentages.xlsx') as writer:
    emotion_log.to_excel(writer, sheet_name='Emotion_Logs', index=False)
    avg_percent_df.to_excel(writer, sheet_name='Average_Percentages', index=False)

# Gasenje kamere i programa
cap.release()
cv2.destroyAllWindows()
