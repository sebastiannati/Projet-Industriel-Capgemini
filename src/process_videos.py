import cv2
import os
import pandas as pd
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Les points qui nous intéressent
right_eye = [33, 133, 160, 144, 159, 145, 158, 153]
left_eye = [263, 362, 387, 373, 386, 374, 385, 380]
mouth = [61, 291, 39, 181, 0, 17, 269, 405]
head = [10, 152]

list_points_no_tuples = right_eye + left_eye + mouth + head

# Les vidéos à traiter
video_paths = [f for f in os.listdir("videos") if f.endswith('.mp4')]
video_paths = [os.path.join("videos", video) for video in video_paths]
print(f"Il y a {len(video_paths)} vidéos à traiter")

# Configuration de la nouvelle API FaceLandmarker de MediaPipe
model_path = 'src/face_landmarker.task'
if not os.path.exists(model_path):
    print(f"Attention : Le fichier de modèle '{model_path}' est requis pour la nouvelle API MediaPipe.")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

if not os.path.exists("videos/videos_traitees.txt"):
    with open("videos/videos_traitees.txt", "a+") as save_video_name:
        print("Création du fichier")

# Ouverture en mode append
with open("videos/videos_traitees.txt", "a+") as save_video_name:
    save_video_name.seek(0)
    nombre_lignes = sum(1 for ligne in save_video_name)

print("Il y a", nombre_lignes, "lignes dans le fichier")
df = 0

# Création du dataframe
if not os.path.exists("csv/videos_coordinates.csv"):
    liste_colonnes = ["nom_video"]
    for j in range(875):
        for k in list_points_no_tuples:
            liste_colonnes.append(f"x_{k}_{j}")
            liste_colonnes.append(f"y_{k}_{j}")
            liste_colonnes.append(f"z_{k}_{j}")
    df = pd.DataFrame(columns=liste_colonnes)
else:
    print("J'ouvre le fichier")
    df = pd.read_csv("csv/videos_coordinates.csv")

print(df.shape)
videos_processed = 0
first_visual_check_done = False

# Parcourir chaque chemin de vidéo
for video_path in video_paths:
    with open("videos/videos_traitees.txt", "a+") as save_video_name:
        save_video_name.seek(0)
        video_saved = save_video_name.read()
        if video_path[7:] in video_saved:
            continue
        else:
            videos_processed += 1

    print(f"Traitement {videos_processed}/{len(video_paths) - nombre_lignes} de : {video_path[7:]}")
    nouvelle_ligne = [video_path[7:]]

    cap = cv2.VideoCapture(video_path.strip())
    fps_reel = cap.get(cv2.CAP_PROP_FPS)
    fps_cible = 25
    intervalle = fps_reel / fps_cible if fps_reel > 0 else 1

    frame_count = 0
    frame_gardees = 0
    prochaine_frame_a_garder = 0.0
    video_illisible = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Je n'arrive pas à lire la vidéo")
            video_illisible = True
            break

        if frame_gardees == 875:
            break

        if frame_count >= prochaine_frame_a_garder:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]
                for landmark_id in list_points_no_tuples:
                    data_point = face_landmarks[landmark_id]
                    nouvelle_ligne.append(data_point.x * frame_rgb.shape[1])
                    nouvelle_ligne.append(data_point.y * frame_rgb.shape[0])
                    nouvelle_ligne.append(data_point.z)

            frame_gardees += 1
            prochaine_frame_a_garder += intervalle

        frame_count += 1

    cap.release()

    # Cas 1 : lecture de la vidéo interrompue prématurément
    if video_illisible:
        with open("videos/videos_illisibles.txt", "a+") as video_non_traitée:
            video_non_traitée.write(video_path[7:] + "\n")
        # On marque quand même la vidéo comme "traitée" pour ne plus jamais la retenter
        with open("videos/videos_traitees.txt", "a+") as save_video_name:
            save_video_name.write(video_path[7:] + "\n")
        continue

    # Cas 2 : trop de valeurs manquantes (détection de visage ratée sur trop de frames)
    nb_zeros_a_ajouter = df.shape[1] - len(nouvelle_ligne)
    if nb_zeros_a_ajouter >= 0.10 * len(nouvelle_ligne):
        with open("videos/videos_illisibles.txt", "a+") as video_non_traitée:
            video_non_traitée.write(video_path[7:] + "\n")
            print("Trop de NULL pour :", video_path[7:], '\n')
        with open("videos/videos_traitees.txt", "a+") as save_video_name:
            save_video_name.write(video_path[7:] + "\n")
        continue

    # Cas 3 : traitement réussi
    nouvelle_ligne += [0] * nb_zeros_a_ajouter
    df = pd.concat([df, pd.DataFrame([nouvelle_ligne], columns=df.columns)], ignore_index=True)

    with open("videos/videos_traitees.txt", "a+") as save_video_name:
        save_video_name.write(video_path[7:] + "\n")

    if videos_processed == 1200:
        break

cv2.destroyAllWindows()
df.to_csv('csv/videos_coordinates.csv', index=False)
print("Terminé !")