"""
get_data.py

Script d'orchestration du pipeline de génération des données :
1. Extraction des coordonnées des landmarks faciaux depuis les vidéos (src/process_videos.py)
2. Calcul des signes de fatigue à partir de ces coordonnées (src/signes.py)

Usage :
    python get_data.py
"""

import subprocess
import sys
import os

import pandas as pd
from src import signes


def extraire_coordonnees():
    """
    Lance src/process_videos.py, qui gère lui-même la reprise sur les vidéos
    déjà traitées (via videos/videos_traitees.txt). On peut donc l'appeler
    à chaque exécution sans risque de retraiter les vidéos déjà faites.
    """
    print(f"\n{'=' * 60}")
    print("Extraction des coordonnées (src/process_videos.py)...")
    print(f"{'=' * 60}\n")

    resultat = subprocess.run([sys.executable, "src/process_videos.py"])

    if resultat.returncode != 0:
        print(f"\n Erreur lors de l'exécution de src/process_videos.py (code {resultat.returncode})")
        sys.exit(resultat.returncode)

    if not os.path.exists("csv/videos_coordinates.csv"):
        print("\n csv/videos_coordinates.csv introuvable après extraction, arrêt du script.")
        sys.exit(1)

    print("\n Extraction des coordonnées terminée.")


def calculer_signes():
    """
    Calcule les signes de fatigue (EAR, MAR, EBR, PERCLOS, HOP) à partir de
    csv/videos_coordinates.csv et enregistre le résultat dans csv/signes.csv.
    """
    print(f"\n{'=' * 60}")
    print("Calcul des signes de fatigue...")
    print(f"{'=' * 60}\n")

    # Ouverture du dataframe des coordonnées par groupe de 1000
    df_coordinates_chunks = pd.read_csv("csv/videos_coordinates.csv", chunksize=1000)

    # Création des colonnes pour le dataframe final
    list_col = ["nom_video"]
    col_signes = ["EAR_left", "EAR_right", "EAR_mean", "MAR", "EBR", "PERCLOS", "HOP_gd", "HOP_hb"]
    for i in range(875):
        for signe in col_signes:
            list_col.append(f"{signe}_{i}")

    # Ouverture du dataframe des résultats
    df = pd.DataFrame(columns=list_col)

    nb_chunk = 0
    # Pour chaque chunk
    for chunk in df_coordinates_chunks:
        print(f"Chunk {nb_chunk + 1}...")
        for ligne in range(chunk.shape[0]):
            # Extraire le nom de la vidéo
            video_name = chunk.iloc[ligne, 0]

            # Initialiser une liste pour stocker le nom de la vidéo et les coordonnées de cette ligne
            video_and_coordinates = [video_name]
            print(f"Processing {ligne + 1}/{chunk.shape[0]} {video_name}...")

            # On itère sur les colonnes
            for i in range(1, chunk.shape[1], 3):
                point_coordinates = [chunk.iloc[ligne, i], chunk.iloc[ligne, i + 1], chunk.iloc[ligne, i + 2]]
                video_and_coordinates.append(point_coordinates)

            # Calcul des signes
            results, list_ebr = signes.calculs_signes(video_and_coordinates)

            row_data = [video_and_coordinates[0]] + results
            df.loc[len(df)] = row_data
            print(df.shape)

        nb_chunk += 1

    
    # Mapping classes according to kss
    mapping_classe = {
        "1-3": 0,
        "6-7": 1,
        "8-9": 2
    }

    df["classe"] = df["nom_video"].str[4:7].map(mapping_classe)

    # Enregistrer le DataFrame dans un fichier CSV
    df.to_csv("csv/signes.csv", index=False)

    print(df["classe"])
    print("\n Calcul des signes terminé, fichier disponible : csv/signes.csv")


def main():
    extraire_coordonnees()
    calculer_signes()
    print(f"\n{'=' * 60}")
    print("Pipeline get_data.py terminé avec succès !")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()