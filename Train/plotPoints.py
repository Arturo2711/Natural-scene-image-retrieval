import cv2
import numpy as np
import matplotlib.pyplot as plt
from featureExtraction import rgb_to_hsi



def plot_Points(path, puntos):
    # Cargar la imagen usando OpenCV
    imagen = cv2.imread(path)

    # Convertir la imagen de BGR a RGB (para usar con matplotlib)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    #imagen_rgb = rgb_to_hsi(imagen)

    # Dibujar los puntos aleatorios sobre la imagen
    for punto in puntos:
        tupla = tuple(punto)
        cv2.circle(imagen_rgb, (tupla[0], tupla[1]), 1, tupla[2], -1)  # -1 para llenar el círculo

    # Mostrar la imagen con los puntos usando matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(imagen_rgb)
    plt.axis('off')  # Para desactivar los ejes
    plt.show()
