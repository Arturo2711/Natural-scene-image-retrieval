import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen usando OpenCV
imagen = cv2.imread('DataBase/TORRALBA_MOD_TODAS/IMAGEN182.jpg')

# Convertir la imagen de BGR a RGB (para usar con matplotlib)
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Generar coordenadas de puntos aleatorios
num_puntos = 30  # Número de puntos aleatorios
ancho, altura, _ = imagen.shape
puntos_x = np.random.randint(0, ancho, size=num_puntos)
puntos_y = np.random.randint(0, altura, size=num_puntos)
puntos = np.stack((puntos_x, puntos_y), axis=-1)

# Dibujar los puntos aleatorios sobre la imagen
for punto in puntos:
    cv2.circle(imagen_rgb, tuple(punto), 1, (0, 255, 255), -1)  # -1 para llenar el círculo

cv2.circle(imagen_rgb, (1,250), 2, (0, 0, 255), -1)

# Mostrar la imagen con los puntos usando matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(imagen_rgb)
plt.axis('off')  # Para desactivar los ejes
plt.show()
