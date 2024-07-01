import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen usando OpenCV
imagen = cv2.imread('DataBase/TORRALBA_MOD_TODAS/IMAGEN182.jpg')

# Convertir la imagen de BGR a RGB (para usar con matplotlib)
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Coordenadas del punto central
punto_central = (100, 150)  # Ejemplo de coordenadas

# Definir tamaño de la vecindad (10x10 píxeles)
ventana_ancho = 10
ventana_alto = 10

# Obtener las coordenadas de la vecindad
x, y = punto_central
x_min = max(0, x - ventana_ancho // 2)
x_max = min(imagen.shape[1] - 1, x + ventana_ancho // 2)
y_min = max(0, y - ventana_alto // 2)
y_max = min(imagen.shape[0] - 1, y + ventana_alto // 2)

# Crear una copia de la imagen original para dibujar la vecindad
imagen_con_vecindad = imagen_rgb.copy()

# Dibujar un rectángulo que representa la vecindad en la copia de la imagen
cv2.rectangle(imagen_con_vecindad, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)  # Color: azul (255, 0, 0), Grosor: 2 píxeles

cv2.circle(imagen_con_vecindad, (100,150), 2, (255, 0, 0), -1)


# Mostrar la imagen con la vecindad dibujada usando matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(imagen_con_vecindad)
plt.axis('off')  # Para desactivar los ejes
plt.title(f'Vecindad de {ventana_ancho}x{ventana_alto} píxeles alrededor del punto ({x}, {y})')
plt.show()
