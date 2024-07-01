import numpy as np

# Crear un arreglo de 5x5 con números del 1 al 25
arr = np.arange(1, 401).reshape(20, 20)
print("Arreglo original:")
print(arr)
print()

# Punto central de interés
pointx = 5
pointy = 5

# Definir el tamaño de la ventana alrededor del punto
window_size = 2  # Tamaño de la ventana (en este caso, 1 fila/columna alrededor del punto)

# Calcular los índices para obtener la ventana alrededor del punto
start_x = max(0, pointx - window_size)
end_x = min(arr.shape[0], pointx + window_size + 1)
start_y = max(0, pointy - window_size)
end_y = min(arr.shape[1], pointy + window_size + 1)

# Obtener el subarreglo alrededor del punto
window = arr[start_x:end_x, start_y:end_y]

print(f"Ventana alrededor del punto ({pointx}, {pointy}):")
print(window)
