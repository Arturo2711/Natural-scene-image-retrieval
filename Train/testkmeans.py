from featureExtraction import loadImage, rgb_to_hsi, get_all_windows, get_window_vector
from plotPoints import plot_Points
import joblib
import numpy as np

def one_Image_Feature_extraction(path):
    dicColors = {0: (132, 45, 210),
                 1: (20, 180, 90),
                 2: (255, 100, 10),
                 3: (75, 200, 255),
                 4: (180, 20, 160),
                 5: (240, 150, 60),
                 6: (50, 160, 230),
                 7: (210, 80, 30),
                 8: (30, 220, 120),
                 9: (160, 60, 190)}
    
    # Cargar el modelo desde el archivo
    model_reloaded = joblib.load('kmeans.pkl')
    
    # Cargar la imagen y convertirla a espacio de color HSI
    image = loadImage(path)
    image = rgb_to_hsi(image)
    
    # Obtener todas las ventanas y puntos de la imagen
    windows, points = get_all_windows(image)
    
    # Preparar lista para almacenar los puntos de los clusters
    clusterPoints = []
    
    # Procesar cada ventana
    for w in windows:
        describing_vector_window = get_window_vector(w)
        describing_vector_window = np.array(describing_vector_window)

        # Predict the cluster label
        clusterPoint_i = model_reloaded.predict(describing_vector_window.reshape(1, -1))[0]  # Assuming single label prediction
        print(clusterPoint_i)
        clusterPoints.append(clusterPoint_i)

    points = list(points)
    # Asignar colores a los puntos basados en los clusters
    for i, p in enumerate(points):
        points[i] = (p[0], p[1], dicColors[clusterPoints[i]])

    # Graficar los puntos sobre la imagen
    plot_Points(path, points)

one_Image_Feature_extraction('DataBase/TORRALBA_MOD_TODAS/IMAGEN470.jpg')
