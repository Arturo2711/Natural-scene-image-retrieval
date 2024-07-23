import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

##### In this program is going to perfom feature extraction of the database ###

def plot_Image(image):
    # Check if the image was loaded correctly
    if image is None:
        print("Error: Could not load the image.")
    else:
        # Define the window name
        window_name = 'Image'

        # Create a window that can be resized
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Resize the window to 800x600 pixels
        cv2.resizeWindow(window_name, 800, 600)

        # Display the image in the resized window
        cv2.imshow(window_name, image)

        # Wait until a key is pressed
        cv2.waitKey(0)

        # Close all windows
        cv2.destroyAllWindows()

### Just to load the image given a path ###

def loadImage(path):
    image = cv2.imread(path)
    return image

### RGB TO HSI ###

def rgb_to_hsi(image):
    # Normalizar la imagen RGB
    image = image / 255.0
    R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]
    
    # Cálculo de la Intensidad I
    I = (R + G + B) / 3.0
    I = np.round(I * 255).astype(np.uint8)
    
    # Cálculo de la Saturación S
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - 3 * min_rgb / (R + G + B + 1e-6)
    S = np.round(S * 255).astype(np.uint8)
    
    # Cálculo del Tono H
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(num / den)
    
    H = np.zeros_like(R)
    H[B <= G] = theta[B <= G]
    H[B > G] = 2 * np.pi - theta[B > G]
    H = H / (2 * np.pi)  # Normalizar H a [0, 1]
    H = np.round(H * 255).astype(np.uint8)  # Escalar y convertir a uint8

    # Combinar los canales en una imagen HSI
    HSI = np.stack([H, S, I], axis=2)
    return HSI


### Get the window pixels, given the coordinates of a point ###

def get_Window(point, channel):
    pointx = point[0]
    pointy = point[1]
    window_size = 5
    # Compute the index around the window
    start_x = max(0, pointx - window_size)
    end_x = min(channel.shape[0], pointx + window_size + 1)
    start_y = max(0, pointy - window_size)
    end_y = min(channel.shape[1], pointy + window_size + 1)

    # Get the window around the point 
    window = channel[start_x:end_x, start_y:end_y]
    return window

### Generate 300 random points ###

def generate_points(image): ## We pass the image just to get the dimension, once we get the dimension, we can limit the random function
    # Generate coordinates of random points
    num_points = 100  # Number of random points
    width, height, _ = image.shape
    points_x = np.random.randint(0, width, size=num_points)
    points_y = np.random.randint(0, height, size=num_points)
    points = np.stack((points_x, points_y), axis=-1)
    return points

### Return all windows, each point will have associated three window, i.e, a windowm for each channel

def get_all_windows(image):
    # Initialize the list of windows (windowh, windowS, windowI)
    windows = []
    # Extract each channel
    H, S, I = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    points = generate_points(image)
    for point in points:
        window_hue = get_Window(point, H)
        window_saturation = get_Window(point, S)
        window_intensity = get_Window(point, I)
        windows.append((window_hue, window_saturation, window_intensity))
    return windows, points

### Get the homogeinity, given a window

def get_homogeneity(window):
    # Ensure the window values are in the range [0, levels-1]
    levels = 256  # Assume 256 gray levels (0-255)
    glcm = graycomatrix(window, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
    homogeneity = graycoprops(glcm, 'homogeneity')
    return homogeneity[0, 0]



### To extract the mean,std and co-occurence matrix of each window

def get_window_vector(window):
    describing_vector_window = []
    for i in range(3):
        current_Window = window[i]
        describing_vector_window.append(np.mean(current_Window))
        describing_vector_window.append(np.std(current_Window))
        describing_vector_window.append(get_homogeneity(current_Window))
    return describing_vector_window


### To generate the csv

def write_Instance(path, image_path, vector):
    with open(path, mode='+a', encoding='utf-8') as file:
        line_to_write = image_path + ',' + str(vector[0]) + ',' + str(vector[1]) + ',' + str(vector[2])+ ',' + str(vector[3]) + ',' + str(vector[4]) + ',' + str(vector[5]) + ',' + str(vector[6]) + ',' + str(vector[7]) + ',' + str(vector[8]) + '\n'
        file.write(line_to_write)




### Feature extraction 

def feature_Extraction():
    for i in range(1244, 1367):
        path = 'DataBase/TORRALBA_MOD_TODAS/IMAGEN{}.jpg'.format(i)
        image = loadImage(path)
        image = rgb_to_hsi(image)
        windows, points = get_all_windows(image)
        for w in windows:
            describing_vector_window = get_window_vector(w)
            write_Instance('Train/clustering.csv', path[28:], describing_vector_window)




#feature_Extraction()









#rgb = loadImage('Train/imagetest.jpg')

#hsi = rgb_to_hsi(rgb)

#print(generate_points(hsi))
#plot_Image(rgb)
#plot_Image(hsi)
#plot_Image(hsi[:, :, 0])





