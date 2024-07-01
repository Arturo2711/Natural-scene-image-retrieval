''' 
def rgb_to_hsi(image):
    ### Accesing to the channels ## 
    image = image / 255.0
    R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]
    
    ### Intensity ###
    I = (R + G + B) / 3.0
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - 3 * min_rgb / (R + G + B + 1e-6) ### Saturation
    
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(num / den)
    
    H = np.zeros_like(R) 
    H[B <= G] = theta[B <= G]
    H[B > G] = 2 * np.pi - theta[B > G]
    H = H / (2 * np.pi)  ### Hue
    
    HSI = np.stack([H, S, I], axis=2)
    return HSI
    
''' 