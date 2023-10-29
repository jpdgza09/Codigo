import math
import numpy as np
from PIL import Image
from scipy.interpolate import griddata


# Define constantes
wavelengths = {'red': 650, 'green': 550, 'blue': 450}
CHIEF_RAY = 0
PARALLEL_RAY = 1

def load_image(path):
    path = r'Downloads\\saturn.jpg' #path 
    """
    Carga una imagen y la devuelve como un arreglo NumPy.
    """
    image = Image.open(path)
    return np.array(image)

def save_image(image, path): """Guarda una imagen en el disco."""
    image = Image.fromarray(image)
    image.save(path, format='PNG')
    return image 


def correct_image_colors(image_array):
    """
    Aplica la corrección cromática básica a la imagen.
    """
    # Normaliza los canales de color
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    # Escala de nuevo a valores entre 0 y 255
    image_array = (image_array * 255).astype(np.uint8)
    return image_array

def interpolate_image(image_array):
    """
    Realiza la interpolación para suavizar la imagen.
    """
    # Convierte la imagen en un arreglo 2D
    arr = image_array[:, :, 0]
    white_coords = np.argwhere(arr == 255)
    nonwhite_coords = np.argwhere(arr != 255)
    nonwhite_pixels = arr[nonwhite_coords[:, 0], nonwhite_coords[:, 1]
    interpolated_pixels = griddata(nonwhite_coords, nonwhite_pixels, white_coords, method='linear', rescale=True)
    interpolated_pixels = np.nan_to_num(interpolated_pixels, nan=127.0)
    arr_out = np.round(interpolated_pixels).astype(int)
    # Asigna los valores interpolados a la imagen original
    for i in range(arr_out.shape[0]):
        image_array[white_coords[i, 0], white_coords[i, 1], 0] = arr_out[i]
    return image_array

def ray_tracing(image_array, width_output, height_output, aberration, interpolate):
    """
    Simula el trazado de rayos a través del sistema óptico.
    """
    pixels = np.ones((width_output, height_output, 3), dtype=np.uint8) * 255
    height, width, _ = image_array.shape

    for rayo in [CHIEF_RAY, PARALLEL_RAY]:
        for i in range(width):
            for j in range(height):
                pixel = image_array[j, i]
                x = i - width / 2
                y = j - height / 2
                r = math.sqrt(x * x + y * y) + 1
                y_objeto = r * res

                if aberration:
                    wavelength = wavelengths['green'] if rayo == CHIEF_RAY else wavelengths['blue']
                    C = 0.0  # Ajusta el valor de C según sea necesario

                    nl_eff = calculate_effective_index(nl, wavelength)
                    f = ((nl - 1) * ((1 / R1) - (1 / R2)))
                    f = f + (C * (y_objeto ** 4 - y_objeto ** 2))
                    f = 1 / f
                    si = (f * so) / (so - f)
                    A = compute_lens_matrix(nl, R1, R2, dl)
                    P2 = np.array([[1, 0], [si / n1, 1]])
                    P1 = np.array([[1, 0], [-so / n1, 1])

                    if rayo == CHIEF_RAY:
                        alpha_entrada = math.atan(y_objeto / so)
                    elif rayo == PARALLEL_RAY:
                        alpha_entrada = 0
                    V_entrada = np.array([n1 * alpha_entrada, y_objeto])
                    V_salida = P2.dot(A.dot(P1.dot(V_entrada)))
                    y_imagen = V_salida[1]

                    if rayo == CHIEF_RAY:
                        Mt = (-1) * y_imagen / y_objeto
                    elif rayo == PARALLEL_RAY:
                        Mt = y_imagen / y_objeto

                    x_prime = Mt * x
                    y_prime = Mt * y
                    pos_x_prime = int(x_prime + width_output / 2)
                    pos_y_prime = int(y_prime + height_output / 2)

                    if 0 <= pos_x_prime < width_output and 0 <= pos_y_prime < height_output:
                        if rayo == CHIEF_RAY:
                            pixels[pos_x_prime, pos_y_prime, :] = pixel
                        elif rayo == PARALLEL_RAY:
                            new_gray = (pixel + pixels[pos_x_prime, pos_y_prime, 0]) // 2
                            pixels[pos_x_prime, pos_y_prime, :] = new_gray

    if interpolate:
        pixels = interpolate_image(pixels)
        print("Interpolación realizada")

    return pixels

if __name__ == "__main":
    # Configuración
    R1 = 0.2
    R2 = -0.2
    dl = 0.01
    nl = 1.5
    aberration = True
    interpolate = True
    res = 0.0001
    f = R1 * R2 / ((R2 - R1) * (nl - 1))
    so = 0.18
    si = (f * so) / (so - f)
    n1 = 1
    Mt = -si / so

    image_array = load_image("saturn.jpg")
    width_output = int(image_array.shape[1] * abs(Mt))
    height_output = int(image_array.shape[0] * abs(Mt))

    pixels = ray_tracing(image_array, width_output, height_output, aberration, interpolate)
    save_image(pixels, 'output.png')




 nl_red = ...    # Debes proporcionar el índice de refracción para el color rojo
    nl_green = ...  # Debes proporcionar el índice de refracción para el color verde
    nl_blue = ...   # Debes proporcionar el índice de refracción para el color azul

    nl_eff = nl_red + (nl_green - nl_red) * (wavelength - wavelengths['red']) / (wavelengths['green'] - wavelengths['red'])
  
    if wavelength >= wavelengths['green']:
        nl_eff = nl_green + (nl_blue - nl_green) * (wavelength - wavelengths['green']) / (wavelengths['blue'] - wavelengths['green'])
    elif wavelength >= wavelengths['red']:
        nl_eff = nl_red + (nl_green - nl_red) * (wavelength - wavelengths['red']) / (wavelengths['green'] - wavelengths['red'])
    else:
        # Handle the case where wavelength is less than red
        nl_eff = nl_red  # Assuming a linear approximation from red to the given wavelength