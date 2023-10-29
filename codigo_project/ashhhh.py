import math
import numpy as np
from PIL import Image
from scipy.interpolate import griddata


# Define una función calculate_effective_index para calcular el índice de refracción efectivo
def calculate_effective_index(nl, wavelength, wavelengths):
    nl_eff_list = []

    for color, nl_color_list in nl.items():
        nl_color1, nl_color2 = nl_color_list  # Tomar los primeros dos índices de refracción
        if wavelength >= wavelengths[color]:
            nl_eff = nl_color1[0] + (nl_color2[0] - nl_color1[0]) * (wavelength - nl_color1[1]) / (nl_color2[1] - nl_color1[1])
        else:
            nl_eff = nl_color1[0]  # Usar el primer índice de refracción si la longitud de onda es menor

        nl_eff_list.append(nl_eff)

    return nl_eff_list

wavelengths = {
    "blue": 450,
    "green": 550,
    "red": 620
}


nl = {
    'red': [(1.5155, 620, 'crown'), (1.7245, 620, 'flint')],
    'green': [(1.5185, 550, 'crown'), (1.7337, 550, 'flint')],
    'blue': [(1.5253, 450, 'crown'), (1.7569, 450, 'flint')]
}

wavelength = 600  # Cambia la longitud de onda según tus necesidades
results = calculate_effective_index(nl, wavelength, wavelengths)

for i, result in enumerate(results):
    print(f"Índice de refracción efectivo {i + 1} en {wavelength} nm para {list(nl.keys())[i]} es {result}")



path = r'Downloads\\saturn.jpg'  # Ruta de la imagen

CHIEF_RAY = 0
PARALLEL_RAY = 1

# Cargar la imagen
image = Image.open(path)
width, height = image.size
image_array = np.array(image)

def corregir_aberracion_cromatica(image_array):
    # Convertir la imagen a un arreglo NumPy
    # Separar los canales B, G y R
    b, g, r = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    # Realizar la ecualización del histograma en cada canal
    b_corrected = np.uint8(255 * (b - b.min()) / (b.max() - b.min()))
    g_corrected = np.uint8(255 * (g - g.min()) / (g.max() - g.min()))
    r_corrected = np.uint8(255 * (r - r.min()) / (r.max() - r.min()))

    # Crear una nueva imagen con los canales ecualizados
    image_corrected = np.stack((b_corrected, g_corrected, r_corrected), axis=-1)

    # Convertir el arreglo NumPy de vuelta a una imagen PILLOW
    image_corrected = Image.fromarray(image_corrected)

    return image_corrected

def interpolation(pixels):
    width_output, height_output, _ = pixels.shape
    arry = np.zeros((width_output, height_output))
    for i in range(width_output):
        for j in range(height_output):
            arry[i, j] = pixels[i, j][0]

    nonwhite_coords = np.argwhere(arry != 255)
    white_coords = np.argwhere(arry == 255)
    nonwhite_pixels = arry[nonwhite_coords[:, 0], nonwhite_coords[:, 1]]
    interpolated_pixels = griddata(nonwhite_coords, nonwhite_pixels, white_coords, method='linear', rescale=True)
    interpolated_pixels = np.nan_to_num(interpolated_pixels, nan=127.0)
    int_out = np.round(interpolated_pixels).astype(int)

    for i in range(int_out.shape[0]):
        pixels[white_coords[i, 0], white_coords[i, 1]] = (int_out[i], int_out[i], int_out[i])
    return pixels

def compute_lens_matrix(nl, R1, R2, dl):
    D1 = (nl - 1) / R1
    D2 = (nl - 1) / (-R2)
    a1 = (1 - (D2 * dl) / nl)
    a2 = -D1 - D2 + (D1 * D2 * dl / nl)
    a3 = dl / nl
    a4 = (1 - (D1 * dl) / nl))
    A = np.array([[a1, a2], [a3, a4]])
    return A

def ray_tracing(image_array, width_output, height_output, aberration, interpolate):
    # Implementa la función ray_tracing con las correcciones necesarias
    pass

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

    path = r'Downloads\\saturn.jpg' #path 
    # Cargar la imagen
    image = Image.open(path)  
    height, width, _ = image.shape
    image_array = np.array(image)
    # Separar los canales RGB
    r_channel = image_array[:, :, 0]
    g_channel = image_array[:, :, 1]
    b_channel = image_array[:, :, 2]

        # Obtener el ancho y alto de cada canal R, G y B
    width_r, height_r = image_array[:, :, 0].shape
    width_g, height_g = image_array[:, :, 1].shape
    width_b, height_b = image_array[:, :, 2].shape

    # Calcular los tamaños de salida para cada canal
    width_output_r = int(width_r * abs(Mt))
    height_output_r = int(height_r * abs(Mt))

    width_output_g = int(width_g * abs(Mt))
    height_output_g = int(height_g * abs(Mt))

    width_output_b = int(width_b * abs(Mt))
    height_output_b = int(height_b * abs(Mt))

    # Luego, puedes usar estos tamaños de salida para cada plano RGB en tus cálculos


    pixels = ray_tracing(image_array, width_output, height_output, aberration, interpolate)
    save_image(pixels, 'output.png')

    image = Image.new("RGB", (width_output, height_output), "white")
    pixels = image.load()

    # Combinar los canales en una sola imagen
    processed_image = np.stack((r_channel_inverted, g_channel, b_channel), axis=-1)

    # Crear una nueva imagen a partir del arreglo NumPy
    processed_image = Image.fromarray(processed_image)

    # Guardar la nueva imagen
    processed_image.save('imagen_procesada.jpg')


    pixels = ray_tracing(image_array, width_output, height_output, CHIEF_RAY, so, n1, image, res, nl, R1, R2, dl, aberration, pixels)
    pixels = ray_tracing(image_array, width_output, height_output, PARALLEL_RAY, so, n1, image, res, nl, R1, R2, dl, aberration, pixels)

    if interpolate:
        pixels = interpolation(pixels)
        print("Interpolation performed")

    image_corrected = corregir_aberracion_cromatica(pixels)
    image_corrected.save('output.png', format='PNG')
