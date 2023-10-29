import math
import numpy as np
from PIL import Image
from scipy.interpolate import griddata
import matplotlib.image as image

from codigo_project.hhh import save_image 

wavelengths = {'red': 620, 'green': 550, 'blue': 450, }
path = r'Downloads\\saturn.jpg' #path 

CHIEF_RAY = 0
PARALLEL_RAY = 1

# Cargar la imagen
image = Image.open(path)  
height, width, _ = image.shape
image_array = np.array(image)


def corregir_aberracion_cromatica(image):
  # Convertir la imagen a un arreglo NumPy
  # Separar los canales B, G y R
    b, g, r = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
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
    a4 = (1 - (D1 * dl) / nl)
    A = np.array([[a1, a2], [a3, a4]])

    return A


def calculate_effective_index(nl, wavelength):
    nl_eff_list = []
    
    for color, nl_color_list in nl.items():
        nl_color1, nl_color2 = nl_color_list  # Tomar los primeros dos índices de refracción
        if wavelength >= wavelengths[color]:
            nl_eff = nl_color1[0] + (nl_color2[0] - nl_color1[0]) * (wavelength - nl_color1[1]) / (nl_color2[1] - nl_color1[1])
        else:
            nl_eff = nl_color1[0]  # Usar el primer índice de refracción si la longitud de onda es menor
            
        nl_eff_list.append(nl_eff)

    return nl_eff_list

# Ejemplo de uso:
wavelengths = {
    'red': 650,
    'green': 550,
    'blue': 450
}


nl = { 
    'red': [(1.5155, wavelengths['red'] 'crown'), (1.7245, wavelengths['red']  'flint' )],
    'green': [(1.5185, wavelengths['green'] '''crown''' ), (1.7337, wavelengths['green'])'''flint''' ],
    'blue': [(1.5253, wavelengths['blue'] '''crown'''), (1.7569, wavelengths['blue']) '''flint''']
    }


wavelength = 600  # Cambia la longitud de onda según tus necesidades
results = calculate_effective_index(nl, wavelength, wavelengths)

for i, result in enumerate(results):
    print(f"Índice de refracción efectivo {i + 1} en {wavelength} nm para {list(nl.keys())[i]} es {result}")
 

def ray_tracing(width, height, rayo, so, n1, obj, res, nl, R1, R2, dl, aberration, pixels):
    for i in range(width):
        for j in range(height):
            pos_x = i
            pos_y = j
            pixel = obj.getpixel((pos_x, pos_y))
            x = pos_x - width / 2
            y = pos_y - height / 2
            r = math.sqrt(x * x + y * y) + 1
            y_objeto = r * res

            if aberration:
                if rayo == CHIEF_RAY:
                    wavelength = wavelengths['green']
                else:
                    wavelength = wavelengths['blue']
                
                C = 0.0  # Asigna un valor adecuado a la constante C
                nl_eff = calculate_effective_index(nl, wavelength)
                f = ((nl - 1) * ((1 / R1) - (1 / R2)))
                f = f + (C * (y_objeto ** 4 - y_objeto ** 2))
                f = 1 / f
                si = (f * so) / (so - f)
                A = compute_lens_matrix(nl, R1, R2, dl)
                P2 = np.array([[1, 0], [si / n1, 1]])
                P1 = np.array([[1, 0], [-so / n1, 1]])

                if rayo == 0:
                    alpha_entrada = math.atan(y_objeto / so)
                elif rayo == 1:
                    alpha_entrada = 0
                V_entrada = np.array([n1 * alpha_entrada, y_objeto])
                V_salida = P2.dot(A.dot(P1.dot(V_entrada)))
                y_imagen = V_salida[1]

                if rayo == 0:
                    Mt = (-1) * y_imagen / y_objeto
                elif rayo == 1:
                    Mt = y_imagen / y_objeto

                x_prime = Mt * x
                y_prime = Mt * y
                pos_x_prime = int(x_prime + width_output / 2)
                pos_y_prime = int(y_prime + height_output / 2)

                if 0 <= pos_x_prime < width_output and 0 <= pos_y_prime < height_output:
                    if rayo == 0:
                        pixels[pos_x_prime, pos_y_prime] = (int(pixel), int(pixel), int(pixel))
                    elif rayo == 1:
                        new_gray = (int(pixel) + pixels[pos_x_prime, pos_y_prime][0]) / 2
                        pix_fin = (int(new_gray), int(new_gray), int(new_gray))
                        pixels[pos_x_prime, pos_y_prime] = pix_fin

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

    obj = Image.open("saturn.jpg", "r")
    height, width, _ = image.shape
    width_output = int(image_array.shape[1] * abs(Mt))
    height_output = int(image_array.shape[0] * abs(Mt))
    pixels = ray_tracing(image_array, width_output, height_output, aberration, interpolate)
    save_image(pixels, 'output.png')

    image = Image.new("RGB", (width_output, height_output), "white")
    pixels = image.load()

    pixels = ray_tracing(width, height, CHIEF_RAY, so, n1, obj, res, nl, R1, R2, dl, aberration, pixels)
    pixels = ray_tracing(width, height, PARALLEL_RAY, so, n1, obj, res, nl, R1, R2, dl, aberration, pixels)

    if interpolate:
        pixels = interpolation(pixels)
        print("Interpolation performed")

    image_corrected = corregir_aberracion_cromatica(pixels)
    image_corrected = Image.fromarray(image_corrected)
    image_corrected.save('output.png', format='PNG')
