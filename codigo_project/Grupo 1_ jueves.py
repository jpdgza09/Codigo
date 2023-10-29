import math
import numpy as np
from PIL import Image
from scipy.interpolate import Rbf

def robust_interpolation(image):
    # Obtener los canales de color R, G y B
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Coordenadas de píxeles no blancos en cada canal
    nonwhite_coords_r = np.argwhere(red_channel != 255)
    nonwhite_coords_g = np.argwhere(green_channel != 255)
    nonwhite_coords_b = np.argwhere(blue_channel != 255)

    # Valores de píxeles no blancos en cada canal
    nonwhite_pixels_r = red_channel[nonwhite_coords_r[:, 0], nonwhite_coords_r[:, 1]]
    nonwhite_pixels_g = green_channel[nonwhite_coords_g[:, 0], nonwhite_coords_g[:, 1]]
    nonwhite_pixels_b = blue_channel[nonwhite_coords_b[:, 0], nonwhite_coords_b[:, 1]]

    # Coordenadas de píxeles blancos en cada canal
    white_coords_r = np.argwhere(red_channel == 255)
    white_coords_g = np.argwhere(green_channel == 255)
    white_coords_b = np.argwhere(blue_channel == 255)

    # Interpolación para cada canal por separado
    interp_r = Rbf(nonwhite_coords_r[:, 0], nonwhite_coords_r[:, 1], nonwhite_pixels_r, function='thin-plate')
    interp_g = Rbf(nonwhite_coords_g[:, 0], nonwhite_coords_g[:, 1], nonwhite_pixels_g, function='thin-plate')
    interp_b = Rbf(nonwhite_coords_b[:, 0], nonwhite_coords_b[:, 1], nonwhite_pixels_b, function='thin-plate')

    # Interpolando los píxeles blancos en cada canal
    interpolated_pixels_r = interp_r(white_coords_r[:, 0], white_coords_r[:, 1])
    interpolated_pixels_g = interp_g(white_coords_g[:, 0], white_coords_g[:, 1])
    interpolated_pixels_b = interp_b(white_coords_b[:, 0], white_coords_b[:, 1])

    # Redondeando y convirtiendo los valores interpolados a enteros
    int_out_r = np.round(interpolated_pixels_r).astype(int)
    int_out_g = np.round(interpolated_pixels_g).astype(int)
    int_out_b = np.round(interpolated_pixels_b).astype(int)

    # Actualizando los píxeles blancos en la imagen original para cada canal
    for i in range(len(white_coords_r)):
        x, y = white_coords_r[i]
        image[x, y] = (int_out_r[i], int_out_g[i], int_out_b[i])

    return image

# Calculo e impresion de la matriz de la lente
def compute_lens_matrix(nl, R1, R2, dl):
    D1 = (nl - 1) / R1
    D2 = (nl - 1) / (-R2)

    
    a1 = 1 - (D2 * dl) / nl
    a2 = -D1 - D2 + (D1 * D2 * dl / nl)
    a3 = dl / nl
    a4 = 1 - (D1 * dl) / nl
    A = np.array([[a1, a2], [a3, a4]])
    return A

# Calculo  e impresion del índice de refracción efectivo para una longitud de onda dada
def calculate_effective_index(nl_color_list, wavelength):
    if len(nl_color_list) >= 2:
        nl_color1, nl_color2 = nl_color_list[:2]  # Tomar los primeros dos índices de refracción
        if isinstance(wavelength, (int, float)):
            if wavelength >= nl_color1[1]:
                nl_eff = nl_color1[0] + (nl_color2[0] - nl_color1[0]) * (wavelength - nl_color1[1]) / (nl_color2[1] - nl_color1[1])
            else:
                nl_eff = nl_color1[0]  # Usar el primer índice de refracción si la longitud de onda es menor
            return nl_eff
        else:
            raise ValueError("La longitud de onda debe ser un número entero o en punto flotante")
    else:
        raise ValueError("nl_color_list debe contener al menos dos valores de índice de refracción")


# Definición de las longitudes de onda para los canales de color
wavelengths = {
    'red': 620,
    'green': 550,
    'blue': 450
}

# Definición de los índices de refracción y la longitud de onda correspondiente para cada color
nl = {

    'red': [(1.5155, wavelengths['red'], 'crown'), (1.7245, wavelengths['red'], 'flint')],
    'green': [(1.5185, wavelengths['green'], 'crown'), (1.7337, wavelengths['green'], 'flint')],
    'blue': [(1.5253, wavelengths['blue'], 'crown'), (1.7569, wavelengths['blue'], 'flint')]
}

# Definir constantes para los rayos
CHIEF_RAY = 0
PARALLEL_RAY = 1


# Función de trazado de rayos
def ray_tracing(width, height, rayo, so, n1, obj, res, R1, R2, dl, pixels, n_rgb, width_output, height_output):
    for i in range(width):
        for j in range(height):
            #Se obtienen las oordenadas del píxel y el valor del píxel en la imagen original
            pos_x = i
            pos_y = j
            pixel = obj.getpixel((pos_x, pos_y))
            x = pos_x - width / 2
            y = pos_y - height / 2
            r = math.sqrt(x * x + y * y) + 1
            y_objeto = r * res

            if aberration==True:  # Si se aplica aberración, se separan las longitudes de onda para los canales R, G y B
                if rayo == CHIEF_RAY:
                    if n_rgb['channel'] == 'red':
                        wavelength = n_rgb['red']
                    elif n_rgb['channel'] == 'green':
                        wavelength = n_rgb['green']
                    else:
                        wavelength = n_rgb['blue']

                C = 0.20  # Asignacion de un valor apropiado a la constante C
                if n_rgb['channel'] == 'red':
                    nl_eff = calculate_effective_index(n_rgb['red'], wavelength)
                elif n_rgb['channel'] == 'green':
                    nl_eff = calculate_effective_index(n_rgb['green'], wavelength)
                else:
                    nl_eff = calculate_effective_index(n_rgb['blue'], wavelength)

                A = compute_lens_matrix(nl_eff, R1, R2, dl)
                P2 = np.array([[1, 0], [si / n1, 1]])  # Esta es q (después del lente)
                P1 = np.array([[1, 0], [-so / n1, 1]])  # Esta es p (antes del lente)

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

#Funcion Main donde se Configuran los parámetros ópticos y las variables de configuración
if __name__ == "__main__":
    # Define las variables de configuración
    aberration = True  # Asigna True o False según corresponda
    interpolate = True  # Asigna True o False según corresponda
    # Resto del código (las funciones definidas previamente)
    # Código de configuración y procesamiento de imágenes
    R1 = 0.2
    R2 = -0.2
    dl = 0.01
    nl = 1.5
    res = 0.0001
    f = R1 * R2 / ((R2 - R1) * (nl - 1))
    so = 0.18
    si = 1.54
    n1 = 1
    Mt = -si / so

# Ruta de la imagen en color
image_path = 'C:/2023-2/Optica/Codigo/codigo_project/mocosvirus.jpg'
# Cargar la imagen en tres canales de color
image = Image.open(image_path)

# Convertir la imagen a un arreglo NumPy
image_array = np.array(image)

# Cargar la imagen
height, width, _ = image_array.shape

# Separar la imagen en los canales RGB
r_channel = image_array[:, :, 0]
g_channel = image_array[:, :, 1]
b_channel = image_array[:, :, 2]

#Calculo del ancho y alto de cada canal R, G y B
width_r, height_r = r_channel.shape
width_g, height_g = g_channel.shape
width_b, height_b = b_channel.shape

# Calculo de los tamaños de salida para cada canal
width_output_r = int(width_r * abs(Mt))
height_output_r = int(height_r * abs(Mt))

width_output_g = int(width_g * abs(Mt))
height_output_g = int(height_g * abs(Mt))

width_output_b = int(width_b * abs(Mt))
height_output_b = int(height_b * abs(Mt))

# n_rgb
n_rgb = {
    'channel': 'red',  # Cambiar a 'green' o 'blue' para los canales G y B
    'red': wavelengths['red'],
    'green': wavelengths['green'],
    'blue': wavelengths['blue']
}

# Llamar a ray_tracing para el rayo principal y se opera para cada canal R, G y B
result_r = ray_tracing(width, height, CHIEF_RAY, so, n1, image, res, R1, R2, dl, r_channel, n_rgb, width_output_r, height_output_r)
result_g = ray_tracing(width, height, CHIEF_RAY, so, n1, image, res, R1, R2, dl, g_channel, n_rgb, width_output_g, height_output_g)
result_b = ray_tracing(width, height, CHIEF_RAY, so, n1, image, res, R1, R2, dl, b_channel, n_rgb, width_output_b, height_output_b)

# Llamar a ray_tracing para el rayo paralelo y se opera para cada canal R, G y B
result_r = ray_tracing(width, height, PARALLEL_RAY, so, n1, image, res, R1, R2, dl, result_r, n_rgb, width_output_r, height_output_r)
result_g = ray_tracing(width, height, PARALLEL_RAY, so, n1, image, res, R1, R2, dl, result_g, n_rgb, width_output_g, height_output_g)
result_b = ray_tracing(width, height, PARALLEL_RAY, so, n1, image, res, R1, R2, dl, result_b, n_rgb, width_output_b, height_output_b)

# Interpolación para los resultados de cada canal R, G y B utilizando la función robust_interpolation
interpolated_image = np.stack((result_r, result_g, result_b), axis=-1)
interpolated_image = robust_interpolation(interpolated_image)
print("Interpolación realizada")

# Crear una imagen PIL a partir del arreglo NumPy
final_image = Image.fromarray(interpolated_image.astype('uint8'))

# Guardar la imagen resultante
final_image.save('output_combined.png')

# Mostrar la imagen resultante
final_image.show()