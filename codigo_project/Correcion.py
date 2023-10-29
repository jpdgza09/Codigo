import math
import numpy as np
from PIL import Image
from scipy.interpolate import Rbf

def robust_interpolation(pixels):
    arry = np.array(pixels[:, :, 0], dtype=float)
    # Coordenadas de píxeles no blancos
    nonwhite_coords = np.argwhere(arry != 255)
    nonwhite_pixels = arry[nonwhite_coords[:, 0], nonwhite_coords[:, 1]]
    # Coordenadas de píxeles blancos
    white_coords = np.argwhere(arry == 255)
    #TPS (Thin Plate Spline)
    interp = Rbf(nonwhite_coords[:, 0], nonwhite_coords[:, 1], nonwhite_pixels, function='thin-plate')
    # Interplando los píxeles blancos
    interpolated_pixels = interp(white_coords[:, 0], white_coords[:, 1])
    # Redondeando y conviertiendo los valores interpolados a enteros
    int_out = np.round(interpolated_pixels).astype(int)
    # Actualizanod los píxeles blancos en la imagen original
    for i in range(len(white_coords)):
        x, y = white_coords[i]
        pixels[x, y] = (int_out[i], int_out[i], int_out[i])
    return pixels


def compute_lens_matrix(nl, R1, R2, dl):
    D1 = (nl - 1) / R1
    D2 = (nl - 1) / (-R2)
    a1 = 1 - (D2 * dl) / nl
    a2 = -D1 - D2 + (D1 * D2 * dl / nl)
    a3 = dl / nl
    a4 = 1 - (D1 * dl) / nl
    A = np.array([[a1, a2], [a3, a4]])
    return A

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

def main():
    # Configuración de parámetros ópticos y variables
    aberration = True
    interpolate = True
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

    # Separar la imagen en los canales RGB
    r_channel, g_channel, b_channel = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]

    # Calcular tamaños de salida para cada canal
    width, height = image_array.shape[1], image_array.shape[0]
    width_output, height_output = int(abs(Mt) * width), int(abs(Mt) * height)

    # n_rgb
    n_rgb = {
        'channel': 'red',  # Cambiar a 'green' o 'blue' para los canales G y B
        'red': wavelengths['red'],
        'green': wavelengths['green'],
        'blue': wavelengths['blue']
    }

    # Llamar a ray_tracing para el rayo principal y operar en cada canal R, G y B
    result_r = ray_tracing(width, height, CHIEF_RAY, so, n1, image, res, R1, R2, dl, r_channel, n_rgb, width_output, height_output)
    result_g = ray_tracing(width, height, CHIEF_RAY, so, n1, image, res, R1, R2, dl, g_channel, n_rgb, width_output, height_output)
    result_b = ray_tracing(width, height, CHIEF_RAY, so, n1, image, res, R1, R2, dl, b_channel, n_rgb, width_output, height_output)

    # Llamar a ray_tracing para el rayo paralelo y operar en cada canal R, G y B
    result_r = ray_tracing(width, height, PARALLEL_RAY, so, n1, image, res, R1, R2, dl, result_r, n_rgb, width_output, height_output)
    result_g = ray_tracing(width, height, PARALLEL_RAY, so, n1, image, res, R1, R2, dl, result_g, n_rgb, width_output, height_output)
    result_b = ray_tracing(width, height, PARALLEL_RAY, so, n1, image, res, R1, R2, dl, result_b, n_rgb, width_output, height_output)

    # Interpolación (si se establece interpolate en True)
    if interpolate:
        interpolated_r = robust_interpolation(result_r)
        interpolated_g = robust_interpolation(result_g)
        interpolated_b = robust_interpolation(result_b)
        print("Interpolación realizada")

        # Combinar canales R, G y B en una sola imagen
        final_image = np.stack((interpolated_r, interpolated_g, interpolated_b), axis=-1)

        # Crear una imagen PIL a partir del arreglo NumPy
        final_image = Image.fromarray(final_image.astype('uint8'))

        # Guardar la imagen resultante
        final_image.save('output_combined.png')

        # Mostrar la imagen resultante
        final_image.show()

if __name__ == "__main__":
    main()
