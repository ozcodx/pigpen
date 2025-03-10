#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para generar imágenes de caracteres del cifrado Pigpen.
Genera las imágenes base y sus rotaciones para formar todas las letras del alfabeto.
"""

import os
import cv2
import numpy as np
import argparse
import random
import string
from datetime import datetime
import math

# Tamaño de las imágenes generadas
IMAGE_SIZE = 100
# Grosor de las líneas
LINE_THICKNESS = 5
# Color de las líneas (blanco)
LINE_COLOR = 255
# Tamaño del punto central
DOT_RADIUS = 7
# Intensidad de las mutaciones (0-1)
MUTATION_INTENSITY = 0.8
# Variables globales
MIN_MUTATIONS = 3
MAX_MUTATIONS = 5

def parse_arguments():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Genera imágenes de caracteres del cifrado Pigpen.')
    parser.add_argument('--output', '-o', default='data/generated', 
                        help='Directorio de salida para las imágenes generadas')
    parser.add_argument('--size', type=int, default=IMAGE_SIZE, 
                        help='Tamaño de las imágenes generadas (ancho=alto)')
    parser.add_argument('--thickness', type=int, default=LINE_THICKNESS, 
                        help='Grosor de las líneas')
    parser.add_argument('--all', action='store_true', 
                        help='Generar todas las letras (no solo las base)')
    parser.add_argument('--debug', action='store_true',
                        help='Mostrar imágenes para depuración')
    parser.add_argument('--mutation-intensity', type=float, default=MUTATION_INTENSITY,
                        help='Intensidad de las mutaciones (0-1)')
    parser.add_argument('--min-mutations', type=int, default=MIN_MUTATIONS,
                        help='Número mínimo de mutaciones a aplicar por imagen')
    parser.add_argument('--max-mutations', type=int, default=MAX_MUTATIONS,
                        help='Número máximo de mutaciones a aplicar por imagen')
    parser.add_argument('--count', '-c', type=int, default=10,
                        help='Número de imágenes a generar por letra (por defecto: 10)')
    return parser.parse_args()

def generate_random_id():
    """Genera un ID alfanumérico aleatorio."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

def apply_mutations(image):
    """
    Aplica mutaciones aleatorias a la imagen para generar variaciones únicas.
    Incluye: wrap, tilt, noise, shift, blur, scale, perspective, erosion, dilation, waves, rotation.
    
    Returns:
        tuple: (imagen_mutada, lista_de_efectos_aplicados)
    """
    # Crear una copia de la imagen para no modificar la original
    mutated_image = image.copy()
    
    # Lista de posibles mutaciones
    mutations = [
        'wrap', 'tilt', 'noise', 'shift', 'blur', 'scale', 'perspective',
        'erosion', 'dilation', 'waves', 'rotation'
    ]
    
    # Seleccionar aleatoriamente MIN_MUTATIONS-MAX_MUTATIONS mutaciones para aplicar
    num_mutations = random.randint(MIN_MUTATIONS, MAX_MUTATIONS)
    selected_mutations = random.sample(mutations, min(num_mutations, len(mutations)))
    
    # Lista para almacenar los efectos aplicados
    applied_effects = []
    
    # Aplicar las mutaciones seleccionadas
    for mutation in selected_mutations:
        if mutation == 'wrap':
            # Aplicar una deformación de tipo "wrap" más intensa
            rows, cols = mutated_image.shape
            # Crear una matriz de mapeo para la deformación
            map_x = np.zeros((rows, cols), dtype=np.float32)
            map_y = np.zeros((rows, cols), dtype=np.float32)
            
            # Calcular la intensidad de la deformación (aumentada)
            intensity = random.uniform(0.01, MUTATION_INTENSITY * 0.05)
            
            # Crear la deformación con frecuencia más alta
            for i in range(rows):
                for j in range(cols):
                    map_x[i, j] = j + intensity * rows * math.sin(i / (random.uniform(3, 10)))
                    map_y[i, j] = i + intensity * cols * math.cos(j / (random.uniform(3, 10)))
            
            # Aplicar la deformación
            mutated_image = cv2.remap(mutated_image, map_x, map_y, cv2.INTER_LINEAR)
            applied_effects.append(f"wrap{intensity:.2f}")
        
        elif mutation == 'tilt':
            # Aplicar una inclinación más pronunciada
            rows, cols = mutated_image.shape
            # Calcular el ángulo de inclinación (entre -15 y 15 grados, aumentado de -5 a 5)
            angle = random.uniform(-15 * MUTATION_INTENSITY, 15 * MUTATION_INTENSITY)
            # Calcular el centro de la imagen
            center = (cols // 2, rows // 2)
            # Crear la matriz de rotación
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Aplicar la rotación
            mutated_image = cv2.warpAffine(mutated_image, M, (cols, rows), borderValue=0)
            applied_effects.append(f"tilt{angle:.1f}")
        
        elif mutation == 'noise':
            # Añadir ruido aleatorio más intenso
            noise = np.zeros(mutated_image.shape, np.uint8)
            # Calcular la intensidad del ruido (aumentada)
            noise_intensity = random.uniform(0.05, MUTATION_INTENSITY * 0.25)
            cv2.randn(noise, 0, noise_intensity * 255)
            # Añadir el ruido a la imagen
            mutated_image = cv2.add(mutated_image, noise)
            applied_effects.append(f"noise{noise_intensity:.2f}")
        
        elif mutation == 'shift':
            # Desplazar más notablemente la imagen
            rows, cols = mutated_image.shape
            # Calcular el desplazamiento (aumentado)
            shift_x = random.randint(-int(cols * MUTATION_INTENSITY * 0.25), int(cols * MUTATION_INTENSITY * 0.25))
            shift_y = random.randint(-int(rows * MUTATION_INTENSITY * 0.25), int(rows * MUTATION_INTENSITY * 0.25))
            # Crear la matriz de transformación
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            # Aplicar el desplazamiento
            mutated_image = cv2.warpAffine(mutated_image, M, (cols, rows), borderValue=0)
            applied_effects.append(f"shift{shift_x}x{shift_y}")
        
        elif mutation == 'blur':
            # Aplicar un desenfoque más notable
            blur_intensity = random.randint(1, max(1, int(5 * MUTATION_INTENSITY)))
            # Asegurar que el valor es impar
            if blur_intensity % 2 == 0:
                blur_intensity += 1
            mutated_image = cv2.GaussianBlur(mutated_image, (blur_intensity, blur_intensity), 0)
            applied_effects.append(f"blur{blur_intensity}")
        
        elif mutation == 'scale':
            # Escalar más notablemente la imagen
            rows, cols = mutated_image.shape
            # Calcular el factor de escala (aumentado)
            scale_factor = random.uniform(1 - MUTATION_INTENSITY * 0.3, 1 + MUTATION_INTENSITY * 0.3)
            # Calcular las nuevas dimensiones
            new_width = int(cols * scale_factor)
            new_height = int(rows * scale_factor)
            # Escalar la imagen
            scaled_image = cv2.resize(mutated_image, (new_width, new_height))
            # Crear una imagen del tamaño original
            result_image = np.zeros((rows, cols), dtype=np.uint8)
            # Calcular las coordenadas para centrar la imagen escalada
            x_offset = max(0, (cols - new_width) // 2)
            y_offset = max(0, (rows - new_height) // 2)
            # Copiar la imagen escalada en la imagen original
            if scale_factor < 1:
                # Si la imagen es más pequeña, centrarla
                result_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled_image
            else:
                # Si la imagen es más grande, recortarla
                crop_x = max(0, (new_width - cols) // 2)
                crop_y = max(0, (new_height - rows) // 2)
                result_image = scaled_image[crop_y:crop_y+rows, crop_x:crop_x+cols]
            
            mutated_image = result_image
            applied_effects.append(f"scale{scale_factor:.2f}")
        
        elif mutation == 'perspective':
            # Aplicar una transformación de perspectiva más pronunciada
            rows, cols = mutated_image.shape
            # Definir los puntos de origen
            pts1 = np.float32([
                [0, 0],
                [cols-1, 0],
                [0, rows-1],
                [cols-1, rows-1]
            ])
            # Definir los puntos de destino con variaciones más grandes
            max_offset = int(MUTATION_INTENSITY * cols * 0.15)  # Aumentado de 0.05 a 0.15
            pts2 = np.float32([
                [random.randint(0, max_offset), random.randint(0, max_offset)],
                [cols-1-random.randint(0, max_offset), random.randint(0, max_offset)],
                [random.randint(0, max_offset), rows-1-random.randint(0, max_offset)],
                [cols-1-random.randint(0, max_offset), rows-1-random.randint(0, max_offset)]
            ])
            # Calcular la matriz de transformación
            M = cv2.getPerspectiveTransform(pts1, pts2)
            # Aplicar la transformación
            mutated_image = cv2.warpPerspective(mutated_image, M, (cols, rows), borderValue=0)
            applied_effects.append(f"persp{max_offset}")
            
        elif mutation == 'erosion':
            # Aplicar erosión para adelgazar las líneas
            kernel_size = random.randint(1, max(1, int(3 * MUTATION_INTENSITY)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mutated_image = cv2.erode(mutated_image, kernel, iterations=1)
            applied_effects.append(f"erosion{kernel_size}")
            
        elif mutation == 'dilation':
            # Aplicar dilatación para engrosar las líneas
            kernel_size = random.randint(1, max(1, int(3 * MUTATION_INTENSITY)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mutated_image = cv2.dilate(mutated_image, kernel, iterations=1)
            applied_effects.append(f"dilation{kernel_size}")
            
        elif mutation == 'waves':
            # Aplicar efecto de ondas
            rows, cols = mutated_image.shape
            # Crear matrices de mapeo
            map_x = np.zeros((rows, cols), dtype=np.float32)
            map_y = np.zeros((rows, cols), dtype=np.float32)
            
            # Calcular parámetros de las ondas
            x_amplitude = random.uniform(3, 10) * MUTATION_INTENSITY
            y_amplitude = random.uniform(3, 10) * MUTATION_INTENSITY
            x_wavelength = random.uniform(30, 100)
            y_wavelength = random.uniform(30, 100)
            
            # Crear el efecto de ondas
            for i in range(rows):
                for j in range(cols):
                    map_x[i, j] = j + x_amplitude * math.sin(i / x_wavelength)
                    map_y[i, j] = i + y_amplitude * math.cos(j / y_wavelength)
            
            # Aplicar la distorsión
            mutated_image = cv2.remap(mutated_image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            applied_effects.append(f"waves{x_amplitude:.1f}_{y_amplitude:.1f}")
            
        elif mutation == 'rotation':
            # Aplicar una rotación aleatoria
            rows, cols = mutated_image.shape
            # Calcular el ángulo de rotación (entre -15 y 15 grados)
            angle = random.uniform(-15 * MUTATION_INTENSITY, 15 * MUTATION_INTENSITY)
            # Calcular el centro de la imagen
            center = (cols // 2, rows // 2)
            # Crear la matriz de rotación
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Aplicar la rotación
            mutated_image = cv2.warpAffine(mutated_image, M, (cols, rows), borderValue=0)
            applied_effects.append(f"rot{angle:.1f}")
    
    return mutated_image, applied_effects

def save_image(image, letter, output_dir):
    """Guarda la imagen con un nombre que incluye la letra y un ID aleatorio."""
    # Crear el directorio específico para la letra
    letter_dir = os.path.join(output_dir, letter)
    os.makedirs(letter_dir, exist_ok=True)
    
    # Aplicar mutaciones aleatorias a la imagen
    mutated_image, applied_effects = apply_mutations(image)
    
    # Generar un ID aleatorio
    random_id = generate_random_id()
    
    # Crear el nombre del archivo con los efectos aplicados
    effects_str = "_".join(applied_effects)
    filename = os.path.join(letter_dir, f"{effects_str}_{random_id}.png")
    
    # Guardar la imagen
    cv2.imwrite(filename, mutated_image)
    
    return filename

def generate_multiple_images(image, letter, output_dir, count=10):
    """Genera múltiples imágenes mutadas para una letra."""
    filenames = []
    for _ in range(count):
        filename = save_image(image, letter, output_dir)
        filenames.append(filename)
    return filenames

def create_base_image():
    """Crea una imagen base en blanco."""
    # Crear una imagen en negro
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    
    # Calcular el margen para centrar el símbolo
    margin = int(IMAGE_SIZE * 0.2)
    
    return image, margin

def draw_letter_A(with_dot=False):
    """
    Dibuja la letra I del cifrado Pigpen:
    - 2 líneas formando un ángulo de 90 grados abajo y a la derecha
    Si with_dot=True, dibuja la letra R (I con un punto en el centro)
    """
    image, margin = create_base_image()
    
    # Puntos para las líneas (ángulo de 90 grados abajo y a la derecha)
    start_x = margin
    start_y = margin
    end_x = IMAGE_SIZE - margin
    end_y = IMAGE_SIZE - margin
    
    # Dibujar línea vertical (abajo)
    cv2.line(image, (start_x, start_y), (start_x, end_y), LINE_COLOR, LINE_THICKNESS)
    
    # Dibujar línea horizontal (derecha)
    cv2.line(image, (start_x, start_y), (end_x, start_y), LINE_COLOR, LINE_THICKNESS)
    
    # Si se requiere, añadir un punto en el centro
    if with_dot:
        center_x = int((start_x + end_x) // 2)
        center_y = int((start_y + end_y) // 2)
        cv2.circle(image, (center_x, center_y), DOT_RADIUS, LINE_COLOR, -1)
    
    return image

def draw_letter_S(with_dot=False):
    """
    Dibuja la letra V del cifrado Pigpen:
    - 2 líneas diagonales formando una V
    Si with_dot=True, dibuja la letra Z (V con un punto en el centro)
    """
    image, margin = create_base_image()
    
    # Puntos para las líneas diagonales
    top = margin
    bottom = IMAGE_SIZE - margin
    left = margin
    right = IMAGE_SIZE - margin
    
    # Dibujar línea diagonal izquierda
    cv2.line(image, (left, bottom), ((left + right) // 2, top), LINE_COLOR, LINE_THICKNESS)
    
    # Dibujar línea diagonal derecha
    cv2.line(image, (right, bottom), ((left + right) // 2, top), LINE_COLOR, LINE_THICKNESS)
    
    # Si se requiere, añadir un punto en el centro
    if with_dot:
        center_x = (left + right) // 2
        center_y = (top + bottom) // 3
        cv2.circle(image, (center_x, center_y*2), DOT_RADIUS, LINE_COLOR, -1)
    
    return image

def draw_letter_B(with_dot=False):
    """
    Dibuja la letra B del cifrado Pigpen:
    - 3 líneas: izquierda, derecha y abajo
    Si with_dot=True, dibuja la letra K (B con un punto en el centro)
    """
    image, margin = create_base_image()
    
    # Puntos para las líneas
    top = margin
    bottom = IMAGE_SIZE - margin
    left = margin
    right = IMAGE_SIZE - margin
    
    # Dibujar línea vertical izquierda
    cv2.line(image, (left, top), (left, bottom), LINE_COLOR, LINE_THICKNESS)
    
    # Dibujar línea vertical derecha
    cv2.line(image, (right, top), (right, bottom), LINE_COLOR, LINE_THICKNESS)
    
    # Dibujar línea horizontal abajo
    cv2.line(image, (left, bottom), (right, bottom), LINE_COLOR, LINE_THICKNESS)
    
    # Si se requiere, añadir un punto en el centro
    if with_dot:
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        cv2.circle(image, (center_x, center_y), DOT_RADIUS, LINE_COLOR, -1)
    
    return image

def draw_letter_E(with_dot=False):
    """
    Dibuja la letra E del cifrado Pigpen:
    - 4 líneas formando un cuadrado
    Si with_dot=True, dibuja la letra N (E con un punto en el centro)
    """
    image, margin = create_base_image()
    
    # Puntos para las líneas del cuadrado
    top = margin
    bottom = IMAGE_SIZE - margin
    left = margin
    right = IMAGE_SIZE - margin
    
    # Dibujar línea horizontal arriba
    cv2.line(image, (left, top), (right, top), LINE_COLOR, LINE_THICKNESS)
    
    # Dibujar línea horizontal abajo
    cv2.line(image, (left, bottom), (right, bottom), LINE_COLOR, LINE_THICKNESS)
    
    # Dibujar línea vertical izquierda
    cv2.line(image, (left, top), (left, bottom), LINE_COLOR, LINE_THICKNESS)
    
    # Dibujar línea vertical derecha
    cv2.line(image, (right, top), (right, bottom), LINE_COLOR, LINE_THICKNESS)
    
    # Si se requiere, añadir un punto en el centro
    if with_dot:
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        cv2.circle(image, (center_x, center_y), DOT_RADIUS, LINE_COLOR, -1)
    
    return image

def flip_horizontal(image):
    """Voltea la imagen horizontalmente."""
    return cv2.flip(image, 1)  # 1 para flip horizontal

def flip_vertical(image):
    """Voltea la imagen verticalmente."""
    return cv2.flip(image, 0)  # 0 para flip vertical

def rotate_90_clockwise(image):
    """Rota la imagen 90 grados en sentido horario."""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def generate_base_letters(output_dir, count=10, debug=False):
    """Genera las imágenes base del cifrado Pigpen."""
    
    # Generar las letras base sin punto
    i_img = draw_letter_A(False)
    generate_multiple_images(i_img, 'I', output_dir, count)
    
    v_img = draw_letter_S(False)
    generate_multiple_images(v_img, 'V', output_dir, count)
    
    b_img = draw_letter_B(False)
    generate_multiple_images(b_img, 'B', output_dir, count)
    
    e_img = draw_letter_E(False)
    generate_multiple_images(e_img, 'E', output_dir, count)
    
    # Generar las letras base con punto
    r_img = draw_letter_A(True)
    generate_multiple_images(r_img, 'R', output_dir, count)
    
    z_img = draw_letter_S(True)
    generate_multiple_images(z_img, 'Z', output_dir, count)
    
    k_img = draw_letter_B(True)
    generate_multiple_images(k_img, 'K', output_dir, count)
    
    n_img = draw_letter_E(True)
    generate_multiple_images(n_img, 'N', output_dir, count)
    
    if debug:
        # Mostrar las imágenes para depuración
        cv2.imshow("I", i_img)
        cv2.imshow("V", v_img)
        cv2.imshow("B", b_img)
        cv2.imshow("E", e_img)
        cv2.imshow("R", r_img)
        cv2.imshow("Z", z_img)
        cv2.imshow("K", k_img)
        cv2.imshow("N", n_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def generate_derived_letters(output_dir, count=10, debug=False):
    """Genera las letras derivadas mediante rotaciones y flips de las letras base."""
    
    # Derivadas de I (antes A)
    i_image = draw_letter_A(False)
    g_image = flip_horizontal(i_image)
    generate_multiple_images(g_image, 'G', output_dir, count)
    
    c_image = flip_vertical(i_image)
    generate_multiple_images(c_image, 'C', output_dir, count)
    
    # Derivadas de G (antes C)
    a_image = flip_vertical(g_image)
    generate_multiple_images(a_image, 'A', output_dir, count)
    
    # Derivadas de B
    b_image = draw_letter_B(False)
    f_image = rotate_90_clockwise(b_image)
    generate_multiple_images(f_image, 'F', output_dir, count)
    
    h_image = flip_vertical(b_image)
    generate_multiple_images(h_image, 'H', output_dir, count)
    
    # Derivadas de F (que es B girada 90 grados)
    d_image = flip_horizontal(f_image)
    generate_multiple_images(d_image, 'D', output_dir, count)
    
    # Derivadas de V (antes S)
    v_image = draw_letter_S(False)
    t_image = rotate_90_clockwise(v_image)
    generate_multiple_images(t_image, 'T', output_dir, count)
    
    s_image = flip_vertical(v_image)
    generate_multiple_images(s_image, 'S', output_dir, count)
    
    # Derivadas de T (que es V girada 90 grados)
    u_image = flip_horizontal(t_image)
    generate_multiple_images(u_image, 'U', output_dir, count)
    
    # Derivadas de R (antes J)
    r_image = draw_letter_A(True)
    p_image = flip_horizontal(r_image)
    generate_multiple_images(p_image, 'P', output_dir, count)
    
    l_image = flip_vertical(r_image)
    generate_multiple_images(l_image, 'L', output_dir, count)
    
    # Derivadas de P (que es R girada horizontalmente)
    j_image = flip_vertical(p_image)
    generate_multiple_images(j_image, 'J', output_dir, count)
    
    # Derivadas de K
    k_image = draw_letter_B(True)
    o_image = rotate_90_clockwise(k_image)
    generate_multiple_images(o_image, 'O', output_dir, count)
    
    q_image = flip_vertical(k_image)
    generate_multiple_images(q_image, 'Q', output_dir, count)
    
    # Derivadas de O (que es K girada 90 grados)
    m_image = flip_horizontal(o_image)
    generate_multiple_images(m_image, 'M', output_dir, count)
    
    # Derivadas de Z (antes W)
    z_image = draw_letter_S(True)
    x_image = rotate_90_clockwise(z_image)
    generate_multiple_images(x_image, 'X', output_dir, count)
    
    w_image = flip_vertical(z_image)
    generate_multiple_images(w_image, 'W', output_dir, count)
    
    # Derivadas de X (que es Z girada 90 grados)
    y_image = flip_horizontal(x_image)
    generate_multiple_images(y_image, 'Y', output_dir, count)

def main():
    """Función principal."""
    args = parse_arguments()
    
    # Actualizar variables globales según los argumentos
    global IMAGE_SIZE, LINE_THICKNESS, MUTATION_INTENSITY, MIN_MUTATIONS, MAX_MUTATIONS
    IMAGE_SIZE = args.size
    LINE_THICKNESS = args.thickness
    MUTATION_INTENSITY = args.mutation_intensity
    MIN_MUTATIONS = args.min_mutations
    MAX_MUTATIONS = args.max_mutations
    
    # Convertir la ruta de salida a una ruta absoluta si es relativa
    output_dir = args.output
    if not os.path.isabs(output_dir):
        # Si la ruta es relativa, convertirla a absoluta basada en el directorio actual
        output_dir = os.path.abspath(output_dir)
    
    print(f"Generando imágenes de caracteres del cifrado Pigpen...")
    print(f"Tamaño de imagen: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Grosor de línea: {LINE_THICKNESS}")
    print(f"Intensidad de mutaciones: {MUTATION_INTENSITY}")
    print(f"Número de mutaciones por imagen: {MIN_MUTATIONS}-{MAX_MUTATIONS}")
    print(f"Imágenes por letra: {args.count}")
    print(f"Directorio de salida: {output_dir}")
    
    # Calcular el número total de imágenes que se generarán
    total_letters = 8  # Letras base (I, V, B, E, R, Z, K, N)
    if args.all:
        total_letters = 26  # Todas las letras (A-Z)
    total_images = total_letters * args.count
    print(f"Se generarán un total de {total_images} imágenes ({total_letters} letras x {args.count} imágenes)")
    
    # Generar las letras base
    generate_base_letters(output_dir, args.count, args.debug)
    
    # Si se solicita, generar todas las letras
    if args.all:
        generate_derived_letters(output_dir, args.count, args.debug)
    
    print("\nProceso completado. Se han generado todas las imágenes solicitadas.")

if __name__ == "__main__":
    main() 