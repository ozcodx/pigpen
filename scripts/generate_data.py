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

def parse_arguments():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Genera imágenes de caracteres del cifrado Pigpen.')
    parser.add_argument('--output', '-o', default='data/unclassified', 
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
    return parser.parse_args()

def generate_random_id():
    """Genera un ID alfanumérico aleatorio."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

def apply_mutations(image):
    """
    Aplica mutaciones aleatorias a la imagen para generar variaciones únicas.
    Incluye: wrap, tilt, noise, shift, y otras transformaciones.
    
    Returns:
        tuple: (imagen_mutada, lista_de_efectos_aplicados)
    """
    # Crear una copia de la imagen para no modificar la original
    mutated_image = image.copy()
    
    # Lista de posibles mutaciones
    mutations = [
        'wrap', 'tilt', 'noise', 'shift', 'blur', 'scale', 'perspective'
    ]
    
    # Seleccionar aleatoriamente 2-4 mutaciones para aplicar (aumentado de 1-3)
    num_mutations = random.randint(2, 4)
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
    
    return mutated_image, applied_effects

def save_image(image, letter, output_dir):
    """Guarda la imagen con un nombre que incluye la letra y un ID aleatorio."""
    # Crear el directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Aplicar mutaciones aleatorias a la imagen
    mutated_image, applied_effects = apply_mutations(image)
    
    # Generar un ID aleatorio
    random_id = generate_random_id()
    
    # Crear el nombre del archivo con los efectos aplicados
    effects_str = "_".join(applied_effects)
    filename = os.path.join(output_dir, f"{letter}_{effects_str}_{random_id}.png")
    
    # Guardar la imagen
    cv2.imwrite(filename, mutated_image)
    
    return filename

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

def generate_base_letters(output_dir, debug=False):
    """Genera las imágenes base del cifrado Pigpen."""
    
    # Generar las letras base sin punto
    i_img = draw_letter_A(False)
    save_image(i_img, 'I', output_dir)
    
    v_img = draw_letter_S(False)
    save_image(v_img, 'V', output_dir)
    
    b_img = draw_letter_B(False)
    save_image(b_img, 'B', output_dir)
    
    e_img = draw_letter_E(False)
    save_image(e_img, 'E', output_dir)
    
    # Generar las letras base con punto
    r_img = draw_letter_A(True)
    save_image(r_img, 'R', output_dir)
    
    z_img = draw_letter_S(True)
    save_image(z_img, 'Z', output_dir)
    
    k_img = draw_letter_B(True)
    save_image(k_img, 'K', output_dir)
    
    n_img = draw_letter_E(True)
    save_image(n_img, 'N', output_dir)
    
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

def generate_derived_letters(output_dir, debug=False):
    """Genera las letras derivadas mediante rotaciones y flips de las letras base."""
    
    # Derivadas de I (antes A)
    i_image = draw_letter_A(False)
    g_image = flip_horizontal(i_image)
    save_image(g_image, 'G', output_dir)
    
    c_image = flip_vertical(i_image)
    save_image(c_image, 'C', output_dir)
    
    # Derivadas de G (antes C)
    a_image = flip_vertical(g_image)
    save_image(a_image, 'A', output_dir)
    
    # Derivadas de B
    b_image = draw_letter_B(False)
    f_image = rotate_90_clockwise(b_image)
    save_image(f_image, 'F', output_dir)
    
    h_image = flip_vertical(b_image)
    save_image(h_image, 'H', output_dir)
    
    # Derivadas de F (que es B girada 90 grados)
    d_image = flip_horizontal(f_image)
    save_image(d_image, 'D', output_dir)
    
    # Derivadas de V (antes S)
    v_image = draw_letter_S(False)
    t_image = rotate_90_clockwise(v_image)
    save_image(t_image, 'T', output_dir)
    
    s_image = flip_vertical(v_image)
    save_image(s_image, 'S', output_dir)
    
    # Derivadas de T (que es V girada 90 grados)
    u_image = flip_horizontal(t_image)
    save_image(u_image, 'U', output_dir)
    
    # Derivadas de R (antes J)
    r_image = draw_letter_A(True)
    p_image = flip_horizontal(r_image)
    save_image(p_image, 'P', output_dir)
    
    l_image = flip_vertical(r_image)
    save_image(l_image, 'L', output_dir)
    
    # Derivadas de P (que es R girada horizontalmente)
    j_image = flip_vertical(p_image)
    save_image(j_image, 'J', output_dir)
    
    # Derivadas de K
    k_image = draw_letter_B(True)
    o_image = rotate_90_clockwise(k_image)
    save_image(o_image, 'O', output_dir)
    
    q_image = flip_vertical(k_image)
    save_image(q_image, 'Q', output_dir)
    
    # Derivadas de O (que es K girada 90 grados)
    m_image = flip_horizontal(o_image)
    save_image(m_image, 'M', output_dir)
    
    # Derivadas de Z (antes W)
    z_image = draw_letter_S(True)
    x_image = rotate_90_clockwise(z_image)
    save_image(x_image, 'X', output_dir)
    
    w_image = flip_vertical(z_image)
    save_image(w_image, 'W', output_dir)
    
    # Derivadas de X (que es Z girada 90 grados)
    y_image = flip_horizontal(x_image)
    save_image(y_image, 'Y', output_dir)

def main():
    """Función principal."""
    args = parse_arguments()
    
    # Actualizar variables globales según los argumentos
    global IMAGE_SIZE, LINE_THICKNESS, MUTATION_INTENSITY
    IMAGE_SIZE = args.size
    LINE_THICKNESS = args.thickness
    MUTATION_INTENSITY = args.mutation_intensity
    
    # Convertir la ruta de salida a una ruta absoluta si es relativa
    output_dir = args.output
    if not os.path.isabs(output_dir):
        # Si la ruta es relativa, convertirla a absoluta basada en el directorio actual
        output_dir = os.path.abspath(output_dir)
    
    print(f"Generando imágenes de caracteres del cifrado Pigpen...")
    print(f"Tamaño de imagen: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Grosor de línea: {LINE_THICKNESS}")
    print(f"Intensidad de mutaciones: {MUTATION_INTENSITY}")
    print(f"Directorio de salida: {output_dir}")
    
    # Generar las letras base
    generate_base_letters(output_dir, args.debug)
    
    # Si se solicita, generar todas las letras
    if args.all:
        generate_derived_letters(output_dir, args.debug)
    
    print("\nProceso completado. Se han generado todas las imágenes solicitadas.")

if __name__ == "__main__":
    main() 