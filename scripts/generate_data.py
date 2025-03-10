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

# Tamaño de las imágenes generadas
IMAGE_SIZE = 100
# Grosor de las líneas
LINE_THICKNESS = 5
# Color de las líneas (blanco)
LINE_COLOR = 255
# Tamaño del punto central
DOT_RADIUS = 7

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
    return parser.parse_args()

def generate_random_id():
    """Genera un ID alfanumérico aleatorio."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

def save_image(image, letter, output_dir):
    """Guarda la imagen con un nombre que incluye la letra y un ID aleatorio."""
    # Crear el directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar un ID aleatorio
    random_id = generate_random_id()
    
    # Crear el nombre del archivo
    filename = os.path.join(output_dir, f"{letter}_{random_id}.png")
    
    # Guardar la imagen
    cv2.imwrite(filename, image)
    
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
    Dibuja la letra A del cifrado Pigpen:
    - 2 líneas formando un ángulo de 90 grados abajo y a la derecha
    Si with_dot=True, dibuja la letra J (A con un punto en el centro)
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
    Dibuja la letra S del cifrado Pigpen:
    - 2 líneas diagonales formando una V
    Si with_dot=True, dibuja la letra W (S con un punto en el centro)
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
    a_img = draw_letter_A(False)
    save_image(a_img, 'A', output_dir)
    
    s_img = draw_letter_S(False)
    save_image(s_img, 'S', output_dir)
    
    b_img = draw_letter_B(False)
    save_image(b_img, 'B', output_dir)
    
    e_img = draw_letter_E(False)
    save_image(e_img, 'E', output_dir)
    
    # Generar las letras base con punto
    j_img = draw_letter_A(True)
    save_image(j_img, 'J', output_dir)
    
    w_img = draw_letter_S(True)
    save_image(w_img, 'W', output_dir)
    
    k_img = draw_letter_B(True)
    save_image(k_img, 'K', output_dir)
    
    n_img = draw_letter_E(True)
    save_image(n_img, 'N', output_dir)
    
    if debug:
        # Mostrar las imágenes para depuración
        cv2.imshow("A", a_img)
        cv2.imshow("S", s_img)
        cv2.imshow("B", b_img)
        cv2.imshow("E", e_img)
        cv2.imshow("J", j_img)
        cv2.imshow("W", w_img)
        cv2.imshow("K", k_img)
        cv2.imshow("N", n_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def generate_derived_letters(output_dir, debug=False):
    """Genera las letras derivadas mediante rotaciones y flips de las letras base."""
    
    # Derivadas de A
    a_image = draw_letter_A(False)
    c_image = flip_horizontal(a_image)
    save_image(c_image, 'C', output_dir)
    
    g_image = flip_vertical(a_image)
    save_image(g_image, 'G', output_dir)
    
    # Derivadas de C (que es A girada horizontalmente)
    i_image = flip_vertical(c_image)
    save_image(i_image, 'I', output_dir)
    
    # Derivadas de B
    b_image = draw_letter_B(False)
    f_image = rotate_90_clockwise(b_image)
    save_image(f_image, 'F', output_dir)
    
    h_image = flip_vertical(b_image)
    save_image(h_image, 'H', output_dir)
    
    # Derivadas de F (que es B girada 90 grados)
    d_image = flip_horizontal(f_image)
    save_image(d_image, 'D', output_dir)
    
    # Derivadas de S
    s_image = draw_letter_S(False)
    u_image = rotate_90_clockwise(s_image)
    save_image(u_image, 'U', output_dir)
    
    v_image = flip_vertical(s_image)
    save_image(v_image, 'V', output_dir)
    
    # Derivadas de U (que es S girada 90 grados)
    t_image = flip_horizontal(u_image)
    save_image(t_image, 'T', output_dir)
    
    # Derivadas de J
    j_image = draw_letter_A(True)
    l_image = flip_horizontal(j_image)
    save_image(l_image, 'L', output_dir)
    
    p_image = flip_vertical(j_image)
    save_image(p_image, 'P', output_dir)
    
    # Derivadas de L (que es J girada horizontalmente)
    r_image = flip_vertical(l_image)
    save_image(r_image, 'R', output_dir)
    
    # Derivadas de K
    k_image = draw_letter_B(True)
    o_image = rotate_90_clockwise(k_image)
    save_image(o_image, 'O', output_dir)
    
    q_image = flip_vertical(k_image)
    save_image(q_image, 'Q', output_dir)
    
    # Derivadas de O (que es K girada 90 grados)
    m_image = flip_horizontal(o_image)
    save_image(m_image, 'M', output_dir)
    
    # Derivadas de W
    w_image = draw_letter_S(True)
    y_image = rotate_90_clockwise(w_image)
    save_image(y_image, 'Y', output_dir)
    
    z_image = flip_vertical(w_image)
    save_image(z_image, 'Z', output_dir)
    
    # Derivadas de Y (que es W girada 90 grados)
    x_image = flip_horizontal(y_image)
    save_image(x_image, 'X', output_dir)

def main():
    """Función principal."""
    args = parse_arguments()
    
    # Actualizar variables globales según los argumentos
    global IMAGE_SIZE, LINE_THICKNESS
    IMAGE_SIZE = args.size
    LINE_THICKNESS = args.thickness
    
    print(f"Generando imágenes de caracteres del cifrado Pigpen...")
    print(f"Tamaño de imagen: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Grosor de línea: {LINE_THICKNESS}")
    print(f"Directorio de salida: {args.output}")
    
    # Generar las letras base
    generate_base_letters(args.output, args.debug)
    
    # Si se solicita, generar todas las letras
    if args.all:
        generate_derived_letters(args.output, args.debug)
    
    print("\nProceso completado. Se han generado todas las imágenes solicitadas.")

if __name__ == "__main__":
    main() 