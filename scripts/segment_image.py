#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para segmentar imágenes que contienen múltiples caracteres del cifrado Pigpen
y dividirlas en imágenes individuales para cada carácter.
"""

import os
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from datetime import datetime

def parse_arguments():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Segmenta una imagen con caracteres Pigpen en caracteres individuales.')
    parser.add_argument('--input', '-i', required=True, help='Ruta a la imagen de entrada')
    parser.add_argument('--output', '-o', default='data/unclassified', help='Directorio de salida para los caracteres segmentados')
    parser.add_argument('--min-size', type=int, default=20, help='Tamaño mínimo de los componentes conectados')
    parser.add_argument('--padding', type=int, default=10, help='Padding alrededor de cada carácter')
    parser.add_argument('--debug', action='store_true', help='Mostrar imágenes de depuración')
    return parser.parse_args()

def preprocess_image(image):
    """
    Preprocesa la imagen para facilitar la segmentación.
    
    Args:
        image: Imagen de entrada en formato BGR
        
    Returns:
        Imagen binaria preprocesada
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Aplicar umbral adaptativo para binarizar la imagen
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Aplicar operaciones morfológicas para eliminar ruido
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary

def segment_characters(binary_image, min_size=20, padding=10, debug=False):
    """
    Segmenta los caracteres individuales de la imagen binaria.
    
    Args:
        binary_image: Imagen binaria preprocesada
        min_size: Tamaño mínimo de los componentes conectados
        padding: Padding alrededor de cada carácter
        debug: Si es True, muestra imágenes de depuración
        
    Returns:
        Lista de imágenes de caracteres segmentados
    """
    # Etiquetar componentes conectados
    labels = measure.label(binary_image, connectivity=2)
    props = measure.regionprops(labels)
    
    # Filtrar regiones por tamaño y extraer caracteres
    characters = []
    
    if debug:
        # Crear una copia de la imagen original para visualización
        debug_img = cv2.cvtColor(binary_image.copy(), cv2.COLOR_GRAY2BGR)
    
    for prop in props:
        if prop.area < min_size:
            continue
        
        # Obtener las coordenadas del bounding box
        minr, minc, maxr, maxc = prop.bbox
        
        # Añadir padding
        minr = max(0, minr - padding)
        minc = max(0, minc - padding)
        maxr = min(binary_image.shape[0], maxr + padding)
        maxc = min(binary_image.shape[1], maxc + padding)
        
        # Extraer el carácter
        char_img = binary_image[minr:maxr, minc:maxc]
        
        # Asegurarse de que la imagen no está vacía
        if char_img.size > 0 and char_img.any():
            characters.append({
                'image': char_img,
                'bbox': (minr, minc, maxr, maxc)
            })
        
        if debug:
            # Dibujar rectángulo en la imagen de depuración
            cv2.rectangle(debug_img, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
    
    if debug and characters:
        plt.figure(figsize=(10, 8))
        plt.imshow(debug_img)
        plt.title('Caracteres Detectados')
        plt.axis('off')
        plt.show()
        
        # Mostrar los caracteres segmentados
        fig, axes = plt.subplots(1, len(characters), figsize=(15, 3))
        if len(characters) == 1:
            axes = [axes]
        
        for i, char in enumerate(characters):
            axes[i].imshow(char['image'], cmap='gray')
            axes[i].set_title(f'Char {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return characters

def save_characters(characters, output_dir):
    """
    Guarda los caracteres segmentados en el directorio de salida.
    
    Args:
        characters: Lista de imágenes de caracteres segmentados
        output_dir: Directorio donde se guardarán las imágenes
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar un timestamp único para este lote de imágenes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar cada carácter como una imagen individual
    for i, char in enumerate(characters):
        filename = os.path.join(output_dir, f'char_{timestamp}_{i+1:03d}.png')
        cv2.imwrite(filename, char['image'])
        print(f'Carácter guardado: {filename}')

def main():
    """Función principal."""
    args = parse_arguments()
    
    # Verificar que la imagen de entrada existe
    if not os.path.isfile(args.input):
        print(f"Error: No se pudo encontrar la imagen de entrada: {args.input}")
        return
    
    # Cargar la imagen
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: No se pudo cargar la imagen: {args.input}")
        return
    
    # Preprocesar la imagen
    binary = preprocess_image(image)
    
    # Segmentar caracteres
    characters = segment_characters(
        binary, 
        min_size=args.min_size, 
        padding=args.padding,
        debug=args.debug
    )
    
    if not characters:
        print("No se detectaron caracteres en la imagen.")
        return
    
    print(f"Se detectaron {len(characters)} caracteres.")
    
    # Guardar los caracteres segmentados
    save_characters(characters, args.output)
    
    print(f"Proceso completado. Los caracteres segmentados se guardaron en: {args.output}")

if __name__ == "__main__":
    main() 