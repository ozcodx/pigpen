#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para descifrar imágenes con cifrado Pigpen utilizando un modelo entrenado.
Puede procesar una imagen completa o una imagen que contenga un solo carácter.
"""

import os
import argparse
import glob
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont

# Mapa de conversión de índices a letras
IDX_TO_CHAR = {i: chr(65 + i) for i in range(26)}  # A-Z

def parse_args():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Descifra imágenes con cifrado Pigpen.')
    parser.add_argument('--input', type=str, required=True,
                        help='Ruta a la imagen a descifrar')
    parser.add_argument('--model', type=str, default=None,
                        help='Ruta al modelo entrenado (si no se especifica, se usa el más reciente)')
    parser.add_argument('--output', type=str, default=None,
                        help='Ruta para guardar la imagen con el texto descifrado')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Umbral para binarización (0.0-1.0)')
    parser.add_argument('--min_area', type=int, default=50,
                        help='Área mínima para considerar un contorno como carácter')
    parser.add_argument('--padding', type=int, default=2,
                        help='Padding alrededor de cada carácter detectado')
    parser.add_argument('--show', action='store_true',
                        help='Mostrar la imagen con el texto descifrado')
    
    return parser.parse_args()

def find_latest_model(models_dir='../models'):
    """Encuentra el modelo más reciente en el directorio de modelos."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"No se encontró el directorio de modelos: {models_dir}")
    
    model_files = list(models_dir.glob('pigpen_model_*.pth'))
    if not model_files:
        raise FileNotFoundError(f"No se encontraron modelos en {models_dir}")
    
    # Ordenar por fecha de modificación (más reciente primero)
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"Usando el modelo más reciente: {latest_model}")
    
    return latest_model

def load_model(model_path, device):
    """Carga un modelo entrenado desde un archivo."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    
    # Cargar el checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determinar el tipo de modelo basado en el nombre del archivo
    model_name = Path(model_path).stem
    if 'resnet18' in model_name:
        model = models.resnet18(weights=None)
    elif 'resnet34' in model_name:
        model = models.resnet34(weights=None)
    elif 'resnet50' in model_name:
        model = models.resnet50(weights=None)
    else:
        # Por defecto, usar resnet18
        model = models.resnet18(weights=None)
    
    # Obtener las clases del checkpoint
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
    else:
        # Si no hay clases en el checkpoint, asumir A-Z
        classes = [chr(65 + i) for i in range(26)]
    
    # Modificar la última capa para el número correcto de clases
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(classes))
    
    # Cargar los pesos del modelo
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, classes

def preprocess_image(image_path, threshold=0.5):
    """Preprocesa la imagen para la detección de caracteres."""
    # Leer la imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral adaptativo para binarizar
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Alternativamente, usar umbral simple
    # _, binary = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY_INV)
    
    return image, binary

def find_characters(binary_image, min_area=100, padding=10):
    """Encuentra los caracteres en la imagen binarizada."""
    # Encontrar contornos
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos por área y obtener bounding boxes
    char_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Añadir padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(binary_image.shape[1] - x, w + 2 * padding)
            h = min(binary_image.shape[0] - y, h + 2 * padding)
            char_regions.append((x, y, w, h))
    
    # Ordenar regiones de izquierda a derecha
    char_regions.sort(key=lambda r: r[0])
    
    return char_regions

def predict_character(model, image_tensor, classes, device):
    """Predice el carácter en una imagen tensor."""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
        
        return classes[predicted.item()], confidence

def process_image(image_path, model, classes, device, args):
    """Procesa una imagen completa para descifrar el texto."""
    # Preprocesar la imagen
    original_image, binary_image = preprocess_image(image_path, args.threshold)
    
    # Encontrar regiones de caracteres
    char_regions = find_characters(binary_image, args.min_area, args.padding)
    
    # Transformación para el modelo
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Crear una copia de la imagen original para dibujar
    result_image = original_image.copy()
    
    # Procesar cada región
    decrypted_text = ""
    for i, (x, y, w, h) in enumerate(char_regions):
        # Extraer la región
        char_image = binary_image[y:y+h, x:x+w]
        
        # Convertir a PIL Image
        pil_image = Image.fromarray(char_image)
        
        # Si la imagen es en escala de grises, convertirla a RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Aplicar transformación
        image_tensor = transform(pil_image)
        
        # Predecir
        char, confidence = predict_character(model, image_tensor, classes, device)
        decrypted_text += char
        
        # Dibujar rectángulo y texto en la imagen
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, char, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    print(f"Texto descifrado: {decrypted_text}")
    
    # Mostrar la imagen con el texto descifrado
    if args.show:
        cv2.imshow("Texto Descifrado", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Guardar la imagen con el texto descifrado
    if args.output:
        cv2.imwrite(args.output, result_image)
        print(f"Imagen con texto descifrado guardada en: {args.output}")
    
    return decrypted_text, result_image

def main():
    """Función principal."""
    args = parse_args()
    
    # Determinar el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar el modelo
    model_path = args.model
    if model_path is None:
        model_path = find_latest_model()
    
    model, classes = load_model(model_path, device)
    print(f"Modelo cargado con {len(classes)} clases: {classes}")
    
    # Procesar la imagen
    decrypted_text, result_image = process_image(args.input, model, classes, device, args)
    
    return decrypted_text

if __name__ == "__main__":
    main() 