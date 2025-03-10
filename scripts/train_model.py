#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para entrenar un modelo de reconocimiento de cifrado Pigpen usando fastai.
Permite usar imágenes de las carpetas 'classified', 'generated' o ambas.
"""

import os
import argparse
import random
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from fastai.vision.all import *

def parse_args():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrena un modelo para reconocer caracteres del cifrado Pigpen.')
    parser.add_argument('--data_source', type=str, default='both', choices=['classified', 'generated', 'both'],
                        help='Fuente de datos para entrenamiento: classified, generated o both')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Número de épocas para entrenar')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamaño del batch para entrenamiento')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Tamaño de las imágenes para el modelo')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Tasa de aprendizaje inicial')
    parser.add_argument('--model_name', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Arquitectura del modelo a utilizar')
    parser.add_argument('--valid_pct', type=float, default=0.2,
                        help='Porcentaje de datos para validación')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Directorio para guardar el modelo entrenado')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla para reproducibilidad')
    
    return parser.parse_args()

def get_data_paths(data_source):
    """Obtiene las rutas de las imágenes según la fuente de datos seleccionada."""
    base_dir = Path(__file__).parent.parent / 'data'
    paths = []
    
    if data_source in ['classified', 'both']:
        classified_dir = base_dir / 'classified'
        if classified_dir.exists():
            paths.append(classified_dir)
            print(f"Usando datos de {classified_dir}")
        else:
            print(f"Advertencia: {classified_dir} no existe")
    
    if data_source in ['generated', 'both']:
        generated_dir = base_dir / 'generated'
        if generated_dir.exists():
            paths.append(generated_dir)
            print(f"Usando datos de {generated_dir}")
        else:
            print(f"Advertencia: {generated_dir} no existe")
    
    if not paths:
        raise ValueError(f"No se encontraron directorios de datos para '{data_source}'")
    
    return paths

def create_dataloaders(data_paths, img_size, batch_size, valid_pct, seed):
    """Crea los dataloaders para entrenamiento y validación."""
    # Configurar semilla para reproducibilidad
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Crear transformaciones de datos
    item_tfms = [Resize(img_size)]
    batch_tfms = [
        *aug_transforms(max_rotate=10, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75),
        Normalize.from_stats(*imagenet_stats)
    ]
    
    # Crear DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )
    
    # Crear dataloaders
    dls = []
    for path in data_paths:
        try:
            dl = dblock.dataloaders(path, bs=batch_size)
            dls.append(dl)
        except Exception as e:
            print(f"Error al crear dataloader para {path}: {e}")
    
    if not dls:
        raise ValueError("No se pudieron crear dataloaders con los datos proporcionados")
    
    # Si hay múltiples dataloaders, combinarlos
    if len(dls) > 1:
        # Combinar los dataloaders (simplemente usamos el primero por ahora)
        # En una implementación más avanzada, se podrían combinar los datasets
        return dls[0]
    else:
        return dls[0]

def train_model(dls, model_name, epochs, lr, output_dir):
    """Entrena el modelo con los dataloaders proporcionados."""
    # Crear el modelo
    learn = vision_learner(dls, eval(model_name), metrics=[error_rate, accuracy])
    
    # Encontrar la tasa de aprendizaje óptima
    print("Buscando tasa de aprendizaje óptima...")
    learn.lr_find()
    
    # Entrenar el modelo
    print(f"Entrenando modelo {model_name} por {epochs} épocas...")
    learn.fine_tune(epochs, base_lr=lr)
    
    # Guardar el modelo
    os.makedirs(output_dir, exist_ok=True)
    model_path = Path(output_dir) / f"pigpen_model_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    learn.export(model_path)
    print(f"Modelo guardado en {model_path}")
    
    # Mostrar resultados
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.savefig(Path(output_dir) / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    return learn, model_path

def main():
    """Función principal."""
    args = parse_args()
    
    print(f"Configuración:")
    print(f"- Fuente de datos: {args.data_source}")
    print(f"- Épocas: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Tamaño de imagen: {args.img_size}")
    print(f"- Modelo: {args.model_name}")
    
    # Obtener rutas de datos
    data_paths = get_data_paths(args.data_source)
    
    # Crear dataloaders
    dls = create_dataloaders(
        data_paths, 
        args.img_size, 
        args.batch_size, 
        args.valid_pct,
        args.seed
    )
    
    # Mostrar algunas imágenes de ejemplo
    print("Mostrando ejemplos de imágenes:")
    dls.show_batch(max_n=9, figsize=(10, 10))
    
    # Entrenar modelo
    learn, model_path = train_model(
        dls, 
        args.model_name, 
        args.epochs, 
        args.lr,
        args.output_dir
    )
    
    print(f"Entrenamiento completado. Modelo guardado en {model_path}")
    
    return learn

if __name__ == "__main__":
    main() 