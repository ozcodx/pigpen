#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para entrenar un modelo de reconocimiento de cifrado Pigpen usando PyTorch.
Permite usar imágenes de las carpetas 'classified', 'generated' o ambas.
"""

import os
import argparse
import random
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
from PIL import Image

# Configuración de semilla para reproducibilidad
def set_seed(seed):
    """Configura semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

class CombinedDataset(Dataset):
    """Dataset que combina múltiples ImageFolder datasets."""
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = [0] + [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]
        self.total_length = sum(self.lengths)
        
        # Combinar las clases de todos los datasets
        self.classes = datasets[0].classes
        self.class_to_idx = datasets[0].class_to_idx
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Determinar a qué dataset pertenece el índice
        dataset_idx = 0
        while dataset_idx < len(self.cumulative_lengths) - 1 and idx >= self.cumulative_lengths[dataset_idx + 1]:
            dataset_idx += 1
        
        # Ajustar el índice para el dataset específico
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]

def create_dataloaders(data_paths, img_size, batch_size, valid_pct, seed):
    """Crea los dataloaders para entrenamiento y validación."""
    # Configurar semilla para reproducibilidad
    set_seed(seed)
    
    # Transformaciones para entrenamiento (con aumento de datos)
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transformaciones para validación (sin aumento de datos)
    valid_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Crear datasets
    train_datasets = []
    valid_datasets = []
    
    for path in data_paths:
        try:
            # Crear dataset completo
            full_dataset = ImageFolder(path, transform=train_transform)
            
            # Dividir en entrenamiento y validación
            train_size = int((1 - valid_pct) * len(full_dataset))
            valid_size = len(full_dataset) - train_size
            
            train_dataset, valid_dataset = random_split(
                full_dataset, [train_size, valid_size],
                generator=torch.Generator().manual_seed(seed)
            )
            
            # Aplicar transformaciones correctas a cada conjunto
            train_dataset.dataset.transform = train_transform
            valid_dataset.dataset.transform = valid_transform
            
            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)
            
            print(f"Dataset de {path}: {len(full_dataset)} imágenes totales, "
                  f"{train_size} para entrenamiento, {valid_size} para validación")
            
        except Exception as e:
            print(f"Error al crear dataset para {path}: {e}")
    
    if not train_datasets:
        raise ValueError("No se pudieron crear datasets con los datos proporcionados")
    
    # Combinar datasets si hay múltiples
    if len(train_datasets) > 1:
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
    else:
        train_dataset = train_datasets[0]
        valid_dataset = valid_datasets[0]
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    
    # Obtener las clases del primer dataset
    classes = full_dataset.classes
    
    return train_loader, valid_loader, classes

def get_model(model_name, num_classes):
    """Crea un modelo preentrenado con la última capa adaptada al número de clases."""
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
    elif model_name == 'resnet34':
        model = models.resnet34(weights='DEFAULT')
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
    else:
        raise ValueError(f"Modelo {model_name} no soportado")
    
    # Modificar la última capa para el número de clases
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena el modelo por una época."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Poner a cero los gradientes
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass y optimización
        loss.backward()
        optimizer.step()
        
        # Estadísticas
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Valida el modelo en el conjunto de validación."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Estadísticas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def plot_confusion_matrix(model, dataloader, classes, device, output_dir):
    """Genera y guarda una matriz de confusión."""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Crear matriz de confusión
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')
    
    # Guardar la figura
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(Path(output_dir) / f"confusion_matrix_{timestamp}.png")
    plt.close()

def show_batch(dataloader, classes):
    """Muestra un batch de imágenes con sus etiquetas."""
    # Obtener un batch
    inputs, labels = next(iter(dataloader))
    
    # Desnormalizar las imágenes
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = inputs * std + mean
    
    # Mostrar imágenes
    plt.figure(figsize=(15, 8))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).clip(0, 1))
        plt.title(f'Clase: {classes[labels[i]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def train_model(train_loader, valid_loader, classes, model_name, epochs, lr, output_dir, seed):
    """Entrena el modelo con los dataloaders proporcionados."""
    # Configurar semilla para reproducibilidad
    set_seed(seed)
    
    # Determinar el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Crear modelo
    model = get_model(model_name, len(classes))
    model = model.to(device)
    
    # Definir criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler para reducir la tasa de aprendizaje
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Entrenamiento
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"Iniciando entrenamiento del modelo {model_name} por {epochs} épocas...")
    
    for epoch in range(epochs):
        # Entrenar una época
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validar
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        
        # Actualizar scheduler
        scheduler.step(val_loss)
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Imprimir progreso
        print(f"Época {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Guardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Crear directorio si no existe
            os.makedirs(output_dir, exist_ok=True)
            
            # Guardar modelo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = Path(output_dir) / f"pigpen_model_{model_name}_{timestamp}.pth"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': classes
            }, model_path)
            
            print(f"Modelo guardado en {model_path}")
    
    # Generar matriz de confusión
    plot_confusion_matrix(model, valid_loader, classes, device, output_dir)
    
    # Graficar historial de entrenamiento
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Pérdida')
    plt.xlabel('Época')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Precisión')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    return model, model_path

def main():
    """Función principal."""
    args = parse_args()
    
    # Configurar semilla para reproducibilidad
    set_seed(args.seed)
    
    print(f"Configuración:")
    print(f"- Fuente de datos: {args.data_source}")
    print(f"- Épocas: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Tamaño de imagen: {args.img_size}")
    print(f"- Modelo: {args.model_name}")
    
    # Obtener rutas de datos
    data_paths = get_data_paths(args.data_source)
    
    # Crear dataloaders
    train_loader, valid_loader, classes = create_dataloaders(
        data_paths, 
        args.img_size, 
        args.batch_size, 
        args.valid_pct,
        args.seed
    )
    
    # Mostrar algunas imágenes de ejemplo
    print("Mostrando ejemplos de imágenes:")
    show_batch(train_loader, classes)
    
    # Entrenar modelo
    model, model_path = train_model(
        train_loader,
        valid_loader,
        classes,
        args.model_name, 
        args.epochs, 
        args.lr,
        args.output_dir,
        args.seed
    )
    
    print(f"Entrenamiento completado. Modelo guardado en {model_path}")
    
    return model

if __name__ == "__main__":
    main() 