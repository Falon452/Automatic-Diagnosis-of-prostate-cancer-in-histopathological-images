import logging
import argparse
import os
import sys
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch import optim
from torch.utils.data import Dataset, random_split, DataLoader
from torch.amp import GradScaler, autocast

import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue
from threading import Thread

IMAGENET_IMAGE_MEAN = [123.68, 116.779, 103.939]
PATCH_SIZE = 224
DEFAULT_LABEL_DICTIONARY = {'BG': 0, 'T': 1, 'N': 2, 'A': 3, 'R1': 4, 'R2': 5, 'R3': 6, 'R4': 7, 'R5': 8}


class AbstractDiagSetDataset(ABC):
    def __init__(self, root_path, partitions, magnification=40, batch_size=32, augment=True,
                 subtract_mean=True, label_dictionary=None, shuffling=True, class_ratios=None,
                 scan_subset=None, buffer_size=64):
        """
        Abstract container for DiagSet-A dataset.

        :param root_path: root directory of the dataset
        :param partitions: list containing all partitions ('train', 'validation' or 'test') that will be loaded
        :param magnification: int in [40, 20, 10, 5] describing scan magnification for which patches will be loaded
        :param batch_size: int, number of images in a single batch
        :param augment: boolean, whether to apply random image augmentations
        :param subtract_mean: boolean, whether to subtract ImageNet mean from every image
        :param label_dictionary: dict assigning int label to every text key, DEFAULT_LABEL_DICTIONARY will
               be used if it is set to None
        :param shuffling: boolean, whether to shuffle the order of batches
        :param class_ratios: dict assigning probability to each int key, specifies ratio of images from a class
               with a given key that will be loaded in each batch (note that it will not always return deterministic
               number of images per class, but will specify the probability of drawing from that class instead).
               Can be None, in which case original dataset
               ratios will be used, otherwise all dict values should sum up to one
        :param scan_subset: subset of scans that will be loaded, either list of strings with scan IDs or
               float in (0, 1), in which case a random subset of scans from given partitions will be selected
        :param buffer_size: number of images from each class that will be stored in buffer
        """
        for partition in partitions:
            assert partition in ['train', 'validation', 'test']

        self.root_path = root_path
        self.partitions = partitions
        self.magnification = magnification
        self.batch_size = batch_size
        self.augment = augment
        self.subtract_mean = subtract_mean
        self.shuffling = shuffling
        self.scan_subset = scan_subset
        self.buffer_size = buffer_size

        if label_dictionary is None:
            logging.info('Using default label dictionary...')
            self.label_dictionary = DEFAULT_LABEL_DICTIONARY
        else:
            self.label_dictionary = label_dictionary

        self.numeric_labels = list(set(self.label_dictionary.values()))
        self.buffers = {}
        self.blob_paths = {}
        self.class_distribution = {}
        for numeric_label in self.numeric_labels:
            self.buffers[numeric_label] = Queue(buffer_size)
            self.blob_paths[numeric_label] = []
            self.class_distribution[numeric_label] = 0
        self.n_images = 0

        self.blobs_path = Path(root_path) / 'blobs' / 'S' / ('%dx' % magnification)
        self.distributions_path = Path(root_path) / 'distributions' / 'S' / ('%dx' % magnification)
        assert self.blobs_path.exists()

        self.scan_names = [path.name for path in self.blobs_path.iterdir()]
        partition_scan_names = []
        for partition in self.partitions:
            partition_path = Path(root_path) / 'partitions' / 'DiagSet-A.2' / ('%s.csv' % partition)
            if partition_path.exists():
                df = pd.read_csv(partition_path)
                partition_scan_names += df['scan_id'].astype(str).tolist()
            else:
                raise ValueError('Partition file not found under "%s".' % partition_path)
        self.scan_names = [scan_name for scan_name in self.scan_names if scan_name in partition_scan_names]

        if self.scan_subset is not None and self.scan_subset != 1.0:
            if isinstance(self.scan_subset, list):
                logging.info('Using given %d out of %d scans...' % (len(self.scan_subset), len(self.scan_names)))
                self.scan_names = self.scan_subset
            else:
                if isinstance(self.scan_subset, float):
                    n_scans = int(self.scan_subset * len(self.scan_names))
                else:
                    n_scans = self.scan_subset
                assert 0 < n_scans <= len(self.scan_names)
                logging.info('Randomly selecting %d out of %d scans...' % (n_scans, len(self.scan_names)))
                self.scan_names = list(np.random.choice(self.scan_names, n_scans, replace=False))

        logging.info('Loading blob paths...')
        for scan_name in self.scan_names:
            for string_label, numeric_label in self.label_dictionary.items():
                blob_names = map(lambda x: x.name,
                                 sorted((self.blobs_path / scan_name / string_label).iterdir()))
                for blob_name in blob_names:
                    self.blob_paths[numeric_label].append(self.blobs_path / scan_name / string_label / blob_name)
            with open(self.distributions_path / ('%s.json' % scan_name), 'r') as f:
                scan_class_distribution = json.load(f)
            self.n_images += sum(scan_class_distribution.values())
            for string_label, numeric_label in self.label_dictionary.items():
                self.class_distribution[numeric_label] += scan_class_distribution[string_label]

        if class_ratios is None:
            self.class_ratios = {}
            for numeric_label in self.numeric_labels:
                self.class_ratios[numeric_label] = self.class_distribution[numeric_label] / self.n_images
        else:
            self.class_ratios = class_ratios

        logging.info('Found %d patches.' % self.n_images)
        class_distribution_text = ', '.join([
            '%s: %.2f%%' % (label, count / self.n_images * 100)
            for label, count in self.class_distribution.items()
        ])
        logging.info('Class distribution: %s.' % class_distribution_text)

        if self.shuffling:
            for numeric_label in self.numeric_labels:
                np.random.shuffle(self.blob_paths[numeric_label])

        for numeric_label in self.numeric_labels:
            if len(self.blob_paths[numeric_label]) > 0:
                Thread(target=self.fill_buffer, daemon=True, args=(numeric_label,)).start()

    @abstractmethod
    def batch(self):
        return

    def length(self):
        return int(np.ceil(self.n_images / self.batch_size))

    def fill_buffer(self, numeric_label):
        while True:
            for blob_path in self.blob_paths[numeric_label]:
                images = self.prepare_images(blob_path)
                for image in images:
                    self.buffers[numeric_label].put(image)
            if self.shuffling:
                np.random.shuffle(self.blob_paths[numeric_label])

    def prepare_images(self, blob_path):
        images = np.load(blob_path)
        if self.shuffling:
            np.random.shuffle(images)

        prepared_images = []
        for i in range(len(images)):
            image = images[i].astype(np.float32)
            if self.augment:
                image = self._augment(image)
            else:
                x = (image.shape[0] - PATCH_SIZE) // 2
                y = (image.shape[1] - PATCH_SIZE) // 2
                image = image[x:(x + PATCH_SIZE), y:(y + PATCH_SIZE)]
            if self.subtract_mean:
                image -= IMAGENET_IMAGE_MEAN
            prepared_images.append(image)

        return np.array(prepared_images)

    def _augment(self, image):
        x_max = image.shape[0] - PATCH_SIZE
        y_max = image.shape[1] - PATCH_SIZE
        x = np.random.randint(x_max)
        y = np.random.randint(y_max)
        image = image[x:(x + PATCH_SIZE), y:(y + PATCH_SIZE)]
        if np.random.choice([True, False]):
            image = np.fliplr(image)
        image = np.rot90(image, k=np.random.randint(4))
        return image


class TrainingDiagSetDataset(AbstractDiagSetDataset):
    def batch(self):
        probabilities = [self.class_ratios[label] for label in self.numeric_labels]
        labels = np.random.choice(self.numeric_labels, self.batch_size, p=probabilities)
        images = np.array([self.buffers[label].get() for label in labels])
        return images, labels


class EvaluationDiagSetDataset(AbstractDiagSetDataset):
    def __init__(self, **kwargs):
        assert kwargs.get('augment', False) is False
        assert kwargs.get('shuffling', False) is False
        assert kwargs.get('class_ratios') is None

        kwargs['augment'] = False
        kwargs['shuffling'] = False
        kwargs['class_ratios'] = None

        self.current_numeric_label_index = 0
        self.current_batch_index = 0

        super().__init__(**kwargs)

    def batch(self):
        labels = []
        images = []
        for _ in range(self.batch_size):
            label = self.numeric_labels[self.current_numeric_label_index]
            while len(self.blob_paths[label]) == 0:
                self.current_numeric_label_index = (self.current_numeric_label_index + 1) % len(self.numeric_labels)
                label = self.numeric_labels[self.current_numeric_label_index]

            image = self.buffers[label].get()
            labels.append(label)
            images.append(image)
            self.current_batch_index += 1

            if self.current_batch_index >= self.class_distribution[label]:
                self.current_batch_index = 0
                self.current_numeric_label_index += 1
                if self.current_numeric_label_index >= len(self.numeric_labels):
                    self.current_numeric_label_index = 0
                    break

        return np.array(images), np.array(labels)


scaler = GradScaler()


def train(model, train_dataset, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    total_batches = train_dataset.length()

    for batch_idx in range(total_batches):
        images_np, labels_np = train_dataset.batch()
        images = torch.from_numpy(images_np).float().permute(0, 3, 1, 2).to(device)
        labels = torch.from_numpy(labels_np).long().to(device)

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            output = model(images)
            loss = criterion(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        correct += (output.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 200 == 0:
            progress = (batch_idx + 1) / total_batches * 100
            logging.info(
                f"Epoch {epoch}/{total_epochs} - "
                f"Batch {batch_idx + 1}/{total_batches} {progress:.2f}%"
            )

    avg_loss = running_loss / total_batches
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, val_dataset, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels = []
    all_preds = []

    total_batches = val_dataset.length()
    with torch.no_grad():
        for _ in range(total_batches):
            images_np, labels_np = val_dataset.batch()
            images = torch.from_numpy(images_np).float().permute(0, 3, 1, 2).to(device)
            labels = torch.from_numpy(labels_np).long().to(device)

            output = model(images)
            loss = criterion(output, labels)
            running_loss += loss.item()

            preds = output.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    precision, recall, f1 = calculate_metrics(all_labels, all_preds)

    return {
        "loss": running_loss / total_batches,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def calculate_metrics(all_labels, all_preds):
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    num_classes = len(np.unique(all_labels))

    precision_per_class = []
    recall_per_class = []
    for c in range(num_classes):
        true_positive = ((all_preds == c) & (all_labels == c)).sum().item()
        predicted_positive = (all_preds == c).sum().item()
        actual_positive = (all_labels == c).sum().item()

        precision = true_positive / predicted_positive if predicted_positive > 0 else 0
        recall = true_positive / actual_positive if actual_positive > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)

    precision = sum(precision_per_class) / num_classes
    recall = sum(recall_per_class) / num_classes
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def validate_sampled(model, val_dataset, criterion, device, sample_fraction=0.2):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    total_batches = val_dataset.length()
    with torch.no_grad():
        for _ in range(total_batches):
            images_np, labels_np = val_dataset.batch()
            if random.random() > sample_fraction:
                continue

            images = torch.from_numpy(images_np).float().permute(0, 3, 1, 2).to(device)
            labels = torch.from_numpy(labels_np).long().to(device)

            output = model(images)
            loss = criterion(output, labels)
            running_loss += loss.item()

            preds = output.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    accuracy = 100.0 * correct / total
    precision, recall, f1 = calculate_metrics(all_labels, all_preds)
    return {
        "loss": running_loss / (total_batches * sample_fraction),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def log_trainable_layers(model):
    logging.info("\n=========== Trainable Layers ===========")
    for name, param in model.named_parameters():
        logging.info(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")
    logging.info("=======================================\n")


def log_args(args):
    logging.info("\n=========== Training Configuration ===========")
    logging.info(f"Output Dir: {args.output_dir}")
    logging.info(f"Dataset Root Directory: {args.root_dir}")
    logging.info(f"Dataset Limit: {args.limit}")
    logging.info(f"Magnification Level: {args.magnification}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Initial LR: {args.lr}")
    logging.info(f"Weight Decay: {args.weight_decay}")
    logging.info(f"Resume from epoch: {args.resume_epoch}")
    logging.info(f"Number of Epochs: {args.epochs}")
    logging.info(f"Initial Frozen Conv Layers: {args.initial_frozen_layers}")
    logging.info(f"Freeze Schedule: {args.freeze_schedule}")
    logging.info("===============================================\n")


def save_checkpoint(model, optimizer, scaler, epoch, loss, output_folder):
    checkpoint_path = os.path.join(output_folder, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint_from_epoch(epoch, model, optimizer, scaler, device, output_folder, test):
    checkpoint_path = os.path.join(output_folder, f"checkpoint_epoch_{epoch}.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint['epoch'], checkpoint['loss']


def set_conv_freeze(model, n_conv_to_freeze, conv_layer_indices):
    base = model.module if hasattr(model, "module") else model
    total_conv = len(conv_layer_indices)

    if n_conv_to_freeze >= total_conv:
        boundary_idx = conv_layer_indices[-1]
    elif n_conv_to_freeze <= 0:
        boundary_idx = -1
    else:
        boundary_idx = conv_layer_indices[n_conv_to_freeze - 1]

    for idx, layer in enumerate(base.features):
        freeze_this = (idx <= boundary_idx)
        for p in layer.parameters():
            p.requires_grad = not freeze_this


def collect_conv_layer_indices(model):
    base = model.module if hasattr(model, "module") else model
    conv_indices = []
    for idx, layer in enumerate(base.features):
        if any(True for _ in layer.parameters()):
            conv_indices.append(idx)
    return conv_indices


"""
TRAINING 40x 1st epoch has 8 conv layers frozen with learning rate 1e-6, later in 30 epoch new layers are unfrozen:
srun python trainobsolete.py --output_dir run_40_test --root_dir DiagSet-A --magnification 40x --batch_size 1024  --weight_decay 1e-5 --epochs 1000 --freeze_schedule "[[1,8,1e-6],[30,3,1e-6]]"

TRAINING 40x 1st epoch has 8 conv layers frozen with learning rate 1e-6, later in 34 epoch new layers are unfrozen:
srun python trainobsolete.py --output_dir run_40_test --root_dir DiagSet-A --magnification 40x --resume_epoch 30 --batch_size 1024  --weight_decay 1e-5 --epochs 1000 --freeze_schedule "[[30,8,1e-6],[34,3,1e-6]]"

TESTING 40x:
srun python trainobsolete.py --test --output_dir run_40_test --root_dir DiagSet-A --magnification 40x --resume_epoch 30 --batch_size 1024  --weight_decay 1e-5 --epochs 1000 --freeze_schedule "[[30,8,1e-6],[34,3,1e-6]]"

"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help="Output Dir")
    parser.add_argument('--root_dir', type=str, default='data', help="Dataset root directory")
    parser.add_argument('--limit', type=int, default=None, help="Limit of items in dataset (default: None)")
    parser.add_argument('--magnification', type=str, choices=['5x', '10x', '20x', '40x'], required=True,
                        help="Magnification level")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-5, help="Initial learning rate for all parameters")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay")
    parser.add_argument('--epochs', type=int, default=40, help="Number of epochs")
    parser.add_argument('--resume_epoch', type=int, default=None, help='Epoch number to resume from')
    parser.add_argument('--initial_frozen_layers', type=int, default=26,
                        help="Number of convolutional layers to freeze at start (default: 26)")
    parser.add_argument('--freeze_schedule', type=str, default=None,
                        help="JSON list of [epoch, num_frozen_conv_layers, learning_rate], \n"
                             "where the earliest epoch entry sets optimizer LR for all trainable parameters;\n"
                             "subsequent entries only set LR for newly unfrozen conv layers.")
    parser.add_argument(
        '--test',
        action='store_true',
        help='If set, load the checkpoint (via --resume_epoch) and run evaluation on the TEST split, then exit.'
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "training.log"))
        ]
    )

    log_args(args)

    if args.freeze_schedule is not None:
        try:
            schedule_list = json.loads(args.freeze_schedule)
            schedule_dict = {}
            for triple in schedule_list:
                if len(triple) != 3:
                    raise ValueError
                epoch_key = int(triple[0])
                n_to_freeze = int(triple[1])
                new_lr = float(triple[2])
                schedule_dict[epoch_key] = (n_to_freeze, new_lr)
            schedule_epochs = sorted(schedule_dict.keys())
            first_schedule_epoch = schedule_epochs[0]
        except Exception:
            logging.error(
                "Failed to parse --freeze_schedule. "
                "Make sure it's valid JSON of [[epoch, n_frozen, lr], ...]."
            )
            sys.exit(1)
    else:
        schedule_dict = {}
        first_schedule_epoch = None

    binary_label_map = {
        "BG": 0, "T": 0, "N": 0,
        "A": 0,
        "R1": 1, "R2": 1, "R3": 1, "R4": 1, "R5": 1
    }

    mag = int(args.magnification.replace('x', ''))

    train_dataset = TrainingDiagSetDataset(
        root_path=args.root_dir,
        partitions=['train'],
        magnification=mag,
        batch_size=args.batch_size,
        label_dictionary=binary_label_map,
        augment=True,
        subtract_mean=True,
        shuffling=True
    )

    val_dataset = EvaluationDiagSetDataset(
        root_path=args.root_dir,
        partitions=['validation'],
        magnification=mag,
        batch_size=args.batch_size,
        label_dictionary=binary_label_map
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, 2)

    if torch.cuda.device_count() > 1:
        logging.info(f"Device count: {torch.cuda.device_count()}. Using nn.DataParallel(model)")
        model = nn.DataParallel(model)
    else:
        logging.info(f"Device count: {torch.cuda.device_count()}. NOT USING nn.DataParallel(model)")

    for i in range(torch.cuda.device_count()):
        logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")

    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_loss = np.inf
    patience = 10
    delta = 0
    epochs_without_improvement = 0

    conv_layer_indices = collect_conv_layer_indices(model)
    total_conv = len(conv_layer_indices)
    new_n_to_freeze, new_lr = schedule_dict[first_schedule_epoch]
    new_n_to_freeze = max(0, min(new_n_to_freeze, total_conv))

    set_conv_freeze(model, n_conv_to_freeze=new_n_to_freeze, conv_layer_indices=conv_layer_indices)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=new_lr,
        weight_decay=args.weight_decay
    )
    curr_lr = new_lr
    no_frozen_layers = new_n_to_freeze
    logging.info(
        f"  [Schedule] Frozen conv blocks = {new_n_to_freeze}, "
        f"set all trainable LRs to {new_lr:.2e}"
    )
    log_trainable_layers(model)

    try:
        start_epoch = 1
        if args.resume_epoch is not None:
            start_epoch, _ = load_checkpoint_from_epoch(
                args.resume_epoch, model, optimizer, scaler, device, args.output_dir, test=args.test
            )
            start_epoch += 1

        if args.test:
            # TESTING
            logging.info(f"TESTING Epoch ${args.resume_epoch}")
            test_dataset = EvaluationDiagSetDataset(
                root_path=args.root_dir,
                partitions=['test'],
                magnification=mag,
                batch_size=args.batch_size,
                label_dictionary=binary_label_map
            )
            val_metrics = validate(model, test_dataset, criterion, device)
            logging.info(f"TEST Accuracy: {val_metrics['accuracy']:.2f}%")
            logging.info(f"TEST Precision: {val_metrics['precision']:.4f}")
            logging.info(f"TEST Recall: {val_metrics['recall']:.4f}")
            logging.info(f"TEST F1-score: {val_metrics['f1_score']:.4f}")
            exit(0)

        for epoch in range(start_epoch, args.epochs + 1):
            logging.info(f"Epoch {epoch}/{args.epochs}")

            # Unfroze some layers?
            if epoch in schedule_dict:
                new_n_to_freeze, new_lr = schedule_dict[epoch]
                new_n_to_freeze = max(0, min(new_n_to_freeze, total_conv))

                if epoch == first_schedule_epoch:
                    set_conv_freeze(model, n_conv_to_freeze=new_n_to_freeze, conv_layer_indices=conv_layer_indices)
                    optimizer = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=new_lr,
                        weight_decay=args.weight_decay
                    )
                    curr_lr = new_lr
                    no_frozen_layers = new_n_to_freeze
                    logging.info(
                        f"  [Schedule] Epoch {epoch} (first): Frozen conv blocks = {new_n_to_freeze}, "
                        f"set all trainable LRs to {new_lr:.2e}"
                    )
                    log_trainable_layers(model)
                elif new_n_to_freeze < no_frozen_layers:
                    unfrozen_indices = conv_layer_indices[new_n_to_freeze:no_frozen_layers]
                    set_conv_freeze(model, n_conv_to_freeze=new_n_to_freeze, conv_layer_indices=conv_layer_indices)
                    base = model.module if hasattr(model, "module") else model
                    newly_unfrozen_params = []
                    for idx in unfrozen_indices:
                        for p in base.features[idx].parameters():
                            if p.requires_grad:
                                newly_unfrozen_params.append(p)

                    optimizer.add_param_group({
                        'params': newly_unfrozen_params,
                        'lr': new_lr,
                        'weight_decay': args.weight_decay
                    })
                    logging.info(
                        f"[Schedule] Epoch {epoch}: Unfroze conv blocks {unfrozen_indices}, "
                        f"added {len(newly_unfrozen_params)} params at lr={new_lr:.2e}"
                    )
                    no_frozen_layers = new_n_to_freeze
                    log_trainable_layers(model)

            # TRAINING
            train_loss, train_acc = train(model, train_dataset, criterion, optimizer, device, epoch, args.epochs)

            # VALIDATING
            if epoch % 5 == 0 or epoch == args.epochs:
                val_metrics = validate(model, val_dataset, criterion, device)
            else:
                val_metrics = validate_sampled(model, val_dataset, criterion, device, sample_fraction=0.5)

            logging.info(f"Train Loss: {train_loss:.4f}")
            logging.info(f"Validation Loss: {val_metrics['loss']:.4f}")
            logging.info(f"Train Accuracy: {train_acc:.2f}%")
            logging.info(f"Validation Accuracy: {val_metrics['accuracy']:.2f}%")
            logging.info(f"Precision: {val_metrics['precision']:.4f}")
            logging.info(f"Recall: {val_metrics['recall']:.4f}")
            logging.info(f"F1-score: {val_metrics['f1_score']:.4f}")

            save_checkpoint(model, optimizer, scaler, epoch, val_metrics['loss'], args.output_dir)

            if val_metrics['loss'] < best_loss - delta:
                best_loss = val_metrics['loss']
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    except KeyboardInterrupt:
        logging.warning("Training interrupted by user!")
        sys.exit(1)

    logging.info("----- TRAINING COMPLETED SUCCESSFULLY ------")
