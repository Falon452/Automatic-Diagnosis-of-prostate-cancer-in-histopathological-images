import argparse
import re
import os
import matplotlib.pyplot as plt


def parse_log_file(log_path):
    epochs = []
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    precision, recall = [], []

    with open(log_path, 'r') as file:
        for line in file:
            if match := re.search(r'Epoch (\d+)/\d+', line):
                epochs.append(int(match.group(1)))
            elif match := re.search(r'Train Loss: ([\d\.]+)', line):
                train_loss.append(float(match.group(1)))
            elif match := re.search(r'Validation Loss: ([\d\.]+)', line):
                val_loss.append(float(match.group(1)))
            elif match := re.search(r'Train Accuracy: ([\d\.]+)%', line):
                train_acc.append(float(match.group(1)))
            elif match := re.search(r'Validation Accuracy: ([\d\.]+)%', line):
                val_acc.append(float(match.group(1)))
            elif match := re.search(r'Precision: ([\d\.]+)', line):
                precision.append(float(match.group(1)))
            elif match := re.search(r'Recall: ([\d\.]+)', line):
                recall.append(float(match.group(1)))

    print(len(epochs))
    min_len = min(len(epochs), len(train_loss), len(val_loss), len(train_acc), len(val_acc))
    return epochs[:min_len], train_loss[:min_len], val_loss[:min_len], train_acc[:min_len], val_acc[:min_len], precision, recall


def plot_loss_and_accuracy(epochs, train_loss, val_loss, train_acc, val_acc, output_folder, filename):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axs[0].plot(epochs, train_loss, label='Train')
    axs[0].plot(epochs, val_loss, label='Validation')
    axs[0].set_title('Loss vs. Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Accuracy
    axs[1].plot(epochs, train_acc, label='Train')
    axs[1].plot(epochs, val_acc, label='Validation')
    axs[1].set_title('Accuracy vs. Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()


def plot_precision_recall_curve(precision, recall, output_folder, filename):
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision–Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, filename))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, required=True, help='Path to training log file')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save plots')
    parser.add_argument('--magnification', type=str, required=True, help='Magnification level (e.g., 10, 20, 40)')
    args = parser.parse_args()

    epochs, train_loss, val_loss, train_acc, val_acc, precision, recall = parse_log_file(args.log_path)

    loss_acc_filename = f'accuracy_and_loss_side_by_side_{args.magnification}.png'
    pr_curve_filename = f'precision_recall_curve_{args.magnification}.png'

    plot_loss_and_accuracy(
        epochs, train_loss, val_loss, train_acc, val_acc,
        args.output_folder, loss_acc_filename
    )

    plot_precision_recall_curve(
        precision, recall, args.output_folder, pr_curve_filename
    )



if __name__ == '__main__':
    r"""
    Example usage:
    python plot_metrics_from_log_file.py --log_path run_10/training.log --output_folder run_10 --magnification 10
    python plot_metrics_from_log_file.py --log_path run_20/training.log --output_folder run_20 --magnification 20
    python plot_metrics_from_log_file.py --log_path run_40/training.log --output_folder run_40 --magnification 40
    
    python plot_metrics_from_log_file.py --log_path 10x/training.log --output_folder 10x --magnification 10
    python plot_metrics_from_log_file.py --log_path 20x/training.log --output_folder 20x --magnification 20
    python plot_metrics_from_log_file.py --log_path 40x/training.log --output_folder 40x --magnification 40
    """
    main()
