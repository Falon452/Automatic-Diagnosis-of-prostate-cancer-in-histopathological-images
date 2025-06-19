import torch
import numpy as np
import argparse
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model, device


def main(model_path, file_path, save_directory):
    model, device = load_model(model_path)
    data = np.load(file_path)

    target_layers = [model.features[-1]]
    input_tensor = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)

    targets = [ClassifierOutputTarget(1) for _ in range(input_tensor.shape[0])]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        os.makedirs(save_directory, exist_ok=True)

        for i in range(len(grayscale_cam)):
            grayscale_cam_image = grayscale_cam[i, :]
            visualization = show_cam_on_image(data[i].astype(np.float32) / 255.0, grayscale_cam_image, use_rgb=True)

            visualization_pil = Image.fromarray(visualization.astype(np.uint8))

            save_path = os.path.join(save_directory, f"gradcam_{i}.png")
            visualization_pil.save(save_path)
            print(f"Image saved to {save_path}")


if __name__ == "__main__":
    r""" 
    python gradcam.py --model model40.pth --file data\40x\0BCCB273-17EA-4DE0-B530-DB721AA010E5\R3\0BCCB273-17EA-4DE0-B530-DB721AA010E5.blob.0.npy --save_dir gradcamy2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the PyTorch model")
    parser.add_argument("--file", required=True, help="Path to the .npy file containing images")
    parser.add_argument("--save_dir", required=True, help="Directory to save the generated Grad-CAM images")
    args = parser.parse_args()

    main(args.model, args.file, args.save_dir)
