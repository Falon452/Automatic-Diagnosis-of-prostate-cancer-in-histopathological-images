#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Traverses a given root directory, finds all `.npy` files within its
subdirectories, and for each file, saves its array slices as PNG images.

The generated PNG images are saved in the *same directory* where the
corresponding .npy file was found.

Expected Directory Structure:
- root_directory/
  - R1/
    - some_file.npy
  - R2/
    - another_file.npy
  - ...and so on

Usage:
    python process_npy_directories.py /path/to/root_directory
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def save_array_slices_as_png(numpy_file_path: str):
    """
    Loads a .npy file and saves each slice as a PNG image in the same directory.

    Args:
        numpy_file_path (str): The full path to the input .npy file.
    """
    # --- CHANGE 1: Determine the output directory ---
    # The output directory is simply the directory where the .npy file is located.
    output_dir = os.path.dirname(numpy_file_path)

    # Get the base name of the file for naming the output images
    # e.g., "my_data" from "/path/to/my_data.npy"
    base_name = os.path.splitext(os.path.basename(numpy_file_path))[0]

    # --- Load the NumPy array ---
    try:
        data = np.load(numpy_file_path)
        print(f"  > Loaded array with shape: {data.shape}")
    except Exception as e:
        print(f"  > Error: Could not load or parse the NumPy file. Skipping. Reason: {e}")
        return  # Exit this function, but allow the script to continue with the next file

    # --- Handle different array dimensions ---
    if data.ndim < 2:
        print(f"  > Warning: Array has {data.ndim} dimension(s). Cannot interpret as image data. Skipping.")
        return

    # If the array is 2D, treat it as a single image by wrapping it in a list.
    images_to_save = [data] if data.ndim == 2 else data
    num_images = len(images_to_save)
    print(f"  > Found {num_images} image(s) to save.")

    # --- Iterate and save each slice as a PNG ---
    for i, image_slice in enumerate(images_to_save):
        # --- CHANGE 2: Construct the output filename to be in the same directory ---
        output_filename = os.path.join(output_dir, f"{base_name}_slice_{i:04d}.png")

        # Determine if a colormap is needed (for grayscale images)
        cmap = 'gray' if image_slice.ndim == 2 else None

        try:
            # Use plt.imsave for direct, no-axes saving.
            plt.imsave(output_filename, image_slice, cmap=cmap)
        except Exception as e:
            print(f"    > Could not save image slice {i}. Reason: {e}")

    print(f"  > Finished processing {os.path.basename(numpy_file_path)}.")


def process_directory_tree(root_dir: str):
    """
    Walks through a directory tree and processes every .npy file found.

    Args:
        root_dir (str): The path to the starting directory.
    """
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory not found at '{root_dir}'")
        sys.exit(1)

    print(f"Starting search in root directory: '{root_dir}'\n")
    npy_file_count = 0

    # os.walk is a generator that traverses a directory tree
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file has a .npy extension (case-insensitive)
            if filename.lower().endswith('.npy'):
                npy_file_count += 1
                full_path = os.path.join(dirpath, filename)
                print(f"Processing file #{npy_file_count}: {full_path}")
                save_array_slices_as_png(full_path)
                print("-" * 20)  # Separator for clarity

    if npy_file_count == 0:
        print("Search complete. No .npy files were found.")
    else:
        print(f"Search complete. Processed a total of {npy_file_count} .npy file(s).")


def main():
    """Main function to handle command-line arguments."""
    if len(sys.argv) != 2:
        print("Usage: python process_npy_directories.py <path_to_root_directory>")
        sys.exit(1)

    root_directory_path = sys.argv[1]
    process_directory_tree(root_directory_path)


if __name__ == "__main__":
    main()
