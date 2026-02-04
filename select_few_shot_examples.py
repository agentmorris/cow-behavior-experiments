#%% Header

"""
Select few-shot examples for cattle behavior VLM classification.

Scans all images in sorted_crops, measures their sizes, then selects
4 images per category spanning the size range (using quartile medians).
Writes results to few_shot_examples.json and image_sizes.json.
"""


#%% Imports and constants

import os
import json
import argparse

from datetime import datetime

from cow_vlm_utils import (
    SORTED_CROPS_DIR,
    OUTPUT_BASE_DIR,
    CATEGORY_FOLDER_MAP,
    IMAGE_EXTENSIONS,
    get_image_dimensions,
)


#%% Scan image sizes

def scan_all_image_sizes(sorted_crops_dir):
    """
    Walk all category subdirectories and measure every image.

    Args:
        sorted_crops_dir: path to the sorted_crops directory

    Returns:
        dict mapping category -> list of dicts with keys:
            'filename', 'path', 'width', 'height', 'long_side'
    """

    sizes_by_category = {}

    for folder_name, category in CATEGORY_FOLDER_MAP.items():
        folder_path = os.path.join(sorted_crops_dir, folder_name)
        if not os.path.isdir(folder_path):
            print('Warning: folder not found: {}'.format(folder_path))
            continue

        category_images = []
        filenames = sorted(os.listdir(folder_path))

        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            image_path = os.path.join(folder_path, filename)
            try:
                width, height = get_image_dimensions(image_path)
                category_images.append({
                    'filename': filename,
                    'path': image_path,
                    'width': width,
                    'height': height,
                    'long_side': max(width, height)
                })
            except Exception as e:
                print('Warning: could not read {}: {}'.format(image_path, e))

        sizes_by_category[category] = category_images
        print('  {}: {} images (long side range: {}-{} px)'.format(
            category,
            len(category_images),
            min(img['long_side'] for img in category_images) if category_images else 0,
            max(img['long_side'] for img in category_images) if category_images else 0,
        ))

    return sizes_by_category

# ...def scan_all_image_sizes()


#%% Select few-shot examples

def select_exemplars(category_images, n_per_category=4):
    """
    Select n_per_category images spanning the size range.

    Sorts images by long_side, divides into n_per_category quartiles,
    and picks the image closest to each quartile's median.

    Args:
        category_images: list of image dicts with 'long_side' key
        n_per_category: number of images to select

    Returns:
        list of selected image dicts
    """

    if len(category_images) <= n_per_category:
        return category_images

    sorted_images = sorted(category_images, key=lambda x: x['long_side'])
    n = len(sorted_images)

    selected = []
    for q in range(n_per_category):

        # Compute the start and end indices for this quartile
        start = int(q * n / n_per_category)
        end = int((q + 1) * n / n_per_category)

        # Find the median index within this quartile
        median_idx = (start + end - 1) // 2

        selected.append(sorted_images[median_idx])

    return selected

# ...def select_exemplars()


#%% Main function

def main():

    parser = argparse.ArgumentParser(
        description='Select few-shot examples for cattle behavior classification.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--crops-dir',
        default=SORTED_CROPS_DIR,
        help='Path to sorted_crops directory (default: {})'.format(
            SORTED_CROPS_DIR
        ),
    )
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_BASE_DIR,
        help='Output directory for JSON files (default: {})'.format(
            OUTPUT_BASE_DIR
        ),
    )
    parser.add_argument(
        '--images-per-category', '-n',
        type=int,
        default=4,
        help='Number of few-shot examples per category (default: 4)',
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: scan all image sizes
    print('Scanning image sizes in {}...'.format(args.crops_dir))
    sizes_by_category = scan_all_image_sizes(args.crops_dir)

    # Save image_sizes.json
    image_sizes_path = os.path.join(args.output_dir, 'image_sizes.json')

    # Flatten for the sizes file
    all_sizes = {}
    for category, images in sizes_by_category.items():
        for img in images:
            # Key by relative path from crops dir for readability
            rel_path = os.path.relpath(img['path'], args.crops_dir)
            all_sizes[rel_path] = {
                'width': img['width'],
                'height': img['height'],
                'long_side': img['long_side'],
                'category': category
            }

    with open(image_sizes_path, 'w') as f:
        json.dump(all_sizes, f, indent=2)

    total_images = sum(len(imgs) for imgs in sizes_by_category.values())
    print('\nSaved sizes for {} images to {}'.format(
        total_images, image_sizes_path
    ))

    # Step 2: select exemplars
    print('\nSelecting {} exemplars per category...'.format(
        args.images_per_category
    ))

    all_examples = []
    for category, images in sorted(sizes_by_category.items()):
        selected = select_exemplars(images, args.images_per_category)
        for img in selected:
            all_examples.append({
                'path': img['path'],
                'filename': img['filename'],
                'category': category,
                'width': img['width'],
                'height': img['height'],
                'long_side': img['long_side']
            })
        sizes = [s['long_side'] for s in selected]
        print('  {}: selected {} images (sizes: {})'.format(
            category,
            len(selected),
            ', '.join(str(s) for s in sizes)
        ))

    # Save few_shot_examples.json
    few_shot_path = os.path.join(args.output_dir, 'few_shot_examples.json')
    few_shot_data = {
        'selection_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'crops_dir': args.crops_dir,
            'images_per_category': args.images_per_category,
            'selection_method': 'quartile_median',
            'total_examples': len(all_examples)
        },
        'examples': all_examples
    }

    with open(few_shot_path, 'w') as f:
        json.dump(few_shot_data, f, indent=2)

    print('\nSaved {} few-shot examples to {}'.format(
        len(all_examples), few_shot_path
    ))

# ...def main()


#%% Command-line entry point

if __name__ == '__main__':
    main()
