#%% Header

"""
Create few-shot example variation JSON files for experimentation.

Generates 5 different sets of few-shot examples with varying selection
strategies. These can be edited manually or regenerated.

Usage:
    python create_few_shot_variations.py --output-dir DIR
"""


#%% Imports

import os
import json
import random
import argparse

from cow_vlm_utils import SORTED_CROPS_DIR, CATEGORY_FOLDER_MAP


#%% Helper functions

def get_all_images_by_category(crops_dir):
    """
    Load all images organized by category with their sizes.

    Returns:
        dict mapping category -> list of image dicts
    """

    from PIL import Image

    images_by_category = {
        'head_up': [],
        'head_down': [],
        'running': [],
        'unknown': [],
    }

    for folder, category in CATEGORY_FOLDER_MAP.items():
        folder_path = os.path.join(crops_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            filepath = os.path.join(folder_path, filename)
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
            except Exception:
                continue

            images_by_category[category].append({
                'path': filepath,
                'filename': filename,
                'category': category,
                'width': width,
                'height': height,
                'long_side': max(width, height),
            })

    return images_by_category


def select_by_quartile(images, n_per_category, seed=42):
    """
    Select images spanning the size range using quartile medians.

    This is the original selection method.
    """

    rng = random.Random(seed)
    selected = []

    for category, items in images.items():
        if not items:
            continue

        # Sort by long side
        sorted_items = sorted(items, key=lambda x: x['long_side'])

        if len(sorted_items) <= n_per_category:
            selected.extend(sorted_items)
            continue

        # Divide into quartiles and pick median of each
        quartile_size = len(sorted_items) // n_per_category
        for i in range(n_per_category):
            start = i * quartile_size
            end = start + quartile_size if i < n_per_category - 1 else len(sorted_items)
            quartile = sorted_items[start:end]
            median_idx = len(quartile) // 2
            selected.append(quartile[median_idx])

    return selected


def select_random(images, n_per_category, seed=42):
    """
    Select images randomly.
    """

    rng = random.Random(seed)
    selected = []

    for category, items in images.items():
        if not items:
            continue

        if len(items) <= n_per_category:
            selected.extend(items)
        else:
            selected.extend(rng.sample(items, n_per_category))

    return selected


def select_largest(images, n_per_category, seed=42):
    """
    Select the largest images (clearest detail).
    """

    selected = []

    for category, items in images.items():
        if not items:
            continue

        sorted_items = sorted(items, key=lambda x: x['long_side'], reverse=True)
        selected.extend(sorted_items[:n_per_category])

    return selected


def select_medium_size(images, n_per_category, seed=42):
    """
    Select medium-sized images (avoiding extremes).
    """

    rng = random.Random(seed)
    selected = []

    for category, items in images.items():
        if not items:
            continue

        # Sort by size and take middle portion
        sorted_items = sorted(items, key=lambda x: x['long_side'])
        n = len(sorted_items)

        # Take middle 50%
        start = n // 4
        end = 3 * n // 4
        middle = sorted_items[start:end]

        if len(middle) <= n_per_category:
            selected.extend(middle)
        else:
            selected.extend(rng.sample(middle, n_per_category))

    return selected


#%% Variation definitions

FEW_SHOT_VARIATIONS = {
    'v1_baseline': {
        'name': 'Baseline (4 per category, quartile selection)',
        'description': 'Current production selection: 4 examples per category spanning size range',
        'n_per_category': 4,
        'selection_method': 'quartile',
        'seed': 42,
    },
    'v2_more_examples': {
        'name': 'More examples (6 per category)',
        'description': 'Increased to 6 examples per category for more coverage',
        'n_per_category': 6,
        'selection_method': 'quartile',
        'seed': 42,
    },
    'v3_fewer_examples': {
        'name': 'Fewer examples (2 per category)',
        'description': 'Reduced to 2 examples per category to test if less is more',
        'n_per_category': 2,
        'selection_method': 'quartile',
        'seed': 42,
    },
    'v4_largest_images': {
        'name': 'Largest images only',
        'description': 'Select only the largest (clearest) images as examples',
        'n_per_category': 4,
        'selection_method': 'largest',
        'seed': 42,
    },
    'v5_different_seed': {
        'name': 'Alternative random selection',
        'description': 'Random selection with different seed for diversity',
        'n_per_category': 4,
        'selection_method': 'random',
        'seed': 123,
    },
}


#%% Main function

def create_few_shot_variations(output_dir, crops_dir=None):
    """
    Write few-shot variation JSON files.

    Args:
        output_dir: directory to write few_shot_variations/ folder
        crops_dir: path to sorted crops (default: SORTED_CROPS_DIR)
    """

    if crops_dir is None:
        crops_dir = SORTED_CROPS_DIR

    print('Loading images from {}...'.format(crops_dir))
    images = get_all_images_by_category(crops_dir)

    for cat, items in images.items():
        print('  {}: {} images'.format(cat, len(items)))

    few_shot_dir = os.path.join(output_dir, 'few_shot_variations')
    os.makedirs(few_shot_dir, exist_ok=True)

    for key, config in FEW_SHOT_VARIATIONS.items():
        method = config['selection_method']
        n = config['n_per_category']
        seed = config['seed']

        if method == 'quartile':
            examples = select_by_quartile(images, n, seed)
        elif method == 'random':
            examples = select_random(images, n, seed)
        elif method == 'largest':
            examples = select_largest(images, n, seed)
        elif method == 'medium':
            examples = select_medium_size(images, n, seed)
        else:
            examples = select_by_quartile(images, n, seed)

        output = {
            'id': key,
            'name': config['name'],
            'description': config['description'],
            'selection_info': {
                'n_per_category': n,
                'selection_method': method,
                'seed': seed,
                'total_examples': len(examples),
            },
            'examples': examples,
        }

        output_path = os.path.join(few_shot_dir, '{}.json'.format(key))
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print('Created: {} ({} examples)'.format(output_path, len(examples)))

    print('')
    print('Created {} few-shot variations in {}'.format(
        len(FEW_SHOT_VARIATIONS), few_shot_dir
    ))


def main():

    parser = argparse.ArgumentParser(
        description='Create few-shot variation JSON files for experimentation.'
    )
    parser.add_argument(
        '--output-dir',
        default='C:/temp/cow-experiments/cow-vlm-experiments/experiments',
        help='Output directory for experiment files'
    )
    parser.add_argument(
        '--crops-dir',
        default=None,
        help='Path to sorted crops directory'
    )

    args = parser.parse_args()
    create_few_shot_variations(args.output_dir, args.crops_dir)


if __name__ == '__main__':
    main()
