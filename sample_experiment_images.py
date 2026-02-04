#%% Header

"""
Sample images for prompt/few-shot experimentation.

Selects a balanced set of correct and incorrect predictions from a model's
results to use as a test set for rapid iteration on prompts and few-shot
examples.

Usage:
    python sample_experiment_images.py path/to/results.json --output-dir DIR

Output:
    experiment_test_set.json containing 60 images:
    - 15 correctly classified head_up
    - 15 correctly classified head_down
    - 15 head_up misclassified as head_down
    - 15 head_down misclassified as head_up
"""


#%% Imports

import os
import json
import random
import argparse


#%% Sampling function

def sample_experiment_images(results_path, output_dir, n_per_category=15,
                              random_seed=42):
    """
    Sample a balanced test set from model results.

    Args:
        results_path: path to a model results JSON file
        output_dir: directory to write experiment_test_set.json
        n_per_category: number of images per category (default 15)
        random_seed: for reproducible sampling

    Returns:
        dict with sampled images and metadata
    """

    with open(results_path, 'r') as f:
        data = json.load(f)

    model_name = data.get('run_info', {}).get('model', 'unknown')
    results = data.get('results', [])

    print('Loaded {} results from {}'.format(len(results), model_name))

    # Categorize results
    correct_head_up = []
    correct_head_down = []
    error_headup_to_headdown = []  # true=head_up, pred=head_down
    error_headdown_to_headup = []  # true=head_down, pred=head_up

    for r in results:
        gt = r.get('ground_truth', '')
        pred = r.get('prediction', '')
        success = r.get('success', False)

        if not success:
            continue

        if gt == 'head_up' and pred == 'head_up':
            correct_head_up.append(r)
        elif gt == 'head_down' and pred == 'head_down':
            correct_head_down.append(r)
        elif gt == 'head_up' and pred == 'head_down':
            error_headup_to_headdown.append(r)
        elif gt == 'head_down' and pred == 'head_up':
            error_headdown_to_headup.append(r)

    print('Category counts:')
    print('  Correct head_up: {}'.format(len(correct_head_up)))
    print('  Correct head_down: {}'.format(len(correct_head_down)))
    print('  Error head_up -> head_down: {}'.format(len(error_headup_to_headdown)))
    print('  Error head_down -> head_up: {}'.format(len(error_headdown_to_headup)))

    # Sample from each category
    rng = random.Random(random_seed)

    def safe_sample(items, n):
        if len(items) <= n:
            return list(items)
        return rng.sample(items, n)

    sampled = {
        'correct_head_up': safe_sample(correct_head_up, n_per_category),
        'correct_head_down': safe_sample(correct_head_down, n_per_category),
        'error_headup_to_headdown': safe_sample(error_headup_to_headdown, n_per_category),
        'error_headdown_to_headup': safe_sample(error_headdown_to_headup, n_per_category),
    }

    # Build output structure
    test_images = []
    for category, items in sampled.items():
        for r in items:
            test_images.append({
                'image_path': r.get('image_path', ''),
                'image_filename': r.get('image_filename', ''),
                'ground_truth': r.get('ground_truth', ''),
                'sample_category': category,
                'original_prediction': r.get('prediction', ''),
            })

    output = {
        'source_model': model_name,
        'source_file': os.path.basename(results_path),
        'sampling_seed': random_seed,
        'n_per_category': n_per_category,
        'category_counts': {
            'correct_head_up': len(sampled['correct_head_up']),
            'correct_head_down': len(sampled['correct_head_down']),
            'error_headup_to_headdown': len(sampled['error_headup_to_headdown']),
            'error_headdown_to_headup': len(sampled['error_headdown_to_headup']),
        },
        'total_images': len(test_images),
        'images': test_images,
    }

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'experiment_test_set.json')

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print('')
    print('Sampled {} images total:'.format(len(test_images)))
    for cat, count in output['category_counts'].items():
        print('  {}: {}'.format(cat, count))
    print('')
    print('Saved to: {}'.format(output_path))

    return output


#%% Main

def main():

    parser = argparse.ArgumentParser(
        description='Sample images for prompt/few-shot experimentation.'
    )
    parser.add_argument(
        'results_file',
        help='Path to model results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='C:/temp/cow-experiments/cow-vlm-experiments/experiments',
        help='Output directory for experiment files'
    )
    parser.add_argument(
        '--n-per-category',
        type=int,
        default=15,
        help='Number of images per category (default: 15)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    args = parser.parse_args()

    sample_experiment_images(
        args.results_file,
        args.output_dir,
        n_per_category=args.n_per_category,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()
