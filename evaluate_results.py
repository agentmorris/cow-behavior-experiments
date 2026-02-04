#%% Header

"""
Evaluate cattle behavior VLM classification results.

Computes accuracy, per-class precision/recall/F1, confusion matrix,
and generates comparison tables across models.

Usage:
    # Evaluate a single results file
    python evaluate_results.py path/to/results.json

    # Compare all results in a directory
    python evaluate_results.py results/ --compare
"""


#%% Imports and constants

import os
import json
import argparse
import glob

from cow_vlm_utils import VALID_CATEGORIES, OUTPUT_BASE_DIR


#%% Metrics computation

def compute_metrics(results):
    """
    Compute classification metrics from a results list.

    Args:
        results: list of result dicts (from a results JSON file)

    Returns:
        dict with keys:
            overall_accuracy, per_class (dict of precision/recall/f1/support),
            confusion_matrix (dict of dicts), total_images,
            successful_predictions, failed_predictions,
            category_distribution (ground truth counts)
    """

    # Filter to successful predictions only
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    # Confusion matrix: confusion[true_label][predicted_label] = count
    all_labels = VALID_CATEGORIES + ['parse_error']
    confusion = {}
    for true_label in all_labels:
        confusion[true_label] = {}
        for pred_label in all_labels:
            confusion[true_label][pred_label] = 0

    for r in successful:
        gt = r.get('ground_truth', 'unknown')
        pred = r.get('prediction', 'parse_error')
        if gt not in confusion:
            confusion[gt] = {label: 0 for label in all_labels}
        if pred not in confusion[gt]:
            confusion[gt][pred] = 0
        confusion[gt][pred] += 1

    # Add failed predictions as parse_error
    for r in failed:
        gt = r.get('ground_truth', 'unknown')
        if gt not in confusion:
            confusion[gt] = {label: 0 for label in all_labels}
        confusion[gt]['parse_error'] = confusion[gt].get('parse_error', 0) + 1

    # Per-class metrics
    per_class = {}
    for category in VALID_CATEGORIES:

        # True positives: predicted as this category AND actually is
        tp = confusion.get(category, {}).get(category, 0)

        # False positives: predicted as this category but actually something else
        fp = sum(
            confusion.get(true_cat, {}).get(category, 0)
            for true_cat in all_labels if true_cat != category
        )

        # False negatives: actually this category but predicted as something else
        fn = sum(
            confusion.get(category, {}).get(pred_cat, 0)
            for pred_cat in all_labels if pred_cat != category
        )

        # Support: total actual instances of this category
        support = sum(confusion.get(category, {}).values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        per_class[category] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': support,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

    # Overall accuracy
    correct = sum(1 for r in results if r.get('correct', False))
    total_successful = len(successful)
    overall_accuracy = (
        correct / total_successful if total_successful > 0 else 0.0
    )

    # Ground truth distribution
    category_distribution = {}
    for r in results:
        gt = r.get('ground_truth', 'unknown')
        category_distribution[gt] = category_distribution.get(gt, 0) + 1

    return {
        'overall_accuracy': round(overall_accuracy, 4),
        'per_class': per_class,
        'confusion_matrix': confusion,
        'total_images': len(results),
        'successful_predictions': total_successful,
        'failed_predictions': len(failed),
        'correct_predictions': correct,
        'category_distribution': category_distribution
    }

# ...def compute_metrics()


#%% Display functions

def format_confusion_matrix(confusion, categories=None):
    """
    Format a confusion matrix as a printable string table.

    Args:
        confusion: dict of dicts (true_label -> predicted_label -> count)
        categories: list of category labels to include

    Returns:
        formatted string
    """

    if categories is None:
        categories = VALID_CATEGORIES + ['parse_error']

    # Filter to categories that have at least one entry
    active_categories = [
        c for c in categories
        if any(confusion.get(c, {}).get(p, 0) > 0 for p in categories)
        or any(confusion.get(t, {}).get(c, 0) > 0 for t in categories)
    ]

    if not active_categories:
        return '(no data)'

    # Column widths
    label_width = max(len(c) for c in active_categories)
    col_width = max(6, label_width)

    # Header row
    header = '{:<{w}}'.format('True \\ Pred', w=label_width + 2)
    for pred in active_categories:
        header += '  {:<{w}}'.format(pred, w=col_width)
    header += '  Total'

    lines = [header]
    lines.append('-' * len(header))

    # Data rows
    for true_cat in active_categories:
        row = '{:<{w}}'.format(true_cat, w=label_width + 2)
        row_total = 0
        for pred_cat in active_categories:
            count = confusion.get(true_cat, {}).get(pred_cat, 0)
            row_total += count
            row += '  {:<{w}}'.format(count, w=col_width)
        row += '  {}'.format(row_total)
        lines.append(row)

    return '\n'.join(lines)

# ...def format_confusion_matrix()


def print_metrics(metrics, model_name=None):
    """
    Print metrics in a readable format.

    Args:
        metrics: dict from compute_metrics()
        model_name: optional model name for the header
    """

    if model_name:
        print('\n=== Results for {} ==='.format(model_name))
    else:
        print('\n=== Classification Results ===')

    print('Total images: {}'.format(metrics['total_images']))
    print('Successful predictions: {}'.format(
        metrics['successful_predictions']
    ))
    print('Failed predictions: {}'.format(metrics['failed_predictions']))
    print('Correct predictions: {}'.format(metrics['correct_predictions']))
    print('Overall accuracy: {:.1f}%'.format(
        100.0 * metrics['overall_accuracy']
    ))

    print('\nPer-class metrics:')
    print('{:<12} {:>10} {:>10} {:>10} {:>10}'.format(
        'Category', 'Precision', 'Recall', 'F1', 'Support'
    ))
    print('-' * 54)
    for category in VALID_CATEGORIES:
        m = metrics['per_class'].get(category, {})
        print('{:<12} {:>10.1f}% {:>10.1f}% {:>10.3f} {:>10}'.format(
            category,
            100.0 * m.get('precision', 0),
            100.0 * m.get('recall', 0),
            m.get('f1', 0),
            m.get('support', 0)
        ))

    print('\nConfusion Matrix:')
    print(format_confusion_matrix(metrics['confusion_matrix']))

# ...def print_metrics()


#%% Model comparison

def compare_models(results_dir):
    """
    Load all result JSON files and compare models.

    Args:
        results_dir: directory containing result JSON files

    Returns:
        list of (model_name, metrics_dict) tuples
    """

    # Find all result JSON files (exclude checkpoints and metadata)
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    result_files = [
        f for f in json_files
        if not os.path.basename(f).startswith('gemini_batch_metadata')
        and not f.endswith('.tmp.json')
    ]

    if not result_files:
        print('No result files found in {}'.format(results_dir))
        return []

    print('Found {} result files in {}'.format(
        len(result_files), results_dir
    ))

    model_results = []

    for filepath in sorted(result_files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            model_name = data.get('run_info', {}).get(
                'model', os.path.basename(filepath)
            )
            results = data.get('results', [])
            metrics = compute_metrics(results)
            model_results.append((model_name, metrics, filepath))

        except (json.JSONDecodeError, KeyError) as e:
            print('Warning: could not load {}: {}'.format(filepath, e))

    if not model_results:
        print('No valid results found.')
        return []

    # Print comparison table
    print('\n=== Model Comparison ===\n')

    # Header
    header = '{:<30} {:>10} {:>10}'.format(
        'Model', 'Accuracy', 'Images'
    )
    for cat in VALID_CATEGORIES:
        header += ' {:>10}'.format(cat)
    print(header)
    print('-' * len(header))

    # Rows
    for model_name, metrics, filepath in model_results:
        row = '{:<30} {:>9.1f}% {:>10}'.format(
            model_name[:30],
            100.0 * metrics['overall_accuracy'],
            metrics['total_images']
        )
        for cat in VALID_CATEGORIES:
            recall = metrics['per_class'].get(cat, {}).get('recall', 0)
            row += ' {:>9.1f}%'.format(100.0 * recall)
        print(row)

    print()

    # Print individual metrics for each model
    for model_name, metrics, filepath in model_results:
        print_metrics(metrics, model_name)

    return model_results

# ...def compare_models()


#%% Main function

def main():

    parser = argparse.ArgumentParser(
        description='Evaluate cattle behavior classification results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python evaluate_results.py path/to/results.json
  python evaluate_results.py results/ --compare
"""
    )

    parser.add_argument(
        'input',
        help='Single results JSON file or directory of result files'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all models in the input directory'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory to save evaluation JSON files'
    )

    args = parser.parse_args()

    if args.compare or os.path.isdir(args.input):
        # Multi-model comparison
        if not os.path.isdir(args.input):
            print('Error: {} is not a directory'.format(args.input))
            return
        model_results = compare_models(args.input)

        # Save evaluation if output-dir specified
        if args.output_dir and model_results:
            os.makedirs(args.output_dir, exist_ok=True)
            for model_name, metrics, filepath in model_results:
                safe_name = sanitize_model_name(model_name)
                eval_path = os.path.join(
                    args.output_dir,
                    'evaluation_{}.json'.format(safe_name)
                )
                with open(eval_path, 'w') as f:
                    json.dump({
                        'model': model_name,
                        'source_file': filepath,
                        'metrics': metrics
                    }, f, indent=2)
                print('Saved evaluation to {}'.format(eval_path))

    else:
        # Single model evaluation
        if not os.path.isfile(args.input):
            print('Error: {} is not a file'.format(args.input))
            return

        with open(args.input, 'r') as f:
            data = json.load(f)

        model_name = data.get('run_info', {}).get('model', 'unknown')
        results = data.get('results', [])
        metrics = compute_metrics(results)
        print_metrics(metrics, model_name)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            safe_name = sanitize_model_name(model_name)
            eval_path = os.path.join(
                args.output_dir,
                'evaluation_{}.json'.format(safe_name)
            )
            with open(eval_path, 'w') as f:
                json.dump({
                    'model': model_name,
                    'source_file': args.input,
                    'metrics': metrics
                }, f, indent=2)
            print('\nSaved evaluation to {}'.format(eval_path))

# ...def main()


#%% Utility import for comparison function

from cow_vlm_utils import sanitize_model_name


#%% Command-line entry point

if __name__ == '__main__':
    main()
