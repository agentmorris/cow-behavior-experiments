#%% Header

"""
Run prompt and few-shot experiments on a test set.

Loads experiment configurations from JSON files and runs classification
on a sampled test set to compare different prompt/few-shot strategies.

Usage:
    # Run all prompt variations with baseline few-shot (Gemini)
    python run_prompt_experiment.py --experiment-dir DIR --vary-prompts \\
        --models gemini-3-flash-preview

    # Run all few-shot variations with baseline prompt (Ollama)
    python run_prompt_experiment.py --experiment-dir DIR --vary-few-shot \\
        --models qwen2.5vl:32b gemma3:27b

    # Run specific combinations
    python run_prompt_experiment.py --experiment-dir DIR \\
        --prompts v1_baseline v3_explicit_examples \\
        --few-shots v1_baseline v2_more_examples \\
        --models gemini-3-flash-preview
"""


#%% Imports

import os
import io
import json
import glob
import base64
import time
import argparse
from datetime import datetime

from PIL import Image

from cow_vlm_utils import (
    load_api_key, resize_image_to_base64, resize_image_to_bytes,
    parse_vlm_response, sanitize_model_name, DEFAULT_IMAGE_MAX_SIZE,
    OUTPUT_BASE_DIR, API_KEY_PATH
)


#%% Configuration loaders

def load_test_set(experiment_dir):
    """Load the experiment test set."""

    path = os.path.join(experiment_dir, 'experiment_test_set.json')
    with open(path, 'r') as f:
        return json.load(f)


def load_prompt_variations(experiment_dir):
    """Load all prompt variation files."""

    prompts_dir = os.path.join(experiment_dir, 'prompt_variations')
    variations = {}

    for filepath in glob.glob(os.path.join(prompts_dir, '*.json')):
        with open(filepath, 'r') as f:
            data = json.load(f)
        variations[data['id']] = data

    return variations


def load_few_shot_variations(experiment_dir):
    """Load all few-shot variation files."""

    few_shot_dir = os.path.join(experiment_dir, 'few_shot_variations')
    variations = {}

    for filepath in glob.glob(os.path.join(few_shot_dir, '*.json')):
        with open(filepath, 'r') as f:
            data = json.load(f)
        variations[data['id']] = data

    return variations


#%% Gemini experiment runner

def run_gemini_experiment(model_name, test_images, system_prompt, few_shot_examples,
                           api_key, image_max_size=DEFAULT_IMAGE_MAX_SIZE):
    """
    Run a Gemini experiment on the test set.

    Args:
        model_name: Gemini model name
        test_images: list of test image dicts
        system_prompt: the system prompt to use
        few_shot_examples: list of few-shot example dicts
        api_key: Gemini API key
        image_max_size: max dimension for images

    Returns:
        dict with results
    """

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Pre-encode few-shot images
    few_shot_parts = []

    # System prompt part
    few_shot_parts.append(types.Part.from_text(text=system_prompt))
    few_shot_parts.append(types.Part.from_text(
        text='\nHere are examples of each category:\n'
    ))

    for ex in few_shot_examples:
        img_path = ex.get('path', '')
        category = ex.get('category', '')

        if not os.path.exists(img_path):
            continue

        img_bytes = resize_image_to_bytes(img_path, image_max_size)
        few_shot_parts.append(types.Part.from_text(
            text='\nExample of {}:'.format(category)
        ))
        few_shot_parts.append(types.Part.from_bytes(
            data=img_bytes, mime_type='image/jpeg'
        ))
        few_shot_parts.append(types.Part.from_text(
            text='This is {}.'.format(category)
        ))

    results = []
    start_time = time.time()
    consecutive_failures = 0
    max_consecutive_failures = 5

    # Process each test image individually for experiments
    for i, img_info in enumerate(test_images):
        img_path = img_info.get('image_path', '')
        img_filename = img_info.get('image_filename', '')
        ground_truth = img_info.get('ground_truth', '')

        if not os.path.exists(img_path):
            results.append({
                'image_path': img_path,
                'image_filename': img_filename,
                'ground_truth': ground_truth,
                'sample_category': img_info.get('sample_category', ''),
                'prediction': 'parse_error',
                'success': False,
                'error': 'Image not found',
            })
            continue

        # Build query
        query_parts = list(few_shot_parts)
        query_parts.append(types.Part.from_text(
            text='\n\nNow classify this image. Respond with ONLY the category '
                 'name (head_up, head_down, running, or unknown):'
        ))

        img_bytes = resize_image_to_bytes(img_path, image_max_size)
        query_parts.append(types.Part.from_bytes(
            data=img_bytes, mime_type='image/jpeg'
        ))

        # Retry logic for truncated/parse error responses
        max_retries = 3
        prediction = 'parse_error'
        response_text = ''
        last_error = None

        for attempt in range(max_retries):
            try:
                # Configure based on model type
                # Gemini 3 Pro requires thinking mode; use LOW to minimize token usage
                if 'pro' in model_name.lower():
                    config = types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=256,
                        thinkingConfig=types.ThinkingConfig(thinkingLevel='LOW'),
                    )
                else:
                    config = types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=256,
                    )
                response = client.models.generate_content(
                    model=model_name,
                    contents=[types.Content(role='user', parts=query_parts)],
                    config=config,
                )

                # Handle None response (quota exceeded, content blocked, etc.)
                if response.text is None:
                    last_error = 'Response text is None (quota exceeded or content blocked)'
                    if attempt < max_retries - 1:
                        time.sleep(2.0)
                    continue

                response_text = response.text.strip().lower()

                # Parse single category response (lenient matching)
                prediction = 'parse_error'
                if 'head_up' in response_text or response_text.startswith('head_u'):
                    prediction = 'head_up'
                elif 'head_down' in response_text or response_text.startswith('head_d'):
                    prediction = 'head_down'
                elif 'running' in response_text or response_text.startswith('run'):
                    prediction = 'running'
                elif 'unknown' in response_text or response_text.startswith('unk'):
                    prediction = 'unknown'

                # If successful, break out of retry loop
                if prediction != 'parse_error':
                    break

                # If parse error, wait briefly before retry
                if attempt < max_retries - 1:
                    time.sleep(1.0)

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(2.0)

        if prediction != 'parse_error':
            consecutive_failures = 0  # Reset on success
            results.append({
                'image_path': img_path,
                'image_filename': img_filename,
                'ground_truth': ground_truth,
                'sample_category': img_info.get('sample_category', ''),
                'prediction': prediction,
                'correct': prediction == ground_truth,
                'success': True,
                'raw_response': response_text,
                'retries': attempt,
            })
        else:
            consecutive_failures += 1
            results.append({
                'image_path': img_path,
                'image_filename': img_filename,
                'ground_truth': ground_truth,
                'sample_category': img_info.get('sample_category', ''),
                'prediction': 'parse_error',
                'success': False,
                'raw_response': response_text,
                'error': last_error or 'Truncated response after {} retries'.format(max_retries),
            })

            # Early termination if too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                print('\n  ERROR: {} consecutive failures - aborting experiment'.format(
                    consecutive_failures
                ))
                print('  Last error: {}'.format(last_error))
                break

        # Progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print('  {}/{} images, {:.1f}s elapsed'.format(
                i + 1, len(test_images), elapsed
            ))

        # Rate limiting
        time.sleep(0.5)

    return results


#%% Ollama experiment runner

def run_ollama_experiment(model_name, test_images, system_prompt, few_shot_examples,
                           server_url='http://localhost:11434',
                           image_max_size=DEFAULT_IMAGE_MAX_SIZE,
                           num_ctx=16384, num_predict=8192):
    """
    Run an Ollama experiment on the test set.

    Args:
        model_name: Ollama model name
        test_images: list of test image dicts
        system_prompt: the system prompt to use
        few_shot_examples: list of few-shot example dicts
        server_url: Ollama server URL
        image_max_size: max dimension for images
        num_ctx: context window size
        num_predict: max output tokens

    Returns:
        dict with results
    """

    import requests

    # Build few-shot messages
    messages = [{'role': 'system', 'content': system_prompt}]

    for ex in few_shot_examples:
        img_path = ex.get('path', '')
        category = ex.get('category', '')

        if not os.path.exists(img_path):
            continue

        img_b64 = resize_image_to_base64(img_path, image_max_size)

        messages.append({
            'role': 'user',
            'content': 'What is the behavior of this cow?',
            'images': [img_b64],
        })
        messages.append({
            'role': 'assistant',
            'content': category,
        })

    results = []
    start_time = time.time()

    # Process each test image individually
    for i, img_info in enumerate(test_images):
        img_path = img_info.get('image_path', '')
        img_filename = img_info.get('image_filename', '')
        ground_truth = img_info.get('ground_truth', '')

        if not os.path.exists(img_path):
            results.append({
                'image_path': img_path,
                'image_filename': img_filename,
                'ground_truth': ground_truth,
                'sample_category': img_info.get('sample_category', ''),
                'prediction': 'parse_error',
                'success': False,
                'error': 'Image not found',
            })
            continue

        # Build query message
        img_b64 = resize_image_to_base64(img_path, image_max_size)

        query_messages = list(messages)
        query_messages.append({
            'role': 'user',
            'content': 'What is the behavior of this cow? Respond with ONLY '
                       'the category name (head_up, head_down, running, or unknown).',
            'images': [img_b64],
        })

        try:
            response = requests.post(
                '{}/api/chat'.format(server_url),
                json={
                    'model': model_name,
                    'messages': query_messages,
                    'stream': False,
                    'options': {
                        'temperature': 0.0,
                        'num_ctx': num_ctx,
                        'num_predict': num_predict,
                    },
                },
                timeout=300,
            )

            if response.status_code != 200:
                raise Exception('HTTP {}: {}'.format(
                    response.status_code, response.text[:200]
                ))

            response_data = response.json()
            response_text = response_data.get('message', {}).get('content', '').strip().lower()

            # Parse single category response (lenient matching)
            prediction = 'parse_error'
            if 'head_up' in response_text or response_text.startswith('head_u'):
                prediction = 'head_up'
            elif 'head_down' in response_text or response_text.startswith('head_d'):
                prediction = 'head_down'
            elif 'running' in response_text or response_text.startswith('run'):
                prediction = 'running'
            elif 'unknown' in response_text or response_text.startswith('unk'):
                prediction = 'unknown'

            results.append({
                'image_path': img_path,
                'image_filename': img_filename,
                'ground_truth': ground_truth,
                'sample_category': img_info.get('sample_category', ''),
                'prediction': prediction,
                'correct': prediction == ground_truth,
                'success': prediction != 'parse_error',
                'raw_response': response_text,
            })

        except Exception as e:
            results.append({
                'image_path': img_path,
                'image_filename': img_filename,
                'ground_truth': ground_truth,
                'sample_category': img_info.get('sample_category', ''),
                'prediction': 'parse_error',
                'success': False,
                'error': str(e),
            })

        # Progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print('  {}/{} images, {:.1f}s elapsed'.format(
                i + 1, len(test_images), elapsed
            ))

    return results


#%% Results computation

def compute_experiment_metrics(results):
    """Compute accuracy metrics for experiment results."""

    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    correct = sum(1 for r in results if r.get('correct', False))

    # Per-category accuracy
    by_sample_category = {}
    for r in results:
        cat = r.get('sample_category', 'unknown')
        if cat not in by_sample_category:
            by_sample_category[cat] = {'total': 0, 'correct': 0}
        by_sample_category[cat]['total'] += 1
        if r.get('correct', False):
            by_sample_category[cat]['correct'] += 1

    category_accuracy = {}
    for cat, counts in by_sample_category.items():
        if counts['total'] > 0:
            category_accuracy[cat] = counts['correct'] / counts['total']
        else:
            category_accuracy[cat] = 0.0

    return {
        'total_images': total,
        'successful_predictions': successful,
        'correct_predictions': correct,
        'overall_accuracy': correct / total if total > 0 else 0.0,
        'category_accuracy': category_accuracy,
    }


#%% Main experiment runner

def run_experiment(experiment_dir, model_name, prompt_id, few_shot_id,
                   output_dir, api_key=None):
    """
    Run a single experiment combination.

    Args:
        experiment_dir: directory with experiment configurations
        model_name: model to use
        prompt_id: which prompt variation to use
        few_shot_id: which few-shot variation to use
        output_dir: where to save results
        api_key: Gemini API key (if using Gemini)

    Returns:
        dict with experiment results
    """

    # Load configurations
    test_set = load_test_set(experiment_dir)
    prompts = load_prompt_variations(experiment_dir)
    few_shots = load_few_shot_variations(experiment_dir)

    if prompt_id not in prompts:
        raise ValueError('Unknown prompt variation: {}'.format(prompt_id))
    if few_shot_id not in few_shots:
        raise ValueError('Unknown few-shot variation: {}'.format(few_shot_id))

    prompt_config = prompts[prompt_id]
    few_shot_config = few_shots[few_shot_id]

    system_prompt = prompt_config['system_prompt']
    few_shot_examples = few_shot_config['examples']
    test_images = test_set['images']

    # Add sample_category to test images for metrics
    for img in test_images:
        if 'sample_category' not in img:
            img['sample_category'] = img.get('ground_truth', 'unknown')

    print('Running experiment:')
    print('  Model: {}'.format(model_name))
    print('  Prompt: {} ({})'.format(prompt_id, prompt_config['name']))
    print('  Few-shot: {} ({})'.format(few_shot_id, few_shot_config['name']))
    print('  Test images: {}'.format(len(test_images)))
    print('  Few-shot examples: {}'.format(len(few_shot_examples)))
    print('')

    # Run experiment
    is_gemini = model_name.startswith('gemini')

    if is_gemini:
        if not api_key:
            api_key = load_api_key(API_KEY_PATH)
        results = run_gemini_experiment(
            model_name, test_images, system_prompt, few_shot_examples, api_key
        )
    else:
        results = run_ollama_experiment(
            model_name, test_images, system_prompt, few_shot_examples
        )

    # Compute metrics
    metrics = compute_experiment_metrics(results)

    # Build output
    output = {
        'experiment_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model_name,
            'prompt_id': prompt_id,
            'prompt_name': prompt_config['name'],
            'system_prompt': system_prompt,
            'few_shot_id': few_shot_id,
            'few_shot_name': few_shot_config['name'],
            'n_few_shot_examples': len(few_shot_examples),
            'n_test_images': len(test_images),
        },
        'metrics': metrics,
        'few_shot_examples': few_shot_examples,
        'results': results,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_safe = sanitize_model_name(model_name)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = '{}_{}_{}.json'.format(model_safe, prompt_id, few_shot_id)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print('')
    print('Results:')
    print('  Overall accuracy: {:.1f}%'.format(100 * metrics['overall_accuracy']))
    print('  By sample category:')
    for cat, acc in sorted(metrics['category_accuracy'].items()):
        print('    {}: {:.1f}%'.format(cat, 100 * acc))
    print('')
    print('Saved to: {}'.format(output_path))

    return output


#%% Main

def main():

    parser = argparse.ArgumentParser(
        description='Run prompt and few-shot experiments.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--experiment-dir',
        default='C:/temp/cow-experiments/cow-vlm-experiments/experiments',
        help='Directory containing experiment configurations'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for results (default: experiment_dir/results)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['gemini-3-flash-preview'],
        help='Models to run experiments on'
    )
    parser.add_argument(
        '--vary-prompts',
        action='store_true',
        help='Run all prompt variations with baseline few-shot'
    )
    parser.add_argument(
        '--vary-few-shot',
        action='store_true',
        help='Run all few-shot variations with baseline prompt'
    )
    parser.add_argument(
        '--prompts',
        nargs='+',
        default=None,
        help='Specific prompt variation IDs to run'
    )
    parser.add_argument(
        '--few-shots',
        nargs='+',
        default=None,
        help='Specific few-shot variation IDs to run'
    )
    parser.add_argument(
        '--baseline-prompt',
        default='v1_baseline',
        help='Baseline prompt ID for --vary-few-shot'
    )
    parser.add_argument(
        '--baseline-few-shot',
        default='v1_baseline',
        help='Baseline few-shot ID for --vary-prompts'
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, 'results')

    # Determine which combinations to run
    prompts_to_run = []
    few_shots_to_run = []

    if args.vary_prompts:
        all_prompts = load_prompt_variations(args.experiment_dir)
        prompts_to_run = list(all_prompts.keys())
        few_shots_to_run = [args.baseline_few_shot]
        print('Running prompt experiment: {} prompts x {} few-shot x {} models'.format(
            len(prompts_to_run), len(few_shots_to_run), len(args.models)
        ))

    elif args.vary_few_shot:
        all_few_shots = load_few_shot_variations(args.experiment_dir)
        prompts_to_run = [args.baseline_prompt]
        few_shots_to_run = list(all_few_shots.keys())
        print('Running few-shot experiment: {} prompts x {} few-shots x {} models'.format(
            len(prompts_to_run), len(few_shots_to_run), len(args.models)
        ))

    elif args.prompts or args.few_shots:
        prompts_to_run = args.prompts or ['v1_baseline']
        few_shots_to_run = args.few_shots or ['v1_baseline']
        print('Running custom experiment: {} prompts x {} few-shots x {} models'.format(
            len(prompts_to_run), len(few_shots_to_run), len(args.models)
        ))

    else:
        print('No experiment specified. Use --vary-prompts, --vary-few-shot, '
              'or specify --prompts and --few-shots')
        return

    print('')

    # Load API key once for Gemini models
    api_key = None
    if any(m.startswith('gemini') for m in args.models):
        api_key = load_api_key(API_KEY_PATH)

    # Run all combinations
    all_results = []
    total = len(args.models) * len(prompts_to_run) * len(few_shots_to_run)
    current = 0

    for model in args.models:
        for prompt_id in prompts_to_run:
            for few_shot_id in few_shots_to_run:
                current += 1
                print('=' * 60)
                print('Experiment {}/{}'.format(current, total))
                print('=' * 60)

                try:
                    result = run_experiment(
                        args.experiment_dir,
                        model,
                        prompt_id,
                        few_shot_id,
                        args.output_dir,
                        api_key=api_key,
                    )
                    all_results.append({
                        'model': model,
                        'prompt_id': prompt_id,
                        'few_shot_id': few_shot_id,
                        'accuracy': result['metrics']['overall_accuracy'],
                        'success': True,
                    })
                except Exception as e:
                    print('ERROR: {}'.format(e))
                    all_results.append({
                        'model': model,
                        'prompt_id': prompt_id,
                        'few_shot_id': few_shot_id,
                        'accuracy': 0.0,
                        'success': False,
                        'error': str(e),
                    })

                print('')

    # Print summary
    print('=' * 60)
    print('EXPERIMENT SUMMARY')
    print('=' * 60)
    print('')
    print('{:<30} {:<15} {:<15} {:>10}'.format(
        'Model', 'Prompt', 'Few-shot', 'Accuracy'
    ))
    print('-' * 70)

    for r in sorted(all_results, key=lambda x: -x['accuracy']):
        if r['success']:
            print('{:<30} {:<15} {:<15} {:>9.1f}%'.format(
                r['model'][:30], r['prompt_id'][:15], r['few_shot_id'][:15],
                100 * r['accuracy']
            ))
        else:
            print('{:<30} {:<15} {:<15} {:>10}'.format(
                r['model'][:30], r['prompt_id'][:15], r['few_shot_id'][:15],
                'ERROR'
            ))


if __name__ == '__main__':
    main()
