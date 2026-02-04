#%% Header

"""
Run classification on all models specified in a configuration file.

This script orchestrates runs across multiple models (Gemini and Ollama),
allowing per-model customization of prompt and few-shot variations while
providing sensible defaults.

Usage:
    # Run all models with default config
    python run_all_models.py

    # Run with custom config
    python run_all_models.py --config my_config.json

    # Test run with limited images
    python run_all_models.py --max-images 5

    # Run specific models only
    python run_all_models.py --models gemini-3-flash-preview qwen2.5vl:32b
"""


#%% Imports

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

from cow_vlm_utils import OUTPUT_BASE_DIR


#%% Constants

DEFAULT_CONFIG_FILE = 'model_run_config.json'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(OUTPUT_BASE_DIR, 'experiments')
PROMPT_VARIATIONS_DIR = os.path.join(EXPERIMENT_DIR, 'prompt_variations')
FEW_SHOT_VARIATIONS_DIR = os.path.join(EXPERIMENT_DIR, 'few_shot_variations')


#%% Configuration loading

def load_config(config_path):
    """Load and validate the configuration file."""

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required fields
    if 'models' not in config:
        raise ValueError('Config must contain "models" list')

    return config


def get_prompt_file(prompt_id):
    """Get the full path to a prompt variation file."""

    path = os.path.join(PROMPT_VARIATIONS_DIR, '{}.json'.format(prompt_id))
    if not os.path.exists(path):
        raise FileNotFoundError('Prompt file not found: {}'.format(path))
    return path


def get_few_shot_file(few_shot_id):
    """Get the full path to a few-shot variation file."""

    path = os.path.join(FEW_SHOT_VARIATIONS_DIR, '{}.json'.format(few_shot_id))
    if not os.path.exists(path):
        raise FileNotFoundError('Few-shot file not found: {}'.format(path))
    return path


#%% Model runners

def run_gemini_model(model_name, prompt_file, few_shot_file, output_dir,
                     max_images=0, sync=True):
    """
    Run classification using Gemini API.

    Args:
        model_name: Gemini model name
        prompt_file: path to prompt variation JSON
        few_shot_file: path to few-shot variation JSON
        output_dir: directory for results
        max_images: max images to process (0 = all)
        sync: use synchronous API (vs batch)

    Returns:
        subprocess return code
    """

    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, 'run_gemini_classification.py'),
        '--model', model_name,
        '--prompt-file', prompt_file,
        '--few-shot-file', few_shot_file,
        '--output-dir', output_dir,
        '--auto-confirm',
    ]

    if sync:
        cmd.append('--sync')

    if max_images > 0:
        cmd.extend(['--max-images', str(max_images)])

    print('Running: {}'.format(' '.join(cmd)))
    print('')

    result = subprocess.run(cmd)
    return result.returncode


def run_ollama_model(model_name, prompt_file, few_shot_file, output_dir,
                     max_images=0):
    """
    Run classification using Ollama API.

    Args:
        model_name: Ollama model name
        prompt_file: path to prompt variation JSON
        few_shot_file: path to few-shot variation JSON
        output_dir: directory for results
        max_images: max images to process (0 = all)

    Returns:
        subprocess return code
    """

    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, 'run_ollama_classification.py'),
        '--model', model_name,
        '--prompt-file', prompt_file,
        '--few-shot-file', few_shot_file,
        '--output-dir', output_dir,
    ]

    if max_images > 0:
        cmd.extend(['--max-images', str(max_images)])

    print('Running: {}'.format(' '.join(cmd)))
    print('')

    result = subprocess.run(cmd)
    return result.returncode


#%% Main orchestration

def run_all_models(config, output_dir, max_images=0, model_filter=None,
                   gemini_sync=True):
    """
    Run classification on all models in the config.

    Args:
        config: loaded configuration dict
        output_dir: directory for results
        max_images: max images per model (0 = all)
        model_filter: list of model names to run (None = all)
        gemini_sync: use sync API for Gemini

    Returns:
        dict of model_name -> return_code
    """

    default_prompt = config.get('default_prompt', 'v1_baseline')
    default_few_shot = config.get('default_few_shot', 'v1_baseline')

    # Resolve default files
    default_prompt_file = get_prompt_file(default_prompt)
    default_few_shot_file = get_few_shot_file(default_few_shot)

    results = {}
    models = config['models']

    # Filter models if specified
    if model_filter:
        models = [m for m in models if m['name'] in model_filter]

    total = len(models)
    print('=' * 60)
    print('Running {} models'.format(total))
    print('Default prompt: {}'.format(default_prompt))
    print('Default few-shot: {}'.format(default_few_shot))
    print('Output directory: {}'.format(output_dir))
    if max_images > 0:
        print('Max images per model: {}'.format(max_images))
    print('=' * 60)
    print('')

    for i, model_config in enumerate(models):
        model_name = model_config['name']
        model_type = model_config.get('type', 'ollama')

        # Get prompt and few-shot (with per-model overrides)
        prompt_id = model_config.get('prompt', default_prompt)
        few_shot_id = model_config.get('few_shot', default_few_shot)

        prompt_file = get_prompt_file(prompt_id)
        few_shot_file = get_few_shot_file(few_shot_id)

        print('=' * 60)
        print('Model {}/{}: {}'.format(i + 1, total, model_name))
        print('  Type: {}'.format(model_type))
        print('  Prompt: {}'.format(prompt_id))
        print('  Few-shot: {}'.format(few_shot_id))
        print('=' * 60)
        print('')

        try:
            if model_type == 'gemini':
                returncode = run_gemini_model(
                    model_name, prompt_file, few_shot_file, output_dir,
                    max_images=max_images, sync=gemini_sync
                )
            else:
                returncode = run_ollama_model(
                    model_name, prompt_file, few_shot_file, output_dir,
                    max_images=max_images
                )

            results[model_name] = returncode

            if returncode == 0:
                print('\n{}: SUCCESS\n'.format(model_name))
            else:
                print('\n{}: FAILED (exit code {})\n'.format(
                    model_name, returncode
                ))

        except Exception as e:
            print('\n{}: ERROR - {}\n'.format(model_name, e))
            results[model_name] = -1

    # Print summary
    print('')
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for model_name, returncode in results.items():
        status = 'OK' if returncode == 0 else 'FAILED ({})'.format(returncode)
        print('  {}: {}'.format(model_name, status))

    return results


#%% Main

def main():

    parser = argparse.ArgumentParser(
        description='Run classification on all configured models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--config', '-c',
        default=os.path.join(SCRIPT_DIR, DEFAULT_CONFIG_FILE),
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory for results (default: timestamped folder)'
    )
    parser.add_argument(
        '--max-images', '-n',
        type=int,
        default=0,
        help='Maximum images per model (0 = all)'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=None,
        help='Specific models to run (default: all in config)'
    )
    parser.add_argument(
        '--gemini-batch',
        action='store_true',
        help='Use batch API for Gemini (default: sync)'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List models in config and exit'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # List models mode
    if args.list_models:
        print('Models in config:')
        for m in config['models']:
            override_info = []
            if 'prompt' in m:
                override_info.append('prompt={}'.format(m['prompt']))
            if 'few_shot' in m:
                override_info.append('few_shot={}'.format(m['few_shot']))
            override_str = ' ({})'.format(', '.join(override_info)) if override_info else ''
            print('  {} [{}]{}'.format(m['name'], m.get('type', 'ollama'), override_str))
        return

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(OUTPUT_BASE_DIR, 'results', timestamp)

    os.makedirs(output_dir, exist_ok=True)

    # Run all models
    results = run_all_models(
        config,
        output_dir,
        max_images=args.max_images,
        model_filter=args.models,
        gemini_sync=not args.gemini_batch,
    )

    # Exit with error if any model failed
    if any(r != 0 for r in results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
