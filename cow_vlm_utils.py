#%% Header

"""
Shared utilities for cattle behavior VLM classification.

Provides image processing, prompt construction, result parsing,
checkpointing, and cost estimation for the cattle behavior
classification pipeline.
"""


#%% Imports and constants

import os
import io
import re
import json
import math
import base64

from datetime import datetime
from PIL import Image


#%% Category constants

VALID_CATEGORIES = ['head_up', 'head_down', 'running', 'unknown']

# Maps folder names in sorted_crops to canonical category names
CATEGORY_FOLDER_MAP = {
    'headdown': 'head_down',
    'headup': 'head_up',
    'running': 'running',
    'unknown': 'unknown'
}

# Maps common model-generated aliases to canonical categories
CATEGORY_ALIASES = {
    'headup': 'head_up',
    'head up': 'head_up',
    'head-up': 'head_up',
    'heads up': 'head_up',
    'looking up': 'head_up',
    'standing': 'head_up',
    'alert': 'head_up',
    'headdown': 'head_down',
    'head down': 'head_down',
    'head-down': 'head_down',
    'heads down': 'head_down',
    'grazing': 'head_down',
    'eating': 'head_down',
    'feeding': 'head_down',
    'looking down': 'head_down',
    'trotting': 'running',
    'galloping': 'running',
    'moving': 'running',
    'walking': 'running',
    'run': 'running',
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}


#%% Path constants

SORTED_CROPS_DIR = r'C:\temp\cow-experiments\sorted_crops'
OUTPUT_BASE_DIR = r'C:\temp\cow-experiments\cow-vlm-experiments'
API_KEY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'GEMINI_API_KEY.txt'
)

DEFAULT_IMAGE_MAX_SIZE = 750
DEFAULT_QUERY_BATCH_SIZE = 5
DEFAULT_CHECKPOINT_INTERVAL = 50


#%% Gemini pricing constants (per 1M tokens, as of January 2026)

GEMINI_30_FLASH_INPUT_PER_M = 0.50
GEMINI_30_FLASH_OUTPUT_PER_M = 3.00
GEMINI_30_PRO_INPUT_PER_M = 2.00
GEMINI_30_PRO_OUTPUT_PER_M = 12.00
BATCH_DISCOUNT = 0.50

# Estimated tokens per image at <= 750px (fits in one 768x768 tile)
ESTIMATED_TOKENS_PER_IMAGE = 258

# Estimated text tokens in the prompt (system + few-shot labels + instructions)
ESTIMATED_TEXT_TOKENS = 500

# Estimated output tokens per request (JSON array for 5 classifications)
ESTIMATED_OUTPUT_TOKENS_PER_REQUEST = 150


#%% Prompt text

SYSTEM_PROMPT = """You are an expert at classifying cattle behavior in camera trap images.

Each cow should be classified into exactly one of these categories:
- head_up: the cow's head is in a position parallel or above the top of the shoulder. It does not matter if the cow is laying down, standing, or walking, it is only based on the vertical relationship between the head and the top of the shoulder.
- head_down: the cow's head is in a position below the top of the shoulder. Similarly, it does not matter if the cow is laying down, standing, or walking.
- running: all four hooves are off the ground, or the cow's legs are in a bounding or suspended stride, or the cow is clearly in rapid motion (often signaled by motion blur).
- unknown: the cow's head is out of frame, or the image is extremely unclear, to the point where you can't make out the posture of the cow.
"""


#%% Image processing functions

def resize_image_to_bytes(image_path, max_size=DEFAULT_IMAGE_MAX_SIZE):
    """
    Resize image so long side <= max_size and return as JPEG bytes.

    Only downscales; if the image is already <= max_size, it is re-encoded
    as JPEG without resizing.  Uses LANCZOS resampling and JPEG quality=85.

    Args:
        image_path: path to the image file
        max_size: maximum dimension for the long side (default 750)

    Returns:
        JPEG image as bytes
    """

    try:
        with Image.open(image_path) as img:

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            width, height = img.size

            # Only downscale, never upscale
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int((height * max_size) / width)
                else:
                    new_height = max_size
                    new_width = int((width * max_size) / height)
                img = img.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS
                )
            # ...if already small enough, keep original size

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
            return img_byte_arr.getvalue()

    except Exception as e:
        raise Exception(
            'Failed to process image {}: {}'.format(image_path, str(e))
        )

# ...def resize_image_to_bytes()


def resize_image_to_base64(image_path, max_size=DEFAULT_IMAGE_MAX_SIZE):
    """
    Resize image so long side <= max_size and return as base64 string.

    Args:
        image_path: path to the image file
        max_size: maximum dimension for the long side (default 750)

    Returns:
        base64-encoded JPEG image string
    """

    image_bytes = resize_image_to_bytes(image_path, max_size)
    return base64.b64encode(image_bytes).decode('utf-8')


def get_image_dimensions(image_path):
    """
    Return (width, height) for an image without loading the full pixel data.

    Args:
        image_path: path to the image file

    Returns:
        tuple of (width, height)
    """

    with Image.open(image_path) as img:
        return img.size


#%% Dataset enumeration

def enumerate_dataset_images(sorted_crops_dir=SORTED_CROPS_DIR):
    """
    Enumerate all images across all category subfolders.

    Args:
        sorted_crops_dir: path to the sorted_crops directory containing
            subfolders named 'headdown', 'headup', 'running', 'unknown'

    Returns:
        sorted list of dicts, each with keys:
            'path': absolute path to the image
            'filename': just the filename
            'ground_truth': canonical category name (e.g. 'head_down')
    """

    all_images = []

    for folder_name, category in CATEGORY_FOLDER_MAP.items():
        folder_path = os.path.join(sorted_crops_dir, folder_name)
        if not os.path.isdir(folder_path):
            print('Warning: folder not found: {}'.format(folder_path))
            continue

        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            all_images.append({
                'path': os.path.join(folder_path, filename),
                'filename': filename,
                'ground_truth': category
            })

    # Sort by filename for deterministic ordering
    all_images.sort(key=lambda x: x['filename'])
    return all_images

# ...def enumerate_dataset_images()


#%% Few-shot example management

def load_few_shot_examples(few_shot_path=None):
    """
    Load few-shot examples from a JSON file.

    Args:
        few_shot_path: path to the few_shot_examples.json file.
            Defaults to OUTPUT_BASE_DIR/few_shot_examples.json.

    Returns:
        list of dicts, each with 'path', 'filename', 'category',
        'width', 'height', 'long_side'
    """

    if few_shot_path is None:
        few_shot_path = os.path.join(OUTPUT_BASE_DIR, 'few_shot_examples.json')

    with open(few_shot_path, 'r') as f:
        data = json.load(f)

    return data['examples']

# ...def load_few_shot_examples()


def get_test_images(all_images, few_shot_examples):
    """
    Filter out few-shot example images from the full dataset.

    Args:
        all_images: list of image dicts from enumerate_dataset_images()
        few_shot_examples: list of example dicts from load_few_shot_examples()

    Returns:
        list of image dicts excluding the few-shot examples
    """

    few_shot_filenames = set(ex['filename'] for ex in few_shot_examples)
    return [img for img in all_images if img['filename'] not in few_shot_filenames]

# ...def get_test_images()


def group_query_images(test_images, batch_size=DEFAULT_QUERY_BATCH_SIZE):
    """
    Group test images into batches of batch_size for multi-image queries.

    Args:
        test_images: list of image dicts
        batch_size: number of images per query (default 5)

    Returns:
        list of lists, each inner list contains up to batch_size image dicts
    """

    groups = []
    for i in range(0, len(test_images), batch_size):
        groups.append(test_images[i:i + batch_size])
    return groups

# ...def group_query_images()


#%% API key management

def load_api_key(api_key_path=None):
    """
    Load Gemini API key from a text file.

    Args:
        api_key_path: path to the API key file (default: GEMINI_API_KEY.txt
            in the same directory as this module)

    Returns:
        API key string

    Raises:
        FileNotFoundError: if the key file does not exist
        ValueError: if the key file is empty
    """

    if api_key_path is None:
        api_key_path = API_KEY_PATH

    if not os.path.exists(api_key_path):
        raise FileNotFoundError(
            'API key file not found: {}. Create this file with your '
            'Gemini API key.'.format(api_key_path)
        )

    with open(api_key_path, 'r') as f:
        api_key = f.read().strip()

    if not api_key:
        raise ValueError(
            'API key file is empty: {}'.format(api_key_path)
        )

    return api_key

# ...def load_api_key()


#%% Filename utilities

def sanitize_model_name(model_name):
    """
    Sanitize a model name for use in filenames.

    Replaces characters that are problematic in filenames (: / \\)
    with hyphens.

    Args:
        model_name: the raw model name string

    Returns:
        sanitized string safe for use in filenames
    """

    name = model_name.replace('models/', '')
    return name.replace(':', '-').replace('/', '-').replace('\\', '-')


#%% Response parsing

def normalize_category(raw_category):
    """
    Normalize a category string returned by a VLM.

    Strips whitespace, lowercases, checks against valid categories
    and known aliases.  Returns 'parse_error' for unrecognized values.

    Args:
        raw_category: the raw category string from the model

    Returns:
        normalized category string, or 'parse_error'
    """

    if not isinstance(raw_category, str):
        return 'parse_error'

    cleaned = raw_category.strip().lower()

    if cleaned in VALID_CATEGORIES:
        return cleaned

    if cleaned in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[cleaned]

    return 'parse_error'

# ...def normalize_category()


def _strip_thinking_tokens(text):
    """
    Remove <think>...</think> blocks from model output.

    Some models (e.g. qwen3-vl) emit thinking tokens wrapped in
    <think> tags before the actual response content.

    Args:
        text: raw model output text

    Returns:
        text with thinking blocks removed
    """

    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _try_parse_truncated_json(text):
    """
    Attempt to recover items from a truncated JSON array.

    When a model's output is cut off mid-array (e.g. due to num_predict
    limit), this extracts whatever complete JSON objects can be found.

    Args:
        text: text containing a possibly-truncated JSON array

    Returns:
        list of parsed dicts, or None if no items could be recovered
    """

    # Find individual JSON objects using a regex
    items = []
    for match in re.finditer(
        r'\{\s*"image_number"\s*:\s*(\d+)\s*,\s*"category"\s*:\s*"([^"]+)"\s*\}',
        text
    ):
        items.append({
            'image_number': int(match.group(1)),
            'category': match.group(2)
        })

    return items if items else None


def parse_vlm_response(response_text):
    """
    Parse a JSON response from a VLM classification query.

    Handles markdown code blocks (```json ... ```), thinking tokens
    (<think>...</think>), and truncated JSON arrays.  Returns as
    many valid items as can be recovered.

    Expected format:
        [
            {"image_number": 1, "category": "head_up"},
            {"image_number": 2, "category": "head_down"},
            ...
        ]

    Args:
        response_text: raw response text from the model

    Returns:
        list of dicts with 'image_number' (int) and 'category' (normalized str)

    Raises:
        ValueError: if the response cannot be parsed as JSON
    """

    text = response_text.strip()

    if not text:
        raise ValueError(
            'No JSON array found in response: {}'.format(text[:200])
        )

    # Strip thinking tokens
    text = _strip_thinking_tokens(text)

    # Strip markdown code block delimiters
    if '```' in text:
        text = re.sub(r'```\w*\n?', '', text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a complete JSON array within the text
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                # Array found but not valid JSON; try to recover items
                recovered = _try_parse_truncated_json(match.group())
                if recovered:
                    parsed = recovered
                else:
                    raise ValueError(
                        'Could not parse JSON from response: {}'.format(
                            text[:200]
                        )
                    )
        else:
            # No complete array found; try to recover from truncated array
            recovered = _try_parse_truncated_json(text)
            if recovered:
                parsed = recovered
            else:
                raise ValueError(
                    'No JSON array found in response: {}'.format(text[:200])
                )

    # Handle case where model returns a single dict instead of array
    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        raise ValueError(
            'Expected JSON array, got {}: {}'.format(
                type(parsed).__name__, str(parsed)[:200]
            )
        )

    # Normalize categories
    results = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        image_number = item.get('image_number', len(results) + 1)
        raw_category = item.get('category', item.get('label', ''))
        results.append({
            'image_number': int(image_number),
            'category': normalize_category(raw_category)
        })

    return results

# ...def parse_vlm_response()


#%% Checkpointing

def save_checkpoint(results, checkpoint_path):
    """
    Save results to a checkpoint file using atomic write pattern.

    Uses backup-delete-rename to avoid corrupting the checkpoint
    if the process is interrupted during write.

    Args:
        results: list of result dicts to save
        checkpoint_path: path to the checkpoint file (should end in .tmp.json)
    """

    backup_path = checkpoint_path.replace('.tmp.json', '.bk.tmp.json')

    checkpoint_data = {
        'checkpoint_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_processed': len(results),
            'is_checkpoint': True
        },
        'results': results
    }

    try:
        # Step 1: write to backup file
        with open(backup_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Step 2: remove old checkpoint if it exists
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        # Step 3: rename backup to checkpoint (atomic on most filesystems)
        os.rename(backup_path, checkpoint_path)

    except Exception as e:
        # Clean up backup on failure
        if os.path.exists(backup_path):
            try:
                os.remove(backup_path)
            except OSError:
                pass
        raise e

# ...def save_checkpoint()


def load_checkpoint(checkpoint_path):
    """
    Load results from a checkpoint file.

    Args:
        checkpoint_path: path to the checkpoint file

    Returns:
        list of result dicts, or empty list if file is missing/malformed
    """

    if not os.path.exists(checkpoint_path):
        return []

    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        results = data.get('results', [])
        print('Loaded checkpoint with {} results from {}'.format(
            len(results), checkpoint_path
        ))
        return results
    except (json.JSONDecodeError, KeyError) as e:
        print('Warning: could not load checkpoint {}: {}'.format(
            checkpoint_path, str(e)
        ))
        return []

# ...def load_checkpoint()


#%% Cost estimation

def estimate_gemini_cost(n_query_images, model_name, is_batch,
                         n_few_shot_images=16,
                         query_batch_size=DEFAULT_QUERY_BATCH_SIZE):
    """
    Estimate the cost of a Gemini classification run.

    Args:
        n_query_images: total number of query images
        model_name: Gemini model name (used to determine pricing tier)
        is_batch: True for batch API (50% discount), False for sync
        n_few_shot_images: number of few-shot example images per request
        query_batch_size: number of query images per request

    Returns:
        dict with keys: n_requests, total_input_tokens, total_output_tokens,
        input_cost, output_cost, total_cost, per_image_cost
    """

    n_requests = math.ceil(n_query_images / query_batch_size)

    input_tokens_per_request = (
        n_few_shot_images * ESTIMATED_TOKENS_PER_IMAGE
        + query_batch_size * ESTIMATED_TOKENS_PER_IMAGE
        + ESTIMATED_TEXT_TOKENS
    )
    output_tokens_per_request = ESTIMATED_OUTPUT_TOKENS_PER_REQUEST

    total_input_tokens = n_requests * input_tokens_per_request
    total_output_tokens = n_requests * output_tokens_per_request

    # Determine pricing tier
    model_lower = model_name.lower()
    if 'flash' in model_lower:
        input_rate = GEMINI_30_FLASH_INPUT_PER_M
        output_rate = GEMINI_30_FLASH_OUTPUT_PER_M
    else:
        input_rate = GEMINI_30_PRO_INPUT_PER_M
        output_rate = GEMINI_30_PRO_OUTPUT_PER_M

    if is_batch:
        input_rate *= BATCH_DISCOUNT
        output_rate *= BATCH_DISCOUNT

    input_cost = total_input_tokens * input_rate / 1_000_000
    output_cost = total_output_tokens * output_rate / 1_000_000
    total_cost = input_cost + output_cost

    return {
        'n_requests': n_requests,
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'input_cost': input_cost,
        'output_cost': output_cost,
        'total_cost': total_cost,
        'per_image_cost': total_cost / max(n_query_images, 1)
    }

# ...def estimate_gemini_cost()


def print_cost_estimate(cost_info, model_name, is_batch):
    """
    Print a formatted cost estimate to the console.

    Args:
        cost_info: dict returned by estimate_gemini_cost()
        model_name: model name for display
        is_batch: True for batch API, False for sync
    """

    mode = 'Batch API' if is_batch else 'Synchronous API'
    print('\n=== Cost Estimate ===')
    print('Model: {} ({})'.format(model_name, mode))
    print('Requests: {} ({} images per request + few-shot examples)'.format(
        cost_info['n_requests'],
        DEFAULT_QUERY_BATCH_SIZE
    ))
    print('Estimated input tokens: {:,}'.format(
        cost_info['total_input_tokens']
    ))
    print('Estimated output tokens: {:,}'.format(
        cost_info['total_output_tokens']
    ))
    print('Input cost: ${:.4f}'.format(cost_info['input_cost']))
    print('Output cost: ${:.4f}'.format(cost_info['output_cost']))
    print('Total estimated cost: ${:.4f}'.format(cost_info['total_cost']))
    print('Per-image cost: ${:.6f}'.format(cost_info['per_image_cost']))
    print()

# ...def print_cost_estimate()


#%% Result saving

def save_results(results, output_path, model_name, processing_method,
                 image_max_size=DEFAULT_IMAGE_MAX_SIZE,
                 query_batch_size=DEFAULT_QUERY_BATCH_SIZE,
                 few_shot_file=None,
                 few_shot_examples=None,
                 prompt_file=None):
    """
    Save classification results to a JSON file.

    Args:
        results: list of result dicts
        output_path: path for the output JSON file
        model_name: model name for metadata
        processing_method: e.g. 'batch_api', 'synchronous_api', 'ollama'
        image_max_size: the max image size used
        query_batch_size: the batch size used
        few_shot_file: path to the few-shot examples file used
        few_shot_examples: list of few-shot example dicts (from
            load_few_shot_examples); stored in the output JSON so
            visualizations can display the example images
        prompt_file: path to the prompt variation file used (if any)
    """

    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    correct = [r for r in successful if r.get('correct', False)]

    overall_accuracy = (
        len(correct) / len(successful) if len(successful) > 0 else 0.0
    )

    output_data = {
        'run_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model_name,
            'processing_method': processing_method,
            'image_max_size': image_max_size,
            'query_batch_size': query_batch_size,
            'few_shot_file': few_shot_file,
            'prompt_file': prompt_file,
            'total_query_images': len(results),
            'successful_predictions': len(successful),
            'failed_predictions': len(failed),
            'overall_accuracy': round(overall_accuracy, 4)
        },
        'results': results
    }

    if few_shot_examples is not None:
        output_data['few_shot_examples'] = [
            {
                'path': ex.get('path', ''),
                'filename': ex.get('filename', ''),
                'category': ex.get('category', ''),
            }
            for ex in few_shot_examples
        ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print('\nResults saved to: {}'.format(output_path))
    print('Total images: {}'.format(len(results)))
    print('Successful: {} ({:.1f}%)'.format(
        len(successful),
        100.0 * len(successful) / max(len(results), 1)
    ))
    print('Correct: {} ({:.1f}% of successful)'.format(
        len(correct),
        100.0 * overall_accuracy
    ))

# ...def save_results()


#%% Test cell

if __name__ == '__main__':

    # Quick sanity checks
    print('VALID_CATEGORIES:', VALID_CATEGORIES)
    print('API_KEY_PATH:', API_KEY_PATH)
    print('SORTED_CROPS_DIR:', SORTED_CROPS_DIR)

    # Test category normalization
    test_cases = [
        ('head_up', 'head_up'),
        ('Head Down', 'head_down'),
        ('RUNNING', 'running'),
        ('grazing', 'head_down'),
        ('galloping', 'running'),
        ('flying', 'parse_error'),
    ]
    for raw, expected in test_cases:
        result = normalize_category(raw)
        status = 'OK' if result == expected else 'FAIL'
        print('{}: normalize_category({!r}) = {!r} (expected {!r})'.format(
            status, raw, result, expected
        ))

    print('\nAll sanity checks passed.')
