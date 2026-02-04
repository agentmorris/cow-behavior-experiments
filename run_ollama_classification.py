#%% Header

"""
Ollama VLM classification for cattle behavior.

Uses the /api/chat endpoint with multi-turn conversation to present
few-shot examples, then queries with up to 5 images per request.

Usage:
    python run_ollama_classification.py --model qwen3-VL:32b
    python run_ollama_classification.py --model qwen2.5vl:72b --resume checkpoint.tmp.json
"""


#%% Imports and constants

import os
import sys
import json
import time
import argparse
import requests as http_requests

from datetime import datetime

from cow_vlm_utils import (
    VALID_CATEGORIES,
    SORTED_CROPS_DIR,
    OUTPUT_BASE_DIR,
    DEFAULT_IMAGE_MAX_SIZE,
    DEFAULT_QUERY_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    SYSTEM_PROMPT,
    sanitize_model_name,
    resize_image_to_base64,
    enumerate_dataset_images,
    load_few_shot_examples,
    get_test_images,
    group_query_images,
    parse_vlm_response,
    save_checkpoint,
    load_checkpoint,
    save_results,
)

IMAGE_PROCESSING_TIMEOUT_SECONDS = 3600
DEFAULT_NUM_CTX = 16384
DEFAULT_NUM_PREDICT = 8192
DEFAULT_MAX_RETRIES = 4


#%% Ollama setup instructions

SETUP_HELP_TEXT = """
=== Ollama Setup Instructions for Cattle Behavior Classification ===

1. If Ollama is not installed, download from https://ollama.com/download
   (Windows installer available).

2. Start the Ollama server in a separate terminal:

   ollama serve

   If you get a bind error, the server is likely already running as a
   Windows service.

3. Pull the models you want to test:

   ollama pull qwen3-VL:32b
   ollama pull qwen2.5vl:72b

   Note: qwen2.5vl:72b requires ~49GB VRAM and will partially spill
   to CPU RAM on a 2x RTX 4090 setup (48GB total). This works but
   will be slower.

4. Verify models are available:

   ollama list

5. If you experience timeouts during model loading, set environment
   variables before starting the server:

   set OLLAMA_KEEP_ALIVE=1h
   set OLLAMA_LOAD_TIMEOUT=30m
   ollama serve

6. Models are stored in %USERPROFILE%\\.ollama\\models by default.
   Remove models you no longer need with:

   ollama rm <model_name>
"""


#%% Few-shot message building

def build_few_shot_messages(few_shot_examples, image_max_size, system_prompt=None):
    """
    Build chat messages for few-shot examples.

    Creates a system message followed by user/assistant pairs,
    one per example image.

    Args:
        few_shot_examples: list of example dicts
        image_max_size: maximum image dimension for resizing
        system_prompt: optional custom system prompt (uses SYSTEM_PROMPT if None)

    Returns:
        list of message dicts for /api/chat
    """

    prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    messages = [
        {'role': 'system', 'content': prompt}
    ]

    for ex in few_shot_examples:
        image_b64 = resize_image_to_base64(ex['path'], image_max_size)
        messages.append({
            'role': 'user',
            'content': 'What is the behavior of this cow?',
            'images': [image_b64]
        })
        messages.append({
            'role': 'assistant',
            'content': ex['category']
        })

    return messages

# ...def build_few_shot_messages()


def build_query_message(query_group, image_max_size):
    """
    Build the final user message with query images.

    Args:
        query_group: list of up to 5 image dicts
        image_max_size: maximum image dimension for resizing

    Returns:
        dict: a single user message with images
    """

    n = len(query_group)
    images_b64 = []
    for img in query_group:
        images_b64.append(resize_image_to_base64(img['path'], image_max_size))

    content = (
        'Now classify the following {} image{}. '
        'The images are provided in order, numbered 1 through {}.\n\n'
        'IMPORTANT: You MUST respond with ONLY a JSON array containing '
        'exactly {} item{}. Do NOT include any other text, explanation, '
        'markdown formatting, or code blocks. Just the raw JSON array.\n\n'
        'Required format:\n'
        '[{{"image_number": 1, "category": "..."}}, '
        '{{"image_number": 2, "category": "..."}}, ...]\n\n'
        'Valid categories are: {}'
    ).format(
        n,
        's' if n > 1 else '',
        n,
        n,
        's' if n > 1 else '',
        ', '.join(VALID_CATEGORIES)
    )

    return {
        'role': 'user',
        'content': content,
        'images': images_b64
    }

# ...def build_query_message()


#%% Ollama processor

class CowOllamaProcessor:
    """Handles VLM inference via Ollama for cattle classification."""

    def __init__(self, server_url='http://localhost:11434',
                 model_name='qwen3-VL:32b',
                 image_max_size=DEFAULT_IMAGE_MAX_SIZE,
                 num_ctx=DEFAULT_NUM_CTX,
                 num_predict=DEFAULT_NUM_PREDICT):
        """
        Initialize Ollama processor.

        Args:
            server_url: Ollama server URL
            model_name: Ollama model name
            image_max_size: maximum image dimension for resizing
            num_ctx: context window size in tokens (default 16384;
                many Ollama vision models default to 4096 which is
                too small for 16 few-shot images + query images)
            num_predict: maximum number of tokens to generate (default 1024;
                some models emit thinking tokens before the JSON response,
                so this needs headroom beyond the JSON output itself)
        """

        self.server_url = server_url
        self.model_name = model_name
        self.image_max_size = image_max_size
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def check_server_health(self):
        """
        Check whether the Ollama server is running and responsive.

        Returns:
            True if server is healthy, False otherwise
        """

        try:
            response = http_requests.get(
                '{}/api/tags'.format(self.server_url), timeout=5
            )
            if response.status_code == 200:
                print('Ollama server is running at {}'.format(self.server_url))
                return True
            else:
                print('Ollama server returned status {}'.format(
                    response.status_code
                ))
                return False
        except http_requests.exceptions.RequestException as e:
            print('Could not connect to Ollama server at {}: {}'.format(
                self.server_url, e
            ))
            return False

    # ...def check_server_health()

    def check_model_available(self):
        """
        Check whether the specified model is available.

        Returns:
            True if model is available, False otherwise
        """

        try:
            response = http_requests.get(
                '{}/api/tags'.format(self.server_url), timeout=10
            )
            if response.status_code == 200:
                models_data = response.json()
                available_models = [
                    model['name']
                    for model in models_data.get('models', [])
                ]

                # Check exact match or partial match
                for available in available_models:
                    if (self.model_name == available
                            or self.model_name in available
                            or available.startswith(self.model_name)):
                        print('Model {} is available'.format(self.model_name))
                        return True

                print('Model {} not found. Available models:'.format(
                    self.model_name
                ))
                for m in available_models:
                    print('  - {}'.format(m))
                return False

        except http_requests.exceptions.RequestException as e:
            print('Could not check model availability: {}'.format(e))
            return False

    # ...def check_model_available()

    def classify_group(self, few_shot_messages, query_group, request_index):
        """
        Classify a group of query images via /api/chat.

        Args:
            few_shot_messages: pre-built few-shot message list
            query_group: list of up to 5 image dicts
            request_index: index of this request

        Returns:
            list of result dicts, one per image in query_group
        """

        query_message = build_query_message(
            query_group, self.image_max_size
        )
        messages = few_shot_messages + [query_message]

        payload = {
            'model': self.model_name,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': 0.1,
                'num_predict': self.num_predict,
                'num_ctx': self.num_ctx
            }
        }

        try:
            response = http_requests.post(
                '{}/api/chat'.format(self.server_url),
                json=payload,
                timeout=IMAGE_PROCESSING_TIMEOUT_SECONDS
            )
            response.raise_for_status()

            response_data = response.json()
            message = response_data.get('message', {})
            response_text = message.get('content', '')

            # If the model produced thinking but no content, it likely
            # ran out of num_predict tokens during the thinking phase.
            # Capture the thinking text for diagnostics.
            thinking_text = message.get('thinking', '')
            done_reason = response_data.get('done_reason', '')
            if not response_text and thinking_text:
                eval_count = response_data.get('eval_count', 0)
                if done_reason == 'length':
                    hint = ' Increase --num-predict or reduce --query-batch-size.'
                else:
                    hint = ' Try reducing --query-batch-size.'
                print('    Warning: model produced {} thinking tokens '
                      'but no content (done_reason={}, eval_count={}).{}'
                      .format(len(thinking_text.split()),
                              done_reason, eval_count, hint))

            return process_response_for_group(
                response_text, query_group, request_index
            )

        except Exception as e:
            results = []
            request_images = [img['filename'] for img in query_group]
            for i, img in enumerate(query_group):
                results.append({
                    'image_path': img['path'],
                    'image_filename': img['filename'],
                    'ground_truth': img['ground_truth'],
                    'prediction': 'parse_error',
                    'correct': False,
                    'success': False,
                    'error_message': 'Ollama error: {}'.format(str(e)),
                    'raw_response': '',
                    'request_images': request_images,
                    'request_index': request_index,
                    'image_index_in_request': i + 1
                })
            return results

    # ...def classify_group()

    def classify_all(self, query_groups, few_shot_messages,
                     checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL,
                     checkpoint_path=None):
        """
        Classify all query image groups with checkpointing.

        Args:
            query_groups: list of lists of image dicts
            few_shot_messages: pre-built few-shot message list
            checkpoint_interval: save checkpoint every N images
            checkpoint_path: path for checkpoint file

        Returns:
            list of all result dicts
        """

        all_results = []
        total_images = sum(len(g) for g in query_groups)
        images_processed = 0

        for req_idx, group in enumerate(query_groups):

            start_time = time.time()
            group_results = self.classify_group(
                few_shot_messages, group, req_idx
            )
            elapsed = time.time() - start_time

            all_results.extend(group_results)
            images_processed += len(group)

            # Progress update
            correct_so_far = sum(
                1 for r in all_results if r.get('correct', False)
            )
            success_so_far = sum(
                1 for r in all_results if r.get('success', False)
            )
            accuracy = (
                correct_so_far / success_so_far
                if success_so_far > 0 else 0.0
            )
            print('  Request {}/{}: {}/{} images, '
                  'accuracy so far: {:.1f}%, '
                  'request time: {:.1f}s'.format(
                      req_idx + 1, len(query_groups),
                      images_processed, total_images,
                      100.0 * accuracy,
                      elapsed
                  ))

            # Checkpoint
            if (checkpoint_path and checkpoint_interval > 0
                    and images_processed % checkpoint_interval == 0):
                save_checkpoint(all_results, checkpoint_path)
                print('  Checkpoint saved ({} images)'.format(
                    images_processed
                ))

        return all_results

    # ...def classify_all()

# ...class CowOllamaProcessor


#%% Response processing (shared with Gemini)

def process_response_for_group(response_text, query_group, request_index):
    """
    Parse a VLM response and match predictions to query images.

    Args:
        response_text: raw text response from the model
        query_group: list of image dicts for this query group
        request_index: index of this request (for tracking)

    Returns:
        list of result dicts, one per image in query_group
    """

    results = []
    request_images = [img['filename'] for img in query_group]

    try:
        parsed = parse_vlm_response(response_text)
    except ValueError as e:
        # All images in this request fail
        for i, img in enumerate(query_group):
            results.append({
                'image_path': img['path'],
                'image_filename': img['filename'],
                'ground_truth': img['ground_truth'],
                'prediction': 'parse_error',
                'correct': False,
                'success': False,
                'error_message': str(e),
                'raw_response': response_text,
                'request_images': request_images,
                'request_index': request_index,
                'image_index_in_request': i + 1
            })
        return results

    # Match parsed results to query images by image_number
    parsed_by_number = {item['image_number']: item for item in parsed}

    for i, img in enumerate(query_group):
        image_number = i + 1
        if image_number in parsed_by_number:
            prediction = parsed_by_number[image_number]['category']
            results.append({
                'image_path': img['path'],
                'image_filename': img['filename'],
                'ground_truth': img['ground_truth'],
                'prediction': prediction,
                'correct': prediction == img['ground_truth'],
                'success': True,
                'raw_response': response_text,
                'request_images': request_images,
                'request_index': request_index,
                'image_index_in_request': image_number
            })
        else:
            results.append({
                'image_path': img['path'],
                'image_filename': img['filename'],
                'ground_truth': img['ground_truth'],
                'prediction': 'parse_error',
                'correct': False,
                'success': False,
                'error_message': 'Image {} not found in model response'
                    .format(image_number),
                'raw_response': response_text,
                'request_images': request_images,
                'request_index': request_index,
                'image_index_in_request': image_number
            })

    return results

# ...def process_response_for_group()


#%% Main function

def main():

    parser = argparse.ArgumentParser(
        description='Ollama cattle behavior classification.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_ollama_classification.py --model qwen3-VL:32b
  python run_ollama_classification.py --model qwen2.5vl:72b
  python run_ollama_classification.py --model qwen3-VL:32b --max-images 10
  python run_ollama_classification.py --setup-help
"""
    )

    parser.add_argument(
        '--model', '-m',
        default='qwen3-VL:32b',
        help='Ollama model name (default: qwen3-VL:32b)'
    )
    parser.add_argument(
        '--server-url',
        default='http://localhost:11434',
        help='Ollama server URL (default: http://localhost:11434)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=DEFAULT_IMAGE_MAX_SIZE,
        help='Maximum image dimension (default: {})'.format(
            DEFAULT_IMAGE_MAX_SIZE
        )
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help='Save checkpoint every N images '
             '(default: {}, 0 to disable)'.format(DEFAULT_CHECKPOINT_INTERVAL)
    )
    parser.add_argument(
        '--resume',
        metavar='FILE',
        help='Resume from checkpoint (.tmp.json)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=os.path.join(OUTPUT_BASE_DIR, 'results'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--few-shot-file',
        default=os.path.join(OUTPUT_BASE_DIR, 'few_shot_examples.json'),
        help='Path to few-shot examples JSON'
    )
    parser.add_argument(
        '--prompt-file',
        default=None,
        help='Path to prompt variation JSON (overrides default SYSTEM_PROMPT)'
    )
    parser.add_argument(
        '--max-images', '-n',
        type=int,
        default=0,
        help='Maximum number of query images (0 = all, for testing)'
    )
    parser.add_argument(
        '--query-batch-size',
        type=int,
        default=DEFAULT_QUERY_BATCH_SIZE,
        help='Number of query images per request (default: {})'.format(
            DEFAULT_QUERY_BATCH_SIZE
        )
    )
    parser.add_argument(
        '--num-ctx',
        type=int,
        default=DEFAULT_NUM_CTX,
        help='Context window size in tokens (default: {}). '
             'Many Ollama vision models default to 4096 which is '
             'too small for 16 few-shot images; increase if you '
             'see empty responses.'.format(DEFAULT_NUM_CTX)
    )
    parser.add_argument(
        '--num-predict',
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help='Maximum output tokens (default: {}). '
             'Some models emit thinking tokens before JSON, '
             'so this needs headroom.'.format(DEFAULT_NUM_PREDICT)
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help='Max retry rounds for failed images (default: {}). '
             'After the initial pass, failed images are re-grouped '
             'and retried up to this many times.'.format(DEFAULT_MAX_RETRIES)
    )
    parser.add_argument(
        '--setup-help',
        action='store_true',
        help='Print Ollama setup instructions and exit'
    )

    args = parser.parse_args()

    if args.setup_help:
        print(SETUP_HELP_TEXT)
        return

    # Initialize processor
    processor = CowOllamaProcessor(
        server_url=args.server_url,
        model_name=args.model,
        image_max_size=args.image_size,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict
    )

    # Health checks
    if not processor.check_server_health():
        print('\nOllama server is not running. Start it with: ollama serve')
        print('Run with --setup-help for full setup instructions.')
        sys.exit(1)

    if not processor.check_model_available():
        print('\nModel not found. Pull it with: ollama pull {}'.format(
            args.model
        ))
        sys.exit(1)

    # Load dataset and few-shot examples
    print('Loading dataset...')
    all_images = enumerate_dataset_images()
    few_shot_examples = load_few_shot_examples(args.few_shot_file)
    test_images = get_test_images(all_images, few_shot_examples)

    # Load custom prompt if specified
    custom_prompt = None
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt_data = json.load(f)
        custom_prompt = prompt_data.get('system_prompt', None)
        if custom_prompt:
            print('Using prompt: {} ({})'.format(
                prompt_data.get('id', 'unknown'),
                prompt_data.get('name', 'unknown')
            ))

    print('Total images: {}'.format(len(all_images)))
    print('Few-shot examples: {}'.format(len(few_shot_examples)))
    print('Test images: {}'.format(len(test_images)))

    # Limit for testing
    if args.max_images > 0:
        test_images = test_images[:args.max_images]
        print('Limited to {} test images'.format(len(test_images)))

    # Handle resume
    existing_results = []
    if args.resume:
        existing_results = load_checkpoint(args.resume)
        if existing_results:
            processed_filenames = set(
                r['image_filename'] for r in existing_results
            )
            test_images = [
                img for img in test_images
                if img['filename'] not in processed_filenames
            ]
            print('Resuming: {} already processed, {} remaining'.format(
                len(processed_filenames), len(test_images)
            ))

    if len(test_images) == 0:
        print('No images to process.')
        if existing_results:
            print('All images already processed in checkpoint.')
        return

    # Group query images
    query_groups = group_query_images(test_images, args.query_batch_size)
    print('Query groups: {} ({} images per group)'.format(
        len(query_groups), args.query_batch_size
    ))

    # Build few-shot messages (done once, reused for every request)
    print('Encoding few-shot example images...')
    few_shot_messages = build_few_shot_messages(
        few_shot_examples, args.image_size, system_prompt=custom_prompt
    )
    n_fs_images = sum(1 for m in few_shot_messages if 'images' in m)
    print('Few-shot messages built ({} messages, {} with images)'.format(
        len(few_shot_messages), n_fs_images
    ))

    # Set up checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_safe = sanitize_model_name(args.model)

    checkpoint_path = os.path.join(
        OUTPUT_BASE_DIR, 'checkpoints',
        '{}_{}.tmp.json'.format(model_safe, timestamp)
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Run classification
    print('\nStarting Ollama classification with {}...'.format(args.model))
    print('(First request may take several minutes while model loads)')

    new_results = processor.classify_all(
        query_groups, few_shot_messages,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=checkpoint_path
    )

    all_results = existing_results + new_results

    # Retry failed images
    for retry_round in range(1, args.max_retries + 1):
        failed_indices = [
            i for i, r in enumerate(all_results)
            if not r.get('success', False)
        ]
        if not failed_indices:
            break

        # Skip retries for server errors (e.g. 500 errors) that won't
        # succeed on retry with the same parameters
        retryable = [
            i for i in failed_indices
            if '500 Server Error' not in all_results[i].get(
                'error_message', '')
        ]
        if not retryable:
            print('\n{} failed images remaining, but all are non-retryable '
                  'server errors.'.format(len(failed_indices)))
            break

        print('\n=== Retry round {}/{}: {} failed images ({} retryable) ==='
              .format(retry_round, args.max_retries,
                      len(failed_indices), len(retryable)))

        retry_images = [
            {
                'path': all_results[i]['image_path'],
                'filename': all_results[i]['image_filename'],
                'ground_truth': all_results[i]['ground_truth']
            }
            for i in retryable
        ]
        retry_groups = group_query_images(
            retry_images, args.query_batch_size
        )
        retry_results = processor.classify_all(
            retry_groups, few_shot_messages,
            checkpoint_interval=0,
            checkpoint_path=None
        )

        # Build lookup by filename for retry results
        retry_by_filename = {}
        for r in retry_results:
            retry_by_filename[r['image_filename']] = r

        # Replace failed results with successful retries
        replaced = 0
        for i in retryable:
            fn = all_results[i]['image_filename']
            if fn in retry_by_filename:
                retry_r = retry_by_filename[fn]
                retry_r['retry_round'] = retry_round
                if retry_r.get('success', False):
                    all_results[i] = retry_r
                    replaced += 1
                else:
                    # Update with latest error info even if still failed
                    all_results[i] = retry_r

        success_count = sum(
            1 for r in all_results if r.get('success', False)
        )
        print('  Retry round {} recovered {} images '
              '(total success: {}/{})'.format(
                  retry_round, replaced, success_count, len(all_results)))

    # Save final results
    output_path = os.path.join(
        args.output_dir,
        '{}_{}.json'.format(model_safe, timestamp)
    )
    save_results(
        all_results, output_path, args.model,
        'ollama', args.image_size, args.query_batch_size,
        args.few_shot_file, few_shot_examples,
        prompt_file=args.prompt_file
    )

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print('Checkpoint cleaned up')
        except OSError:
            pass

# ...def main()


#%% Command-line entry point

if __name__ == '__main__':
    main()
