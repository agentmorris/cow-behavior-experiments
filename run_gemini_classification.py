#%% Header

"""
Gemini VLM classification for cattle behavior.

Supports batch API (async, 50% cheaper) and synchronous API.
Each request contains 16 few-shot example images plus up to 5 query images.
Results include image paths, ground truth labels, and predicted categories.

Usage:
    # Batch mode (default)
    python run_gemini_classification.py --model gemini-3-flash-preview

    # Synchronous mode
    python run_gemini_classification.py --model gemini-3-pro-preview --sync

    # Cancel a batch job
    python run_gemini_classification.py --cancel batches/xyz789

    # Resume from checkpoint or batch metadata
    python run_gemini_classification.py --resume path/to/checkpoint.tmp.json
"""


#%% Imports and constants

import os
import sys
import json
import time
import base64
import argparse

from google import genai
from datetime import datetime

from cow_vlm_utils import (
    VALID_CATEGORIES,
    SORTED_CROPS_DIR,
    OUTPUT_BASE_DIR,
    DEFAULT_IMAGE_MAX_SIZE,
    DEFAULT_QUERY_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    SYSTEM_PROMPT,
    load_api_key,
    sanitize_model_name,
    resize_image_to_bytes,
    resize_image_to_base64,
    enumerate_dataset_images,
    load_few_shot_examples,
    get_test_images,
    group_query_images,
    parse_vlm_response,
    save_checkpoint,
    load_checkpoint,
    estimate_gemini_cost,
    print_cost_estimate,
    save_results,
)

DEFAULT_MAX_RETRIES = 4


#%% Few-shot prompt building

def build_few_shot_parts(few_shot_examples, image_max_size, system_prompt=None):
    """
    Build the content parts list for few-shot examples.

    Returns a list of dicts suitable for Gemini API content parts,
    alternating text labels and inline_data image entries.  The images
    are grouped by category for clarity.

    Args:
        few_shot_examples: list of example dicts from load_few_shot_examples()
        image_max_size: maximum image dimension for resizing
        system_prompt: optional custom system prompt (uses SYSTEM_PROMPT if None)

    Returns:
        list of content part dicts (text and inline_data)
    """

    prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    parts = [
        {'text': prompt},
        {'text': '\n== LABELED EXAMPLES ==\n\n'
                 'Below are labeled examples of each behavior category.\n'}
    ]

    # Group examples by category, maintaining order
    by_category = {}
    for ex in few_shot_examples:
        cat = ex['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(ex)

    for category in VALID_CATEGORIES:
        examples = by_category.get(category, [])
        if not examples:
            continue

        parts.append(
            {'text': '\nCategory: {}\n'.format(category)}
        )

        for i, ex in enumerate(examples):
            image_b64 = resize_image_to_base64(ex['path'], image_max_size)
            parts.append(
                {'text': 'Example {} ({}):'.format(i + 1, category)}
            )
            parts.append({
                'inline_data': {
                    'mime_type': 'image/jpeg',
                    'data': image_b64
                }
            })
            parts.append(
                {'text': 'This cow is classified as: {}\n'.format(category)}
            )

    return parts

# ...def build_few_shot_parts()


def build_query_parts(query_group, image_max_size):
    """
    Build content parts for a group of query images.

    Args:
        query_group: list of image dicts (up to 5)
        image_max_size: maximum image dimension for resizing

    Returns:
        list of content part dicts
    """

    n = len(query_group)
    parts = [
        {'text': '\n== QUERY IMAGES ==\n\n'
                 'Now classify the following {} image{}.\n\n'
                 'IMPORTANT: You MUST respond with ONLY a JSON array '
                 'containing exactly {} item{}. Do NOT include any '
                 'other text, explanation, markdown formatting, or '
                 'code blocks. Just the raw JSON array.\n\n'
                 'Required format:\n'
                 '[{{"image_number": 1, "category": "..."}}, '
                 '{{"image_number": 2, "category": "..."}}, ...]\n\n'
                 'Valid categories are: {}\n'.format(
                     n,
                     's' if n > 1 else '',
                     n,
                     's' if n > 1 else '',
                     ', '.join(VALID_CATEGORIES)
                 )}
    ]

    for i, img in enumerate(query_group):
        image_b64 = resize_image_to_base64(img['path'], image_max_size)
        parts.append(
            {'text': 'Query image {}:'.format(i + 1)}
        )
        parts.append({
            'inline_data': {
                'mime_type': 'image/jpeg',
                'data': image_b64
            }
        })

    parts.append(
        {'text': '\nClassify all {} query image{} above. '
                 'Respond with ONLY the JSON array of exactly {} item{}.'.format(
                     n, 's' if n > 1 else '',
                     n, 's' if n > 1 else ''
                 )}
    )

    return parts

# ...def build_query_parts()


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
                'error_message': 'Image {} not found in model response'.format(
                    image_number
                ),
                'raw_response': response_text,
                'request_images': request_images,
                'request_index': request_index,
                'image_index_in_request': image_number
            })

    return results

# ...def process_response_for_group()


#%% Synchronous Gemini processor

class CowGeminiSyncProcessor:
    """Handles synchronous Gemini API for cattle classification."""

    def __init__(self, api_key, model_name='gemini-3-flash-preview',
                 image_max_size=DEFAULT_IMAGE_MAX_SIZE):
        """
        Initialize sync processor.

        Args:
            api_key: Gemini API key
            model_name: Gemini model name
            image_max_size: maximum image dimension for resizing
        """

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.image_max_size = image_max_size

    def classify_group(self, few_shot_parts, query_group, request_index,
                        max_api_retries=3):
        """
        Classify a group of query images in a single sync API call.

        Retries with exponential backoff on 429 (rate limit) errors.

        Args:
            few_shot_parts: pre-built few-shot content parts
            query_group: list of up to 5 image dicts
            request_index: index of this request
            max_api_retries: max retries on rate limit errors

        Returns:
            list of result dicts, one per image in query_group
        """

        query_parts = build_query_parts(query_group, self.image_max_size)
        all_parts = few_shot_parts + query_parts

        last_error = None
        for attempt in range(max_api_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=all_parts
                )
                response_text = response.text
                return process_response_for_group(
                    response_text, query_group, request_index
                )
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Retry on rate limiting (429) with exponential backoff
                if '429' in error_str and attempt < max_api_retries:
                    wait_time = 2 ** (attempt + 2)  # 4, 8, 16 seconds
                    print('    Rate limited (429), waiting {}s '
                          '(attempt {}/{})...'.format(
                              wait_time, attempt + 1, max_api_retries))
                    time.sleep(wait_time)
                    continue
                break

        # All retries exhausted or non-retryable error
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
                'error_message': 'API error: {}'.format(str(last_error)),
                'raw_response': '',
                'request_images': request_images,
                'request_index': request_index,
                'image_index_in_request': i + 1
            })
        return results

    # ...def classify_group()

    def classify_all(self, query_groups, few_shot_parts,
                     checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL,
                     checkpoint_path=None, rate_limit_pause=2):
        """
        Classify all query image groups with checkpointing.

        Args:
            query_groups: list of lists of image dicts
            few_shot_parts: pre-built few-shot content parts
            checkpoint_interval: save checkpoint every N images
            checkpoint_path: path for checkpoint file
            rate_limit_pause: seconds to pause between requests

        Returns:
            list of all result dicts
        """

        all_results = []
        total_images = sum(len(g) for g in query_groups)
        images_processed = 0

        for req_idx, group in enumerate(query_groups):

            group_results = self.classify_group(
                few_shot_parts, group, req_idx
            )
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
            print('  Request {}/{}: {}/{} images, accuracy so far: {:.1f}%'
                  .format(
                      req_idx + 1, len(query_groups),
                      images_processed, total_images,
                      100.0 * accuracy
                  ))

            # Checkpoint
            if (checkpoint_path and checkpoint_interval > 0
                    and images_processed % checkpoint_interval == 0):
                save_checkpoint(all_results, checkpoint_path)
                print('  Checkpoint saved ({} images)'.format(
                    images_processed
                ))

            # Rate limiting
            if req_idx < len(query_groups) - 1:
                time.sleep(rate_limit_pause)

        return all_results

    # ...def classify_all()

# ...class CowGeminiSyncProcessor


#%% Batch Gemini processor

class CowGeminiBatchProcessor:
    """Handles Gemini Batch API workflow for cattle classification."""

    def __init__(self, api_key, model_name='gemini-3-flash-preview',
                 image_max_size=DEFAULT_IMAGE_MAX_SIZE):
        """
        Initialize batch processor.

        Args:
            api_key: Gemini API key
            model_name: Gemini model name
            image_max_size: maximum image dimension for resizing
        """

        self.client = genai.Client(api_key=api_key)

        # Batch API needs 'models/' prefix
        if not model_name.startswith('models/'):
            model_name = 'models/' + model_name
        self.model_name = model_name
        self.image_max_size = image_max_size

    def prepare_batch_requests(self, query_groups, few_shot_parts):
        """
        Prepare batch requests, one per query group.

        Args:
            query_groups: list of lists of image dicts
            few_shot_parts: pre-built few-shot content parts

        Returns:
            list of (request_dict, group_metadata) tuples
        """

        print('Preparing batch requests for {} groups...'.format(
            len(query_groups)
        ))

        batch_requests = []

        for i, group in enumerate(query_groups):
            if (i + 1) % 50 == 0:
                print('  Prepared {}/{} requests...'.format(
                    i + 1, len(query_groups)
                ))

            query_parts = build_query_parts(group, self.image_max_size)
            all_parts = few_shot_parts + query_parts

            request = {
                'contents': [{
                    'parts': all_parts,
                    'role': 'user'
                }]
            }

            group_meta = {
                'request_index': i,
                'images': [
                    {
                        'path': img['path'],
                        'filename': img['filename'],
                        'ground_truth': img['ground_truth']
                    }
                    for img in group
                ]
            }

            batch_requests.append((request, group_meta))

        print('Prepared {} batch requests'.format(len(batch_requests)))
        return batch_requests

    # ...def prepare_batch_requests()

    def submit_batch_job(self, batch_requests_and_meta):
        """
        Submit batch job to Gemini.

        Args:
            batch_requests_and_meta: list of (request_dict, group_meta) tuples

        Returns:
            (batch_job object, list of group_meta dicts)
        """

        print('Submitting batch job to Gemini...')

        requests = [req for req, _ in batch_requests_and_meta]
        group_metas = [meta for _, meta in batch_requests_and_meta]

        try:
            batch_job = self.client.batches.create(
                model=self.model_name,
                src=requests,
                config={
                    'display_name': 'cow-classification-{}'.format(
                        datetime.now().strftime('%Y%m%d-%H%M%S')
                    ),
                }
            )

            print('  Batch job created: {}'.format(batch_job.name))
            print('  Status: {}'.format(batch_job.state.name))
            print('  Model: {}'.format(self.model_name))
            print('  Request count: {}'.format(len(requests)))

            return batch_job, group_metas

        except Exception as e:
            raise Exception(
                'Failed to submit batch job: {}'.format(str(e))
            )

    # ...def submit_batch_job()

    def save_batch_metadata(self, batch_job, group_metas, output_dir):
        """
        Save batch job metadata for resume capability.

        Args:
            batch_job: batch job object
            group_metas: list of group metadata dicts
            output_dir: directory for metadata file

        Returns:
            path to the saved metadata file
        """

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_safe = sanitize_model_name(self.model_name)
        metadata_path = os.path.join(
            output_dir,
            'gemini_batch_metadata_{}_{}.json'.format(model_safe, timestamp)
        )

        metadata = {
            'batch_job_name': batch_job.name,
            'model': self.model_name,
            'status': batch_job.state.name,
            'submission_time': timestamp,
            'total_requests': len(group_metas),
            'group_metas': group_metas
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print('  Batch metadata saved: {}'.format(metadata_path))
        return metadata_path

    # ...def save_batch_metadata()

    def poll_batch_completion(self, batch_job, poll_interval=60):
        """
        Poll batch job until completion.

        Args:
            batch_job: batch job object
            poll_interval: seconds between polls

        Returns:
            completed batch job object
        """

        print('Polling batch job: {}'.format(batch_job.name))
        print('Poll interval: {} seconds'.format(poll_interval))

        start_time = time.time()
        poll_count = 0

        completed_states = {
            'JOB_STATE_SUCCEEDED',
            'JOB_STATE_FAILED',
            'JOB_STATE_CANCELLED',
            'JOB_STATE_EXPIRED',
        }

        while True:
            try:
                current_job = self.client.batches.get(name=batch_job.name)
                poll_count += 1
                elapsed = time.time() - start_time

                print('  Poll #{} ({:.1f}h elapsed): Status = {}'.format(
                    poll_count, elapsed / 3600, current_job.state.name
                ))

                if current_job.state.name in completed_states:
                    if current_job.state.name == 'JOB_STATE_SUCCEEDED':
                        print('  Batch job completed successfully!')
                        return current_job
                    elif current_job.state.name == 'JOB_STATE_FAILED':
                        error = getattr(current_job, 'error', 'Unknown error')
                        raise Exception(
                            'Batch job failed: {}'.format(error)
                        )
                    elif current_job.state.name == 'JOB_STATE_CANCELLED':
                        raise Exception('Batch job was cancelled')
                    elif current_job.state.name == 'JOB_STATE_EXPIRED':
                        raise Exception('Batch job expired')

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                print('\nPolling interrupted by user')
                print('Batch job name: {}'.format(batch_job.name))
                print('To cancel this job, run:')
                print('  python run_gemini_classification.py --cancel {}'
                      .format(batch_job.name))
                print('The job will continue running on Google\'s servers '
                      'until cancelled')
                raise

            except Exception as e:
                error_msg = str(e).lower()
                if any(kw in error_msg for kw in
                       ['cancelled', 'expired', 'failed']):
                    raise
                else:
                    print('  Error during polling: {}'.format(e))
                    print('  Will retry in {} seconds...'.format(
                        poll_interval
                    ))
                    time.sleep(poll_interval)

    # ...def poll_batch_completion()

    def download_and_process_results(self, batch_job, group_metas):
        """
        Download and process batch results.

        Args:
            batch_job: completed batch job object
            group_metas: list of group metadata dicts

        Returns:
            list of result dicts, one per image
        """

        print('Processing batch results...')

        if not (batch_job.dest and batch_job.dest.inlined_responses):
            raise Exception('No results found in batch job')

        responses = batch_job.dest.inlined_responses
        print('Found {} responses'.format(len(responses)))

        if len(responses) != len(group_metas):
            raise Exception(
                'Response count mismatch: expected {} but got {}'.format(
                    len(group_metas), len(responses)
                )
            )

        all_results = []

        for resp_idx, result in enumerate(responses):
            group_meta = group_metas[resp_idx]
            query_group = group_meta['images']

            try:
                if (result.response and hasattr(result.response, 'text')
                        and result.response.text):
                    response_text = result.response.text.strip()
                    group_results = process_response_for_group(
                        response_text,
                        query_group,
                        group_meta['request_index']
                    )
                    all_results.extend(group_results)
                else:
                    # No response text: mark all images as failed
                    request_images = [img['filename'] for img in query_group]
                    for i, img in enumerate(query_group):
                        all_results.append({
                            'image_path': img['path'],
                            'image_filename': img['filename'],
                            'ground_truth': img['ground_truth'],
                            'prediction': 'parse_error',
                            'correct': False,
                            'success': False,
                            'error_message': 'No text in batch response',
                            'raw_response': '',
                            'request_images': request_images,
                            'request_index': group_meta['request_index'],
                            'image_index_in_request': i + 1
                        })
            except Exception as e:
                print('  Warning: error processing response {}: {}'.format(
                    resp_idx, e
                ))
                request_images = [img['filename'] for img in query_group]
                for i, img in enumerate(query_group):
                    all_results.append({
                        'image_path': img['path'],
                        'image_filename': img['filename'],
                        'ground_truth': img['ground_truth'],
                        'prediction': 'parse_error',
                        'correct': False,
                        'success': False,
                        'error_message': str(e),
                        'raw_response': '',
                        'request_images': request_images,
                        'request_index': group_meta['request_index'],
                        'image_index_in_request': i + 1
                    })

        successful = [r for r in all_results if r.get('success', False)]
        correct = [r for r in successful if r.get('correct', False)]
        print('Processed {} images: {} successful, {} correct'.format(
            len(all_results), len(successful), len(correct)
        ))

        return all_results

    # ...def download_and_process_results()

# ...class CowGeminiBatchProcessor


#%% Batch job cancellation

def cancel_batch_job(api_key, job_name):
    """
    Cancel a running batch job.

    Args:
        api_key: Gemini API key
        job_name: batch job name (e.g. 'batches/xyz789')
    """

    client = genai.Client(api_key=api_key)

    try:
        client.batches.cancel(name=job_name)
        print('Cancellation requested for job: {}'.format(job_name))
        print('It may take a few minutes for the job to fully stop.')
    except Exception as e:
        print('Error cancelling job: {}'.format(e))

# ...def cancel_batch_job()


#%% Main function

def main():

    parser = argparse.ArgumentParser(
        description='Gemini cattle behavior classification (batch or sync).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Batch mode (default)
  python run_gemini_classification.py --model gemini-3-flash-preview

  # Sync mode with cost auto-confirm
  python run_gemini_classification.py --model gemini-3-pro-preview --sync -y

  # Test with 10 images
  python run_gemini_classification.py --sync --max-images 10 -y

  # Cancel a batch job
  python run_gemini_classification.py --cancel batches/xyz789
"""
    )

    parser.add_argument(
        '--model', '-m',
        default='gemini-3-flash-preview',
        help='Gemini model name (default: gemini-3-flash-preview)'
    )
    parser.add_argument(
        '--sync',
        action='store_true',
        help='Use synchronous API instead of batch'
    )
    parser.add_argument(
        '--cancel',
        metavar='JOB_NAME',
        help='Cancel a running batch job'
    )
    parser.add_argument(
        '--resume',
        metavar='FILE',
        help='Resume from checkpoint (.tmp.json) or batch metadata (.json)'
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
        '--auto-confirm', '-y',
        action='store_true',
        help='Skip cost confirmation prompt'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help='Save checkpoint every N images, sync mode only '
             '(default: {}, 0 to disable)'.format(DEFAULT_CHECKPOINT_INTERVAL)
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
        '--max-retries',
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help='Max retry rounds for failed images after the initial pass '
             '(default: {}). Set to 0 to disable.'.format(DEFAULT_MAX_RETRIES)
    )
    parser.add_argument(
        '--poll-interval', '-p',
        type=int,
        default=60,
        help='Batch polling interval in seconds (default: 60)'
    )

    args = parser.parse_args()

    # Handle cancellation
    if args.cancel:
        api_key = load_api_key()
        cancel_batch_job(api_key, args.cancel)
        return

    # Validate arguments
    if args.sync and args.resume and not args.resume.endswith('.tmp.json'):
        print('Error: sync mode resume requires a .tmp.json checkpoint file')
        sys.exit(1)

    # Load API key
    api_key = load_api_key()

    # Handle resume for batch mode
    if args.resume and not args.sync:
        if args.resume.endswith('.tmp.json'):
            print('Error: batch mode resume requires a batch metadata .json '
                  'file, not a checkpoint')
            sys.exit(1)

        print('Resuming from batch metadata: {}'.format(args.resume))
        with open(args.resume, 'r') as f:
            metadata = json.load(f)

        processor = CowGeminiBatchProcessor(
            api_key=api_key,
            model_name=metadata['model'],
            image_max_size=args.image_size
        )

        # Create a minimal batch job object for polling
        class MinimalBatchJob:
            def __init__(self, name):
                self.name = name
        batch_job = MinimalBatchJob(metadata['batch_job_name'])

        try:
            completed_job = processor.poll_batch_completion(
                batch_job, args.poll_interval
            )
            all_results = processor.download_and_process_results(
                completed_job, metadata['group_metas']
            )

            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_safe = sanitize_model_name(metadata['model'])
            output_path = os.path.join(
                args.output_dir,
                '{}_batch_{}.json'.format(model_safe, timestamp)
            )
            resume_few_shot = load_few_shot_examples(args.few_shot_file)
            save_results(
                all_results, output_path, metadata['model'],
                'batch_api', args.image_size, args.query_batch_size,
                args.few_shot_file, resume_few_shot,
                prompt_file=args.prompt_file
            )
        except Exception as e:
            print('Error during resume: {}'.format(e))
            sys.exit(1)

        return

    # Load dataset and few-shot examples
    print('Loading dataset...')
    all_images = enumerate_dataset_images()
    few_shot_examples = load_few_shot_examples(args.few_shot_file)
    test_images = get_test_images(all_images, few_shot_examples)

    print('Total images: {}'.format(len(all_images)))
    print('Few-shot examples: {}'.format(len(few_shot_examples)))
    print('Test images: {}'.format(len(test_images)))

    # Limit for testing
    if args.max_images > 0:
        test_images = test_images[:args.max_images]
        print('Limited to {} test images'.format(len(test_images)))

    # Group query images
    query_groups = group_query_images(test_images, args.query_batch_size)
    print('Query groups: {} ({} images per group)'.format(
        len(query_groups), args.query_batch_size
    ))

    # Load custom prompt if provided
    custom_prompt = None
    if args.prompt_file:
        print('Loading custom prompt from {}'.format(args.prompt_file))
        with open(args.prompt_file, 'r') as f:
            prompt_data = json.load(f)
        custom_prompt = prompt_data.get('system_prompt', None)
        if custom_prompt:
            print('Using prompt: {} ({})'.format(
                prompt_data.get('id', 'unknown'),
                prompt_data.get('name', 'unknown')
            ))

    # Build few-shot parts (done once, reused for every request)
    print('Encoding few-shot example images...')
    few_shot_parts = build_few_shot_parts(
        few_shot_examples, args.image_size, system_prompt=custom_prompt
    )
    print('Few-shot parts built ({} parts)'.format(len(few_shot_parts)))

    # Cost estimation
    is_batch = not args.sync
    cost_info = estimate_gemini_cost(
        len(test_images), args.model, is_batch,
        len(few_shot_examples), args.query_batch_size
    )
    print_cost_estimate(cost_info, args.model, is_batch)

    if not args.auto_confirm:
        confirm = input('Continue? (y/N): ').strip().lower()
        if confirm != 'y':
            print('Cancelled.')
            return

    # Set up output
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_safe = sanitize_model_name(args.model)

    if args.sync:

        # === Synchronous mode ===
        processor = CowGeminiSyncProcessor(
            api_key=api_key,
            model_name=args.model,
            image_max_size=args.image_size
        )

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
                query_groups = group_query_images(
                    test_images, args.query_batch_size
                )
                print('Resuming: {} already processed, {} remaining'.format(
                    len(processed_filenames), len(test_images)
                ))

        checkpoint_path = os.path.join(
            OUTPUT_BASE_DIR, 'checkpoints',
            '{}_sync_{}.tmp.json'.format(model_safe, timestamp)
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        print('\nStarting synchronous classification...')
        new_results = processor.classify_all(
            query_groups, few_shot_parts,
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

            print('\n=== Retry round {}/{}: {} failed images ==='
                  .format(retry_round, args.max_retries,
                          len(failed_indices)))

            retry_images = [
                {
                    'path': all_results[i]['image_path'],
                    'filename': all_results[i]['image_filename'],
                    'ground_truth': all_results[i]['ground_truth']
                }
                for i in failed_indices
            ]
            retry_groups = group_query_images(
                retry_images, args.query_batch_size
            )

            # Use longer pauses during retries to avoid rate limits
            retry_results = processor.classify_all(
                retry_groups, few_shot_parts,
                checkpoint_interval=0,
                checkpoint_path=None,
                rate_limit_pause=max(4, 2 ** retry_round)
            )

            # Replace failed results with successful retries
            retry_by_filename = {}
            for r in retry_results:
                retry_by_filename[r['image_filename']] = r

            replaced = 0
            for i in failed_indices:
                fn = all_results[i]['image_filename']
                if fn in retry_by_filename:
                    retry_r = retry_by_filename[fn]
                    retry_r['retry_round'] = retry_round
                    if retry_r.get('success', False):
                        all_results[i] = retry_r
                        replaced += 1
                    else:
                        all_results[i] = retry_r

            success_count = sum(
                1 for r in all_results if r.get('success', False)
            )
            print('  Retry round {} recovered {} images '
                  '(total success: {}/{})'.format(
                      retry_round, replaced,
                      success_count, len(all_results)))

        # Save final results
        output_path = os.path.join(
            args.output_dir,
            '{}_sync_{}.json'.format(model_safe, timestamp)
        )
        save_results(
            all_results, output_path, args.model,
            'synchronous_api', args.image_size, args.query_batch_size,
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

    else:

        # === Batch mode ===
        processor = CowGeminiBatchProcessor(
            api_key=api_key,
            model_name=args.model,
            image_max_size=args.image_size
        )

        # Prepare batch requests
        batch_requests = processor.prepare_batch_requests(
            query_groups, few_shot_parts
        )

        # Submit
        batch_job, group_metas = processor.submit_batch_job(batch_requests)

        # Save metadata immediately for resume
        processor.save_batch_metadata(
            batch_job, group_metas, args.output_dir
        )

        # Poll for completion
        try:
            completed_job = processor.poll_batch_completion(
                batch_job, args.poll_interval
            )
        except KeyboardInterrupt:
            print('\nUse --resume with the metadata file to resume later.')
            sys.exit(0)

        # Process results
        all_results = processor.download_and_process_results(
            completed_job, group_metas
        )

        # Save final results
        output_path = os.path.join(
            args.output_dir,
            '{}_batch_{}.json'.format(model_safe, timestamp)
        )
        save_results(
            all_results, output_path, args.model,
            'batch_api', args.image_size, args.query_batch_size,
            args.few_shot_file, few_shot_examples,
            prompt_file=args.prompt_file
        )

# ...def main()


#%% Command-line entry point

if __name__ == '__main__':
    main()
