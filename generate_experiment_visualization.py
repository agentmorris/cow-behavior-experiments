#%% Header

"""
Generate HTML visualizations for prompt/few-shot experiment results.

Produces:
1. Master summary page with all experiments in a sortable table
2. Per-experiment detail pages with confusion matrix drill-down
3. Confusion cell detail pages showing actual images

Usage:
    python generate_experiment_visualization.py

    # Custom directories
    python generate_experiment_visualization.py --results-dir path/to/results
                                                 --output-dir path/to/output
"""


#%% Imports and constants

import os
import json
import argparse
import glob

from PIL import Image

from cow_vlm_utils import VALID_CATEGORIES, OUTPUT_BASE_DIR, sanitize_model_name


THUMBNAIL_MAX_SIZE = 200
DEFAULT_RESULTS_DIR = os.path.join(OUTPUT_BASE_DIR, 'experiments', 'results')
DEFAULT_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'experiments', 'visualizations')


CATEGORY_COLORS = {
    'head_up': '#4a90d9',
    'head_down': '#50b050',
    'running': '#e67e22',
    'unknown': '#999999',
    'parse_error': '#e74c3c',
}

SAMPLE_CATEGORY_COLORS = {
    'correct_head_up': '#27ae60',
    'correct_head_down': '#27ae60',
    'error_headup_to_headdown': '#e74c3c',
    'error_headdown_to_headup': '#e74c3c',
}


#%% Image utility

def copy_and_resize_image(source_path, output_folder, max_size=THUMBNAIL_MAX_SIZE):
    """
    Copy an image to the output folder, resizing so long side <= max_size.

    If the thumbnail already exists on disk it is not regenerated.

    Args:
        source_path: absolute path to the original image
        output_folder: directory where the thumbnail will be written
        max_size: maximum pixel dimension for the long side

    Returns:
        relative path suitable for an HTML <img> src attribute,
        or empty string on failure
    """

    safe_filename = os.path.basename(source_path)
    safe_filename = safe_filename.replace('#', '_').replace(':', '_')

    # Ensure consistent .jpg extension
    base, ext = os.path.splitext(safe_filename)
    if ext.lower() not in ('.jpg', '.jpeg'):
        safe_filename = base + '.jpg'

    dest_path = os.path.join(output_folder, safe_filename)

    # Skip if already created
    if os.path.exists(dest_path):
        return '{}/{}'.format(os.path.basename(output_folder), safe_filename)

    try:
        with Image.open(source_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            width, height = img.size
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
            img.save(dest_path, format='JPEG', quality=85, optimize=True)
        return '{}/{}'.format(os.path.basename(output_folder), safe_filename)

    except Exception as e:
        print('Warning: could not process image {}: {}'.format(
            source_path, e
        ))
        return ''


#%% CSS styles

def _css_common():
    """Return common CSS shared by all pages."""

    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, sans-serif;
            max-width: 1600px; margin: 0 auto; padding: 20px;
            background: #f5f5f5; color: #333;
        }
        h1 { font-size: 1.5em; margin-bottom: 8px; }
        h2 { font-size: 1.2em; margin: 20px 0 10px; }
        .header { background: #fff; padding: 20px; border-radius: 8px;
                   margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .stats { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px; }
        .stat { background: #f0f4f8; padding: 8px 16px; border-radius: 6px; }
        .stat-value { font-size: 1.3em; font-weight: bold; }
        .stat-label { font-size: 0.85em; color: #666; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { padding: 8px 12px; border: 1px solid #ddd; text-align: center; }
        th { background: #f0f4f8; font-weight: 600; cursor: pointer; }
        th:hover { background: #e0e8f0; }
        .section { background: #fff; padding: 20px; border-radius: 8px;
                   margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .filter-bar { margin: 12px 0; }
        .filter-btn {
            padding: 6px 14px; margin-right: 6px; border: 1px solid #ccc;
            border-radius: 4px; background: #fff; cursor: pointer;
            font-size: 0.9em;
        }
        .filter-btn.active { background: #4a90d9; color: #fff; border-color: #4a90d9; }
        .badge {
            display: inline-block; padding: 2px 10px; border-radius: 12px;
            color: #fff; font-size: 0.85em; font-weight: 600; margin-right: 6px;
        }
        a { color: #4a90d9; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .cm-cell-nonzero { font-weight: bold; }
        .cm-cell-diag { background: #e8f5e9; }
        .cm-cell-offdiag-nonzero { background: #fde8e8; }
        td a { color: inherit; text-decoration: underline; }
        td a:hover { color: #4a90d9; }
        .accuracy-high { color: #27ae60; font-weight: bold; }
        .accuracy-mid { color: #f39c12; }
        .accuracy-low { color: #e74c3c; }
        .sortable { position: relative; }
        .sortable::after { content: ' \\2195'; font-size: 0.8em; color: #999; }
    """


def _css_detail():
    """Return CSS for detail pages."""

    return """
        .image-card {
            background: #fff; padding: 12px; margin-bottom: 12px;
            border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            display: flex; gap: 16px; align-items: flex-start;
        }
        .image-card img { width: 200px; height: auto; border-radius: 4px; }
        .image-card .info { flex: 1; }
        .image-card .filename {
            font-size: 0.8em; color: #888; margin-top: 6px;
            word-break: break-all;
        }
        .image-card.incorrect { background: #fde8e8; border: 1px solid #f5c6c6; }
        .back-link { margin-bottom: 12px; }
        .sample-cat-badge {
            display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 0.75em; font-weight: 600; margin-left: 8px;
        }
    """


#%% Load experiment results

def load_experiment_results(results_dir):
    """
    Load all experiment result JSON files from a directory.

    Args:
        results_dir: directory containing result JSON files

    Returns:
        list of dicts with experiment info and results
    """

    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    experiments = []

    for filepath in sorted(json_files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Support both run_info (old format) and experiment_info (new format)
            exp_info = data.get('experiment_info', data.get('run_info', {}))
            results = data.get('results', [])

            if not results:
                continue

            # Calculate accuracy (excluding parse errors)
            parse_errors = sum(1 for r in results if r.get('prediction') == 'parse_error')
            successfully_parsed = len(results) - parse_errors
            correct = sum(1 for r in results if r.get('correct', False))
            accuracy = correct / successfully_parsed if successfully_parsed > 0 else 0
            parse_error_rate = parse_errors / len(results) if results else 0

            # Calculate per-sample-category accuracy (excluding parse errors)
            by_sample_cat = {}
            for r in results:
                sample_cat = r.get('sample_category', 'unknown')
                if sample_cat not in by_sample_cat:
                    by_sample_cat[sample_cat] = {'correct': 0, 'total': 0, 'parsed': 0}
                by_sample_cat[sample_cat]['total'] += 1
                if r.get('prediction') != 'parse_error':
                    by_sample_cat[sample_cat]['parsed'] += 1
                    if r.get('correct', False):
                        by_sample_cat[sample_cat]['correct'] += 1

            # Extract fields (support both old and new key names)
            model = exp_info.get('model', 'unknown')
            prompt = exp_info.get('prompt_id', exp_info.get('prompt', 'unknown'))
            prompt_desc = exp_info.get('prompt_name', exp_info.get('prompt_description', ''))
            few_shot = exp_info.get('few_shot_id', exp_info.get('few_shot', 'unknown'))
            few_shot_desc = exp_info.get('few_shot_name', exp_info.get('few_shot_description', ''))

            experiments.append({
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'model': model,
                'prompt': prompt,
                'prompt_description': prompt_desc,
                'few_shot': few_shot,
                'few_shot_description': few_shot_desc,
                'accuracy': accuracy,
                'parse_error_rate': parse_error_rate,
                'total_images': len(results),
                'successfully_parsed': successfully_parsed,
                'parse_errors': parse_errors,
                'correct_count': correct,
                'by_sample_category': by_sample_cat,
                'results': results,
            })

        except (json.JSONDecodeError, KeyError) as e:
            print('Warning: could not load {}: {}'.format(filepath, e))

    return experiments


#%% Compute confusion matrix

def compute_confusion_matrix(results):
    """
    Compute confusion matrix from results.

    Args:
        results: list of result dicts with ground_truth and prediction

    Returns:
        dict of dicts (true_label -> predicted_label -> count)
    """

    confusion = {}
    for r in results:
        gt = r.get('ground_truth', 'unknown')
        pred = r.get('prediction', 'parse_error')

        if gt not in confusion:
            confusion[gt] = {}
        if pred not in confusion[gt]:
            confusion[gt][pred] = 0
        confusion[gt][pred] += 1

    return confusion


#%% Build confusion matrix HTML

def build_confusion_matrix_html(confusion, link_prefix=None, exp_safe=None):
    """
    Build an HTML <table> for a confusion matrix.

    Args:
        confusion: dict of dicts (true_label -> predicted_label -> count)
        link_prefix: if provided, cell counts become links to detail pages
        exp_safe: sanitized experiment name for link filenames

    Returns:
        HTML string
    """

    categories = VALID_CATEGORIES + ['parse_error']

    # Filter to active categories
    active = [
        c for c in categories
        if any(confusion.get(c, {}).get(p, 0) > 0 for p in categories)
        or any(confusion.get(t, {}).get(c, 0) > 0 for t in categories)
    ]

    if not active:
        return '<p>(no data)</p>'

    lines = ['<table>']
    # Header row
    lines.append('<tr><th>True \\ Predicted</th>')
    for pred in active:
        lines.append('<th>{}</th>'.format(pred))
    lines.append('<th>Total</th></tr>')

    # Data rows
    for true_cat in active:
        lines.append('<tr><th>{}</th>'.format(true_cat))
        row_total = 0
        for pred_cat in active:
            count = confusion.get(true_cat, {}).get(pred_cat, 0)
            row_total += count
            css_classes = []
            if true_cat == pred_cat:
                css_classes.append('cm-cell-diag')
            if count > 0:
                css_classes.append('cm-cell-nonzero')
                if true_cat != pred_cat:
                    css_classes.append('cm-cell-offdiag-nonzero')
            cls = ' class="{}"'.format(' '.join(css_classes)) if css_classes else ''

            # Make count a link if link_prefix provided
            if (link_prefix and exp_safe and count > 0
                    and true_cat != 'parse_error' and pred_cat != 'parse_error'):
                link_url = '{}{}_{}_{}.html'.format(
                    link_prefix, exp_safe, true_cat, pred_cat
                )
                cell_content = '<a href="{}">{}</a>'.format(link_url, count)
            else:
                cell_content = str(count)

            lines.append('<td{}>{}</td>'.format(cls, cell_content))
        lines.append('<td><b>{}</b></td>'.format(row_total))
        lines.append('</tr>')

    lines.append('</table>')
    return '\n'.join(lines)


#%% Generate confusion detail pages

def generate_confusion_detail_pages(experiment, exp_safe, detail_dir, vis_images_folder):
    """
    Generate HTML pages for each confusion matrix cell.

    Args:
        experiment: experiment dict
        exp_safe: sanitized experiment name for filenames
        detail_dir: directory to write detail HTML pages
        vis_images_folder: directory containing shared thumbnails
    """

    os.makedirs(detail_dir, exist_ok=True)
    results = experiment['results']

    # Group images by (true_label, predicted_label)
    cells = {}
    for r in results:
        gt = r.get('ground_truth', 'unknown')
        pred = r.get('prediction', 'parse_error')

        if gt == 'parse_error' or pred == 'parse_error':
            continue

        key = (gt, pred)
        if key not in cells:
            cells[key] = []
        cells[key].append(r)

    # Generate a page for each cell
    for (true_cat, pred_cat), cell_results in cells.items():
        if not cell_results:
            continue

        filename = '{}_{}_{}.html'.format(exp_safe, true_cat, pred_cat)
        output_path = os.path.join(detail_dir, filename)

        is_correct = (true_cat == pred_cat)
        title = '{} / {} / {}: {} -> {}'.format(
            experiment['model'], experiment['prompt'], experiment['few_shot'],
            true_cat, pred_cat
        )

        html = '<!DOCTYPE html>\n<html>\n<head>\n'
        html += '<meta charset="UTF-8">\n'
        html += '<title>{}</title>\n'.format(title)
        html += '<style>{}\n{}</style>\n'.format(_css_common(), _css_detail())
        html += '</head>\n<body>\n'

        # Header
        html += '<div class="header">\n'
        html += '<div class="back-link"><a href="../{}.html">&larr; Back to experiment</a></div>\n'.format(exp_safe)
        html += '<h1>{} &rarr; {}</h1>\n'.format(true_cat, pred_cat)
        html += '<p><b>Model:</b> {} | <b>Prompt:</b> {} | <b>Few-shot:</b> {}</p>\n'.format(
            experiment['model'], experiment['prompt'], experiment['few_shot']
        )
        html += '<p>{} images</p>\n'.format(len(cell_results))
        html += '</div>\n'

        # Image list
        for r in cell_results:
            img_path = r.get('image_path', '')
            img_filename = r.get('image_filename', os.path.basename(img_path))
            sample_cat = r.get('sample_category', '')

            # Get thumbnail path
            safe_thumb = img_filename.replace('#', '_').replace(':', '_')
            base, ext = os.path.splitext(safe_thumb)
            if ext.lower() not in ('.jpg', '.jpeg'):
                safe_thumb = base + '.jpg'

            thumb_rel = '../vis_images/{}'.format(safe_thumb)

            card_class = 'image-card' if is_correct else 'image-card incorrect'

            # Sample category badge
            sample_cat_display = {
                'correct_head_up': 'Easy head_up',
                'correct_head_down': 'Easy head_down',
                'error_headup_to_headdown': 'Hard head_up',
                'error_headdown_to_headup': 'Hard head_down',
            }
            sample_cat_html = ''
            if sample_cat:
                sample_color = SAMPLE_CATEGORY_COLORS.get(sample_cat, '#888')
                display_name = sample_cat_display.get(sample_cat, sample_cat)
                sample_cat_html = '<span class="sample-cat-badge" style="background:{};color:#fff;">{}</span>'.format(
                    sample_color, display_name
                )

            html += '<div class="{}">\n'.format(card_class)
            html += '<img src="{}" alt="{}">\n'.format(thumb_rel, img_filename)
            html += '<div class="info">\n'
            html += '<div><b>Ground truth:</b> <span class="badge" style="background:{}">{}</span></div>\n'.format(
                CATEGORY_COLORS.get(true_cat, '#333'), true_cat
            )
            html += '<div><b>Prediction:</b> <span class="badge" style="background:{}">{}</span>{}</div>\n'.format(
                CATEGORY_COLORS.get(pred_cat, '#333'), pred_cat, sample_cat_html
            )
            html += '<div class="filename">{}</div>\n'.format(img_filename)
            html += '</div>\n</div>\n'

        html += '</body>\n</html>\n'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)


#%% Generate experiment detail page

def generate_experiment_detail_page(experiment, exp_safe, output_dir, vis_images_folder):
    """
    Generate HTML detail page for a single experiment.

    Args:
        experiment: experiment dict
        exp_safe: sanitized experiment name
        output_dir: directory for output HTML
        vis_images_folder: directory for thumbnails
    """

    output_path = os.path.join(output_dir, '{}.html'.format(exp_safe))
    results = experiment['results']

    # Generate thumbnails for all images
    for r in results:
        img_path = r.get('image_path', '')
        if img_path and os.path.exists(img_path):
            copy_and_resize_image(img_path, vis_images_folder)

    # Generate confusion detail pages
    detail_dir = os.path.join(output_dir, 'confusion_details')
    generate_confusion_detail_pages(experiment, exp_safe, detail_dir, vis_images_folder)

    # Compute confusion matrix
    confusion = compute_confusion_matrix(results)

    html = '<!DOCTYPE html>\n<html>\n<head>\n'
    html += '<meta charset="UTF-8">\n'
    html += '<title>{} / {} / {}</title>\n'.format(
        experiment['model'], experiment['prompt'], experiment['few_shot']
    )
    html += '<style>{}\n{}</style>\n'.format(_css_common(), _css_detail())
    html += '</head>\n<body>\n'

    # Header
    html += '<div class="header">\n'
    html += '<div class="back-link"><a href="index.html">&larr; Back to summary</a></div>\n'
    html += '<h1>Experiment Details</h1>\n'
    html += '<div class="stats">\n'
    html += '<div class="stat"><div class="stat-value">{}</div><div class="stat-label">Model</div></div>\n'.format(
        experiment['model']
    )
    html += '<div class="stat"><div class="stat-value">{}</div><div class="stat-label">Prompt</div></div>\n'.format(
        experiment['prompt']
    )
    html += '<div class="stat"><div class="stat-value">{}</div><div class="stat-label">Few-shot</div></div>\n'.format(
        experiment['few_shot']
    )
    html += '<div class="stat"><div class="stat-value">{:.1f}%</div><div class="stat-label">Accuracy</div></div>\n'.format(
        100 * experiment['accuracy']
    )
    html += '<div class="stat"><div class="stat-value">{:.1f}%</div><div class="stat-label">Parse Error Rate</div></div>\n'.format(
        100 * experiment.get('parse_error_rate', 0)
    )
    html += '</div>\n'
    html += '<p style="margin-top:8px;color:#666;font-size:0.9em;">Accuracy = {}/{} correct / successfully parsed (excludes {} parse errors)</p>\n'.format(
        experiment['correct_count'], experiment.get('successfully_parsed', experiment['total_images']),
        experiment.get('parse_errors', 0)
    )

    if experiment.get('prompt_description'):
        html += '<p style="margin-top:10px;color:#666;"><b>Prompt:</b> {}</p>\n'.format(
            experiment['prompt_description']
        )
    if experiment.get('few_shot_description'):
        html += '<p style="color:#666;"><b>Few-shot:</b> {}</p>\n'.format(
            experiment['few_shot_description']
        )

    html += '</div>\n'

    # Per-sample-category accuracy
    html += '<div class="section">\n'
    html += '<h2>Accuracy by Sample Category</h2>\n'
    html += '<p style="color:#666;font-size:0.9em;margin-bottom:10px;">Accuracy excludes parse errors</p>\n'
    html += '<table>\n'
    html += '<tr><th>Sample Category</th><th>Correct</th><th>Parsed</th><th>Parse Errors</th><th>Accuracy</th></tr>\n'

    by_sample = experiment.get('by_sample_category', {})
    # Map internal names to display names
    sample_cat_display = {
        'correct_head_up': 'Easy head_up',
        'correct_head_down': 'Easy head_down',
        'error_headup_to_headdown': 'Hard head_up',
        'error_headdown_to_headup': 'Hard head_down',
    }
    for cat in ['correct_head_up', 'correct_head_down', 'error_headup_to_headdown', 'error_headdown_to_headup']:
        if cat in by_sample:
            data = by_sample[cat]
            parsed = data.get('parsed', data['total'])
            parse_errors = data['total'] - parsed
            acc = data['correct'] / parsed if parsed > 0 else 0
            color = SAMPLE_CATEGORY_COLORS.get(cat, '#888')
            display_name = sample_cat_display.get(cat, cat)
            html += '<tr><td><span class="badge" style="background:{}">{}</span></td>'.format(color, display_name)
            html += '<td>{}</td><td>{}</td><td>{}</td><td>{:.1f}%</td></tr>\n'.format(
                data['correct'], parsed, parse_errors, 100 * acc
            )

    html += '</table>\n</div>\n'

    # Confusion matrix
    html += '<div class="section">\n'
    html += '<h2>Confusion Matrix</h2>\n'
    html += '<p style="color:#666;font-size:0.9em;margin-bottom:10px;">Click any cell to see the images.</p>\n'
    html += build_confusion_matrix_html(confusion, link_prefix='confusion_details/', exp_safe=exp_safe)
    html += '</div>\n'

    # Per-image results
    html += '<div class="section">\n'
    html += '<h2>Per-Image Results</h2>\n'

    # Filter buttons
    html += '<div class="filter-bar">\n'
    n_correct = sum(1 for r in results if r.get('correct', False))
    n_incorrect = len(results) - n_correct
    html += '<button class="filter-btn active" onclick="filterRows(\'all\')">All ({})</button>\n'.format(len(results))
    html += '<button class="filter-btn" onclick="filterRows(\'correct\')">Correct ({})</button>\n'.format(n_correct)
    html += '<button class="filter-btn" onclick="filterRows(\'incorrect\')">Incorrect ({})</button>\n'.format(n_incorrect)

    # Add sample category filters
    sample_cat_display = {
        'correct_head_up': 'Easy head_up',
        'correct_head_down': 'Easy head_down',
        'error_headup_to_headdown': 'Hard head_up',
        'error_headdown_to_headup': 'Hard head_down',
    }
    for cat in ['correct_head_up', 'correct_head_down', 'error_headup_to_headdown', 'error_headdown_to_headup']:
        count = sum(1 for r in results if r.get('sample_category') == cat)
        if count > 0:
            display_name = sample_cat_display.get(cat, cat)
            html += '<button class="filter-btn" onclick="filterRows(\'{}\')">{} ({})</button>\n'.format(
                cat, display_name, count
            )

    html += '</div>\n'

    html += '<div id="image-list">\n'

    for r in results:
        img_path = r.get('image_path', '')
        img_filename = r.get('image_filename', os.path.basename(img_path))
        gt = r.get('ground_truth', 'unknown')
        pred = r.get('prediction', 'parse_error')
        correct = r.get('correct', False)
        sample_cat = r.get('sample_category', '')

        # Thumbnail path
        safe_thumb = img_filename.replace('#', '_').replace(':', '_')
        base, ext = os.path.splitext(safe_thumb)
        if ext.lower() not in ('.jpg', '.jpeg'):
            safe_thumb = base + '.jpg'
        thumb_rel = 'vis_images/{}'.format(safe_thumb)

        card_class = 'image-card' if correct else 'image-card incorrect'

        # Sample category badge
        sample_cat_display = {
            'correct_head_up': 'Easy head_up',
            'correct_head_down': 'Easy head_down',
            'error_headup_to_headdown': 'Hard head_up',
            'error_headdown_to_headup': 'Hard head_down',
        }
        sample_cat_html = ''
        if sample_cat:
            sample_color = SAMPLE_CATEGORY_COLORS.get(sample_cat, '#888')
            display_name = sample_cat_display.get(sample_cat, sample_cat)
            sample_cat_html = '<span class="sample-cat-badge" style="background:{};color:#fff;">{}</span>'.format(
                sample_color, display_name
            )

        result_icon = '&#10003;' if correct else '&#10007;'
        result_color = '#27ae60' if correct else '#e74c3c'

        html += '<div class="{}" data-correct="{}" data-sample-cat="{}">\n'.format(
            card_class, 'true' if correct else 'false', sample_cat
        )
        html += '<img src="{}" alt="{}">\n'.format(thumb_rel, img_filename)
        html += '<div class="info">\n'
        html += '<div><b>Ground truth:</b> <span class="badge" style="background:{}">{}</span>'.format(
            CATEGORY_COLORS.get(gt, '#333'), gt
        )
        html += '<b>Prediction:</b> <span class="badge" style="background:{}">{}</span>'.format(
            CATEGORY_COLORS.get(pred, '#333'), pred
        )
        html += '<span style="color:{};">{}</span>{}</div>\n'.format(result_color, result_icon, sample_cat_html)
        html += '<div class="filename">{}</div>\n'.format(img_filename)
        html += '</div>\n</div>\n'

    html += '</div>\n</div>\n'

    # JavaScript for filters
    html += """
<script>
function filterRows(mode) {
    var rows = document.querySelectorAll('.image-card');
    var buttons = document.querySelectorAll('.filter-btn');
    buttons.forEach(function(btn) { btn.classList.remove('active'); });
    event.target.classList.add('active');
    rows.forEach(function(row) {
        var isCorrect = row.getAttribute('data-correct') === 'true';
        var sampleCat = row.getAttribute('data-sample-cat');
        if (mode === 'all') {
            row.style.display = '';
        } else if (mode === 'correct') {
            row.style.display = isCorrect ? '' : 'none';
        } else if (mode === 'incorrect') {
            row.style.display = isCorrect ? 'none' : '';
        } else {
            row.style.display = (sampleCat === mode) ? '' : 'none';
        }
    });
}
</script>
"""

    html += '</body>\n</html>\n'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


#%% Generate master summary page

def generate_summary_page(experiments, output_dir, vis_images_folder):
    """
    Generate the master summary HTML page with all experiments.

    Args:
        experiments: list of experiment dicts
        output_dir: directory for output HTML
        vis_images_folder: directory for thumbnails
    """

    output_path = os.path.join(output_dir, 'index.html')

    # Sort by accuracy descending
    sorted_exps = sorted(experiments, key=lambda x: x['accuracy'], reverse=True)

    html = '<!DOCTYPE html>\n<html>\n<head>\n'
    html += '<meta charset="UTF-8">\n'
    html += '<title>Experiment Summary</title>\n'
    html += '<style>{}</style>\n'.format(_css_common())
    html += '</head>\n<body>\n'

    # Header
    html += '<div class="header">\n'
    html += '<h1>Prompt / Few-Shot Experiment Summary</h1>\n'
    html += '<div class="stats">\n'
    html += '<div class="stat"><div class="stat-value">{}</div><div class="stat-label">Total Experiments</div></div>\n'.format(
        len(experiments)
    )

    # Count unique models, prompts, few-shots
    models = set(e['model'] for e in experiments)
    prompts = set(e['prompt'] for e in experiments)
    few_shots = set(e['few_shot'] for e in experiments)
    html += '<div class="stat"><div class="stat-value">{}</div><div class="stat-label">Models</div></div>\n'.format(len(models))
    html += '<div class="stat"><div class="stat-value">{}</div><div class="stat-label">Prompts</div></div>\n'.format(len(prompts))
    html += '<div class="stat"><div class="stat-value">{}</div><div class="stat-label">Few-shot Sets</div></div>\n'.format(len(few_shots))

    best_acc = max(e['accuracy'] for e in experiments) if experiments else 0
    html += '<div class="stat"><div class="stat-value">{:.1f}%</div><div class="stat-label">Best Accuracy</div></div>\n'.format(
        100 * best_acc
    )
    html += '</div>\n'
    html += '<p style="margin-top:10px;color:#666;font-size:0.9em;">Accuracy = correct / successfully parsed (excludes parse errors)</p>\n'
    html += '</div>\n'

    # Summary table
    html += '<div class="section">\n'
    html += '<h2>All Experiments (click column headers to sort)</h2>\n'
    html += '<table id="summary-table">\n'
    html += '<tr>'
    html += '<th class="sortable" onclick="sortTable(0)">Model</th>'
    html += '<th class="sortable" onclick="sortTable(1)">Prompt</th>'
    html += '<th class="sortable" onclick="sortTable(2)">Few-shot</th>'
    html += '<th class="sortable" onclick="sortTable(3)">Accuracy</th>'
    html += '<th class="sortable" onclick="sortTable(4)">Parse Errors</th>'
    html += '<th class="sortable" onclick="sortTable(5)">Easy Head-Up</th>'
    html += '<th class="sortable" onclick="sortTable(6)">Easy Head-Down</th>'
    html += '<th class="sortable" onclick="sortTable(7)">Hard Head-Up</th>'
    html += '<th class="sortable" onclick="sortTable(8)">Hard Head-Down</th>'
    html += '</tr>\n'

    for exp in sorted_exps:
        # Create safe filename for link
        exp_safe = '{}_{}_{}' .format(
            sanitize_model_name(exp['model']),
            exp['prompt'],
            exp['few_shot']
        )

        # Accuracy class
        acc = exp['accuracy']
        if acc >= 0.70:
            acc_class = 'accuracy-high'
        elif acc >= 0.55:
            acc_class = 'accuracy-mid'
        else:
            acc_class = 'accuracy-low'

        # Parse error rate class
        pe_rate = exp.get('parse_error_rate', 0)
        if pe_rate <= 0.05:
            pe_class = 'accuracy-high'
        elif pe_rate <= 0.20:
            pe_class = 'accuracy-mid'
        else:
            pe_class = 'accuracy-low'

        # Per-sample-category accuracies (excluding parse errors)
        by_sample = exp.get('by_sample_category', {})

        def get_cat_acc(cat):
            if cat in by_sample:
                data = by_sample[cat]
                parsed = data.get('parsed', data['total'])
                return data['correct'] / parsed if parsed > 0 else 0
            return 0

        html += '<tr>'
        html += '<td style="text-align:left;">{}</td>'.format(exp['model'])
        html += '<td><a href="{}.html">{}</a></td>'.format(exp_safe, exp['prompt'])
        html += '<td>{}</td>'.format(exp['few_shot'])
        html += '<td class="{}">{:.1f}%</td>'.format(acc_class, 100 * acc)
        html += '<td class="{}">{:.1f}% ({}/{})</td>'.format(
            pe_class, 100 * pe_rate, exp.get('parse_errors', 0), exp['total_images']
        )
        # Easy = baseline got it right, Hard = baseline got it wrong
        # "error_headup_to_headdown" means ground truth is head_up but baseline said head_down (hard head_up)
        # "error_headdown_to_headup" means ground truth is head_down but baseline said head_up (hard head_down)
        html += '<td>{:.1f}%</td>'.format(100 * get_cat_acc('correct_head_up'))
        html += '<td>{:.1f}%</td>'.format(100 * get_cat_acc('correct_head_down'))
        html += '<td>{:.1f}%</td>'.format(100 * get_cat_acc('error_headup_to_headdown'))
        html += '<td>{:.1f}%</td>'.format(100 * get_cat_acc('error_headdown_to_headup'))
        html += '</tr>\n'

    html += '</table>\n</div>\n'

    # Best by model
    html += '<div class="section">\n'
    html += '<h2>Best Configuration per Model</h2>\n'
    html += '<p style="color:#666;font-size:0.9em;margin-bottom:10px;">Accuracy = correct / successfully parsed (excludes parse errors)</p>\n'
    html += '<table>\n'
    html += '<tr><th>Model</th><th>Best Prompt</th><th>Best Few-shot</th><th>Accuracy</th><th>Parse Errors</th></tr>\n'

    for model in sorted(models):
        model_exps = [e for e in experiments if e['model'] == model]
        best = max(model_exps, key=lambda x: x['accuracy'])
        acc_class = 'accuracy-high' if best['accuracy'] >= 0.70 else ('accuracy-mid' if best['accuracy'] >= 0.55 else 'accuracy-low')
        pe_rate = best.get('parse_error_rate', 0)
        pe_class = 'accuracy-high' if pe_rate <= 0.05 else ('accuracy-mid' if pe_rate <= 0.20 else 'accuracy-low')
        html += '<tr><td style="text-align:left;">{}</td><td>{}</td><td>{}</td><td class="{}">{:.1f}%</td><td class="{}">{:.1f}%</td></tr>\n'.format(
            model, best['prompt'], best['few_shot'], acc_class, 100 * best['accuracy'],
            pe_class, 100 * pe_rate
        )

    html += '</table>\n</div>\n'

    # JavaScript for sorting
    html += """
<script>
var sortDir = {};
function sortTable(col) {
    var table = document.getElementById('summary-table');
    var rows = Array.from(table.rows).slice(1);
    sortDir[col] = !sortDir[col];
    rows.sort(function(a, b) {
        var aVal = a.cells[col].textContent.replace('%', '');
        var bVal = b.cells[col].textContent.replace('%', '');
        var aNum = parseFloat(aVal);
        var bNum = parseFloat(bVal);
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return sortDir[col] ? aNum - bNum : bNum - aNum;
        }
        return sortDir[col] ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });
    rows.forEach(function(row) { table.appendChild(row); });
}
</script>
"""

    html += '</body>\n</html>\n'

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print('Saved summary page to {}'.format(output_path))


#%% Main function

def main():

    parser = argparse.ArgumentParser(
        description='Generate HTML visualizations for prompt/few-shot experiments.'
    )

    parser.add_argument(
        '--results-dir',
        default=DEFAULT_RESULTS_DIR,
        help='Directory containing experiment result JSON files'
    )
    parser.add_argument(
        '--output-dir',
        default=DEFAULT_OUTPUT_DIR,
        help='Directory for output HTML and thumbnails'
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    vis_images_folder = os.path.join(args.output_dir, 'vis_images')
    os.makedirs(vis_images_folder, exist_ok=True)

    # Load experiments
    print('Loading experiments from {}'.format(args.results_dir))
    experiments = load_experiment_results(args.results_dir)

    if not experiments:
        print('No experiments found.')
        return

    print('Found {} experiments'.format(len(experiments)))

    # Generate summary page
    generate_summary_page(experiments, args.output_dir, vis_images_folder)

    # Generate detail pages for each experiment
    print('Generating detail pages...')
    for exp in experiments:
        exp_safe = '{}_{}_{}'.format(
            sanitize_model_name(exp['model']),
            exp['prompt'],
            exp['few_shot']
        )
        generate_experiment_detail_page(exp, exp_safe, args.output_dir, vis_images_folder)
        print('  Generated: {}.html'.format(exp_safe))

    print('\nDone! Open {} to view results.'.format(
        os.path.join(args.output_dir, 'index.html')
    ))


#%% Command-line entry point

if __name__ == '__main__':
    main()
