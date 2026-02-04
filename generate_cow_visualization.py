#%% Header

"""
Generate HTML visualizations for cattle behavior classification results.

Produces interactive HTML dashboards for single-model evaluation and
multi-model comparison.  Thumbnails are resized and stored in a shared
vis_images/ folder so they can be reused across multiple HTML files.

Usage:
    # Single model visualization
    python generate_cow_visualization.py path/to/results.json

    # Multi-model comparison
    python generate_cow_visualization.py results/ --compare

    # Sample a subset of images
    python generate_cow_visualization.py path/to/results.json --sample 100
"""


#%% Imports and constants

import os
import io
import json
import random
import argparse
import glob

from PIL import Image

from cow_vlm_utils import VALID_CATEGORIES, OUTPUT_BASE_DIR, sanitize_model_name
from evaluate_results import compute_metrics


THUMBNAIL_MAX_SIZE = 200


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

# ...def copy_and_resize_image()


#%% HTML building blocks

CATEGORY_COLORS = {
    'head_up': '#4a90d9',
    'head_down': '#50b050',
    'running': '#e67e22',
    'unknown': '#999999',
    'parse_error': '#e74c3c',
}


def _css_common():
    """Return common CSS shared by single-model and comparison pages."""

    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, sans-serif;
            max-width: 1400px; margin: 0 auto; padding: 20px;
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
        th, td { padding: 6px 10px; border: 1px solid #ddd; text-align: center; }
        th { background: #f0f4f8; font-weight: 600; }
        .section { background: #fff; padding: 20px; border-radius: 8px;
                   margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .filter-bar { margin: 12px 0; }
        .filter-btn {
            padding: 6px 14px; margin-right: 6px; border: 1px solid #ccc;
            border-radius: 4px; background: #fff; cursor: pointer;
            font-size: 0.9em;
        }
        .filter-btn.active { background: #4a90d9; color: #fff; border-color: #4a90d9; }
        .image-row {
            display: flex; align-items: center; gap: 16px;
            padding: 10px; margin-bottom: 8px; border-radius: 6px;
            background: #fff; border: 1px solid #eee;
        }
        .image-row.incorrect { background: #fde8e8; border-color: #f5c6c6; }
        .image-row img { width: 200px; height: auto; border-radius: 4px; }
        .image-row .info { flex: 1; }
        .image-row .filename { font-size: 0.8em; color: #888;
                               word-break: break-all; margin-top: 4px; }
        .badge {
            display: inline-block; padding: 2px 10px; border-radius: 12px;
            color: #fff; font-size: 0.85em; font-weight: 600; margin-right: 6px;
        }
        .result-icon { font-size: 1.1em; margin-left: 6px; }
        .cm-cell-nonzero { font-weight: bold; }
        .cm-cell-diag { background: #e8f5e9; }
        .cm-cell-offdiag-nonzero { background: #fde8e8; }
        td a { color: inherit; text-decoration: underline; }
        td a:hover { color: #4a90d9; }
        .few-shot-gallery {
            display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px;
        }
        .few-shot-card {
            text-align: center;
        }
        .few-shot-card img {
            width: 150px; height: auto; border-radius: 4px;
            border: 2px solid #ddd;
        }
        .few-shot-label {
            font-size: 0.7em; color: #888; margin-top: 2px;
            max-width: 150px; overflow: hidden; text-overflow: ellipsis;
            white-space: nowrap;
        }
        @media (max-width: 768px) {
            .image-row { flex-direction: column; align-items: flex-start; }
            .image-row img { width: 100%; max-width: 300px; }
        }
    """

# ...def _css_common()


def _build_confusion_matrix_html(confusion, categories=None, link_prefix=None,
                                  model_safe=None):
    """
    Build an HTML <table> for a confusion matrix.

    Args:
        confusion: dict of dicts  (true_label -> predicted_label -> count)
        categories: ordered list of labels to include
        link_prefix: if provided, cell counts become links to detail pages
                     (e.g., 'confusion_details/' for relative links)
        model_safe: sanitized model name for link filenames

    Returns:
        HTML string
    """

    if categories is None:
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

            # Make count a link if link_prefix provided and not parse_error
            if (link_prefix and model_safe and count > 0
                    and true_cat != 'parse_error' and pred_cat != 'parse_error'):
                link_url = '{}{}_{}_{}.html'.format(
                    link_prefix, model_safe, true_cat, pred_cat
                )
                cell_content = '<a href="{}">{}</a>'.format(link_url, count)
            else:
                cell_content = str(count)

            lines.append('<td{}>{}</td>'.format(cls, cell_content))
        lines.append('<td><b>{}</b></td>'.format(row_total))
        lines.append('</tr>')

    lines.append('</table>')
    return '\n'.join(lines)

# ...def _build_confusion_matrix_html()


def _build_metrics_table_html(per_class):
    """
    Build an HTML <table> for per-class precision / recall / F1.

    Args:
        per_class: dict from compute_metrics()['per_class']

    Returns:
        HTML string
    """

    lines = ['<table>']
    lines.append(
        '<tr><th>Category</th><th>Precision</th><th>Recall</th>'
        '<th>F1</th><th>Support</th></tr>'
    )
    for cat in VALID_CATEGORIES:
        m = per_class.get(cat, {})
        color = CATEGORY_COLORS.get(cat, '#333')
        lines.append(
            '<tr>'
            '<td style="text-align:left;">'
            '<span class="badge" style="background:{color}">{cat}</span>'
            '</td>'
            '<td>{prec:.1f}%</td>'
            '<td>{rec:.1f}%</td>'
            '<td>{f1:.3f}</td>'
            '<td>{sup}</td>'
            '</tr>'.format(
                color=color,
                cat=cat,
                prec=100.0 * m.get('precision', 0),
                rec=100.0 * m.get('recall', 0),
                f1=m.get('f1', 0),
                sup=m.get('support', 0),
            )
        )
    lines.append('</table>')
    return '\n'.join(lines)

# ...def _build_metrics_table_html()


def _css_confusion_detail():
    """Return CSS for confusion detail pages."""

    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         Roboto, sans-serif;
            max-width: 1000px; margin: 0 auto; padding: 20px;
            background: #f5f5f5; color: #333;
        }
        h1 { font-size: 1.3em; margin-bottom: 8px; }
        .header { background: #fff; padding: 20px; border-radius: 8px;
                   margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .back-link { margin-bottom: 12px; }
        .back-link a { color: #4a90d9; text-decoration: none; }
        .back-link a:hover { text-decoration: underline; }
        .badge {
            display: inline-block; padding: 2px 10px; border-radius: 12px;
            color: #fff; font-size: 0.85em; font-weight: 600; margin-right: 6px;
        }
        .image-card {
            background: #fff; padding: 12px; margin-bottom: 12px;
            border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .image-card img { max-width: 100%; height: auto; border-radius: 4px; }
        .image-card .filename {
            font-size: 0.8em; color: #888; margin-top: 6px;
            word-break: break-all;
        }
    """


def generate_confusion_detail_pages(results, model_name, model_safe,
                                     confusion_dir, vis_images_folder):
    """
    Generate HTML pages for each confusion matrix cell.

    Creates one page per (true_label, predicted_label) pair showing all
    images that fall into that cell.

    Args:
        results: list of result dicts from JSON
        model_name: display name of the model
        model_safe: sanitized model name for filenames
        confusion_dir: directory to write the detail HTML pages
        vis_images_folder: directory containing shared thumbnails

    Returns:
        None (writes files to confusion_dir)
    """

    os.makedirs(confusion_dir, exist_ok=True)

    # Group images by (true_label, predicted_label)
    cells = {}
    for r in results:
        gt = r.get('ground_truth', 'unknown')
        pred = r.get('prediction', 'parse_error')

        # Skip parse errors
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

        filename = '{}_{}_{}.html'.format(model_safe, true_cat, pred_cat)
        output_path = os.path.join(confusion_dir, filename)

        is_correct = (true_cat == pred_cat)
        title = '{}: {} → {}'.format(model_name, true_cat, pred_cat)
        if is_correct:
            description = 'Correctly classified as {}'.format(true_cat)
        else:
            description = 'Misclassified: true={}, predicted={}'.format(
                true_cat, pred_cat
            )

        html = '<!DOCTYPE html>\n<html>\n<head>\n'
        html += '<meta charset="UTF-8">\n'
        html += '<title>{}</title>\n'.format(title)
        html += '<style>{}</style>\n'.format(_css_confusion_detail())
        html += '</head>\n<body>\n'

        # Header
        html += '<div class="header">\n'
        html += '<div class="back-link"><a href="../{}.html">← Back to {}</a></div>\n'.format(
            model_safe, model_name
        )
        html += '<h1>{}</h1>\n'.format(title)
        html += '<p>{} ({} images)</p>\n'.format(description, len(cell_results))

        true_color = CATEGORY_COLORS.get(true_cat, '#333')
        pred_color = CATEGORY_COLORS.get(pred_cat, '#333')
        html += '<p style="margin-top:8px;">'
        html += 'Ground truth: <span class="badge" style="background:{}">'.format(true_color)
        html += '{}</span> '.format(true_cat)
        html += 'Prediction: <span class="badge" style="background:{}">'.format(pred_color)
        html += '{}</span></p>\n'.format(pred_cat)
        html += '</div>\n'

        # Image list
        for r in cell_results:
            img_path = r.get('image_path', '')
            img_filename = r.get('image_filename', os.path.basename(img_path))

            # Get thumbnail path (relative to confusion_dir, so ../vis_images/)
            safe_thumb = img_filename.replace('#', '_').replace(':', '_')
            base, ext = os.path.splitext(safe_thumb)
            if ext.lower() not in ('.jpg', '.jpeg'):
                safe_thumb = base + '.jpg'

            thumb_rel = '../vis_images/{}'.format(safe_thumb)

            html += '<div class="image-card">\n'
            html += '<img src="{}" alt="{}">\n'.format(thumb_rel, img_filename)
            html += '<div class="filename">{}</div>\n'.format(img_filename)
            html += '</div>\n'

        html += '</body>\n</html>\n'

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

    n_pages = len(cells)
    if n_pages > 0:
        print('  Generated {} confusion detail pages for {}'.format(
            n_pages, model_name
        ))


# ...def generate_confusion_detail_pages()


def _load_prompt_from_file(prompt_id, experiment_dir=None):
    """
    Load a prompt from the prompt_variations folder.

    Args:
        prompt_id: the prompt variation ID (e.g., 'v1_baseline')
        experiment_dir: base experiment directory (defaults to standard location)

    Returns:
        dict with 'system_prompt', 'name', 'id', or None if not found
    """

    if not prompt_id:
        return None

    if experiment_dir is None:
        experiment_dir = os.path.join(OUTPUT_BASE_DIR, 'experiments')

    prompt_file = os.path.join(
        experiment_dir, 'prompt_variations', '{}.json'.format(prompt_id)
    )

    if not os.path.exists(prompt_file):
        return None

    try:
        with open(prompt_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _load_few_shot_from_file(few_shot_id, experiment_dir=None):
    """
    Load few-shot examples from the few_shot_variations folder.

    Args:
        few_shot_id: the few-shot variation ID (e.g., 'v1_baseline')
        experiment_dir: base experiment directory (defaults to standard location)

    Returns:
        list of example dicts with 'path', 'filename', 'category', or empty list if not found
    """

    if not few_shot_id:
        return []

    if experiment_dir is None:
        experiment_dir = os.path.join(OUTPUT_BASE_DIR, 'experiments')

    few_shot_file = os.path.join(
        experiment_dir, 'few_shot_variations', '{}.json'.format(few_shot_id)
    )

    if not os.path.exists(few_shot_file):
        return []

    try:
        with open(few_shot_file, 'r') as f:
            data = json.load(f)
        return data.get('examples', [])
    except Exception:
        return []


def _build_prompt_html(system_prompt, prompt_id=None, prompt_name=None):
    """
    Build an HTML section showing the system prompt text.

    Args:
        system_prompt: the system prompt text
        prompt_id: optional prompt variation ID
        prompt_name: optional prompt variation name

    Returns:
        HTML string, or empty string if no prompt provided
    """

    if not system_prompt:
        return ''

    html = '<div class="section">\n'
    html += '<h2>System Prompt'
    if prompt_name:
        html += ' <span style="color:#666;font-weight:normal;font-size:0.85em;">'
        html += '({}'.format(prompt_name)
        if prompt_id:
            html += ' / {}'.format(prompt_id)
        html += ')</span>'
    html += '</h2>\n'
    html += '<pre style="background:#f8f9fa;padding:16px;border-radius:6px;'
    html += 'font-family:monospace;font-size:0.9em;white-space:pre-wrap;'
    html += 'word-wrap:break-word;line-height:1.5;border:1px solid #e9ecef;">'
    # Escape HTML entities in the prompt text
    escaped_prompt = system_prompt.replace('&', '&amp;')
    escaped_prompt = escaped_prompt.replace('<', '&lt;')
    escaped_prompt = escaped_prompt.replace('>', '&gt;')
    html += escaped_prompt
    html += '</pre>\n'
    html += '</div>\n'
    return html


def _build_few_shot_html(few_shot_examples, vis_images_folder):
    """
    Build an HTML section showing few-shot example images grouped by category.

    Args:
        few_shot_examples: list of dicts with 'path', 'filename', 'category'
        vis_images_folder: directory for shared thumbnails

    Returns:
        HTML string, or empty string if no examples provided
    """

    if not few_shot_examples:
        return ''

    # Group by category
    by_category = {}
    for ex in few_shot_examples:
        cat = ex.get('category', 'unknown')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(ex)

    html = '<div class="section">\n'
    html += '<h2>Few-Shot Examples</h2>\n'
    html += '<p style="color:#666;margin-bottom:12px;">These example images '
    html += 'were included in every prompt to guide the model.</p>\n'

    for cat in VALID_CATEGORIES:
        examples = by_category.get(cat, [])
        if not examples:
            continue

        color = CATEGORY_COLORS.get(cat, '#333')
        html += '<div style="margin-bottom:16px;">\n'
        html += '<h3><span class="badge" style="background:{}">{}</span>'
        html += ' ({} examples)</h3>\n'
        html = html.format(color, cat, len(examples))
        html += '<div class="few-shot-gallery">\n'

        for ex in examples:
            img_path = ex.get('path', '')
            filename = ex.get('filename', os.path.basename(img_path))

            if os.path.exists(img_path):
                thumb_rel = copy_and_resize_image(
                    img_path, vis_images_folder
                )
            else:
                thumb_rel = ''

            if thumb_rel:
                html += (
                    '<div class="few-shot-card">'
                    '<img src="{}" alt="{}">'
                    '<div class="few-shot-label">{}</div>'
                    '</div>\n'
                ).format(thumb_rel, filename, filename[:30])
            else:
                html += (
                    '<div class="few-shot-card">'
                    '<div style="width:150px;height:112px;background:#eee;'
                    'display:flex;align-items:center;justify-content:center;'
                    'border-radius:4px;color:#999;">Not found</div>'
                    '<div class="few-shot-label">{}</div>'
                    '</div>\n'
                ).format(filename[:30])

        html += '</div>\n</div>\n'

    html += '</div>\n'
    return html

# ...def _build_few_shot_html()


#%% Single-model HTML generation

def generate_single_model_html(json_path, output_html_path,
                                vis_images_folder, sample=None,
                                random_seed=42):
    """
    Generate a single-model evaluation HTML page.

    The page includes: summary stats, confusion matrix, per-class metrics,
    filter buttons (all / correct / incorrect), and a per-image list with
    thumbnails.

    Args:
        json_path: path to the results JSON file
        output_html_path: path for the output HTML file
        vis_images_folder: directory for shared thumbnails
        sample: if not None, randomly sample this many images for display
        random_seed: seed for reproducible sampling
    """

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Model name can be in run_info (regular runs) or experiment_info (experiments)
    model_name = (
        data.get('run_info', {}).get('model') or
        data.get('experiment_info', {}).get('model') or
        'unknown'
    )
    results = data.get('results', [])

    if not results:
        print('No results in {}'.format(json_path))
        return

    metrics = compute_metrics(results)

    # Sampling
    display_results = list(results)
    sample_note = ''
    if sample is not None and sample < len(display_results):
        rng = random.Random(random_seed)
        display_results = rng.sample(display_results, sample)
        sample_note = ' (showing {} of {} images)'.format(
            sample, len(results)
        )

    # Create thumbnail directory
    os.makedirs(vis_images_folder, exist_ok=True)

    # Copy ALL images to vis_images (needed for confusion detail pages)
    # This happens before sampling so detail pages can link to any image
    for r in results:
        img_path = r.get('image_path', '')
        if img_path and os.path.exists(img_path):
            copy_and_resize_image(img_path, vis_images_folder)

    # Generate confusion detail pages
    model_safe = sanitize_model_name(model_name)
    output_dir = os.path.dirname(output_html_path)
    confusion_dir = os.path.join(output_dir, 'confusion_details')
    generate_confusion_detail_pages(
        results, model_name, model_safe, confusion_dir, vis_images_folder
    )

    # Generate thumbnails and build image rows (for sampled display)
    image_rows_html = []
    for r in display_results:
        img_path = r.get('image_path', '')
        filename = r.get('image_filename', os.path.basename(img_path))
        gt = r.get('ground_truth', 'unknown')
        pred = r.get('prediction', 'parse_error')
        correct = r.get('correct', False)
        success = r.get('success', False)

        # Create thumbnail
        if os.path.exists(img_path):
            thumb_rel = copy_and_resize_image(img_path, vis_images_folder)
        else:
            thumb_rel = ''

        row_class = 'image-row' + ('' if correct else ' incorrect')
        result_text = 'Correct' if correct else 'Incorrect'
        result_icon = '&#10003;' if correct else '&#10007;'
        result_color = '#27ae60' if correct else '#e74c3c'

        gt_color = CATEGORY_COLORS.get(gt, '#333')
        pred_color = CATEGORY_COLORS.get(pred, '#333')

        img_tag = ''
        if thumb_rel:
            img_tag = '<img src="{}" alt="{}">'.format(thumb_rel, filename)
        else:
            img_tag = (
                '<div style="width:200px;height:150px;background:#eee;'
                'display:flex;align-items:center;justify-content:center;'
                'border-radius:4px;color:#999;">Image not found</div>'
            )

        row_html = (
            '<div class="{row_class}" data-correct="{correct_str}">'
            '{img_tag}'
            '<div class="info">'
            '<div>'
            'Ground truth: <span class="badge" style="background:{gt_color}">'
            '{gt}</span>'
            'Prediction: <span class="badge" style="background:{pred_color}">'
            '{pred}</span>'
            '<span class="result-icon" style="color:{result_color}">'
            '{result_icon} {result_text}</span>'
            '</div>'
            '<div class="filename">{filename}</div>'
            '</div>'
            '</div>'
        ).format(
            row_class=row_class,
            correct_str='true' if correct else 'false',
            img_tag=img_tag,
            gt_color=gt_color,
            gt=gt,
            pred_color=pred_color,
            pred=pred,
            result_color=result_color,
            result_icon=result_icon,
            result_text=result_text,
            filename=filename,
        )
        image_rows_html.append(row_html)

    # Build full HTML
    html = '<!DOCTYPE html>\n<html>\n<head>\n'
    html += '<meta charset="UTF-8">\n'
    html += '<title>Results: {}</title>\n'.format(model_name)
    html += '<style>{}</style>\n'.format(_css_common())
    html += '</head>\n<body>\n'

    # Header
    html += '<div class="header">\n'
    html += '<h1>Cattle Behavior Classification: {}</h1>\n'.format(model_name)
    html += '<div class="stats">\n'
    html += (
        '<div class="stat"><div class="stat-value">{}</div>'
        '<div class="stat-label">Total images</div></div>\n'
    ).format(metrics['total_images'])
    html += (
        '<div class="stat"><div class="stat-value">{:.1f}%</div>'
        '<div class="stat-label">Accuracy</div></div>\n'
    ).format(100.0 * metrics['overall_accuracy'])
    html += (
        '<div class="stat"><div class="stat-value">{}</div>'
        '<div class="stat-label">Correct</div></div>\n'
    ).format(metrics['correct_predictions'])
    html += (
        '<div class="stat"><div class="stat-value">{}</div>'
        '<div class="stat-label">Failed</div></div>\n'
    ).format(metrics['failed_predictions'])
    html += '</div>\n</div>\n'

    # System prompt (if available)
    run_info = data.get('run_info', {})
    experiment_info = data.get('experiment_info', {})
    system_prompt = run_info.get('system_prompt') or experiment_info.get('system_prompt')
    prompt_id = experiment_info.get('prompt_id')
    prompt_name = experiment_info.get('prompt_name')

    # Try to extract prompt_id from run_info.prompt_file path
    if not prompt_id:
        prompt_file = run_info.get('prompt_file', '')
        if prompt_file:
            # Extract ID from path like ".../prompt_variations/v5_negative_guidance.json"
            import re
            match = re.search(r'prompt_variations[/\\]([^/\\]+)\.json$', prompt_file)
            if match:
                prompt_id = match.group(1)

    # If prompt not embedded in results, try to load from prompt_variations file
    if not system_prompt and prompt_id:
        prompt_data = _load_prompt_from_file(prompt_id)
        if prompt_data:
            system_prompt = prompt_data.get('system_prompt')
            if not prompt_name:
                prompt_name = prompt_data.get('name')

    # For regular runs without experiment prompt, use the baseline prompt
    if not system_prompt:
        prompt_data = _load_prompt_from_file('v1_baseline')
        if prompt_data:
            system_prompt = prompt_data.get('system_prompt')
            prompt_name = prompt_data.get('name')
            prompt_id = 'v1_baseline'

    html += _build_prompt_html(system_prompt, prompt_id, prompt_name)

    # Few-shot examples
    few_shot_examples = data.get('few_shot_examples', [])
    few_shot_id = experiment_info.get('few_shot_id')

    # Try to extract few_shot_id from run_info.few_shot_file path
    if not few_shot_id:
        few_shot_file = run_info.get('few_shot_file', '')
        if few_shot_file:
            # Extract ID from path like ".../few_shot_variations/v5_different_seed.json"
            import re
            match = re.search(r'few_shot_variations[/\\]([^/\\]+)\.json$', few_shot_file)
            if match:
                few_shot_id = match.group(1)

    # If few-shot examples not embedded in results, try to load from file
    if not few_shot_examples and few_shot_id:
        few_shot_examples = _load_few_shot_from_file(few_shot_id)

    html += _build_few_shot_html(few_shot_examples, vis_images_folder)

    # Confusion matrix (with links to detail pages)
    html += '<div class="section">\n'
    html += '<h2>Confusion Matrix</h2>\n'
    html += '<p style="color:#666;font-size:0.9em;margin-bottom:10px;">'
    html += 'Click any cell count to see all images in that category.</p>\n'
    html += _build_confusion_matrix_html(
        metrics['confusion_matrix'],
        link_prefix='confusion_details/',
        model_safe=model_safe
    )
    html += '</div>\n'

    # Per-class metrics
    html += '<div class="section">\n'
    html += '<h2>Per-Class Metrics</h2>\n'
    html += _build_metrics_table_html(metrics['per_class'])
    html += '</div>\n'

    # Per-image results with filter
    html += '<div class="section">\n'
    html += '<h2>Per-Image Results{}</h2>\n'.format(sample_note)
    html += '<div class="filter-bar">\n'
    html += (
        '<button class="filter-btn active" onclick="filterRows(\'all\')">All'
        ' ({})</button>\n'
    ).format(len(display_results))
    n_correct = sum(1 for r in display_results if r.get('correct', False))
    n_incorrect = len(display_results) - n_correct
    html += (
        '<button class="filter-btn" onclick="filterRows(\'correct\')">Correct'
        ' ({})</button>\n'
    ).format(n_correct)
    html += (
        '<button class="filter-btn" onclick="filterRows(\'incorrect\')">'
        'Incorrect ({})</button>\n'
    ).format(n_incorrect)
    html += '</div>\n'

    html += '<div id="image-list">\n'
    html += '\n'.join(image_rows_html)
    html += '\n</div>\n'
    html += '</div>\n'

    # JavaScript for filter buttons
    html += """
<script>
function filterRows(mode) {
    var rows = document.querySelectorAll('.image-row');
    var buttons = document.querySelectorAll('.filter-btn');
    buttons.forEach(function(btn) { btn.classList.remove('active'); });
    event.target.classList.add('active');
    rows.forEach(function(row) {
        var isCorrect = row.getAttribute('data-correct') === 'true';
        if (mode === 'all') {
            row.style.display = '';
        } else if (mode === 'correct') {
            row.style.display = isCorrect ? '' : 'none';
        } else {
            row.style.display = isCorrect ? 'none' : '';
        }
    });
}
</script>
"""

    html += '</body>\n</html>\n'

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print('Saved single-model visualization to {}'.format(output_html_path))

# ...def generate_single_model_html()


#%% Multi-model comparison HTML generation

def generate_comparison_html(results_dir, output_html_path,
                              vis_images_folder, sample=None,
                              random_seed=42):
    """
    Generate a multi-model comparison HTML page.

    Loads all result JSON files from a directory, computes metrics, and
    produces a comparison dashboard with: summary accuracy table,
    per-model confusion matrices, and a per-image comparison table
    showing all models' predictions with disagreements highlighted.

    Args:
        results_dir: directory containing result JSON files
        output_html_path: path for the output HTML file
        vis_images_folder: directory for shared thumbnails
        sample: if not None, randomly sample this many images for display
        random_seed: seed for reproducible sampling
    """

    # Find result files
    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    result_files = [
        f for f in json_files
        if not os.path.basename(f).startswith('gemini_batch_metadata')
        and not f.endswith('.tmp.json')
    ]

    if not result_files:
        print('No result files found in {}'.format(results_dir))
        return

    # Load all models
    models = []
    for filepath in sorted(result_files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            model_name = data.get('run_info', {}).get(
                'model', os.path.basename(filepath)
            )
            results = data.get('results', [])
            metrics = compute_metrics(results)
            models.append({
                'name': model_name,
                'results': results,
                'metrics': metrics,
                'filepath': filepath,
                'few_shot_examples': data.get('few_shot_examples', []),
            })
        except (json.JSONDecodeError, KeyError) as e:
            print('Warning: could not load {}: {}'.format(filepath, e))

    if not models:
        print('No valid result files found.')
        return

    print('Loaded {} models for comparison'.format(len(models)))

    # Build per-image lookup:  filename -> { model_name: result_dict }
    all_filenames = set()
    model_by_filename = {}
    for model in models:
        model_by_filename[model['name']] = {}
        for r in model['results']:
            fn = r.get('image_filename', '')
            all_filenames.add(fn)
            model_by_filename[model['name']][fn] = r

    sorted_filenames = sorted(all_filenames)

    # Sampling
    sample_note = ''
    if sample is not None and sample < len(sorted_filenames):
        rng = random.Random(random_seed)
        sorted_filenames = rng.sample(sorted_filenames, sample)
        sorted_filenames.sort()
        sample_note = ' (showing {} of {} images)'.format(
            sample, len(all_filenames)
        )

    os.makedirs(vis_images_folder, exist_ok=True)

    # Copy ALL images from all models to vis_images (for confusion detail pages)
    output_dir = os.path.dirname(output_html_path)
    confusion_dir = os.path.join(output_dir, 'confusion_details')

    for model in models:
        model_safe = sanitize_model_name(model['name'])

        # Copy images
        for r in model['results']:
            img_path = r.get('image_path', '')
            if img_path and os.path.exists(img_path):
                copy_and_resize_image(img_path, vis_images_folder)

        # Generate confusion detail pages for this model
        generate_confusion_detail_pages(
            model['results'], model['name'], model_safe,
            confusion_dir, vis_images_folder
        )

    # ---- Build HTML ----
    html = '<!DOCTYPE html>\n<html>\n<head>\n'
    html += '<meta charset="UTF-8">\n'
    html += '<title>Model Comparison</title>\n'
    html += '<style>\n{}\n'.format(_css_common())
    # Extra styles for comparison page
    html += """
        .comparison-table { font-size: 0.85em; }
        .comparison-table td, .comparison-table th { padding: 4px 8px; }
        .comparison-table img { width: 120px; height: auto; border-radius: 3px; }
        .disagree { background: #fff3cd; }
        .model-section { margin-bottom: 30px; }
    """
    html += '</style>\n</head>\n<body>\n'

    # ---- Header ----
    html += '<div class="header">\n'
    html += '<h1>Cattle Behavior Classification: Model Comparison</h1>\n'
    html += '<div class="stats">\n'
    html += (
        '<div class="stat"><div class="stat-value">{}</div>'
        '<div class="stat-label">Models</div></div>\n'
    ).format(len(models))
    html += (
        '<div class="stat"><div class="stat-value">{}</div>'
        '<div class="stat-label">Total unique images</div></div>\n'
    ).format(len(all_filenames))
    html += '</div>\n</div>\n'

    # ---- Summary accuracy table ----
    html += '<div class="section">\n'
    html += '<h2>Model Summary</h2>\n'
    html += '<table>\n'
    html += '<tr><th>Model</th><th>Accuracy</th><th>Images</th>'
    for cat in VALID_CATEGORIES:
        html += '<th>{} recall</th>'.format(cat)
    html += '</tr>\n'

    for model in models:
        m = model['metrics']
        model_safe = sanitize_model_name(model['name'])
        html += '<tr><td style="text-align:left;font-weight:600;">'
        html += '<a href="{}.html">{}</a></td>'.format(model_safe, model['name'])
        html += '<td>{:.1f}%</td>'.format(100.0 * m['overall_accuracy'])
        html += '<td>{}</td>'.format(m['total_images'])
        for cat in VALID_CATEGORIES:
            recall = m['per_class'].get(cat, {}).get('recall', 0)
            html += '<td>{:.1f}%</td>'.format(100.0 * recall)
        html += '</tr>\n'

    html += '</table>\n</div>\n'

    # ---- Per-model confusion matrices ----
    html += '<div class="section">\n'
    html += '<h2>Confusion Matrices</h2>\n'
    html += '<p style="color:#666;font-size:0.9em;margin-bottom:10px;">'
    html += 'Click any cell count to see all images in that category.</p>\n'
    for model in models:
        model_safe = sanitize_model_name(model['name'])
        html += '<div class="model-section">\n'
        html += '<h3>{}</h3>\n'.format(model['name'])
        html += _build_confusion_matrix_html(
            model['metrics']['confusion_matrix'],
            link_prefix='confusion_details/',
            model_safe=model_safe
        )
        html += '</div>\n'
    html += '</div>\n'

    # ---- Per-image comparison table ----
    html += '<div class="section">\n'
    html += '<h2>Per-Image Comparison{}</h2>\n'.format(sample_note)

    # Filter bar
    html += '<div class="filter-bar">\n'
    html += (
        '<button class="filter-btn active" '
        'onclick="filterCompare(\'all\')">All ({})</button>\n'
    ).format(len(sorted_filenames))

    # Count disagreements (where models don't all agree)
    n_disagree = 0
    for fn in sorted_filenames:
        preds = set()
        for model in models:
            r = model_by_filename[model['name']].get(fn)
            if r:
                preds.add(r.get('prediction', 'parse_error'))
        if len(preds) > 1:
            n_disagree += 1

    html += (
        '<button class="filter-btn" '
        'onclick="filterCompare(\'disagree\')">Disagreements ({})</button>\n'
    ).format(n_disagree)
    html += (
        '<button class="filter-btn" '
        'onclick="filterCompare(\'agree\')">Agreements ({})</button>\n'
    ).format(len(sorted_filenames) - n_disagree)
    html += '</div>\n'

    html += '<table class="comparison-table" id="compare-table">\n'
    html += '<tr><th>Image</th><th>Ground Truth</th>'
    for model in models:
        html += '<th>{}</th>'.format(model['name'])
    html += '</tr>\n'

    for fn in sorted_filenames:
        # Determine ground truth (from first model that has this image)
        gt = ''
        img_path = ''
        preds = {}
        for model in models:
            r = model_by_filename[model['name']].get(fn)
            if r:
                if not gt:
                    gt = r.get('ground_truth', '')
                    img_path = r.get('image_path', '')
                preds[model['name']] = r.get('prediction', 'parse_error')

        pred_values = set(preds.values())
        has_disagreement = len(pred_values) > 1

        # Generate thumbnail
        thumb_html = ''
        if img_path and os.path.exists(img_path):
            thumb_rel = copy_and_resize_image(img_path, vis_images_folder)
            if thumb_rel:
                thumb_html = '<img src="{}">'.format(thumb_rel)

        if not thumb_html:
            thumb_html = '<span style="color:#999;font-size:0.8em;">{}</span>'.format(
                fn[:40]
            )

        disagree_attr = 'true' if has_disagreement else 'false'
        html += '<tr data-disagree="{}">\n'.format(disagree_attr)
        html += '<td>{}</td>\n'.format(thumb_html)
        gt_color = CATEGORY_COLORS.get(gt, '#333')
        html += '<td><span class="badge" style="background:{}">{}</span></td>\n'.format(
            gt_color, gt
        )

        for model in models:
            pred = preds.get(model['name'], '')
            if not pred:
                html += '<td>-</td>\n'
                continue
            pred_color = CATEGORY_COLORS.get(pred, '#333')
            is_correct = (pred == gt)
            cell_class = ''
            if has_disagreement:
                cell_class = ' class="disagree"'
            elif not is_correct:
                cell_class = ' class="cm-cell-offdiag-nonzero"'

            icon = '&#10003;' if is_correct else '&#10007;'
            icon_color = '#27ae60' if is_correct else '#e74c3c'
            html += (
                '<td{}><span class="badge" style="background:{}">{}</span>'
                '<span style="color:{}">{}</span></td>\n'
            ).format(cell_class, pred_color, pred, icon_color, icon)

        html += '</tr>\n'

    html += '</table>\n</div>\n'

    # JavaScript for comparison filter
    html += """
<script>
function filterCompare(mode) {
    var rows = document.querySelectorAll('#compare-table tr[data-disagree]');
    var buttons = document.querySelectorAll('.filter-btn');
    buttons.forEach(function(btn) { btn.classList.remove('active'); });
    event.target.classList.add('active');
    rows.forEach(function(row) {
        var isDisagree = row.getAttribute('data-disagree') === 'true';
        if (mode === 'all') {
            row.style.display = '';
        } else if (mode === 'disagree') {
            row.style.display = isDisagree ? '' : 'none';
        } else {
            row.style.display = isDisagree ? 'none' : '';
        }
    });
}
</script>
"""

    html += '</body>\n</html>\n'

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print('Saved comparison visualization to {}'.format(output_html_path))

# ...def generate_comparison_html()


#%% Main function

def main():

    parser = argparse.ArgumentParser(
        description='Generate HTML visualizations for cattle behavior '
                    'classification results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python generate_cow_visualization.py path/to/results.json
  python generate_cow_visualization.py results/ --compare
  python generate_cow_visualization.py path/to/results.json --sample 100
"""
    )

    parser.add_argument(
        'input',
        help='Single results JSON file or directory of result files'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Generate multi-model comparison visualization'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory for output HTML and thumbnails '
             '(default: visualizations/ under results dir)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Randomly sample N images for display (default: show all)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif os.path.isdir(args.input):
        output_dir = os.path.join(
            os.path.dirname(args.input.rstrip(os.sep)),
            'visualizations'
        )
    else:
        output_dir = os.path.join(
            os.path.dirname(args.input),
            'visualizations'
        )

    os.makedirs(output_dir, exist_ok=True)
    vis_images_folder = os.path.join(output_dir, 'vis_images')

    if args.compare or os.path.isdir(args.input):
        # Multi-model comparison
        results_dir = args.input
        if not os.path.isdir(results_dir):
            print('Error: {} is not a directory'.format(results_dir))
            return

        output_html = os.path.join(output_dir, 'comparison.html')
        generate_comparison_html(
            results_dir, output_html, vis_images_folder,
            sample=args.sample, random_seed=args.random_seed
        )

        # Also generate individual model pages
        json_files = glob.glob(os.path.join(results_dir, '*.json'))
        result_files = [
            f for f in json_files
            if not os.path.basename(f).startswith('gemini_batch_metadata')
            and not f.endswith('.tmp.json')
        ]

        for filepath in sorted(result_files):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                model_name = (
                    data.get('run_info', {}).get('model') or
                    data.get('experiment_info', {}).get('model') or
                    os.path.splitext(os.path.basename(filepath))[0]
                )
                safe_name = sanitize_model_name(model_name)
                model_html = os.path.join(
                    output_dir, '{}.html'.format(safe_name)
                )
                generate_single_model_html(
                    filepath, model_html, vis_images_folder,
                    sample=args.sample, random_seed=args.random_seed
                )
            except (json.JSONDecodeError, KeyError) as e:
                print('Warning: could not process {}: {}'.format(filepath, e))

    else:
        # Single-model visualization
        if not os.path.isfile(args.input):
            print('Error: {} is not a file'.format(args.input))
            return

        with open(args.input, 'r') as f:
            data = json.load(f)
        model_name = (
            data.get('run_info', {}).get('model') or
            data.get('experiment_info', {}).get('model') or
            'unknown'
        )
        safe_name = sanitize_model_name(model_name)
        output_html = os.path.join(
            output_dir, '{}.html'.format(safe_name)
        )

        generate_single_model_html(
            args.input, output_html, vis_images_folder,
            sample=args.sample, random_seed=args.random_seed
        )

# ...def main()


#%% Command-line entry point

if __name__ == '__main__':
    main()
