# Cattle Behavior VLM Classification - Implementation Plan

## Overview

Evaluate VLMs on classifying cow behavior (head_up, head_down, running, unknown) in camera
trap image crops using few-shot prompting. Compare Gemini 3.0 (Pro/Flash) and Ollama
(qwen3-VL:32b, qwen2.5vl:72b).

**Hardware:** 2x RTX 4090 (48GB VRAM total). This was sufficient to run qwen2.5vl:72b
via Ollama in the hero-images project.

## Key Design Decisions

- **4 few-shot examples per category** (16 total), selected to span the image size range
- **5 query images per API request** (with all 16 few-shot examples included in each request)
- **Images >750px** on long side downscaled; smaller images NOT upscaled
- **Gemini**: both batch API (async, 50% cheaper) and sync API supported via CLI flag
- **Ollama**: uses `/api/chat` with multi-turn few-shot (user/assistant pairs)
- **Checkpointing**: atomic write pattern (backup -> delete -> rename)
- **No category leakage**: images sent as base64 bytes only, never file paths, so the VLM
  cannot see the folder name (headdown/headup/etc.)

## Dataset

Source: `C:\temp\cow-experiments\sorted_crops\` (headdown=604, headup=468, running=11,
unknown=234; total=1,317). After removing 16 few-shot examples, 1,301 test images remain
(261 requests at 5 images/request).

## File Structure

All code in `c:\git\agentmorrisprivate\archive\cow_experiments\`. All output/results to
`C:\temp\cow-experiments\cow-vlm-experiments\`.

### New Files to Create

| File | Purpose |
|------|---------|
| `README.md` | Project documentation: overview, setup (including Ollama), CLI examples. Modeled after hero-images README. |
| `implementation-plan.md` | Copy of this plan, for reference by future agents/developers |
| `cow_vlm_utils.py` | Shared utilities: image processing, prompt building, response parsing, checkpointing, cost estimation, constants |
| `select_few_shot_examples.py` | One-time: measure image sizes, select 4 exemplars per category |
| `run_gemini_classification.py` | Gemini classification (batch + sync modes) |
| `run_ollama_classification.py` | Ollama classification via /api/chat |
| `evaluate_results.py` | Compute accuracy, confusion matrix, per-class metrics, model comparison |
| `generate_cow_visualization.py` | HTML visualization: single-model + multi-model comparison |
| `requirements.txt` | Python dependencies |

### Existing Files (unchanged)

- `GEMINI_API_KEY.txt` - already exists, loaded by `cow_vlm_utils.load_api_key()`
- `prepare_cow_crops.py`, `validate_cow_labels.py` - existing scripts, not modified
- `.gitignore` - already excludes GEMINI_API_KEY.txt

### Output Directory Layout

```
C:\temp\cow-experiments\cow-vlm-experiments\
    image_sizes.json
    few_shot_examples.json
    results\
        gemini-3-flash-preview_batch_YYYYMMDD_HHMMSS.json
        gemini-3-pro-preview_sync_YYYYMMDD_HHMMSS.json
        qwen3-VL-32b_YYYYMMDD_HHMMSS.json
        qwen2.5vl-72b_YYYYMMDD_HHMMSS.json
    checkpoints\
        *.tmp.json
    visualizations\
        vis_images\          (shared thumbnails)
        *.html               (per-model + comparison)
```

---

## Module Details

### 1. `cow_vlm_utils.py` - Shared Utilities

Adapted from `C:\git\hero-images\hero_images\image_processor.py` and shared functions in
`gemini_labeling.py` / `ollama_local_labeling.py`.

**Constants:**
- `VALID_CATEGORIES = ['head_up', 'head_down', 'running', 'unknown']`
- `CATEGORY_FOLDER_MAP = {'headdown': 'head_down', 'headup': 'head_up', ...}`
- Category alias normalization map (e.g. 'grazing' -> 'head_down', 'galloping' -> 'running')
- Gemini pricing constants for cost estimation
- Default paths (`SORTED_CROPS_DIR`, `OUTPUT_BASE_DIR`, `API_KEY_PATH`)

**Functions:**
- `resize_image_to_bytes(image_path, max_size=750)` - only downscales, never upscales
- `resize_image_to_base64(image_path, max_size=750)`
- `get_image_dimensions(image_path)` - returns (width, height)
- `enumerate_dataset_images(sorted_crops_dir)` - returns list of dicts with path, filename,
  ground_truth for all images
- `load_api_key(api_key_path)` - reads from GEMINI_API_KEY.txt
- `sanitize_model_name(model_name)` - replace `:` `/` with `-`
- `load_few_shot_examples(path)` - load few_shot_examples.json
- `get_test_images(all_images, few_shot_examples)` - exclude few-shot images
- `group_query_images(test_images, batch_size=5)` - chunk into groups
- `parse_vlm_response(response_text)` - extract JSON array, normalize categories
- `normalize_category(raw)` - map aliases, flag unknown as 'parse_error'
- `save_checkpoint(results, checkpoint_path)` - atomic write
- `load_checkpoint(checkpoint_path)` - load partial results
- `estimate_gemini_cost(n_images, model, is_batch)` - compute and display cost

### 2. `select_few_shot_examples.py` - Few-Shot Selection

**Algorithm:** For each category, sort images by long-side dimension, divide into 4
quartiles, pick the image closest to each quartile's median. Ensures size diversity
(~100px, ~400px, ~800px, ~2000px representatives).

**Output:** `few_shot_examples.json` with selection metadata and per-image details
(path, filename, category, width, height, long_side). Also saves `image_sizes.json`.

**CLI:** `python select_few_shot_examples.py [--crops-dir] [--output-dir] [--images-per-category N]`

### 3. `run_gemini_classification.py` - Gemini Classification

Two classes: `CowGeminiBatchProcessor` and `CowGeminiSyncProcessor`.

**Prompt structure (Gemini):** Single content message with interleaved parts:
```
[system text] -> [example label text] -> [example image inline_data] ->
[label confirmation] -> ... (16 examples) ...
[query instruction: "classify the following N images"] ->
[query image 1 inline_data] -> [query image 2 inline_data] -> ...
[final instruction: "respond with JSON array"]
```

Few-shot image bytes are pre-encoded once in `__init__` and reused for every request.

**Batch mode:** `client.batches.create()` with 261 request dicts. Poll with
`client.batches.get()`. Results come back in submission order. Batch metadata saved for
resume after disconnection.

**Sync mode:** Sequential requests with `model.generate_content()`. Rate limiting (2s pause
between requests). Checkpoint every 50 images.

**Cost estimation** displayed before submission; user must confirm (or use `--auto-confirm`).
Estimated costs: Flash batch ~$0.45, Flash sync ~$0.89, Pro batch ~$1.77, Pro sync ~$3.56.

**CLI:**
```
python run_gemini_classification.py
    [--model MODEL]           # default: gemini-3-flash-preview
    [--sync]                  # sync instead of batch
    [--cancel JOB_NAME]       # cancel batch job
    [--resume FILE]           # resume from checkpoint/metadata
    [--image-size N]          # default: 750
    [--auto-confirm / -y]     # skip cost prompt
    [--checkpoint-interval N] # default: 50 images
    [--output-dir DIR]
    [--few-shot-file FILE]
    [--max-images N]          # for testing
    [--query-batch-size N]    # default: 5
```

### 4. `run_ollama_classification.py` - Ollama Classification

Class: `CowOllamaProcessor`. Uses `/api/chat` endpoint.

**Prompt structure (Ollama):** Multi-turn conversation:
```
system: [classification instructions and category descriptions]
user: "What is the behavior of this cow?" + [image: headup_example_1]
assistant: "head_up"
user: "What is the behavior of this cow?" + [image: headup_example_2]
assistant: "head_up"
... (16 example pairs) ...
user: "Classify the following N images. Respond with JSON array..." +
      [image: query_1, image: query_2, ..., image: query_5]
```

**Health checks:** `check_server_health()` (GET /api/tags) and
`check_model_available()` before processing.

**CLI:**
```
python run_ollama_classification.py
    [--model MODEL]           # default: qwen3-VL:32b
    [--server-url URL]        # default: http://localhost:11434
    [--image-size N]          # default: 750
    [--checkpoint-interval N] # default: 50
    [--resume FILE]
    [--output-dir DIR]
    [--few-shot-file FILE]
    [--max-images N]
    [--query-batch-size N]    # default: 5
    [--setup-help]            # print Ollama setup instructions
```

### 5. `evaluate_results.py` - Metrics and Comparison

**Functions:** `compute_metrics(results)` returns overall accuracy, per-class
precision/recall/F1, confusion matrix. `compare_models(results_dir)` loads all result JSONs
and prints a comparison table.

**CLI:** `python evaluate_results.py INPUT [--compare] [--output-dir DIR]`

### 6. `generate_cow_visualization.py` - HTML Output

**Single-model mode:** Confusion matrix at top, per-class metrics table, filter buttons
(all/correct/incorrect), then per-image table with 200px thumbnails, ground truth,
prediction, and correct/incorrect indicator. Incorrect rows highlighted in pink.

**Multi-model comparison mode:** Summary accuracy table, side-by-side confusion matrices,
per-image comparison table showing all models' predictions with disagreements highlighted.

Shared `vis_images/` thumbnail folder. Supports `--sample N` for random subsampling.

**CLI:** `python generate_cow_visualization.py INPUT [--compare] [--output-dir DIR]
[--sample N] [--random-seed N]`

---

## JSON Output Format (all models)

```json
{
    "run_info": {
        "timestamp": "...",
        "model": "gemini-3-flash-preview",
        "processing_method": "batch_api",
        "image_max_size": 750,
        "query_batch_size": 5,
        "few_shot_file": "...",
        "total_query_images": 1301,
        "total_requests": 261,
        "successful_predictions": 1295,
        "failed_predictions": 6,
        "overall_accuracy": 0.873
    },
    "results": [
        {
            "image_path": "...",
            "image_filename": "...",
            "ground_truth": "head_down",
            "prediction": "head_down",
            "correct": true,
            "success": true,
            "request_index": 0,
            "image_index_in_request": 1
        }
    ]
}
```

---

## Edge Cases

- **Last batch <5 images:** prompt says "the following N images" dynamically
- **Unexpected categories:** normalized via alias map; truly unknown -> 'parse_error'
- **Fewer items returned than expected:** missing images marked as `success: false`
- **Malformed JSON response:** all images in that request marked as failed, raw response logged
- **Very small images (<100px):** not upscaled, sent at original size
- **Category leakage:** images sent as base64 only, no file paths in prompts

---

## Coding Conventions (per MegaDetector developers.md)

- `#%%` cell markers for interactive execution
- snake_case functions/variables, CamelCase classes, UPPER_SNAKE_CASE constants
- Google-style docstrings, no type hints
- Single quotes, string.format() over f-strings
- 100-120 char max line length
- Module header docstrings

---

## README.md Structure (modeled after hero-images README)

```
# Cattle behavior classification with VLMs
## Project overview
  - What this project does, dataset description, models tested
## Scripts
  - Table of scripts and their purposes
## Usage
  ### Setup
    - pip install -r requirements.txt
    - Gemini API key setup
    - Ollama setup (server, model pulling, VRAM notes, troubleshooting)
  ### Select few-shot examples
    - CLI example for select_few_shot_examples.py
  ### Classify images
    #### Gemini (batch and sync modes)
      - CLI examples, cost estimates, cancel/resume
    #### Ollama
      - CLI examples, model recommendations, checkpoint/resume
  ### Evaluate results
    - Single model and multi-model comparison examples
  ### Visualize results
    - Single model and comparison HTML examples
## Gemini batch job management
  - Cancel, resume instructions
## Data pipeline diagram
## Technical notes
  - Image preprocessing, output format, hardware requirements
```

README.md will be created early and updated as each module is implemented.

## Implementation Order

1. `requirements.txt` + pip install
2. `README.md` (initial version with setup instructions)
3. `implementation-plan.md` (copy of this plan)
4. `cow_vlm_utils.py` (core shared module)
5. `select_few_shot_examples.py` (run it, verify output)
6. `run_gemini_classification.py` (test with --max-images 10 --sync -y)
7. `run_ollama_classification.py` (test with --max-images 10)
8. `evaluate_results.py`
9. `generate_cow_visualization.py`
10. Update `README.md` with all CLI examples and final documentation

---

## Verification

1. Run `select_few_shot_examples.py` -> check `image_sizes.json` has 1,317 entries,
   `few_shot_examples.json` has 16 images spanning sizes, visually inspect selections
2. Run Gemini sync with `--max-images 10 -y` -> verify 10 results, ground truth matches
   folders, predictions are valid categories
3. Run Gemini sync with `--max-images 7 -y` -> verify last batch handles 2 images correctly
4. Test checkpoint/resume: start with `--max-images 20`, Ctrl+C midway, resume with
   `--resume`, verify all 20 images in final results
5. Run Ollama with `--max-images 10` -> verify similar output
6. Full runs on all 4 models
7. `evaluate_results.py results/ --compare` -> confusion matrix row sums match category
   counts, comparison table shows all 4 models
8. `generate_cow_visualization.py results/ --compare` -> open HTML, verify thumbnails load,
   filter buttons work, comparison page correct

---

## Ollama Setup Instructions (for README and --setup-help)

```
1. Start Ollama server:        ollama serve
2. Pull models:                ollama pull qwen3-VL:32b
                               ollama pull qwen2.5vl:72b
3. Verify:                     ollama list

Notes:
- This machine has 2x RTX 4090 (48GB VRAM total)
- qwen2.5vl:72b needs ~49GB VRAM; will partially spill to CPU RAM
  (this worked for the hero-images project on this hardware)
- qwen3-VL:32b should fit comfortably in GPU memory
- First request per model may take minutes while loading into GPU memory
- If timeouts occur during model loading, set environment variables:
    set OLLAMA_KEEP_ALIVE=1h
    set OLLAMA_LOAD_TIMEOUT=30m
  then restart ollama serve
- Models stored in %USERPROFILE%\.ollama\models by default
- List models: ollama list
- Remove models: ollama rm <model_name>
```

---

## Dependencies (requirements.txt)

```
pillow>=11.3.0
google-generativeai>=0.8.5
google-genai>=1.38.0
requests>=2.31.0
```
