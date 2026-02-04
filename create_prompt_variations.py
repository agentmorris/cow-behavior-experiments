#%% Header

"""
Create prompt variation JSON files for experimentation.

Generates 5 different system prompts that vary in how they describe the
head_up/head_down distinction. These can be edited manually or regenerated.

Usage:
    python create_prompt_variations.py --output-dir DIR
"""


#%% Imports

import os
import json
import argparse


#%% Prompt definitions

PROMPT_VARIATIONS = {
    'v1_baseline': {
        'name': 'Baseline (current)',
        'description': 'The current production prompt',
        'prompt': """You are an expert at classifying cattle behavior in camera trap images.

Each cow should be classified into exactly one of these categories:
- head_up: the cow's head is in a position parallel or above the top of the shoulder. It does not matter if the cow is laying down, standing, or walking, it is only based on the vertical relationship between the head and the top of the shoulder.
- head_down: the cow's head is in a position below the top of the shoulder. Similarly, it does not matter if the cow is laying down, standing, or walking.
- running: all four hooves are off the ground, or the cow's legs are in a bounding or suspended stride, or the cow is clearly in rapid motion (often signaled by motion blur).
- unknown: the cow's head is out of frame, or the image is extremely unclear, to the point where you can't make out the posture of the cow.
"""
    },

    'v2_back_reference': {
        'name': 'Back line reference',
        'description': 'Uses "top of the back" instead of "shoulder" as reference point',
        'prompt': """You are an expert at classifying cattle behavior in camera trap images.

Each cow should be classified into exactly one of these categories:
- head_up: the cow's head is at or above the level of the top of its back (the horizontal line along the spine). This applies whether the cow is standing, walking, or lying down.
- head_down: the cow's head is below the level of the top of its back. This typically indicates grazing or drinking. This applies whether the cow is standing, walking, or lying down.
- running: all four hooves are off the ground, or the cow's legs are in a bounding or suspended stride, or the cow is clearly in rapid motion (often signaled by motion blur).
- unknown: the cow's head is out of frame, or the image is extremely unclear, to the point where you can't make out the posture of the cow.
"""
    },

    'v3_explicit_examples': {
        'name': 'Explicit behavioral examples',
        'description': 'Adds concrete behavioral examples for each category',
        'prompt': """You are an expert at classifying cattle behavior in camera trap images.

Each cow should be classified into exactly one of these categories:
- head_up: the cow's head is level with or raised above the top of its back. Examples: looking around alertly, walking with head forward, standing at rest with head level, lying down but with head held up.
- head_down: the cow's head is lowered below the level of its back. Examples: grazing on grass, drinking water, sniffing the ground. The head is clearly angled downward.
- running: all four hooves are off the ground, or the cow's legs are in a bounding or suspended stride, or the cow is clearly in rapid motion (often signaled by motion blur).
- unknown: the cow's head is out of frame, or the image is extremely unclear, to the point where you can't make out the posture of the cow.

Important: Focus on where the HEAD is relative to the BACK, not on what activity the cow appears to be doing.
"""
    },

    'v4_chain_of_thought': {
        'name': 'Chain of thought guidance',
        'description': 'Guides the model through a reasoning process',
        'prompt': """You are an expert at classifying cattle behavior in camera trap images.

For each image, determine the cow's head position using this process:
1. First, locate the cow's back (the top line of the body from shoulders to hips)
2. Then, locate where the cow's head is positioned
3. Compare: Is the head at/above back level, or below it?

Categories:
- head_up: the head is at the same level as the back or higher. This includes cows looking forward, standing alert, or lying with head raised.
- head_down: the head is clearly below back level. This includes grazing, drinking, or any posture where the head droops below the back line.
- running: all four hooves are off the ground, or the cow is in a bounding stride, or there's motion blur indicating rapid movement.
- unknown: the head is not visible (out of frame or obstructed), or the image is too unclear to determine posture.
"""
    },

    'v5_negative_guidance': {
        'name': 'Negative guidance',
        'description': 'Explicitly states what NOT to do and common mistakes',
        'prompt': """You are an expert at classifying cattle behavior in camera trap images.

Each cow should be classified into exactly one of these categories:
- head_up: the cow's head is at or above the level of its back/shoulders.
- head_down: the cow's head is below the level of its back/shoulders.
- running: all four hooves are off the ground, or legs in a bounding stride, or clear motion blur.
- unknown: head out of frame, or image too unclear to determine posture.

IMPORTANT - Common mistakes to avoid:
- A cow lying down can be EITHER head_up or head_down depending on head position - don't assume lying = head_down
- A cow walking or moving slowly is NOT running unless hooves are off the ground
- If the head is roughly level with the back, classify as head_up (head_down requires the head to be clearly BELOW the back)
- Only use unknown if you truly cannot see the head or determine its position

Focus purely on the vertical relationship between head and back, regardless of what activity the cow appears to be doing.
"""
    },
}


#%% Main function

def create_prompt_variations(output_dir):
    """
    Write prompt variation JSON files.

    Args:
        output_dir: directory to write prompt_variations/ folder
    """

    prompts_dir = os.path.join(output_dir, 'prompt_variations')
    os.makedirs(prompts_dir, exist_ok=True)

    for key, data in PROMPT_VARIATIONS.items():
        output_path = os.path.join(prompts_dir, '{}.json'.format(key))

        output = {
            'id': key,
            'name': data['name'],
            'description': data['description'],
            'system_prompt': data['prompt'],
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print('Created: {}'.format(output_path))

    print('')
    print('Created {} prompt variations in {}'.format(
        len(PROMPT_VARIATIONS), prompts_dir
    ))


def main():

    parser = argparse.ArgumentParser(
        description='Create prompt variation JSON files for experimentation.'
    )
    parser.add_argument(
        '--output-dir',
        default='C:/temp/cow-experiments/cow-vlm-experiments/experiments',
        help='Output directory for experiment files'
    )

    args = parser.parse_args()
    create_prompt_variations(args.output_dir)


if __name__ == '__main__':
    main()
