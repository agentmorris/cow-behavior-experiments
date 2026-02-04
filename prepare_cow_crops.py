#%% Header

"""
0. Rename label files to match images.

1. Associate each label with a MegaDetector detection, probably writing the result to a self-contained .json file that contains both box locations and
labels, and probably unique detection IDs that make it easy to figure out later which detection corresponds to each cropped image.

2. Crop the cows out to separate images (possibly padding with some extra pixels around the outside of each box), preserving the detection IDs from step
(1) in the filenames.  Creating separate images for each cow is not strictly required, but I think it will make the next steps easier.  At this stage I
would probably write the crops out to separate folders for each category.  That's also not strictly necessary, but I think it will also make the next steps easier.

3. Semi-randomly choose maybe three from each category to use as few-shot examples.  I.e., I think in the next step, we'll submit a query to a VLM that
says something like "here are three examples each for cows that are head-up, head-down, running, and unknown, now here's 10 more that I want you to assign the
same categories to".  I say "semi-randomly" because you probably want a bit of visual diversity in the few-shot examples, e.g. if you happen to pick three
far-away cows for the "head-down" examples, that's not optimal.

4. Run those queries through a few different VLMs and evaluate the results.  Note to selves: if the file names contain the categories (e.g. the cropped
images are in folders called "head down", "head up", etc.), make sure the VLM doesn't get to see the full filenames.  I think this will be a non-issue,
because I think the images will get submitted with no metadata, as just a byte stream, but just keep this in mind.

"""


#%% Imports and constants

import os
import shutil
import json

from collections import defaultdict
from tqdm import tqdm

from megadetector.detection.run_detector_batch import \
    load_and_run_detector_batch, write_results_to_file

base_folder = 'c:/temp/cow-experiments'
label_folder = os.path.join(base_folder,'Label JSONS')
image_folder = os.path.join(base_folder,'Original Cow Images')
preview_folder = os.path.join(base_folder,'label-preview')
crop_folder = os.path.join(base_folder,'crops')
sorted_crop_folder = os.path.join(base_folder,'sorted_crops')

assert os.path.isdir(label_folder)
assert os.path.isdir(image_folder)

md_results_file_raw = os.path.join(image_folder,'md_results_raw.json')
md_results_file_with_detection_ids = os.path.join(image_folder,'md_results_with_detection_ids.json')
md_results_file_with_labels = os.path.join(image_folder,'md_results_with_labels.json')


#%% Match labels to files, copy into labels folder

images_relative = sorted(os.listdir(image_folder))
labels_relative = sorted(os.listdir(label_folder))

label_to_count = defaultdict(int)

overwrite_json_files = False

for label_fn_relative in tqdm(labels_relative):
    assert label_fn_relative.startswith('anno_')
    expected_image_name = label_fn_relative.replace('anno_','').replace('.json','.JPG')
    assert expected_image_name in images_relative

    fn_src_abs = os.path.join(label_folder,label_fn_relative)
    fn_dst_relative = expected_image_name.replace('.JPG','.json')
    fn_dst_abs = os.path.join(image_folder,fn_dst_relative)
    # shutil.copyfile(fn_src_abs,fn_dst_abs)
    with open(fn_src_abs,'r') as f:
        d = json.load(f)
    d['imagePath'] = expected_image_name
    for shape in d['shapes']:
        assert len(shape['points'][0]) == 2
        assert shape['shape_type'] == 'point'
        label_to_count[shape['label']] += 1
    if (not os.path.isfile(fn_dst_abs)) or (overwrite_json_files):
        with open(fn_dst_abs,'w') as f:
            json.dump(d,f,indent=1)

print('Copied labels for {} images:'.format(len(labels_relative)))

for label in label_to_count:
    print('{}: {}'.format(label,label_to_count[label]))


#%% Create classification categories

from megadetector.utils.ct_utils import invert_dictionary
classification_category_id_to_name = {}
for category_name in label_to_count.keys():
    classification_category_id_to_name[str(len(classification_category_id_to_name))] = category_name
classification_category_name_to_id = invert_dictionary(classification_category_id_to_name)


#%% Run MegaDetector

model_name = 'mdv5a'

results = load_and_run_detector_batch(model_file=model_name,
                                      image_file_names=image_folder,
                                      quiet=True,
                                      include_image_size=True,
                                      use_threads_for_queue=True)

_ = write_results_to_file(results,
                          output_file=md_results_file_raw,
                          relative_path_base=image_folder,
                          detector_file=model_name,
                          info=None)


#%% Add unique IDs to each detection

with open(md_results_file_raw,'r') as f:
    d = json.load(f)

for i_image,im in enumerate(d['images']):
    for i_det,det in enumerate(im['detections']):
        id_string = 'im_' + str(i_image).zfill(4) + '_det_' + str(i_det).zfill(4)
        det['crop_id'] = id_string

with open(md_results_file_with_detection_ids,'w') as f:
    json.dump(d,f,indent=1)


#%% Associate labelme categories with detections

detection_threshold = 0.2
fake_bbox_width_normalized = 0.03
fake_bbox_height_normalized = fake_bbox_width_normalized

with open(md_results_file_with_detection_ids,'r') as f:
    d = json.load(f)

unmatched_label_detection_category_id = '100'
ambiguous_label_detection_category_id = '101'
d['detection_categories'][unmatched_label_detection_category_id] = 'unmatched label'
d['detection_categories'][ambiguous_label_detection_category_id] = 'ambiguous label'

labels_without_matching_boxes = []
labels_with_multiple_matching_boxes = []
labels_matched_successfully = []

for i_image,im in enumerate(d['images']):

    image_filename_relative = im['file']
    label_filename_relative = image_filename_relative.replace('.JPG','.json')
    label_filename_abs = os.path.join(image_folder,label_filename_relative)
    assert os.path.isfile(label_filename_abs)

    detections_above_threshold = [det for det in im['detections'] if \
                  ((det['category'] == '1') and (det['conf'] >= detection_threshold))]
    detection_id_to_detection = {det['crop_id']:det for det in detections_above_threshold}

    with open(label_filename_abs,'r') as f:

        labelme_dict = json.load(f)
        assert labelme_dict['imageWidth'] == im['width']
        assert labelme_dict['imageHeight'] == im['height']

    # These will hold "fake" detections for unmatched or ambiguously-matched labels
    new_detections = []

    for i_shape,shape in enumerate(labelme_dict['shapes']):

        label_string = 'image_{}_label_{}'.format(
            str(i_image).zfill(4),str(i_shape).zfill(4))

        assert len(shape['points'][0]) == 2
        assert shape['shape_type'] == 'point'
        # Labelme points are absolute x/y
        labelme_x_normalized = shape['points'][0][0] / im['width']
        labelme_y_normalized = shape['points'][0][1] / im['height']

        assert shape['label'] in classification_category_name_to_id
        classification_category_id = classification_category_name_to_id[shape['label']]

        # Find detections that contain this point
        matching_detection_ids = []
        for det in detections_above_threshold:
            bbox = det['bbox']
            if (labelme_x_normalized >= bbox[0]) and (labelme_x_normalized <= (bbox[0] + bbox[2])) and \
                (labelme_y_normalized >= bbox[1]) and (labelme_y_normalized <= (bbox[1] + bbox[3])):
                matching_detection_ids.append(det['crop_id'])

        # If there is a unique match...
        if len(matching_detection_ids) == 1:
            labels_matched_successfully.append(label_string)
            matching_detection = detection_id_to_detection[matching_detection_ids[0]]
            matching_detection['classifications'] = [[classification_category_id,1.0]]
        # If there are no matches
        elif len(matching_detection_ids) == 0:
            labels_without_matching_boxes.append(label_string)
            fake_det = {}
            fake_det['category'] = unmatched_label_detection_category_id
            fake_det['conf'] = 1.0
            fake_det['bbox'] = [labelme_x_normalized - (fake_bbox_width_normalized)/2.0,
                                labelme_y_normalized - (fake_bbox_height_normalized)/2.0,
                                fake_bbox_width_normalized,
                                fake_bbox_height_normalized]
            fake_det['classifications'] = [[classification_category_id,1.0]]
            new_detections.append(fake_det)
        else:
            assert len(matching_detection_ids) > 1
            labels_with_multiple_matching_boxes.append(label_string)
            fake_det = {}
            fake_det['category'] = ambiguous_label_detection_category_id
            fake_det['conf'] = 1.0
            fake_det['bbox'] = [labelme_x_normalized - (fake_bbox_width_normalized)/2.0,
                                labelme_y_normalized - (fake_bbox_height_normalized)/2.0,
                                fake_bbox_width_normalized,
                                fake_bbox_height_normalized]
            fake_det['classifications'] = [[classification_category_id,1.0]]
            new_detections.append(fake_det)

    # ...for each label

    im['detections'].extend(new_detections)

# ...for each image

print('Successfully matched {} labels'.format(len(labels_matched_successfully)))
print('Ambiguously matched {} labels'.format(len(labels_with_multiple_matching_boxes)))
print('Failed to match {} labels'.format(len(labels_without_matching_boxes)))

d['classification_categories'] = classification_category_id_to_name

with open(md_results_file_with_labels,'w') as f:
    json.dump(d,f,indent=1)


#%% Preview with labelme categories

from megadetector.visualization.visualize_detector_output import \
    visualize_detector_output, DEFAULT_BOX_THICKNESS, default_box_sort_order

html_output_file = os.path.join(preview_folder,'index.html')

visualize_detector_output(detector_output_path=md_results_file_with_labels,
                          out_dir=preview_folder,
                          images_dir=image_folder,
                          confidence_threshold=0.199999999999,
                          sample=-1,
                          output_image_width=1600,
                          random_seed=None,
                          render_detections_only=False,
                          classification_confidence_threshold=0.1,
                          html_output_file=html_output_file,
                          html_output_options=None,
                          preserve_path_structure=True,
                          parallelize_rendering=True,
                          parallelize_rendering_n_cores=10,
                          parallelize_rendering_with_threads=True,
                          box_sort_order=default_box_sort_order,
                          category_names_to_blur=None,
                          link_images_to_originals=False,
                          detector_label_map=None,
                          box_thickness=DEFAULT_BOX_THICKNESS,
                          box_expansion=0)

from megadetector.utils.path_utils import open_file
open_file(html_output_file)


#%% Create crops

# Only write animal crops, not fake detections
from megadetector.postprocessing.create_crop_folder import \
    CreateCropFolderOptions, create_crop_folder

crop_options = CreateCropFolderOptions()

#: Confidence threshold determining which detections get written
crop_options.confidence_threshold = 0.2

#: Number of pixels to expand each crop
crop_options.expansion = 20

#: JPEG quality to use for saving crops (None for default)
crop_options.quality = 95

#: Whether to overwrite existing images
crop_options.overwrite = True

#: Number of concurrent workers
crop_options.n_workers = 8

#: Whether to use processes ('process') or threads ('thread') for parallelization
crop_options.pool_type = 'thread'

#: Include only these categories, or None to include all
#:
#: options.category_names_to_include = ['animal']
crop_options.category_names_to_include = None

create_crop_folder(input_file=md_results_file_with_labels,
                   input_folder=image_folder,
                   output_folder=crop_folder,
                   output_file=None,
                   crops_output_file=None,
                   options=crop_options)


#%% Copy crops to folders

from megadetector.utils.path_utils import find_images

## Map crop IDs to labels

with open(md_results_file_with_labels,'r') as f:
    d = json.load(f)

crop_id_to_label = {}

detection_category_name_to_id = invert_dictionary(d['detection_categories'])
classification_category_name_to_id = invert_dictionary(d['classification_categories'])

for im in d['images']:

    for det in im['detections']:

        if 'crop_id' not in det:
            continue
        crop_id = det['crop_id']
        if det['category'] != detection_category_name_to_id['animal']:
            assert 'classifications' not in det
            continue
        if 'classifications' not in det:
            continue
        classifications = det['classifications']
        assert len(classifications) == 1
        assert len(classifications[0]) == 2
        assert classifications[0][1] == 1.0
        category_name = classification_category_id_to_name[classifications[0][0]]
        assert crop_id not in crop_id_to_label
        crop_id_to_label[crop_id] = category_name

    # ...for each detection

# ...for each image


## Map crop IDs to relative paths

crop_images_relative = find_images(crop_folder,recursive=True,return_relative_paths=True)

crop_id_to_crop_filename_relative = {}

for fn in crop_images_relative:
     # '~scratch~general~vast~u6064781~HD 1~2024~NM~Toriette~05_29_2024-07_04_2024~8206-06~100_BTCF~IMG_3416.crop_im_0277_det_0001.JPG'
    tokens = os.path.splitext(fn)[0].split('crop_')
    assert len(tokens) == 2
    if not tokens[1].startswith('im'):
        continue
    crop_id = tokens[1]
    assert crop_id not in crop_id_to_crop_filename_relative
    crop_id_to_crop_filename_relative[crop_id] = fn


## Copy files

for crop_id in tqdm(crop_id_to_label):

    assert crop_id in crop_id_to_crop_filename_relative
    category_name = crop_id_to_label[crop_id]
    input_fn_relative = crop_id_to_crop_filename_relative[crop_id]
    input_fn_abs = os.path.join(crop_folder,input_fn_relative)
    assert os.path.isfile(input_fn_abs)
    output_fn_abs = os.path.join(sorted_crop_folder,category_name,input_fn_relative)
    os.makedirs(os.path.dirname(output_fn_abs),exist_ok=True)
    shutil.copyfile(input_fn_abs,output_fn_abs)

