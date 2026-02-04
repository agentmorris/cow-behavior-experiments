#%% Imports and constants

import os

base_folder = r'G:\temp\Mini Experiment'
label_folder = os.path.join(base_folder,'Label JSONS')
image_folder = os.path.join(base_folder,'Original Cow Images')

assert os.path.isdir(label_folder)
assert os.path.isdir(image_folder)


#%% Match labels to files, copy into labels folder

import shutil
import json

from collections import defaultdict

images_relative = sorted(os.listdir(image_folder))
labels_relative = sorted(os.listdir(label_folder))

label_to_count = defaultdict(int)

for label_fn_relative in labels_relative:
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
    with open(fn_dst_abs,'w') as f:
        json.dump(d,f,indent=1)

print('Copied labels for {} images:'.format(len(labels_relative)))

for label in label_to_count:
    print('{}: {}'.format(label,label_to_count[label]))


#%% Label

cmd = 'python labelme "{}"'.format(image_folder)
print(cmd)
import clipboard; clipboard.copy(cmd)
