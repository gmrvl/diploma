import argparse
import base64
import json
import os
import os.path as osp

import PIL.Image
import yaml

from labelme.logger import logger
from labelme import utils

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', default=None)
    # set label names file path
    parser.add_argument('-n','--names',default=None)
    args = parser.parse_args()
    zzz = 0
    for root, dirs, files in os.walk("labelme.val"): #Путь к файлам json
        for file in files:
            if ('.json' in file):
                main(args, root + '/' + file, str(zzz))
                zzz = zzz + 1

def main(args, json_file, zzz):
    if args.out is None:
        out_dir = osp.basename(json_file).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    
    label_name_to_value = {'_background_': 0}
    
    # add label names to dict    
    if args.names is not None:
        with open(args.names, 'r') as f:
            lines = f.readlines()
        
        count = 1
        for line in lines:
            line = line.strip()
            if line == '':
                continue
                
            label_name_to_value[line] = count
            count += 1
        
    data = json.load(open(json_file))
    imageData = data.get('imageData')

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)

    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    lbl_viz = utils.draw_label(lbl, img, label_names)

    PIL.Image.fromarray(img).save(osp.join(out_dir, zzz + 'img.png'))
    utils.lblsave(osp.join(out_dir, zzz + 'label.png'), lbl)
    # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, zzz + 'label_viz.png'))

    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name + '\n')

    #logger.warning('info.yaml is being replaced by label_names.txt')
    #info = dict(label_names=label_names)
    #with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
    #    yaml.safe_dump(info, f, default_flow_style=False)

    #logger.info('Saved to: {}'.format(out_dir))


if __name__ == '__main__':
    test()