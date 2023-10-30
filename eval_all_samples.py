import json
import os
import shutil
import yaml
from omegaconf import OmegaConf

from fatezero import function_test


def load_style_txt(file_path):
    with open(file_path, "r") as f:
        styles = f.readlines()
    return [style.strip() for style in styles]


def process_json_and_images(data, base_directory, new_directory):

    genre_classes = load_style_txt("data/genre_class.txt")
    artist_classes = load_style_txt("data/artist_class.txt")
    style_classes = load_style_txt("data/style_class.txt")

    for key, item in data.items():
        for style_type, classes in [('genre_class', genre_classes), ('artist_class', artist_classes),
                                    ('style_class', style_classes)]:
            style_str = classes[item[style_type]].split(' ')[-1].lower().replace('_', ' ')

            # 更新source_prompt
            modified_prompt = f"a {style_str} style of {item['source_prompt']}"

            old_image_path = os.path.join(base_directory, item["image_path"])
            new_folder_path = os.path.join(new_directory, key + "_" + style_str)

            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            new_image_path = os.path.join(new_folder_path, os.path.basename(item["image_path"]))
            shutil.copy2(old_image_path, new_image_path)  # 复制图像到新的文件夹

            modify_yaml_and_run('./config/teaser/jeep_watercolor.yaml', new_folder_path, modified_prompt, item['source_prompt'])


def modify_yaml_and_run(input_yaml, image_folder, prompt, source_prompt):
    with open(input_yaml, 'r') as f:
        config = yaml.safe_load(f)

    config['dataset_config']['path'] = image_folder
    config['dataset_config']['prompt'] = source_prompt
    config['editing_config']['editing_prompts'] = [source_prompt, prompt]

    with open(input_yaml, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # run_your_program()
    config = "./config/teaser/jeep_watercolor.yaml"
    Omegadict = OmegaConf.load(config)
    if 'unet' in os.listdir(Omegadict['pretrained_model_path']):
        function_test(config=config, logdir=os.path.join('./result', image_folder.split('/')[-1]), **Omegadict)


with open("./data/image700_source2edit_prompt.json", "r") as f:
    data = json.load(f)

base_directory = "./data/annotation_images/"
new_directory = "/tmp"
process_json_and_images(data, base_directory, new_directory)
