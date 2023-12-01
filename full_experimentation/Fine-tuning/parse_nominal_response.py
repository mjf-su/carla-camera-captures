import os
import numpy as np
import json
import copy
import time

folder = os.path.dirname(__file__)
target = "nominal_stop_sign"
target_folder = os.path.join(folder, "data", target)
response_file = "vlm_responses.npz"
trim_response_file = "trim_vlm_responses.npz"

incorrect_labels = {}

incorrect_format = 0 # answer doesn't include 'overall scenario classification
incorrect_cls = 0 # false positive response
trim_insufficient = 0 # trimming didn't bring word count down enough
word_limit = 358

for exp in os.listdir(target_folder):
    if "exp" not in exp:
        continue

    incorrect_labels[exp] = []
    experiment = np.load(os.path.join(target_folder, exp, response_file))
    
    responses = experiment["vlm_responses"]
    image_names = experiment["image_names"]

    trim_responses = []
    trim_image_names = []
    for i, response in enumerate(responses):
        if "overall scenario classification:" not in response.lower():
            print("Response on " + image_names[i] + " in " + exp + " needs your review.")
            incorrect_labels[exp].append(image_names[i]) # store image to ignore in fine-tuning
            incorrect_format += 1
        else:
            paragraphs = response.split("\n\n") # split response into paragraphs

            cls_list = []
            for p in paragraphs: # process each paragraph in response
                split = p.split(":")
                cls = split[-1]
                cls_list.append(cls)

            cut_index = -1
            for j, p in enumerate(copy.copy(paragraphs[1:])): # retain object detection
                if "overall scenario classification" in p.lower(): # ignore after final classification
                    cut_index = j+2
                    break
                elif "classification" not in p.lower() or "3." not in p.lower():
                    continue # ignore other commentary
                else:
                    split = p.split('\n')
                    paragraphs[j+1] = split[0] + "\n" + split[-3][3:] + "\n" + split[-1] # retain object desc, last question and object classification
            paragraphs = paragraphs[:cut_index]
            trim_response = "\n\n".join(paragraphs)

            if word_limit < len(trim_response.split()):
                trim_insufficient += 1

            scene_classification = []
            for cls in cls_list: 
                if "normal" in cls.lower():
                    scene_classification.append(False)
                elif "anomaly" in cls.lower():
                    scene_classification.append(True)
            if np.any(scene_classification):
                print("false_positive! on " + image_names[i] + " in " + exp) 
                incorrect_labels[exp].append(image_names[i]) # store image to ignore in fine-tuning
                incorrect_cls += 1
            
            trim_responses.append(trim_response)
            trim_image_names.append(image_names[i])
    np.savez(os.path.join(target_folder, exp, trim_response_file), vlm_responses = trim_responses, image_names = trim_image_names, vlm_prompt = experiment["vlm_prompt"])

with open(os.path.join(target_folder, "incorrect_labels.json"), "w") as fp:
    json.dump(incorrect_labels, fp)

print(incorrect_format, "answers with incorrect formatting.")
print(incorrect_cls, "false positives.")
print(trim_insufficient, "responses over word limit")


      




