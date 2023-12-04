import os
import numpy as np
import json
import copy

folder = os.path.dirname(__file__)
target = "nominal_traffic_light"
target_folder = os.path.join(folder, "data", target)
response_file = "vlm_responses.npz"
trim_response_file = "trim_nom_vlm_responses.npz"

with open(os.path.join(target_folder, "incorrect_labels.json"), 'r') as io:
    incorrect_labels = json.load(io)

for exp in os.listdir(target_folder):
    if "exp" not in exp:
        continue

    experiment = np.load(os.path.join(target_folder, exp, response_file))
    
    responses = experiment["vlm_responses"]
    image_names = experiment["image_names"]

    trim_responses = []
    trim_image_names = []
    for i, response in enumerate(responses):
        if image_names[i] in incorrect_labels[exp]:
            continue # ignore incorrect format and false positives
        else:
            paragraphs = response.split("\n\n") # split response into paragraphs

            cut_index = -1
            for j, p in enumerate(copy.copy(paragraphs[1:])): # retain object detection
                if "overall scenario classification" in p.lower(): # ignore after final classification
                    cut_index = j+2
                    break
                elif "classification" not in p.lower() or "4." not in p.lower():
                    continue # ignore other commentary
                else:
                    split = p.split('\n')
                    paragraphs[j+1] = split[0] + "\n" + split[-2][3:] + "\n" + split[-1] # retain object desc, last question and object classification
            paragraphs = paragraphs[:cut_index]
            trim_response = "\n\n".join(paragraphs)
            
            trim_responses.append(trim_response)
            trim_image_names.append(image_names[i])
    assert len(trim_image_names) > 0 and len(trim_responses) > 0
    np.savez(os.path.join(target_folder, exp, trim_response_file), vlm_responses = trim_responses, image_names = trim_image_names, vlm_prompt = experiment["vlm_prompt"])