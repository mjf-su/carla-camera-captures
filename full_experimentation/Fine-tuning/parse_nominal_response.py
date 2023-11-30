import os
import numpy as np
import json

folder = os.path.dirname(__file__)
target = "nominal_stop_sign"
target_folder = os.path.join(folder, "data", target)
response_file = "vlm_responses.npz"
incorrect_labels = {}

incorrect_format = 0 # answer doesn't include 'overall scenario classification
incorrect_cls = 0 # false positive response

for exp in os.listdir(target_folder):
    if "exp" not in exp:
        continue

    incorrect_labels[exp] = []
    experiment = np.load(os.path.join(target_folder, exp, response_file))
    
    responses = experiment["vlm_responses"]
    image_names = experiment["image_names"]
    for i, response in enumerate(responses):
        if "overall scenario classification:" not in response.lower():
            print("Response on " + image_names[i] + " in " + exp + " needs your review.")
            incorrect_labels[exp].append(image_names[i]) # store image to ignore in fine-tuning
            incorrect_format += 1
        else:
            paragraphs = response.split("\n\n") # split response into paragraphs

            obj_list = []
            cls_list = []
            for p in paragraphs: # process each paragraph in response
                split = p.split(":")
                obj = split[0]
                cls = split[-1]
                obj_list.append(obj)
                cls_list.append(cls)

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

with open(os.path.join(target_folder, "incorrect_labels.json"), "w") as fp:
    json.dump(incorrect_labels, fp)

print(incorrect_format, " answers with incorrect formatting.")
print(incorrect_cls, " false positives.")


      




