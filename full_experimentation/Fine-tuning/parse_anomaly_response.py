import os
import numpy as np
import time

folder = os.path.dirname(__file__)
target = "nominal_stop_sign"
target_folder = os.path.join(folder, "data", target)
response_file = "gpt4v_responses_newFormat.npz"

for exp in os.listdir(target_folder):
    if "exp" not in exp:
        continue
    experiment = np.load(os.path.join(target_folder, exp, "image_data", response_file))
    
    responses = experiment["gpt4v_responses"]
    image_names = experiment["image_names"]
    for i, response in enumerate(responses):
        if "overall scenario classification:" not in response.lower() and "overall scene classification:" not in response.lower():
            print("Response on " + image_names[i] + " in " + exp + " needs your review.")
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




