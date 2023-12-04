import os
import numpy as np
import copy
import json

folder = os.path.dirname(__file__)
scenario = "anomaly_stop_sign"
scenario_folder = os.path.join(folder, "data", scenario)
response_file = "vlm_responses.npz"
# trim_response_file = "trim_vlm_responses.npz"

incorrect_labels = {}
false_negatives = {}

incorrect_format = 0
fp = 0
frame_fn = 0
total_fn = 0

total_nominals = 0
total_anomalies = 0

for exp in os.listdir(scenario_folder):
    if "exp" not in exp:
        continue
    experiment = np.load(os.path.join(scenario_folder, exp, response_file))
    
    responses = experiment["vlm_responses"]
    image_names = experiment["image_names"]
    anomaly_detected = experiment["anomaly_detected"]
    
    incorrect_labels[exp] = []
    false_negatives[exp] = []

    # trim_responses = []
    # trim_image_names = []

    total_anomalies += np.sum(anomaly_detected)
    total_nominals += np.sum(np.invert(anomaly_detected))

    if len(anomaly_detected) > len(image_names):
        break

    for i, response in enumerate(responses):
        if "overall scenario classification:" not in response.lower():
            print("Response on " + image_names[i] + " in " + exp + " needs your review. True label is: ", anomaly_detected[i])
            incorrect_format += 1
            incorrect_labels[exp].append(image_names[i])
        else: # characterize frame-wise accuracy
            paragraphs = response.split("\n\n") # split response into paragraphs

            cls_list = []
            for p in paragraphs: # process each paragraph in response
                split = p.split(":")
                cls = split[-1]
                cls_list.append(cls)

            scene_classification = []
            for cls in cls_list: 
                if "normal" in cls.lower():
                    scene_classification.append(False)
                elif "anomaly" in cls.lower():
                    scene_classification.append(True)

            if anomaly_detected[i]:
                if np.any(scene_classification):
                    continue # correct classification
                else:
                    frame_fn += 1
                    false_negatives[exp].append(image_names[i])
            else:
                if np.any(scene_classification):
                    fp += 1
                    incorrect_labels[exp].append(image_names[i]) # store image to ignore in fine-tuning
                else:
                    continue # correct classification

        # else: # trim response for fine-tuning
        #     paragraphs = response.split("\n\n") # split response into paragraphs

        #     cut_index = -1
        #     for j, p in enumerate(copy.copy(paragraphs[1:])): # retain object detection
        #         if "overall scenario classification" in p.lower(): # ignore after final classification
        #             cut_index = j+2
        #             break
        #         elif "classification" not in p.lower() or "3." not in p.lower():
        #             continue # ignore other commentary
        #         else:
        #             split = p.split('\n')
        #             paragraphs[j+1] = split[0] + "\n" + split[-3][3:] + "\n" + split[-1] # retain object desc, last question and object classification
        #     paragraphs = paragraphs[:cut_index]
        #     trim_response = "\n\n".join(paragraphs)

        #     trim_responses.append(trim_response)
        #     trim_image_names.append(image_names[i])
    
    exp_anomaly = []
    for response, image_name in zip(responses[anomaly_detected], image_names[anomaly_detected]): # parse anomalous examples
        if image_name in incorrect_labels[exp]:
            continue
        else:
            paragraphs = response.split("\n\n") # split response into paragraphs

            cls_list = []
            for p in paragraphs: # process each paragraph in response
                split = p.split(":")
                cls = split[-1]
                cls_list.append(cls)

            frame_anomaly = False
            for cls in cls_list: 
                if "anomaly" in cls.lower():
                    frame_anomaly = True # anomaly detected
                    break
            exp_anomaly.append(frame_anomaly)
    
    if np.any(anomaly_detected): # anomaly exists in experiment
        assert len(exp_anomaly) > 0
        total_fn += 0 if np.any(exp_anomaly) else 1

    # np.savez(os.path.join(scenario_folder, exp, trim_response_file), vlm_responses = trim_responses, image_names = trim_image_names, vlm_prompt = experiment["vlm_prompt"])

with open(os.path.join(scenario_folder, "incorrect_labels.json"), "w") as io:
    json.dump(incorrect_labels, io)
with open(os.path.join(scenario_folder, "false_negatives.json"), "w") as io:
    json.dump(false_negatives, io)

print(incorrect_format, "answers with incorrect formatting.")
print(fp, "false positives.")
print(total_fn, "total false negatives")
print(frame_fn, "framewise false negatives")

print(total_nominals, "total nominal examples")
print(total_anomalies, "total anomalies")

