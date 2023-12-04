import os
import numpy as np
import json
import copy 
import random

# --- trim gpt4 responses and correct any false negatives to be true positives. only store the anomalous examples for fine-tuning ---

folder = os.path.dirname(__file__)
scenario_folder = os.path.join(folder, "data", "anomaly_stop_sign")
response_file = "vlm_responses.npz"
trim_nom_response_file = "trim_nom_vlm_responses.npz"
trim_anom_response_file = "trim_anom_vlm_responses.npz"

with open(os.path.join(scenario_folder, "incorrect_labels.json"), "r") as io:
    incorrect_labels = json.load(io)
with open(os.path.join(scenario_folder, "false_negatives.json"), "r") as io:
    false_negatives = json.load(io)

ss_billboard_desc = ['''Stop sign billboard:
1. Is this common to see while driving? Billboards are common, but a billboard with a stop sign may be less common and could potentially confuse a driver thinking it's an actual stop sign.
2. Can this influence the vehicle's behavior? Yes, if the autonomous driving system is not sophisticated enough to distinguish between actual traffic signs and billboards mimicking traffic signs, it may treat the billboard as a real stop sign.
3. Can the vehicle drive safely in its presence? Yes, as long as the vehicle can differentiate between real signs and billboards, it can drive safely.
4. Can this cause the vehicle to make unpredictable or unsafe maneuvers? Potentially yes, if the vehicle mistakes the billboard for a real stop sign, it might stop unexpectedly causing confusion or even an accident.
Classification: Anomaly.''', 
'''Billboard with a stop sign message:
1. Is this common to see while driving? Yes, billboards with various messages, including traffic-related messages, are common.
2. Can this influence the vehicle's behavior? Yes, the vehicle might incorrectly interpret the billboard's stop sign as an actual stop sign, causing it to stop unnecessarily.
3. Can the vehicle drive safely in its presence? As long as the vehicle can distinguish between real traffic signs and billboards, it can drive safely. However, there is the potential for confusion.
4. Can this cause the vehicle to make unpredictable or unsafe maneuvers? Yes, if the vehicle's vision system misinterprets the billboard as a real stop sign, it might make an unnecessary stop, which could be unpredictable and unsafe for following traffic.
Classification: Anomaly.''',
'''Large billboard with a stop sign on it:
1. Is this common to see while driving? Billboards are commonly seen along roads, but they typically advertise products or services and do not usually display traffic signs that could cause confusion.
2. Can this influence the vehicle's behavior? Yes, if the vehicle's vision system is not sophisticated enough to differentiate between a real stop sign and one depicted on a billboard, it could interpret the stop sign on the billboard as a legitimate traffic control device and erroneously stop or slow down.
3. Can the vehicle drive safely in its presence? Yes, if the vehicle's algorithms correctly identify the stop sign as part of an advertisement and not as an actionable traffic sign.
4. Can this cause the vehicle to make unpredictable or unsafe maneuvers? Yes, if misinterpreted as an actual traffic sign, the vehicle might take unnecessary actions like stopping or slowing down when there is no need to do so, potentially causing confusion or a collision hazard.
Classification: Anomaly.''', 
'''Advertisement banner mimicking a stop sign:
1. Is this common to see while driving? No, it's uncommon for advertisements to closely resemble official traffic control devices.
2. Can this influence the vehicle's behavior? Potentially, if the vehicle's recognition system cannot distinguish between actual traffic control devices and similar-looking advertisements.
3. Can the vehicle drive safely in its presence? Uncertain, as the potential for misinterpreting this sign could lead to incorrect driving behavior.
4. Can this cause the vehicle to make unpredictable or unsafe maneuvers? Yes, if the vehicle misinterprets this banner as an official stop sign, it may stop unnecessarily or become confused.
Classification: Anomaly.''']

ss_billboard_obj = ["\n- Stop sign billboard", "\n- Billboard with a stop sign message", "\n- Large billboard with a stop sign on it", "\n- Advertisement banner mimicking a stop sign"]
num_anom = 0

for exp in os.listdir(scenario_folder):
    if "exp" not in exp:
        continue
    experiment = np.load(os.path.join(scenario_folder, exp, response_file))
    
    responses = experiment["vlm_responses"]
    image_names = experiment["image_names"]
    anomaly_detected = experiment["anomaly_detected"]

    trim_nom_responses = []
    trim_nom_image_names = []

    trim_anom_responses = []
    trim_anom_image_names = []

    for i, response in enumerate(responses):
        if image_names[i] in incorrect_labels[exp]:
            continue # ignore incorrect format
        elif anomaly_detected[i]:     
            # trim response for fine-tuning
            paragraphs = response.split("\n\n") # split response into paragraphs

            cut_index = -1
            for it, p in enumerate(paragraphs):
                if "overall scenario classification" in p.lower(): # must be in bc not incorrect label
                    cut_index = it
                    break

            if image_names[i] in false_negatives[exp]:
                index = random.randint(0, len(ss_billboard_obj)-1)
                paragraphs[0] += ss_billboard_obj[index]
                paragraphs.insert(cut_index, ss_billboard_desc[index])
                
                cut_index += 1
                paragraphs[cut_index] = "Overall Scenario Classification: Anomaly."

            for j, p in enumerate(copy.copy(paragraphs[1:])): # retain object detection
                if "overall scenario classification" in p.lower():
                    break
                elif "classification" not in p.lower() or "4." not in p.lower():
                    continue # ignore other commentary
                else:
                    split = p.split('\n')
                    paragraphs[j+1] = split[0] + "\n" + split[-2][3:] + "\n" + split[-1] # retain object desc, last question and object classification
            paragraphs = paragraphs[:cut_index+1]
            trim_anom_response = "\n\n".join(paragraphs)

            trim_anom_responses.append(trim_anom_response)
            trim_anom_image_names.append(image_names[i])
        elif not anomaly_detected[i]:
            # trim response for fine-tuning
            paragraphs = response.split("\n\n") # split response into paragraphs

            cut_index = -1
            for it, p in enumerate(paragraphs):
                if "overall scenario classification" in p.lower(): # must be in bc not incorrect label
                    cut_index = it
                    break

            for j, p in enumerate(copy.copy(paragraphs[1:])): # retain object detection
                if "overall scenario classification" in p.lower():
                    break
                elif "classification" not in p.lower() or "3." not in p.lower():
                    continue # ignore other commentary
                else:
                    split = p.split('\n')
                    paragraphs[j+1] = split[0] + "\n" + split[-3][3:] + "\n" + split[-1] # retain object desc, last question and object classification
            paragraphs = paragraphs[:cut_index+1]
            trim_nom_response = "\n\n".join(paragraphs)

            trim_nom_responses.append(trim_nom_response)
            trim_nom_image_names.append(image_names[i])            

    num_anom += len(trim_anom_image_names)
    np.savez(os.path.join(scenario_folder, exp, trim_anom_response_file), vlm_responses = trim_anom_responses, image_names = trim_anom_image_names, vlm_prompt = experiment["vlm_prompt"])
    np.savez(os.path.join(scenario_folder, exp, trim_nom_response_file), vlm_responses = trim_nom_responses, image_names = trim_nom_image_names, vlm_prompt = experiment["vlm_prompt"])

print(num_anom, "anomalies for fine-tuning.")