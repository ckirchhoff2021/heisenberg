import os
import json
import numpy as np


root = "datas/aiops/"

curve_labels = ["normal", "offset", "increase", "fluctuate", "spike", "decline"]
recover_labels = ["no_recover", "recovering", "recovered", "normal"] 
label_dict = {
    "down": "decline", "up": "increase", "level_shift": "offset", "fluctuation": "fluctuate", 
    "spike": "spike", "normal": "normal", "offset": "offset", "fluctuate": "fluctuate"
    }

recover_normal_state = 'recovered'

def count_valids():
    datas = json.load(open(os.path.join(root, "files/jsons/seed-valid.json"), 'r'))
    counts = len(datas)
    valids = 0
    
    curves = dict()
    recovers = dict()
    for i in range(counts):
        x = datas[i]
        c1 = x['label1']
        c2 = x['label2']
        
        r1 = x['recover1']
        r2 = x['recover2']
        
        if c1 == c2 and r1 == r2:
            valids += 1
            curves[c1] = curves.get(c1, 0) + 1
            recovers[r1] = recovers.get(r1, 0) + 1
      
    print(curves)  # {'normal': 277, 'level_shift': 150, 'increase': 171, 'fluctuation': 80, 'spike': 74, 'decline': 11}
    print(recovers) # {'normal': 277, 'no_recover': 378, 'recovering': 41, 'recovered': 67}
    print(f'valids: {valids} / {counts} = {valids / counts}') # valids: 763 / 763 = 1.0
    
    
def count_trains():
    datas = json.load(open(os.path.join(root, "files/jsons/seed-train.json"), 'r'))
    curves = dict()
    recovers = dict()
    for x in datas:
        c = x['result_label'] 
        r = x['recover_label']
        
        curves[c] = curves.get(c, 0) + 1
        recovers[r] = recovers.get(r, 0) + 1
        
    print(curves) # {'offset': 446, 'normal': 221, 'spike': 436, 'increase': 522, 'decline': 126}
    print(recovers) # {'no_recover': 970, 'recovering': 426, 'recovered': 354, 'other': 1}
    print(f'counts: {len(datas)}')  # counts: 1751

      
def process_gen_datas():
    images = os.listdir(os.path.join(root, "files/gen"))
    ts = json.load(open(os.path.join(root, "files/gen/series.json"), 'r'))
    datas = list()
    for x in images:
        if not x.endswith(".png"):
            continue
        file_name = x.split('.')[0]
        c = label_dict[file_name.split('_')[0]]
        ts_data = ts[file_name]
        current = np.array(ts_data['current']).astype(np.float32)
        reference = np.array(ts_data['reference']).astype(np.float32)
        
        max_value = max(np.max(current), np.max(reference))
        min_value = min(np.min(current), np.min(reference))
        
        diffs = np.abs(current - reference)
        last_diff = diffs[-4:].mean()
        prev_diff = diffs[-8:-4].mean()
        
        ratio = abs((last_diff - prev_diff) / (prev_diff + 1e-8))
        
        r = 'other'
        if last_diff < (max_value - min_value) * 0.05:
            r = "recovered"
        
        if last_diff < prev_diff and ratio > 0.1:
            r = "recovering"
            
        if last_diff < prev_diff and ratio < 0.1:
            r = "no_recover"
            
        if last_diff > prev_diff:
            r = "no_recover"
        
        if r not in recover_labels or c not in curve_labels:
            print(c, r, x)
            continue

        datas.append({ "curve": c, "recover": r if c != 'normal' else recover_normal_state, "image": x })
    
    with open(os.path.join(root, "files/jsons/argos-dp-gens.json"), 'w') as f:
        json.dump(datas, f, indent=4)        
        

def process_labeled_datas():
    images = os.listdir(os.path.join(root, "files/labeled"))
    valids = json.load(open(os.path.join(root, "files/jsons/seed-valid.json"), 'r'))
    
    valid_datas = list()
    for x in valids:
        c = label_dict.get(x['label1'], x['label1'])
        r = x['recover1']
        image = x['image_file']
        
        if (image not in images) or (c not in curve_labels) or (r not in recover_labels):
            print(c, r, image)
            continue
        
        r = recover_normal_state if r == 'normal' else r
        valid_datas.append({
            "curve": c,
            "recover": r if c != 'normal' else recover_normal_state,
            "image": image,
        })
    
    with open(os.path.join(root, "files/jsons/argos-dp-valids.json"), 'w') as f:
        json.dump(valid_datas, f, indent=4, ensure_ascii=False)        
    print(f'valids: {len(valid_datas)}')
    
    trains = json.load(open(os.path.join(root, "files/jsons/seed-train.json"), 'r'))
    train_datas = list()
    for x in trains:
        thinking = x["reason_summary"]
        output = x["output_text"]
        image = x["image"].split('/')[-1]
        c = x['result_label'] 
        r = x['recover_label']
        
        if (image not in images) or (c not in curve_labels) or (r not in recover_labels):
            print(c, r, image)
            continue
        
        r = recover_normal_state if r == 'normal' else r
        train_datas.append({
            "thinking": thinking,
            "output": output,
            "curve": c,
            "recover": r if c != 'normal' else recover_normal_state,
            "image": image,
        })        
    
    with open(os.path.join(root, "files/jsons/argos-dp-trains.json"), 'w') as f:
        json.dump(train_datas, f, indent=4, ensure_ascii=False)     
    print(f'trains: {len(train_datas)}')
           
    
def extrac_prompt():
    sft_datas = json.load(open(os.path.join(root, "files/jsons/seed-sft.json"), 'r'))
    x = sft_datas[0]
    prompt = x['messages'][0]['content']
    image = "<image>"
    prompt = prompt[len(image):]
    
    print(prompt)
    with open(os.path.join(root, "files/prompts/argos-dp-prompt.md"), 'w') as f:
        f.write(prompt)


def count_argos(file):
    datas = json.load(open(file, 'r'))
    print(f'counts: {len(datas)}')
    curves = dict()
    recovers = dict()
    for x in datas:
        c = x['curve']
        r = x['recover']
        curves[c] = curves.get(c, 0) + 1
        recovers[r] = recovers.get(r, 0) + 1
    print(curves)
    print(recovers)
    

def merge_labeled_gen():
    gens = json.load(open(os.path.join(root, "files/jsons/argos-dp-gens.json"), 'r'))
    trains = json.load(open(os.path.join(root, "files/jsons/argos-dp-trains.json"), 'r'))
    valids = json.load(open(os.path.join(root, "files/jsons/argos-dp-valids.json"), 'r'))
    
    import random
    
    for x in gens:
        x["from"] = "gen"
    for x in trains:
        x["from"] = "labeled"
    for x in valids:
        x["from"] = "labeled"
        
    all_datas = gens + valids
    
    def distribute(datas):
        curves = dict()
        recovers = dict()
        for x in datas:
            c = x['curve']
            r = x['recover']
            curves[c] = curves.get(c, 0) + 1
            recovers[r] = recovers.get(r, 0) + 1
        print(len(datas))
        print(curves)
        print(recovers)
    
    random.shuffle(all_datas)
    valid_datas = all_datas[:800]
    train_datas = all_datas[800:] + trains
    
    distribute(valid_datas)
    distribute(train_datas)
    
    with open(os.path.join(root, "files/jsons/argos-trains.json"), 'w') as f:
        json.dump(train_datas, f, indent=4, ensure_ascii=False)        
    
    with open(os.path.join(root, "files/jsons/argos-valids.json"), 'w') as f:
        json.dump(valid_datas, f, indent=4, ensure_ascii=False)        
    

if __name__ == '__main__':
    # count_valids()
    # count_trains()
    # process_gen_datas()
    # process_labeled_datas()
    # extrac_prompt()
    
    # count_argos(os.path.join(root, "files/jsons/argos-dp-valids.json"))
    # count_argos(os.path.join(root, "files/jsons/argos-dp-trains.json"))
    # count_argos(os.path.join(root, "files/jsons/argos-dp-gens.json"))
    
    merge_labeled_gen()
