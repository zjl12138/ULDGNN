import json
import os

def correct_dataset(rootDir, artboard_idx, node_indices, correct_idx):
    which_json = {}
    for id in correct_idx:
        id = int(id)
        graph_idx = node_indices[id].item()
        json_path = os.path.join(rootDir, f"{artboard_idx}-{graph_idx}.json")
        #print("correcting data in file: ", json_path)
        which_json[graph_idx] = {"data":json.load(open(json_path, "r")),"path":json_path}

    for id in correct_idx:
        id =int(id)
        graph_idx = node_indices[id].item()
        fetch_graph_data =  which_json[graph_idx]['data']
        tmp = id - 1
        while tmp >= 0 and node_indices[tmp] == graph_idx:
            tmp = tmp -1
        correct_label_idx = id - (tmp + 1) 
        print(f"correcting {correct_label_idx}-th label in file {which_json[graph_idx]['path']}, ")
        fetch_graph_data['labels'][correct_label_idx] = 1
    
    for id in correct_idx:
        id = int(id)
        graph_idx = node_indices[id].item()
        json.dump(which_json[graph_idx]['data'], open(which_json[graph_idx]['path'],"w"))

