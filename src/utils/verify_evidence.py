import json
import os

def preview_evidence(json_path, repo_root):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for item in data:
        print(f"\n[ID: {item['id']}] Question: {item['question']}")
        for ev in item['evidence']:
            file_path = os.path.join(repo_root, ev['file'])
            if os.path.exists(file_path):
                with open(file_path, 'r') as f_code:
                    lines = f_code.readlines()
                    snippet = lines[ev['line_start']-1 : ev['line_end']]
                    print(f"--- Source: {ev['file']} (Lines {ev['line_start']}-{ev['line_end']}) ---")
                    print("".join(snippet[:5]) + "...")
            else:
                print(f"!! File not found: {file_path}")