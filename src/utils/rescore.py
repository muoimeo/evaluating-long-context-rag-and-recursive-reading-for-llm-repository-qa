import json
import os
import glob
from verify_evidence import calculate_citation_overlap

def rescore_all():
    print("Re-scoring all 30-sample results using updated verify_evidence.py...")
    
    with open('qa_dataset/seed_v0.json', 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    gt_map = {item['id']: item for item in qa_data}
    
    for file_path in glob.glob('results/eval_method_*.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        total_accuracy = 0
        total_citation = 0
        total_latency = 0
        count = 0
        
        for res in results:
            if not res.get('success', False):
                continue
            
            q_id = res['id']
            if q_id not in gt_map:
                continue
                
            gt_evidence = gt_map[q_id].get('evidence', [])
            pred_evidence = res.get('predicted_evidence', [])
            
            # Recalculate citation
            new_citation = calculate_citation_overlap(pred_evidence, gt_evidence)
            res['citation_score'] = new_citation
            
            total_accuracy += res.get('accuracy_score', 0)
            total_citation += new_citation
            total_latency += res.get('latency_sec', 0)
            count += 1
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        if count > 0:
            avg_acc = total_accuracy / count
            avg_cit = total_citation / count
            avg_lat = total_latency / count
            method_name = file_path.split('_')[-1].split('.')[0].upper()
            print(f"--- Method {method_name} (N={count}) ---")
            print(f"Accuracy: {avg_acc:.3f}")
            print(f"Citation: {avg_cit:.3f}")
            print(f"Latency:  {avg_lat:.2f}s")
            print("")

if __name__ == "__main__":
    rescore_all()
