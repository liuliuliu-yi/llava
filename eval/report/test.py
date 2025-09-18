# 假设原文件为 batch_result1.json，修正后保存为 batch_result1_fixed.json
import json
with open('/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/eval/report/batch_result1_fixed.json', 'r') as f:
    lines = f.readlines()
items = [json.loads(line) for line in lines]
with open('/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/eval/report/batch_result1_fixed1.json', 'w') as f:
    json.dump(items, f, ensure_ascii=False, indent=2)