

import argparse
import json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def multi_label_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return {"precision": precision, "recall": recall, "f1": f1}

def nlg_metrics(gen_texts, ref_texts):
    cc = SmoothingFunction()
    bleu1 = np.mean([
        sentence_bleu([ref.split()], hyp.split(), weights=(1,0,0,0), smoothing_function=cc.method1)
        for ref, hyp in zip(ref_texts, gen_texts)
    ])
    bleu4 = np.mean([
        sentence_bleu([ref.split()], hyp.split(), weights=(0.25,0.25,0.25,0.25), smoothing_function=cc.method1)
        for ref, hyp in zip(ref_texts, gen_texts)
    ])
    rouge = Rouge()
    rouge_l_scores = [rouge.get_scores(h, r)[0]["rouge-l"]["f"] for h, r in zip(gen_texts, ref_texts)]
    rouge_l = np.mean(rouge_l_scores)
    return {"BLEU-1": bleu1, "BLEU-4": bleu4, "ROUGE-L": rouge_l}

def main():
    parser = argparse.ArgumentParser(description="ECG报告CE和NLG评估（标签npy，参考报告csv，预测标签用mlb编码）")
    parser.add_argument("--json_result", type=str, default='/data_C/sdb1/lyi/ecg-chat/ECG-Chat-master/llava/llava/eval/report/batch_result_array.json', help="推理结果json文件")
    parser.add_argument("--label_npy", type=str, default='/data_C/sdb1/lyi/ECG-Chat-master/llava/llava/eval/report/label_test_bin.npy', help="真实标签npy (N, C)")
    parser.add_argument("--mlb", type=str, default='/data_C/sdb1/lyi/ked3/control-spiderman-ECGFM-KED-456810e/dataset/shaoxing/mlb.pkl', help="MultiLabelBinarizer类别名pkl文件")
    parser.add_argument("--ref_report", type=str,  default='/data_C/sdb1/lyi/ECG-Chat-master/llava/llava/eval/report/report_test_before_rr.csv', help="人工参考报告csv文件（含 report列）")
    parser.add_argument("--report_column", type=str, default="report", help="人工参考报告的列名")
    parser.add_argument("--model_report_field", type=str, default="model_output", help="模型报告字段名")
    args = parser.parse_args()

    print("加载标签/报告/推理结果...")
    y_true = np.load(args.label_npy) #真实二值化标签
    with open(args.mlb, "rb") as f:
        mlb = pickle.load(f)
    with open(args.json_result, "r", encoding="utf-8") as f:  #推理结果列表
        pred_list = json.load(f)
    #ref_reports = pd.read_csv(args.ref_report)[args.report_column].tolist() #参考报告
    with open(args.ref_report, encoding='utf-8') as f:
        ref_reports = [line.strip() for line in f if line.strip()][1:]  # 跳过表头
    
    

    assert len(pred_list) == len(y_true) == len(ref_reports), "样本数不一致！"

    # 预测标签二值化
    pred_labels_list = []
    gen_texts = []
    for d in pred_list:
        pred_lbls = d.get("prelabels", [])
        if isinstance(pred_lbls, (int, float)):
            pred_lbls = []
        pred_labels_list.append(pred_lbls)
        gen_texts.append(d.get(args.model_report_field, ""))

    y_pred = mlb.transform(pred_labels_list) #预测的二值化标签

    print("==== CE指标 ====")
    ce_result = multi_label_metrics(y_true, y_pred)
    print("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
        ce_result["precision"], ce_result["recall"], ce_result["f1"]
    ))

    print("==== NLG指标 ====")
    nlg_result = nlg_metrics(gen_texts, ref_reports)
    for k, v in nlg_result.items():
        print(f"{k}: {v:.4f}")

    with open("log.txt", "a", encoding="utf-8") as f:
        f.write("==== CE指标 ====\n")
        f.write("Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}\n".format(
            ce_result["precision"], ce_result["recall"], ce_result["f1"]
        ))

        f.write("==== NLG指标 ====\n")
        for k, v in nlg_result.items():
            f.write(f"{k}: {v:.4f}\n")

if __name__ == "__main__":
    main()

