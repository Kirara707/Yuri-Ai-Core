import os, glob, re, torch, pandas as pd
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from utils import load_config

# === 1. 配置加载 ===
cfg = load_config()
CKPT_ROOT = cfg["bert_checkpoint"]
# 输入输出路径
TXT_IN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), cfg["txt_test_cleaned_dir"])
OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), cfg["csv_prediction_dir"], "model_BERT_prediction.csv")
# 【关键】详细日志路径
OUT_DETAIL_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), cfg["csv_prediction_dir"], "bert_detailed_log.csv")

MAX_LEN = cfg.get("bert_max_len", 512)
STRIDE  = cfg.get("bert_stride", 128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 2. 模型加载 ===
BASE = os.path.dirname(os.path.dirname(__file__))
CKPT_ABS = os.path.abspath(os.path.join(BASE, CKPT_ROOT))
tokenizer = BertTokenizerFast.from_pretrained(CKPT_ABS, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(CKPT_ABS, local_files_only=True).to(device)

# === 3. 辅助函数 ===
def extract_golden_sentence(text, max_chars=100):
    """【新增】智能截取金句，防止全文展示"""
    text = text.replace(" ", "").replace("\n", "") # 去空格
    if len(text) <= max_chars: return text
    # 取中间部分
    mid = len(text) // 2
    start = max(0, mid - max_chars // 2)
    end = min(len(text), mid + max_chars // 2)
    return "..." + text[start:end] + "..."

def slide_window(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    windows = []
    for i in range(0, len(tokens), STRIDE):
        chunk = tokens[i:i+MAX_LEN-2]
        if len(chunk) < 30: continue 
        windows.append([tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id])
    return windows

# === 4. 主流程 ===
def main():
    txt_files = sorted(glob.glob(os.path.join(TXT_IN_DIR, "*.txt")))
    if not txt_files: return

    pool = []
    for f in txt_files:
        # 兼容文件名解析
        fname = os.path.basename(f)[:-4]
        if '_' in fname:
            parts = fname.split('_')
            book_id, vol_id = int(parts[0]), int(parts[1])
        else:
            # 如果没有下划线，整个文件名就是book_id，vol_id默认为1
            try:
                book_id = int(fname)
                vol_id = 1
            except ValueError:
                book_id, vol_id = 0, 0
            
        text = open(f, encoding='utf-8', errors='ignore').read()
        for w in slide_window(text):
            pool.append({
                'input_ids': w, 
                'book_id': book_id, 
                'vol_id': vol_id, 
                'raw_filename': fname
            })

    if not pool: return

    # 推理
    ds = Dataset.from_list(pool)
    ds = ds.map(lambda x: {'input_ids': x['input_ids'][:MAX_LEN]}, num_proc=1)
    ds.set_format(type='torch', columns=['input_ids'])
    
    args = TrainingArguments(output_dir=".", per_device_eval_batch_size=16, report_to=[], remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer))
    
    print("开始推理...")
    logits = trainer.predict(ds).predictions
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

    # === 5. 保存详细数据（核心修改）===
    print("正在生成可视化数据...")
    detailed_records = []
    block_counter = {}
    
    for sample, prob in zip(pool, probs):
        fname = sample['raw_filename']
        if fname not in block_counter: block_counter[fname] = 0
        
        # 解码并截取
        raw_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        short_text = extract_golden_sentence(raw_text) # 调用截取函数
        
        detailed_records.append({
            "filename": fname,
            "block_id": block_counter[fname],
            "text": short_text,    # 存截取后的文本
            "score": float(prob),  # 存分数
            "book_id": sample['book_id'],
            "vol_id": sample['vol_id']
        })
        block_counter[fname] += 1
        
    # 保存详细日志
    pd.DataFrame(detailed_records).to_csv(OUT_DETAIL_CSV, index=False, encoding='utf-8-sig')
    print(f"[OK] 详细日志已保存: {OUT_DETAIL_CSV}")

    # 保存原本的卷级汇总（兼容旧逻辑）
    df_vol = pd.DataFrame(detailed_records)
    (df_vol.groupby(['book_id', 'vol_id'], as_index=False)
           .agg(pred_prob=('score', 'mean'))
           .assign(filename=lambda x: x['book_id'].astype(str) + '_' + x['vol_id'].astype(str))
           [['filename', 'book_id', 'vol_id', 'pred_prob']]
           .to_csv(OUT_CSV, index=False, encoding='utf-8-sig'))

if __name__ == '__main__':
    main()
