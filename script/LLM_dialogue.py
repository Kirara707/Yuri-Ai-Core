import os
import re
import sys
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from utils import load_config

# -------------------- 配置 --------------------
config = load_config()
input_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["csv_cut_dialogue_dir"])
output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["csv_prediction_dir"])
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "LLM_dialogue_prediction.csv")  # 具体输出文件名

# 优先从环境变量读取API Key，其次从config.json读取
api_key = os.environ.get('MOONSHOT_API_KEY') or config.get("api_key")
if not api_key:
    print("[ERROR] ❌ 未配置 Moonshot API Key，请在 GUI 侧边栏中输入")
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.moonshot.cn/v1",
)

# -------------------- 限速参数 --------------------
MAX_THREADS = config["LLM_threads_dialogue"]    # 最大并发数
MAX_RETRIES = 1        # 每个请求最大重试次数（减少重试次数）

lock = Lock()          # 写 CSV 时加锁

empty_files = []

# -------------------- 获取文件列表 --------------------
# -------------------- 获取txt_test中的书籍列表（用于过滤） --------------------
txt_test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), config["txt_test_dir"])
txt_test_books = set()
if os.path.exists(txt_test_dir):
    for f in os.listdir(txt_test_dir):
        if f.endswith('.txt'):
            # 提取书籍ID（去掉.txt后缀）
            book_id = f.replace('.txt', '')
            txt_test_books.add(book_id)

print(f"[INFO] txt_test 中有 {len(txt_test_books)} 本书: {txt_test_books}")

# -------------------- 获取文件列表并过滤 --------------------
all_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
# 只保留在txt_test中存在的书籍
file_list = [f for f in all_files if f.replace('.csv', '') in txt_test_books]

if len(all_files) > len(file_list):
    print(f"[FILTER] 过滤掉 {len(all_files) - len(file_list)} 个不在txt_test中的文件")

# -------------------- 保持数字排序逻辑 --------------------
def get_leading_number(fname):
    # 提取文件名开头连续数字
    match = re.match(r'(\d+)', fname)
    return int(match.group(1)) if match else float('inf')

file_list.sort(key=get_leading_number)

# -------------------- 删除已存在的输出文件 --------------------
if os.path.exists(output_file):
    os.remove(output_file)

# -------------------- 处理单个 block 的函数 --------------------
def process_block(filename, block_id, block_df):
    dialogues = block_df["dialogue"].tolist()

    prompt_text = f"""以下是 {len(dialogues)} 条台词，请统计其中体现"百合氛围"（暧昧、亲密、暗恋、浪漫等）的数量，只输出数字，不要额外文字：\n"""
    for i, line in enumerate(dialogues, start=1):
        prompt_text += f"{i}. {line}\n"

    retry_count = 0
    count = 0
    elapsed = 0
    
    while retry_count < MAX_RETRIES:
        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model="kimi-k2-0905-preview",
                messages=[
                    {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，擅长中文和英文对话。"},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.0,
                timeout=30  # 添加超时设置
            )
            
            text = completion.choices[0].message.content
            match = re.search(r'\d+', text)
            count = int(match.group()) if match else 0
            elapsed = time.time() - start_time
            tqdm.write(f"Block {block_id}: 成功处理，百合台词数 {count}")
            break
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate_limit' in error_msg or '429' in error_msg:
                wait = 2 + random.random() * 2  # 增加等待时间
                tqdm.write(f"Block {block_id} 限速，等待 {wait:.1f}s 重试... 错误: {e}")
                time.sleep(wait)
                retry_count += 1
            elif 'timeout' in error_msg or 'connection' in error_msg:
                wait = 1 + random.random()
                tqdm.write(f"Block {block_id} 网络错误，等待 {wait:.1f}s 重试... 错误: {e}")
                time.sleep(wait)
                retry_count += 1
            else:
                tqdm.write(f"Block {block_id} 处理失败: {e}")
                count = 0
                elapsed = time.time() - start_time
                break
    
    # 如果重试次数用完，返回默认值
    if retry_count >= MAX_RETRIES:
        tqdm.write(f"Block {block_id} 重试次数用尽，返回默认值")
        count = 0
        elapsed = time.time() - start_time
    
    return block_id, count, elapsed

# -------------------- 遍历文本 --------------------
with tqdm(total=len(file_list), desc="Processing all texts") as pbar_file:
    for filename in file_list:
        file_path = os.path.join(input_folder, filename)

        # 跳过空文件
        if os.path.getsize(file_path) == 0:
            empty_files.append(filename)
            tqdm.write(f"跳过空文件: {filename}")
            pbar_file.update(1)
            continue

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                empty_files.append(filename)
                tqdm.write(f"跳过空内容文件: {filename}")
                pbar_file.update(1)
                continue
        except pd.errors.EmptyDataError:
            empty_files.append(filename)
            tqdm.write(f" 跳过无法读取的空文件: {filename}")
            pbar_file.update(1)
            continue

        yuri_count = 0
        total_lines = len(df)
        futures = []

        blocks = list(df.groupby("block_id"))

        # -------------------- 多线程处理 blocks --------------------
        # 减少并发数以避免API限速
        actual_threads = min(MAX_THREADS, max(1, len(blocks) // 2))
        tqdm.write(f"文件 {filename} 使用 {actual_threads} 个线程处理 {len(blocks)} 个blocks")
        
        with ThreadPoolExecutor(max_workers=actual_threads) as executor:
            with tqdm(total=len(blocks), desc=f"Processing blocks in {filename}", leave=False) as pbar_block:
                for block_id, block_df in blocks:
                    futures.append(executor.submit(process_block, filename, block_id, block_df))

                for future in as_completed(futures):
                    try:
                        block_id, count, elapsed = future.result(timeout=60)  # 添加超时
                        yuri_count += count
                        tqdm.write(f"Text {filename}, Block {block_id}: {count} 百合台词, 耗时 {elapsed:.2f}s")
                    except Exception as e:
                        tqdm.write(f"处理Block时出错: {e}")
                    finally:
                        pbar_block.update(1)

        yuri_concentration = yuri_count / total_lines if total_lines > 0 else 0

        # -------------------- 写入 CSV --------------------
        row = pd.DataFrame([{
            "text_id": filename.replace(".csv", ""),
            "total_lines": total_lines,
            "yuri_lines": yuri_count,
            "yuri_concentration": round(yuri_concentration, 3)
        }])

        with lock:
            if not os.path.exists(output_file):
                row.to_csv(output_file, index=False, encoding="utf-8-sig")
            else:
                row.to_csv(output_file, index=False, encoding="utf-8-sig", mode='a', header=False)

        pbar_file.update(1)

print(f"百合浓度计算完成！共跳过 {len(empty_files)} 个空文件")
