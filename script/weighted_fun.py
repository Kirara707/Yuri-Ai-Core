import pandas as pd
import re
import os
import json
from utils import load_config
from datetime import datetime

config = load_config()

def read_csv_safe(path, **kwargs):
    """安全读取CSV文件，尝试多种编码"""
    # 先检查文件是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] 必需的文件不存在: {path}")
    
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if enc == "latin1":  # 最后一次尝试才打印错误
                raise Exception(f"[ERROR] 无法读取文件: {path}, 错误: {e}")
            continue
    raise Exception(f"[ERROR] 无法读取文件: {path}, 所有编码都失败")

def call_moonshot_api(prompt: str, max_tokens: int = 2000):
    """调用Moonshot API进行网络信息收集和分析"""
    try:
        from openai import OpenAI
        import os
        
        # 优先从环境变量读取API Key，其次从config.json读取
        api_key = os.environ.get('MOONSHOT_API_KEY') or config.get('api_key')
        if not api_key:
            print("[ERROR] ❌ 未配置 Moonshot API Key，请在 GUI 侧边栏中输入")
            return None
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        )
        
        completion = client.chat.completions.create(
            model="kimi-k2-0905-preview",
            messages=[
                {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，擅长中文文本分析、剧情概括和人物关系提取。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[ERROR] Moonshot API调用失败: {e}")
        return None

def generate_comprehensive_analysis(book_id: str, bert_score: float, dialogue_score: float, verb_score: float, final_score: float):
    """生成综合分析，包括网络信息收集、剧情分析、人物关系图、高光时刻"""
    
    # 确定轻重度等级
    if final_score >= 0.9:
        level = "超重度百合"
    elif final_score >= 0.7:
        level = "重度百合"
    elif final_score >= 0.5:
        level = "中度百合"
    elif final_score >= 0.3:
        level = "轻度百合"
    else:
        level = "微百合/友情向"
    
    # 读取对话和动词数据
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dialogue_file = os.path.join(base_dir, "csv", "cut_dialogue", f"{book_id}.csv")
    verb_file = os.path.join(base_dir, "csv", "cut_verb", f"{book_id}.csv")
    
    dialogues = []
    verbs = []
    
    if os.path.exists(dialogue_file):
        df_dialogue = read_csv_safe(dialogue_file)
        if not df_dialogue.empty and 'dialogue' in df_dialogue.columns:
            dialogues = df_dialogue['dialogue'].tolist()[:10]
    
    if os.path.exists(verb_file):
        df_verb = read_csv_safe(verb_file)
        if not df_verb.empty and 'verb' in df_verb.columns:
            verbs = df_verb['verb'].tolist()[:15]
    
    # 构建网络信息收集和分析提示
    dialogue_sample = "\n".join(dialogues[:5]) if dialogues else ""
    verb_sample = ", ".join(verbs[:10]) if verbs else ""
    
    prompt = f"""
请基于以下数据，为这部百合作品进行深度分析：

书名：书籍{book_id}
轻重度评分：{final_score:.3f} ({level})
BERT评分：{bert_score:.3f}
对话评分：{dialogue_score:.3f}
动词评分：{verb_score:.3f}

对话样本：
{dialogue_sample}

动词样本：
{verb_sample}

请从以下角度进行分析，并利用你的网络知识库补充相关信息：

1. **剧情概括与评语**：
   - 基于文本内容概括剧情
   - 结合网络上的同类作品信息进行对比分析
   - 评价作品的文学价值和情感表达

2. **人物关系图构建**：
   - 从对话中提取主要人物
   - 分析人物之间的关系类型和互动模式
   - 构建人物关系网络

3. **高光时刻选择**：
   - 选择BERT+LLM分析中分数最高的3个片段
   - 详细描述每个高光时刻的剧情
   - 分析为什么这些时刻是情感高潮

请以JSON格式返回：
{{
    "plot_summary": "详细的剧情概括，包含网络信息对比",
    "character_relationships": {{
        "characters": [
            {{"name": "人物名", "role": "角色描述", "importance": "high/medium/low"}}
        ],
        "relationships": [
            {{"from": "人物A", "to": "人物B", "type": "关系类型", "strength": 0.9}}
        ]
    }},
    "highlights": [
        {{"text": "高光时刻1", "reason": "选择理由", "score": 0.95}},
        {{"text": "高光时刻2", "reason": "选择理由", "score": 0.92}},
        {{"text": "高光时刻3", "reason": "选择理由", "score": 0.88}}
    ]
}}
"""
    
    # 调用LLM进行分析
    print(f"[AI] 正在为书籍{book_id}调用LLM进行综合分析...")
    api_response = call_moonshot_api(prompt)
    
    if not api_response:
        # API失败时返回基础分析
        return {
            "book_id": book_id,
            "final_score": final_score,
            "level": level,
            "plot_summary": f"书籍{book_id}的轻重度评分为{final_score:.3f}，属于{level}。由于API调用失败，无法提供详细分析。",
            "character_relationships": {"characters": [], "relationships": []},
            "highlights": []
        }
    
    try:
        # 解析JSON响应
        analysis = json.loads(api_response)
        analysis["book_id"] = book_id
        analysis["final_score"] = final_score
        analysis["level"] = level
        return analysis
    except json.JSONDecodeError:
        # JSON解析失败时返回基础分析
        return {
            "book_id": book_id,
            "final_score": final_score,
            "level": level,
            "plot_summary": f"书籍{book_id}的轻重度评分为{final_score:.3f}，属于{level}。" + api_response[:200],
            "character_relationships": {"characters": [], "relationships": []},
            "highlights": []
        }

def main():
    """主函数"""
    print("[START] 开始简化加权分析流程")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # 0. 获取txt_test中的书籍列表（用于过滤）
    txt_test_dir = os.path.join(base_dir, config["txt_test_dir"])
    txt_test_books = set()
    if os.path.exists(txt_test_dir):
        for f in os.listdir(txt_test_dir):
            if f.endswith('.txt'):
                book_id = f.replace('.txt', '')
                txt_test_books.add(book_id)
    
    print(f"[INFO] txt_test 中有 {len(txt_test_books)} 本书: {txt_test_books}")
    
    # 1. 读取BERT预测结果
    bert_file = os.path.join(base_dir, "csv", "prediction", "model_BERT_prediction.csv")
    df_bert = read_csv_safe(bert_file)
    
    # 2. 读取LLM对话分析结果
    dialogue_file = os.path.join(base_dir, "csv", "prediction", "LLM_dialogue_prediction.csv")
    df_dialogue = read_csv_safe(dialogue_file)
    
    # 3. 读取LLM动词分析结果
    verb_file = os.path.join(base_dir, "csv", "prediction", "LLM_verb_prediction.csv")
    df_verb = read_csv_safe(verb_file)
    
    # 4. 合并数据进行加权计算
    print("[DATA] 开始合并和加权计算...")
    
    # 准备数据框 - 从BERT数据中提取filename和book_id
    if 'pred_prob' in df_bert.columns and 'book_id' in df_bert.columns:
        bert_data = df_bert[['filename', 'book_id', 'pred_prob']].copy()
        # 只保留在txt_test中的书籍
        bert_data = bert_data[bert_data['book_id'].astype(str).isin(txt_test_books)]
        print(f"[FILTER] BERT数据过滤后剩余 {len(bert_data)} 条记录")
    elif 'pred_prob' in df_bert.columns:
        # 如果没有book_id列，从filename提取
        bert_data = df_bert[['filename', 'pred_prob']].copy()
        bert_data['book_id'] = bert_data['filename'].str.split('_').str[0]
    else:
        bert_data = pd.DataFrame()
    
    dialogue_data = pd.DataFrame()
    verb_data = pd.DataFrame()
    
    # 处理对话数据
    if 'yuri_concentration' in df_dialogue.columns:
        dialogue_data = df_dialogue[['text_id', 'yuri_concentration']].rename(columns={'yuri_concentration': 'dialogue_score'})
    elif 'yuri_score' in df_dialogue.columns:
        dialogue_data = df_dialogue[['text_id', 'yuri_score']].rename(columns={'yuri_score': 'dialogue_score'})
    else:
        raise ValueError("[ERROR] LLM对话分析结果缺少必要的评分列")
    # 确保text_id是字符串类型
    dialogue_data['text_id'] = dialogue_data['text_id'].astype(str)
    
    # 处理动词数据
    if 'yuri_concentration' in df_verb.columns:
        verb_data = df_verb[['text_id', 'yuri_concentration']].rename(columns={'yuri_concentration': 'verb_score'})
    elif 'yuri_score' in df_verb.columns:
        verb_data = df_verb[['text_id', 'yuri_score']].rename(columns={'yuri_score': 'verb_score'})
    else:
        raise ValueError("[ERROR] LLM动作分析结果缺少必要的评分列")
    # 确保text_id是字符串类型
    verb_data['text_id'] = verb_data['text_id'].astype(str)
    
    # 合并数据 - 使用book_id作为主键（按书籍聚合）
    merged = bert_data.copy()
    
    # 按book_id分组，计算每本书的平均BERT分数
    merged = merged.groupby('book_id', as_index=False).agg({
        'pred_prob': 'mean',
        'filename': 'first'  # 保留第一个filename作为参考
    })
    merged.rename(columns={'book_id': 'text_id', 'pred_prob': 'bert_score'}, inplace=True)
    
    # 确保text_id是字符串类型
    merged['text_id'] = merged['text_id'].astype(str)
    
    # 合并对话和动词数据（都使用纯book_id）
    merged = merged.merge(dialogue_data, on='text_id', how='left')
    merged = merged.merge(verb_data, on='text_id', how='left')
    
    # 填充缺失值
    merged['dialogue_score'] = merged['dialogue_score'].fillna(0.0)
    merged['verb_score'] = merged['verb_score'].fillna(0.0)
    
    # 计算加权评分
    merged['weighted'] = (
        0.4 * merged['bert_score'] +
        0.35 * merged['dialogue_score'] +
        0.25 * merged['verb_score']
    )
    
    print(f"[OK] 加权计算完成，共处理{len(merged)}条记录")
    
    # 5. 保存加权结果
    weighted_dir = os.path.join(base_dir, "csv", "weighted")
    os.makedirs(weighted_dir, exist_ok=True)
    
    detail_path = os.path.join(weighted_dir, "weighted_detail.csv")
    simple_path = os.path.join(weighted_dir, "weighted.csv")
    
    merged.to_csv(detail_path, index=False, encoding='utf-8-sig')
    merged[['text_id', 'weighted']].to_csv(simple_path, index=False, encoding='utf-8-sig')
    
    print(f"[SAVE] 加权结果已保存到: {detail_path}")
    
    # 6. 生成综合分析（每本书一条记录）
    print("[AI] 开始生成LLM综合分析...")
    analyses = []
    
    for _, row in merged.iterrows():
        # text_id 现在就是纯 book_id
        book_id = str(row['text_id'])
        
        bert_score = row['bert_score']
        dialogue_score = row['dialogue_score']
        verb_score = row['verb_score']
        final_score = row['weighted']
        
        print(f"[AI] 正在为书籍{book_id}调用LLM进行综合分析...")
        analysis = generate_comprehensive_analysis(
            book_id, bert_score, dialogue_score, verb_score, final_score
        )
        analyses.append(analysis)
        print(f"[OK] 已完成书籍{book_id}的综合分析")
    
    # 7. 保存分析结果
    if analyses:
        result_dir = os.path.join(base_dir, "csv", "result")
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存CSV格式
        csv_path = os.path.join(result_dir, "llm_analysis.csv")
        df_analysis = pd.DataFrame(analyses)
        df_analysis.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 保存JSON格式
        json_path = os.path.join(result_dir, "llm_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analyses, f, ensure_ascii=False, indent=2)
        
        print(f"[SAVE] LLM分析结果已保存到: {csv_path}")
        print(f"[SAVE] LLM分析JSON已保存到: {json_path}")
        
        # 8. 生成书籍排行榜
        book_results = []
        for analysis in analyses:
            book_results.append({
                'filename': analysis['book_id'],
                'title': f'书籍{analysis["book_id"]}',
                'weighted': analysis['final_score'],
                'level': analysis['level']
            })
        
        df_books = pd.DataFrame(book_results)
        df_books = df_books.sort_values('weighted', ascending=False)
        
        book_result_path = os.path.join(result_dir, "book_result_sorted.csv")
        df_books.to_csv(book_result_path, index=False, encoding='utf-8-sig')
        
        print(f"[RANK] 书籍排行榜已保存到: {book_result_path}")
        
        # 9. 更新历史排行榜
        history_dir = os.path.join(base_dir, "csv", "history_rank_book")
        os.makedirs(history_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(history_dir, f"rank_{timestamp}.csv")
        df_books.to_csv(history_file, index=False, encoding='utf-8-sig')
        
        # 更新large文件
        large_file = os.path.join(history_dir, "history_rank_book_large.csv")
        if os.path.exists(large_file):
            df_existing = read_csv_safe(large_file)
            df_combined = pd.concat([df_existing, df_books], ignore_index=True)
        else:
            df_combined = df_books
        
        df_combined.to_csv(large_file, index=False, encoding='utf-8-sig')
        print(f"[RANK] 历史排行榜已更新: {large_file}")
    
    print("[COMPLETE] 简化加权分析流程完成!")

if __name__ == "__main__":
    main()
