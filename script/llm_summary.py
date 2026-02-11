import pandas as pd
import os
import json
from openai import OpenAI
from utils import load_config
from typing import Dict, List, Optional
import re
import time

config = load_config()

# 初始化 OpenAI 客户端（使用 Moonshot API）
client = OpenAI(
    api_key=config.get('api_key', config.get('moonshot_api_key')),
    base_url="https://api.moonshot.cn/v1",
)

def read_csv_safe(path, **kwargs):
    for enc in ["utf-8", "utf-8-sig", "gbk", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法读取文件: {path}, 尝试的编码都失败了")

def call_moonshot_api(prompt: str, max_tokens: int = 3000, max_retries: int = 3, enable_search: bool = True) -> Optional[str]:
    """
    调用Moonshot API（使用 openai 库，支持联网搜索）
    
    Args:
        prompt: 输入提示
        max_tokens: 最大token数
        max_retries: 最大重试次数
        enable_search: 是否启用联网搜索（通过$web_search标记）
        
    Returns:
        API响应文本或None
    """
    for retry in range(max_retries):
        try:
            print(f"  [API] 调用中... (尝试 {retry + 1}/{max_retries})" + (" [联网搜索已启用]" if enable_search else ""))
            
            # 构建消息 - Moonshot使用$web_search标记启用搜索
            if enable_search:
                system_content = "你是一个专业的轻小说剧情分析师，专门深度解读百合作品的剧情发展、人物情感和关系演变。你可以使用 $web_search 功能搜索网络资料。请只输出JSON格式，不要输出其他内容。"
                # 在用户提示中添加搜索标记
                search_prompt = f"$web_search(请先搜索该书籍的相关资料)\n\n{prompt}"
            else:
                system_content = "你是一个专业的轻小说剧情分析师，专门深度解读百合作品的剧情发展、人物情感和关系演变。请只输出JSON格式，不要输出其他内容。"
                search_prompt = prompt
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": search_prompt}
            ]
            
            completion = client.chat.completions.create(
                model="moonshot-v1-128k",  # 使用支持搜索的模型
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens,
                timeout=300
            )
            
            result = completion.choices[0].message.content.strip()
            print(f"  [API] 调用成功!")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"  [API] 错误: {e}")
            
            if 'rate_limit' in error_msg or '429' in error_msg:
                wait_time = 10 * (retry + 1)
                print(f"  [API] 限速，等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
            elif 'timeout' in error_msg:
                wait_time = 10 * (retry + 1)
                print(f"  [API] 超时，等待 {wait_time}s 后重试...")
                time.sleep(wait_time)
            else:
                time.sleep(5)
    
    print(f"  [API] 重试 {max_retries} 次后仍失败")
    return None

def generate_llm_analysis(book_title: str, full_text: str, dialogues: List[str], verbs: List[str], 
                         bert_score: float, dialogue_score: float, verb_score: float, 
                         final_score: float) -> Dict:
    """
    生成LLM分析
    
    Args:
        book_title: 书名
        full_text: 完整原文（如果可用）
        dialogues: 对话列表（备用）
        verbs: 动词列表（备用）
        bert_score: BERT评分
        dialogue_score: 对话评分
        verb_score: 动词评分
        final_score: 最终评分
        
    Returns:
        包含分析结果的字典
    """
    # 确定轻重度等级
    if final_score >= 0.8:
        level = "超重度"
    elif final_score >= 0.6:
        level = "重度"
    elif final_score >= 0.4:
        level = "中度"
    elif final_score >= 0.2:
        level = "轻度"
    else:
        level = "微百合"
    
    # 准备文本内容
    if full_text:
        # 如果有完整原文，取前8000字（避免超过token限制）
        text_sample = full_text[:8000] if len(full_text) > 8000 else full_text
        text_info = f"【完整文本前8000字】：\n{text_sample}\n\n（文本总长度：{len(full_text)} 字）"
    else:
        # 否则使用CSV片段
        dialogue_sample = "\n".join(dialogues[:50]) if dialogues else "（无对话数据）"
        verb_sample = ", ".join(verbs[:100]) if verbs else "（无动作数据）"
        text_info = f"【对话片段】：\n{dialogue_sample}\n\n【动作关键词】：\n{verb_sample}"
    
    # 构建分析提示 - 启用联网搜索，要求更详细具体的分析
    prompt = f"""
📚 **任务说明**：请对以下百合轻小说进行深度剧情分析和人物关系解析。

书名：《{book_title}》

---

**第一步：联网搜索资料（STFW原则）**

请使用 $web_search 搜索以下信息：
1. 书籍的完整剧情介绍、分卷内容、故事梗概
2. 主要角色的详细信息（全名、性格特征、人物设定、成长轨迹）
3. 重要情节转折点、经典场景、名场面
4. 角色关系发展脉络、情感线索
5. 读者评价、经典台词、情感分析

搜索关键词："{book_title} 剧情"、"{book_title} 角色"、"{book_title} 人物关系"、"{book_title} 百合 分析"

---

**第二步：结合实际文本验证和补充**

提供的文本片段（用于验证搜索结果的准确性和提取原文引用）：

{text_info}

【技术评分】：BERT={bert_score:.3f} | 对话={dialogue_score:.3f} | 动作={verb_score:.3f} | 综合={final_score:.3f}（{level}）

---

**第三步：输出详细分析（要求）**

**1. 剧情概括（400-600字）**
   ✅ 必须包含：
   - 故事发生的具体背景（时间、地点、社会环境）
   - 主角们的初次相遇场景和契机
   - 情感发展的3-5个关键阶段（陌生→熟悉→暧昧→确认等）
   - 主要冲突和矛盾点
   - 高潮场景的具体描述
   - 结局走向或开放式结局的解读
   - 百合元素的具体体现（肢体接触、对话暧昧、心理描写等）
   
   ❌ 避免：笼统概括、缺少具体情节、过于简短

**2. 人物关系分析**
   A) **文字描述（200-300字）**：
      - 每个主要角色的性格特点、行为模式、内心世界
      - 角色之间的互动方式、对话特点
      - 关系的微妙变化和情感张力
      - 双方在关系中的角色定位（主动/被动、攻/受等）
   
   B) **结构化数据**：
      - characters: 列出3-5个主要角色，包含详细性格描述（不要只写关键词）
      - relationships: 详细说明关系类型、发展过程、亲密度评分依据

**3. 高光时刻（3-5个场景）**
   每个场景必须包含：
   
   **text（场景描述，150-250字）**：
   - 如果文本中有原文片段，优先引用或改写原文
   - 详细描述场景的环境、氛围、人物动作、对话
   - 生动还原场景细节（神态、语气、肢体接触等）
   - 例如："夕阳下的屋顶，A轻轻握住B冰凉的手指，'我不想再逃避了'，声音颤抖却坚定..."
   
   **reason（情感分析，80-120字）**：
   - 分析该场景在情感发展中的转折意义
   - 解读人物心理状态的变化
   - 说明这个场景为何是"高光时刻"
   - 例如："这是她们从'若即若离'到'心意相通'的临界点：通过身体接触打破心理隔阂，用直白的告白替代暧昧的试探，标志着关系质变的开始。"

---

**输出JSON格式**：
{{
    "plot_summary": "详细的剧情概括，包含具体情节、场景、对话等细节（400-600字）",
    "character_relationships": {{
        "description": "深入的人物关系文字分析，包含性格、互动、情感张力等（200-300字）",
        "characters": [
            {{"name": "角色完整姓名", "role": "主角", "traits": "详细性格描述，不要只写关键词，要写完整句子"}},
            {{"name": "角色完整姓名", "role": "主角", "traits": "详细性格描述"}},
            {{"name": "其他角色", "role": "配角", "traits": "性格描述"}}
        ],
        "relationships": [
            {{"source": "角色A", "target": "角色B", "type": "关系类型（恋人/暗恋/挚友/师徒等）", "strength": 0.95, "description": "详细关系描述，必须包含：两人如何认识、关系如何发展、关键互动场景、情感变化轨迹（100-150字）"}},
            {{"source": "角色B", "target": "角色A", "type": "相互关系", "strength": 0.90, "description": "关系描述，包含发展过程和互动细节（100-150字）"}}
        ]
    }},
    "highlights": [
        {{
            "text": "【必须包含超详细的场景描写，300-500字】要像写小说一样完整还原这个场景：包含详细的环境描写（时间、地点、天气、光线、氛围）、人物的外貌和表情、具体的动作和姿态、完整的对话内容、细腻的心理活动、情绪的细微变化。不要只写梗概，要让读者能身临其境地感受每一个细节。示例：'那是一个阴雨绵绵的午后，教室里只剩下主角一个人在整理书包。突然，门口传来轻微的脚步声，主角抬起头，看见佐伯同学站在门口，湿漉漉的头发贴在脸颊上，校服的袖口还在滴水。佐伯没有说话，只是静静地看着主角，眼神中带着复杂的情绪。主角放下书包，缓缓走向佐伯...'",
            "reason": "【超详细的百合情感分析，200-300字】不仅要分析这个场景的情感意义，还要深入探讨：(1)这个场景如何推动人物关系的发展 (2)体现了哪些百合元素和情感张力 (3)场景中的象征意义和隐喻手法 (4)与之前剧情的呼应和对后续的铺垫 (5)这个场景在整部作品中的独特价值。要像专业评论家一样深入剖析。",
            "context": "【详细的背景说明，150-200字】要详细交代：(1)这个场景发生在故事的第几章/哪个阶段 (2)在此之前两人经历了什么重要事件 (3)当时人物之间的关系处于什么状态（亲密/疏远/矛盾等）(4)两人各自的心理状态和情绪 (5)场景发生的具体时间、地点、周围环境。要让读者完全理解这个场景的重要性。",
            "characters_involved": ["主角名字", "佐伯同学"],
            "yuri_intensity": 0.85
        }},
        {{
            "text": "【超详细场景描写，300-500字，务必像写小说一样生动详细】",
            "reason": "【深度百合分析，200-300字，从多个维度分析】",
            "context": "【详细背景故事，150-200字，包含时间地点人物状态】",
            "characters_involved": ["角色B", "角色C"],
            "yuri_intensity": 0.90
        }},
        {{
            "text": "【详细场景描写，150-250字以上，务必包含足够信息量】",
            "reason": "【情感分析，100-150字以上】",
            "context": "【背景故事，80-120字】",
            "characters_involved": ["角色A", "角色C"],
            "yuri_intensity": 0.80
        }}
    ]
}}

⚠️ **对 character_relationships 字段的严格要求**：
1. **必须包含 relationships 数组**，至少要有 2-3 条关键的人物关系
2. **每条 relationship 必须包含**：
   - source 或 from: 人物A的名字（两种字段名都可以）
   - target 或 to: 人物B的名字（两种字段名都可以）
   - type: 关系类型（不要空值，比如"百合恋爱关系"或"相互暗恋"）
   - strength: 0-1 之间的数值，表示关系强度
   - description: 100-150字的详细关系描述，必须包含：两人如何认识、关系如何发展、关键互动场景、情感变化轨迹、关系的特殊之处
3. **relationships 数组不能为空**，必须至少有 1 条关系数据
4. **characters 数组也必须有**，列出主要人物及其详细特征

⚠️ **对 highlights 字段的严格要求**：
1. **text 字段务必是 300-500 字以上的超详细场景描写**：不要只摘录原文的一句话！要基于原文和你的理解，完整还原整个场景的前后经过、环境描写、人物外貌、对话内容、动作细节、心理活动、情绪变化等全部要素。要像写小说一样详细生动，让读者能身临其境地感受到这个场景的每一个细节和情感张力。
2. **reason 字段必须是 200-300 字以上的深度百合分析**：不仅要阐述这个场景的情感意义和为什么是高光时刻，还要深入分析：(1)涉及的人物关系发生了什么微妙变化 (2)体现了哪些百合元素（暗示、张力、情感共鸣等）(3)场景中的象征意义和隐喻 (4)对整部作品情节和情感线的推动作用 (5)与其他经典百合作品的相似或独特之处。
3. **context 字段必须是 150-200 字的详细背景说明**：要详细交代：(1)这个场景发生在故事的哪个阶段 (2)之前发生了什么重要事件导致这个场景 (3)当时人物的关系处于什么状态 (4)人物的情绪和心理状态 (5)场景发生的具体时间、地点、环境氛围。
4. **characters_involved 必须是非空列表**，列出涉及的所有人物名字（至少2个）
5. **yuri_intensity 必须是 0-1 之间的数值**，精确反映场景的百合强度
6. **优先使用搜索到的权威资料获取完整剧情和人物信息**
7. **从提供的文本中提取原文引用作为基础，但必须大幅扩充和补充细节**
8. **确保分析极其具体、详细、有深度，绝对避免笼统和简短的描述**
9. **高光时刻要像专业小说评论家一样生动还原场景，让读者能完全沉浸其中**
"""
    
    # 调用API（启用联网搜索）
    api_response = call_moonshot_api(prompt, max_tokens=3000, enable_search=True)
    if not api_response:
        # 如果API调用失败，返回基础分析（兼容新旧格式）
        return {
            "plot_summary": f"《{book_title}》是一部百合题材的轻小说作品。由于API调用失败，无法提供详细剧情分析。",
            "character_relationships": {
                "description": "由于API调用失败，无法提供人物关系分析。",
                "characters": [],
                "relationships": []
            },
            "highlights": [
                {
                    "text": "无法加载场景数据",
                    "reason": "API调用失败",
                    "context": "",
                    "characters_involved": [],
                    "yuri_intensity": 0.5
                }
            ]
        }
    
    try:
        # 尝试解析JSON响应
        analysis = json.loads(api_response)
        
        # 数据验证和修复
        if 'character_relationships' in analysis:
            rel_data = analysis['character_relationships']
            # 确保 relationships 字段存在
            if 'relationships' not in rel_data or not rel_data['relationships']:
                # 如果没有 relationships，创建一个简单的默认关系
                print("[WARNING] 缺少 relationships 字段，生成默认值")
                rel_data['relationships'] = []
                if 'characters' in rel_data and len(rel_data['characters']) >= 2:
                    # 从 characters 列表生成一个基本的关系
                    chars = rel_data['characters']
                    if len(chars) >= 2:
                        rel_data['relationships'].append({
                            "source": chars[0].get('name', '角色A'),
                            "target": chars[1].get('name', '角色B'),
                            "type": "百合关系",
                            "strength": 0.85,
                            "description": f"{chars[0].get('name', '角色A')} 与 {chars[1].get('name', '角色B')} 之间存在深厚的情感联系，是本作的核心人物关系线。"
                        })
        
        return analysis
    except json.JSONDecodeError:
        # 如果JSON解析失败，尝试提取文本内容
        return {
            "plot_summary": f"《{book_title}》是一部百合题材的轻小说作品。" + (api_response[:150] + "..." if len(api_response) > 150 else ""),
            "character_relationships": {
                "description": "由于响应格式问题，无法提供详细的人物关系分析。",
                "characters": [],
                "relationships": []
            },
            "highlights": []
        }

def process_book_analysis():
    """处理所有书籍的LLM分析"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # 从多个来源获取书籍列表
    books_to_analyze = {}  # {book_id: {'title': xxx, 'final_score': xxx, ...}}
    
    # 来源1：book_result_sorted.csv（如果存在）
    book_result_path = os.path.join(base_dir, "csv", "result", "book_result_sorted.csv")
    if os.path.exists(book_result_path):
        try:
            df_books = read_csv_safe(book_result_path)
            for _, row in df_books.iterrows():
                book_id = str(row.get('filename', row.get('book_id', '')))
                if book_id and book_id != 'nan':
                    books_to_analyze[book_id] = {
                        'title': row.get('title', f'书籍{book_id}'),
                        'final_score': row.get('weighted', row.get('final', 0)),
                        'bert_score': row.get('norm_bert', row.get('bert', 0)),
                        'dialogue_score': row.get('norm_dialogue', row.get('dialogue', 0)),
                        'verb_score': row.get('norm_verb', row.get('verb', 0)),
                    }
        except Exception as e:
            print(f"[WARN] 读取 book_result_sorted.csv 失败: {e}")
    
    # 来源2：bert_detailed_log.csv（补充缺失的书籍）
    bert_log_path = os.path.join(base_dir, "csv", "prediction", "bert_detailed_log.csv")
    if os.path.exists(bert_log_path):
        try:
            df_bert = read_csv_safe(bert_log_path)
            for filename in df_bert['filename'].unique():
                book_id = str(int(filename)) if pd.notna(filename) else None
                if book_id and book_id not in books_to_analyze:
                    # 计算该书籍的平均BERT分数
                    book_data = df_bert[df_bert['filename'] == filename]
                    avg_score = book_data['score'].mean() if 'score' in book_data.columns else 0
                    books_to_analyze[book_id] = {
                        'title': f'书籍{book_id}',  # 默认标题
                        'final_score': avg_score,
                        'bert_score': avg_score,
                        'dialogue_score': 0,
                        'verb_score': 0,
                    }
        except Exception as e:
            print(f"[WARN] 读取 bert_detailed_log.csv 失败: {e}")
    
    # 来源3：history_rank_book（获取书籍标题）
    rank_path = os.path.join(base_dir, "csv", "history_rank_book", "history_rank_book_large.csv")
    if os.path.exists(rank_path):
        try:
            df_rank = read_csv_safe(rank_path)
            for _, row in df_rank.iterrows():
                aid = row.get('aid')
                if pd.notna(aid):
                    book_id = str(int(aid))
                    if book_id in books_to_analyze:
                        title = row.get('title', '')
                        if title and str(title) != 'nan':
                            books_to_analyze[book_id]['title'] = str(title)
        except Exception as e:
            print(f"[WARN] 读取 history_rank_book_large.csv 失败: {e}")
    
    if not books_to_analyze:
        print("[ERROR] 没有找到任何需要分析的书籍")
        return
    
    print(f"[INFO] 找到 {len(books_to_analyze)} 本书籍需要分析")
    
    # 读取对话和动词数据目录
    dialogue_dir = os.path.join(base_dir, "csv", "cut_dialogue")
    verb_dir = os.path.join(base_dir, "csv", "cut_verb")
    
    # 加载已有分析结果（增量处理）
    json_path = os.path.join(base_dir, "csv", "result", "llm_analysis.json")
    existing_analyses = []
    existing_ids = set()
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_analyses = json.load(f)
                existing_ids = {str(a.get('book_id', '')) for a in existing_analyses}
            print(f"[INFO] 已有 {len(existing_analyses)} 条分析记录")
        except:
            pass
    
    analyses = existing_analyses.copy()
    
    for book_id, book_info in books_to_analyze.items():
        # 跳过已分析的书籍
        if book_id in existing_ids:
            print(f"[SKIP] 书籍 {book_id} 已有分析记录")
            continue
        
        book_title = book_info['title']
        final_score = book_info['final_score']
        bert_score = book_info['bert_score']
        dialogue_score = book_info['dialogue_score']
        verb_score = book_info['verb_score']
        
        # 优先读取完整原文
        full_text = ""
        text_sources = [
            os.path.join(base_dir, "txt_test", f"{book_id}.txt"),
            os.path.join(base_dir, "assets", "txt_test_cleaned", f"{book_id}.txt"),
            os.path.join(base_dir, "txt", f"{book_id}.txt"),
        ]
        
        for text_path in text_sources:
            if os.path.exists(text_path):
                try:
                    with open(text_path, 'r', encoding='utf-8') as f:
                        full_text = f.read()
                    print(f"[INFO] 从 {text_path} 读取完整文本 ({len(full_text)} 字)")
                    break
                except:
                    try:
                        with open(text_path, 'r', encoding='gbk') as f:
                            full_text = f.read()
                        print(f"[INFO] 从 {text_path} 读取完整文本 ({len(full_text)} 字)")
                        break
                    except Exception as e:
                        print(f"[WARN] 读取文本文件失败 {text_path}: {e}")
        
        # 如果没有原文，则使用CSV文件
        dialogues = []
        verbs = []
        
        if not full_text:
            print(f"[WARN] 未找到书籍 {book_id} 的原文，使用CSV片段")
            # 读取对话数据
            dialogue_file = os.path.join(dialogue_dir, f"{book_id}.csv")
            if os.path.exists(dialogue_file):
                try:
                    df_dialogue = read_csv_safe(dialogue_file)
                    dialogues = df_dialogue['dialogue'].tolist()[:50]  # 增加到50条
                except Exception as e:
                    print(f"[WARN] 读取对话文件失败 {dialogue_file}: {e}")
            
            # 读取动词数据
            verb_file = os.path.join(verb_dir, f"{book_id}.csv")
            if os.path.exists(verb_file):
                try:
                    df_verb = read_csv_safe(verb_file)
                    verbs = df_verb['verb'].tolist()[:100]  # 增加到100个
                except Exception as e:
                    print(f"[WARN] 读取动词文件失败 {verb_file}: {e}")
        
        # 生成LLM分析
        print(f"[PROCESS] 正在分析书籍 {book_id}: {book_title}")
        analysis = generate_llm_analysis(
            book_title, full_text, dialogues, verbs,
            bert_score, dialogue_score, verb_score, final_score
        )
        
        analysis['book_id'] = book_id
        analysis['book_title'] = book_title
        analysis['final_score'] = final_score
        analysis['level'] = (
            "超重度" if final_score >= 0.8 else
            "重度" if final_score >= 0.6 else
            "中度" if final_score >= 0.4 else
            "轻度" if final_score >= 0.2 else
            "微百合"
        )
        
        analyses.append(analysis)
        print(f"[OK] 已完成书籍 {book_id} 的分析")
    
    # 保存分析结果
    if analyses:
        analysis_df = pd.DataFrame(analyses)
        analysis_path = os.path.join(base_dir, "csv", "result", "llm_analysis.csv")
        analysis_df.to_csv(analysis_path, index=False, encoding='utf-8-sig')
        print(f"[SAVE] LLM分析已保存到: {analysis_path}")
        
        # 同时保存JSON格式供GUI使用
        json_path = os.path.join(base_dir, "csv", "result", "llm_analysis.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analyses, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] LLM分析JSON已保存到: {json_path}")
        
    else:
        print("[ERROR] 没有生成任何分析")

def generate_single_book_analysis(book_id: str) -> Optional[Dict]:
    """为单本书生成分析"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    # 读取书籍结果
    book_result_path = os.path.join(base_dir, "csv", "result", "book_result_sorted.csv")
    if not os.path.exists(book_result_path):
        return None
    
    df_books = read_csv_safe(book_result_path)
    book_row = df_books[df_books['filename'] == book_id]
    
    if book_row.empty:
        return None
    
    book_row = book_row.iloc[0]
    book_title = book_row.get('title', f'书籍{book_id}')
    
    # 获取评分数据
    bert_score = book_row.get('bert', 0)
    dialogue_score = book_row.get('dialogue', 0)
    verb_score = book_row.get('verb', 0)
    final_score = book_row.get('final', 0)
    
    # 读取对话和动词数据
    dialogue_dir = os.path.join(base_dir, "csv", "cut_dialogue")
    verb_dir = os.path.join(base_dir, "csv", "cut_verb")
    
    dialogues = []
    verbs = []
    
    # 读取对话数据
    dialogue_file = os.path.join(dialogue_dir, f"{book_id}.csv")
    if os.path.exists(dialogue_file):
        try:
            df_dialogue = read_csv_safe(dialogue_file)
            dialogues = df_dialogue['dialogue'].tolist()[:10]
        except Exception as e:
            print(f"[WARN] 读取对话文件失败 {dialogue_file}: {e}")
    
    # 读取动词数据
    verb_file = os.path.join(verb_dir, f"{book_id}.csv")
    if os.path.exists(verb_file):
        try:
            df_verb = read_csv_safe(verb_file)
            verbs = df_verb['verb'].tolist()[:20]
        except Exception as e:
            print(f"[WARN] 读取动词文件失败 {verb_file}: {e}")
    
    # 生成分析
    analysis = generate_llm_analysis(
        book_title, dialogues, verbs,
        bert_score, dialogue_score, verb_score, final_score
    )
    
    analysis['book_id'] = book_id
    analysis['book_title'] = book_title
    analysis['final_score'] = final_score
    
    return analysis

if __name__ == "__main__":
    process_book_analysis()
