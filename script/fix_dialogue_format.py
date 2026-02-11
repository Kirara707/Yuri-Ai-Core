import pandas as pd
import re
import os
from pathlib import Path

def extract_speaker_from_dialogue(dialogue_text):
    """
    从对话内容中推断说话者
    基于对话中的称呼、语气和内容特征
    """
    dialogue_text = dialogue_text.strip()
    
    # 定义角色特征
    speaker_patterns = {
        '玲奈': [
            r'玲奈亲', r'玲奈子', r'我.*玲奈', r'姊姊',
            r'你这女人', r'你这家伙', r'呜呜', r'唉唉',
            r'紫阳花同学', r'小香穗', r'真唯'
        ],
        '紫阳花': [
            r'小玲奈', r'玲奈子', r'小紫', r'紫阳花',
            r'对不起', r'抱歉', r'呵呵', r'嗯',
            r'没关系', r'别担心'
        ],
        '香穗': [
            r'玲奈亲', r'喵', r'小香穗', r'～',
            r'呢', r'啊', r'哇'
        ],
        '真唯': [
            r'玲奈子', r'甘织', r'纱月', r'紫阳花',
            r'没什么', r'是这样吗', r'我知道了'
        ],
        '纱月': [
            r'甘织', r'玲奈子', r'无聊', r'麻烦',
            r'为什么', r'什么意思', r'别说了'
        ],
        '遥奈': [
            r'姊姊', r'玲奈', r'游戏', r'FPS',
            r'白金', r'排位赛'
        ]
    }
    
    # 计算每个角色的匹配分数
    scores = {}
    for speaker, patterns in speaker_patterns.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, dialogue_text))
            score += matches
        scores[speaker] = score
    
    # 返回得分最高的角色，如果没有匹配则返回"未知"
    if max(scores.values()) == 0:
        return "未知"
    
    return max(scores, key=scores.get)

def fix_dialogue_file(file_path):
    """
    修复单个对话文件，添加speaker列
    """
    try:
        # 读取原始文件
        df = pd.read_csv(file_path)
        
        # 检查是否已经有speaker列
        if 'speaker' in df.columns:
            print(f"文件 {file_path} 已经包含speaker列，跳过处理")
            return True
        
        # 添加speaker列
        speakers = []
        for dialogue in df['dialogue']:
            speaker = extract_speaker_from_dialogue(str(dialogue))
            speakers.append(speaker)
        
        df['speaker'] = speakers
        
        # 保存修复后的文件
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"成功修复文件: {file_path}")
        print(f"添加了 {len(speakers)} 个说话者标识")
        
        # 显示说话者分布
        speaker_counts = df['speaker'].value_counts()
        print("说话者分布:")
        for speaker, count in speaker_counts.items():
            print(f"  {speaker}: {count} 行")
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False

def main():
    """
    主函数：修复所有对话文件
    """
    print("开始修复对话数据格式...")
    
    # 对话文件目录
    dialogue_dir = Path("csv/cut_dialogue")
    
    if not dialogue_dir.exists():
        print(f"目录 {dialogue_dir} 不存在")
        return
    
    # 处理所有CSV文件
    csv_files = list(dialogue_dir.glob("*.csv"))
    
    if not csv_files:
        print("没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个文件需要处理")
    
    success_count = 0
    for file_path in csv_files:
        if fix_dialogue_file(file_path):
            success_count += 1
    
    print(f"处理完成！成功修复 {success_count}/{len(csv_files)} 个文件")

if __name__ == "__main__":
    main()
