#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合推理启动脚本
基于GUI中的推理流程，整合所有步骤到一个脚本中
支持完整的百合轻小说分析流程
"""

import subprocess
import sys
import os
import time
import json
from datetime import datetime
from typing import List, Tuple, Optional
import pandas as pd

# ================== 配置区 ==================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PYTHON_EXE = sys.executable

# 推理流程步骤定义（与GUI保持一致）
SCRIPTS_RUN_MODE = [
    ("Step 1", "清洗文本", "script/clean_txt.py"),
    ("Step 2", "BERT推理", "script/model_BERT_infer.py"),
    ("Step 3", "对话切分", "script/dialogue_cut.py"),
    ("Step 4", "剔除台词", "script/drop_dialogue.py"),
    ("Step 5", "动词提取", "script/verb_cut.py"),
    ("Step 6", "LLM对话", "script/LLM_dialogue.py"),
    ("Step 7", "LLM动作", "script/LLM_verb.py"),
    ("Step 8", "动词归一", "script/verb_normalizer.New.py"),
    ("Step 9", "对话归一", "script/dialogue_normalizer.New.py"),
    ("Step 10", "加权计算+LLM分析", "script/weighted_fun.py"),
    ("Step 11", "数据合并", "script/merge.py"),
    ("Step 12", "书级匹配", "script/book_result_match.py"),
    ("Step 13", "书级排序", "script/sort_book.py"),
    ("Step 14", "更新书榜", "script/result_book_merge.py"),
]

# 颜色输出
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_step_header(step_num: int, total_steps: int, name: str, description: str):
    """打印步骤标题"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}[{step_num}/{total_steps}] {name}{Colors.END}")
    print(f"{Colors.YELLOW}{description}{Colors.END}")
    print(f"{Colors.CYAN}{'='*60}{Colors.END}")

def print_success(message: str):
    """打印成功消息"""
    print(f"{Colors.GREEN}[OK] {message}{Colors.END}")

def print_error(message: str):
    """打印错误消息"""
    print(f"{Colors.RED}[ERROR] {message}{Colors.END}")

def print_warning(message: str):
    """打印警告消息"""
    print(f"{Colors.YELLOW}[WARN] {message}{Colors.END}")

def print_info(message: str):
    """打印信息消息"""
    print(f"{Colors.BLUE}[INFO] {message}{Colors.END}")

def check_prerequisites():
    """检查运行前提条件"""
    print_info("检查运行环境...")
    
    # 检查Python环境
    try:
        import torch
        import transformers
        import pandas
        import openai
        print_success("Python环境检查通过")
    except ImportError as e:
        print_error(f"缺少必要的Python包: {e}")
        return False
    
    # 检查配置文件
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    if not os.path.exists(config_path):
        print_error("配置文件config.json不存在")
        return False
    
    # 检查API密钥
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            api_key = config.get('api_key') or config.get('moonshot_api_key')
            if not api_key:
                print_error("未配置API密钥")
                return False
        print_success("API配置检查通过")
    except Exception as e:
        print_error(f"配置文件读取失败: {e}")
        return False
    
    # 检查模型文件
    model_path = config.get('bert_checkpoint', './models/checkpoint-47200')
    model_abs_path = os.path.join(PROJECT_ROOT, model_path)
    if not os.path.exists(model_abs_path):
        print_error(f"BERT模型文件不存在: {model_abs_path}")
        return False
    
    print_success("所有前提条件检查通过")
    return True

def run_script(script_path: str, step_info: Tuple[str, str, str], log_file=None) -> bool:
    """运行单个脚本"""
    step_num, name, description = step_info
    script_full_path = os.path.join(PROJECT_ROOT, script_path)
    
    print_step_header(step_num.split()[1], len(SCRIPTS_RUN_MODE), name, description)
    
    if not os.path.exists(script_full_path):
        print_error(f"脚本文件不存在: {script_full_path}")
        return False
    
    try:
        start_time = time.time()
        print_info(f"开始执行: {script_path}")
        
        # 写入步骤开始标记到日志文件
        if log_file:
            step_start_time = datetime.now().strftime('%H:%M:%S')
            log_file.write(f"\n[{step_start_time}] 开始执行: {step_num} - {name}\n")
            log_file.flush()
        
        # 使用subprocess运行脚本
        process = subprocess.Popen(
            [PYTHON_EXE, script_full_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
            cwd=PROJECT_ROOT
        )
        
        # 实时输出日志
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output = output.strip()
                output_lines.append(output)
                print(f"  {output}")
                
                # 写入日志文件
                if log_file:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    log_file.write(f"[{timestamp}] {output}\n")
                    log_file.flush()
        
        # 等待进程完成
        return_code = process.poll()
        elapsed_time = time.time() - start_time
        
        # 写入步骤完成标记到日志文件
        if log_file:
            step_end_time = datetime.now().strftime('%H:%M:%S')
            if return_code == 0:
                log_file.write(f"[{step_end_time}] [OK] {step_num} 执行完成 (耗时: {elapsed_time:.2f}秒)\n")
            else:
                log_file.write(f"[{step_end_time}] [ERROR] {step_num} 执行失败 (返回码: {return_code})\n")
            log_file.flush()
        
        if return_code == 0:
            print_success(f"执行完成，耗时: {elapsed_time:.2f}秒")
            return True
        else:
            print_error(f"执行失败，返回码: {return_code}")
            if output_lines:
                print("最后几行输出:")
                for line in output_lines[-5:]:
                    print(f"  {line}")
            return False
            
    except Exception as e:
        print_error(f"执行脚本时出错: {e}")
        if log_file:
            error_time = datetime.now().strftime('%H:%M:%S')
            log_file.write(f"[{error_time}] [ERROR] 执行脚本时出错: {e}\n")
            log_file.flush()
        return False

def generate_summary_report():
    """生成处理结果摘要报告"""
    print_info("\n生成处理结果摘要报告...")
    
    try:
        # 检查关键输出文件
        result_files = {
            "BERT预测结果": "csv/prediction/model_BERT_prediction.csv",
            "BERT详细日志": "csv/prediction/bert_detailed_log.csv",
            "LLM对话分析": "csv/prediction/LLM_dialogue_prediction.csv",
            "LLM动词分析": "csv/prediction/LLM_verb_prediction.csv",
            "加权结果": "csv/weighted/weighted.csv",
            "书籍结果": "csv/result/book_result_sorted.csv",
            "LLM分析": "csv/result/llm_analysis.csv"
        }
        
        print(f"\n{Colors.BOLD}[DATA] 处理结果摘要{Colors.END}")
        print(f"{Colors.CYAN}{'='*50}{Colors.END}")
        
        for name, file_path in result_files.items():
            full_path = os.path.join(PROJECT_ROOT, file_path)
            if os.path.exists(full_path):
                try:
                    df = pd.read_csv(full_path)
                    file_size = os.path.getsize(full_path) / 1024  # KB
                    print(f"{Colors.GREEN}[OK]{Colors.END} {name}: {len(df)}行, {file_size:.1f}KB")
                except Exception as e:
                    print(f"{Colors.YELLOW}[WARN]{Colors.END} {name}: 文件存在但读取失败 ({e})")
            else:
                print(f"{Colors.RED}[ERROR]{Colors.END} {name}: 文件不存在")
        
        # 如果有书籍结果，显示排行榜前5名
        book_result_path = os.path.join(PROJECT_ROOT, "csv/result/book_result_sorted.csv")
        if os.path.exists(book_result_path):
            try:
                df_books = pd.read_csv(book_result_path)
                if not df_books.empty:
                    print(f"\n{Colors.BOLD}[TOP] 百合轻小说排行榜 TOP 5{Colors.END}")
                    print(f"{Colors.CYAN}{'='*50}{Colors.END}")
                    
                    for i, (_, row) in enumerate(df_books.head(5).iterrows(), 1):
                        title = row.get('title', f'书籍{row.get("filename", i)}')
                        score = row.get('weighted', row.get('final', 0))
                        
                        if score >= 0.8:
                            level = "超重度"
                            emoji = "[RED]"
                        elif score >= 0.6:
                            level = "重度"
                            emoji = "[ORANGE]"
                        elif score >= 0.4:
                            level = "中度"
                            emoji = "[YELLOW]"
                        elif score >= 0.2:
                            level = "轻度"
                            emoji = "[GREEN]"
                        else:
                            level = "微百合"
                            emoji = "[WHITE]"
                        
                        print(f"{i}. {emoji} {title} - {score:.3f} ({level})")
                        
            except Exception as e:
                print_warning(f"读取书籍结果失败: {e}")
        
        print(f"\n{Colors.GREEN}[COMPLETE] 处理流程完成！{Colors.END}")
        print(f"{Colors.INFO}[TIP] 你可以运行GUI应用查看详细结果: streamlit run gui_app.py{Colors.END}")
        
    except Exception as e:
        print_error(f"生成摘要报告失败: {e}")

def main():
    """主函数"""
    print(f"{Colors.BOLD}{Colors.MAGENTA}")
    print("🌸 Yuri AI Core - 整合推理系统")
    print("=" * 50)
    print("基于BERT+LLM的百合轻小说智能分析")
    print("=" * 50)
    print(f"{Colors.END}")
    
    # 检查前提条件
    if not check_prerequisites():
        print_error("前提条件检查失败，程序退出")
        sys.exit(1)
    
    # 询问用户是否继续
    try:
        response = input(f"\n{Colors.YELLOW}是否开始完整的推理流程? (y/N): {Colors.END}").strip().lower()
        if response not in ['y', 'yes', '是']:
            print_info("用户取消操作")
            return
    except KeyboardInterrupt:
        print_info("\n用户中断操作")
        return
    
    # 记录开始时间
    start_time = time.time()
    
    print(f"\n{Colors.BOLD}[START] 开始执行推理流程...{Colors.END}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建日志文件
    log_filename = f"inference_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_filepath = os.path.join(PROJECT_ROOT, "logs", log_filename)
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    
    # 写入日志头部
    with open(log_filepath, 'w', encoding='utf-8') as log_file:
        log_file.write(f"Yuri AI Core - 推理日志\n")
        log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*60 + "\n\n")
        log_file.flush()
    
    print_info(f"[LOG] 详细日志将保存到: {log_filepath}")
    
    # 执行所有步骤
    success_count = 0
    failed_steps = []
    
    with open(log_filepath, 'a', encoding='utf-8') as log_file:
        for i, (step_name, description, script_path) in enumerate(SCRIPTS_RUN_MODE, 1):
            step_info = (step_name, description, script_path)
            
            if run_script(script_path, step_info, log_file):
                success_count += 1
                print_success(f"步骤 {i} 完成")
            else:
                failed_steps.append((i, step_name, script_path))
                print_error(f"步骤 {i} 失败")
                
                # 询问是否继续
                try:
                    response = input(f"{Colors.YELLOW}步骤 {i} 失败，是否继续执行后续步骤? (y/N): {Colors.END}").strip().lower()
                    if response not in ['y', 'yes', '是']:
                        print_info("用户选择停止执行")
                        break
                except KeyboardInterrupt:
                    print_info("\n用户中断操作")
                    break
    
    # 计算总耗时
    total_time = time.time() - start_time
    
    # 写入日志尾部
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_filepath, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"结束时间: {end_time}\n")
        log_file.write(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)\n")
        log_file.write(f"成功步骤: {success_count}/{len(SCRIPTS_RUN_MODE)}\n")
        log_file.write(f"日志文件: {log_filepath}\n")
    
    # 显示执行结果
    print(f"\n{Colors.BOLD}[DATA] 执行结果统计{Colors.END}")
    print(f"{Colors.CYAN}{'='*50}{Colors.END}")
    print(f"总步骤数: {len(SCRIPTS_RUN_MODE)}")
    print(f"成功步骤: {success_count}")
    print(f"失败步骤: {len(failed_steps)}")
    print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    print(f"[LOG] 详细日志: {log_filepath}")
    
    if failed_steps:
        print(f"\n{Colors.RED}失败的步骤:{Colors.END}")
        for step_num, step_name, script_path in failed_steps:
            print(f"  {step_num}. {step_name} - {script_path}")
    
    # 生成摘要报告
    if success_count == len(SCRIPTS_RUN_MODE):
        print_success("所有步骤执行成功！")
        generate_summary_report()
    else:
        print_warning("部分步骤执行失败，请检查错误信息")
        if success_count > 0:
            print_info("已完成的步骤结果仍然可以查看")
            generate_summary_report()

def show_help():
    """显示帮助信息"""
    help_text = f"""
{Colors.BOLD}{Colors.CYAN}Yuri AI Core - 整合推理系统{Colors.END}

{Colors.BOLD}用法:{Colors.END}
    python integrated_inference.py [选项]

{Colors.BOLD}选项:{Colors.END}
    -h, --help     显示此帮助信息
    --check-only   仅检查环境，不执行推理
    --step N       仅执行指定步骤 (1-{len(SCRIPTS_RUN_MODE)})

{Colors.BOLD}步骤列表:{Colors.END}
"""
    for i, (step_name, description, _) in enumerate(SCRIPTS_RUN_MODE, 1):
        help_text += f"    {i:2d}. {step_name} - {description}\n"
    
    help_text += f"""
{Colors.BOLD}示例:{Colors.END}
    python integrated_inference.py              # 执行完整流程
    python integrated_inference.py --check-only # 仅检查环境
    python integrated_inference.py --step 2    # 仅执行BERT推理

{Colors.BOLD}注意事项:{Colors.END}
    1. 确保已安装所有必要的Python包
    2. 确保config.json中已配置API密钥
    3. 确保BERT模型文件存在
    4. 建议在GPU环境下运行以获得更好性能
"""
    print(help_text)

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            show_help()
            sys.exit(0)
        elif sys.argv[1] == '--check-only':
            if check_prerequisites():
                print_success("环境检查通过，可以开始推理")
                sys.exit(0)
            else:
                print_error("环境检查失败")
                sys.exit(1)
        elif sys.argv[1] == '--step':
            if len(sys.argv) < 3:
                print_error("请指定步骤号")
                sys.exit(1)
            try:
                step_num = int(sys.argv[2])
                if 1 <= step_num <= len(SCRIPTS_RUN_MODE):
                    step_name, description, script_path = SCRIPTS_RUN_MODE[step_num - 1]
                    step_info = (step_name, description, script_path)
                    
                    # 为单步执行创建日志文件
                    log_filename = f"step_{step_num}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    log_filepath = os.path.join(PROJECT_ROOT, "logs", log_filename)
                    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
                    
                    with open(log_filepath, 'w', encoding='utf-8') as log_file:
                        log_file.write(f"Yuri AI Core - 单步执行日志\n")
                        log_file.write(f"步骤: {step_name} - {description}\n")
                        log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        log_file.write("="*60 + "\n\n")
                        log_file.flush()
                        
                        if run_script(script_path, step_info, log_file):
                            log_file.write(f"\n{'='*60}\n")
                            log_file.write(f"步骤执行完成\n")
                            log_file.write(f"日志文件: {log_filepath}\n")
                            print_success(f"步骤 {step_num} 执行成功")
                            print_info(f"日志保存到: {log_filepath}")
                            sys.exit(0)
                        else:
                            log_file.write(f"\n{'='*60}\n")
                            log_file.write(f"步骤执行失败\n")
                            log_file.write(f"日志文件: {log_filepath}\n")
                            print_error(f"步骤 {step_num} 执行失败")
                            print_info(f"日志保存到: {log_filepath}")
                            sys.exit(1)
                else:
                    print_error(f"步骤号必须在1-{len(SCRIPTS_RUN_MODE)}之间")
                    sys.exit(1)
            except ValueError:
                print_error("步骤号必须是数字")
                sys.exit(1)
        else:
            print_error(f"未知参数: {sys.argv[1]}")
            print("使用 -h 或 --help 查看帮助信息")
            sys.exit(1)
    else:
        main()
