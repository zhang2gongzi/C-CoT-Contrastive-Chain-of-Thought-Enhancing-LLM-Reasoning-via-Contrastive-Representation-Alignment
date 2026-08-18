import json
import sys

def count_matching_answers(json_file_path):
    """
    计算JSON文件中true_answer与model_answer相等的条目数量
    
    参数:
        json_file_path (str): JSON文件的路径
        
    返回:
        int: 匹配的条目数量
    """
    try:
        # 打开并读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # 确保数据是列表格式
            if not isinstance(data, list):
                print("错误: JSON数据不是列表格式")
                return 0
            
            # 初始化计数器
            match_count = 0
            
            # 遍历每个条目并检查答案是否匹配
            for item in data:
                # 检查条目是否包含必要的键
                if "true_answer" in item and "model_answer" in item:
                    if item["true_answer"] == item["model_answer"]:
                        match_count += 1
        
        return match_count
    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file_path}")
        return 0
    except json.JSONDecodeError:
        print(f"错误: 文件 {json_file_path} 不是有效的JSON格式")
        return 0
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return 0

if __name__ == "__main__":
    # 检查是否提供了文件路径参数
    if len(sys.argv) != 2:
        print("用法: python count_matching_answers.py <json_file_path>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    count = count_matching_answers(json_file)
    print(f"true_answer与model_answer相等的条目数量: {count}")
