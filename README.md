# Recursive LLM System (遞迴式大型語言模型系統)

[English](#english) | [繁體中文](#traditional-chinese)

---

## <a name="english"></a>English

# Recursive LLM System

A system for improving Chain of Thought (CoT) reasoning by enabling a Large Language Model to call itself recursively, breaking down complex problems into simpler sub-problems.

## Overview

This proof-of-concept system enhances LLM reasoning by:
1. Breaking down initial queries into multiple sub-problems
2. Generating targeted prompts for each sub-problem
3. Recursively solving complex sub-problems
4. Aggregating responses into a comprehensive final answer

The system keeps track of the entire thought process, saving each step for review and analysis.

## Requirements

- Python 3.7+
- Ollama installed and running locally
- The `deepseek-r1:14b` model loaded in Ollama
- `argparse` module (included in Python standard library)

## Installation

```bash
# Clone the repository
git clone https://github.com/Gary0302/recursive_llm.git
cd recursive_llm

# Install dependencies
pip install requests
```

## Usage

### Command Line Interface

The system provides a convenient command-line interface with several options:

```bash
python recursive_llm.py [-r RECURSION_DEPTH] [-p PROMPT] [-m MODEL_NAME] [-s]
```

Arguments:
- `-r`: Maximum recursion depth (1-4, default: 3)
- `-p`: Custom prompt/query (default: "What are the environmental and economic impacts of renewable energy adoption globally?")
- `-m`: Model name to use (default: "deepseek-r1:14b")
- `-s`: Enable saving thought process to a file

Examples:
```bash
# Use default settings
python recursive_llm.py

# Custom query with depth of 2
python recursive_llm.py -r 2 -p "Explain the impact of quantum computing on cybersecurity"

# Use a different model and save thought process
python recursive_llm.py -m "llama3:8b" -s

# Maximum recursion with custom prompt
python recursive_llm.py -r 4 -p "Compare and contrast different machine learning approaches for time series analysis" -s
```

### As a Module

You can also use the system programmatically:

```python
from recursive_llm import RecursiveLLM

# Initialize the system
system = RecursiveLLM(
    model_name="deepseek-r1:14b",
    max_recursion_depth=3,
    save_thoughts=True
)

# Process a query
query = "What are the environmental and economic impacts of renewable energy adoption globally?"
result = system.process_query(query)

print(result)

# Save the thought process
system.save_thought_process("thought_process.json")
```

## How It Works

1. **Problem Decomposition**: The LLM breaks down complex queries into 2-5 sub-questions
2. **Dynamic Prompting**: Generates specific prompts for each sub-question
3. **Recursive Processing**: For complex sub-questions, repeats the process recursively
4. **Solution Aggregation**: Combines all answers into a coherent final response
5. **Thought Logging**: Records all steps and decisions for transparency

## Customization

- Adjust `max_recursion_depth` to control the depth of problem decomposition (1-4 recommended)
- Modify the complexity assessment logic in `assess_complexity()`
- Change the Ollama model by setting a different `model_name`
- Enable/disable thought process logging with `save_thoughts`

## Limitations

- The system relies on the underlying model's ability to break down problems effectively
- Very deep recursion may introduce inconsistencies or drift from the original question
- Simple similarity metrics may not always prevent recursion loops
- Performance will vary based on the model used and the complexity of queries

---

## <a name="traditional-chinese"></a>繁體中文

# 遞迴式大型語言模型系統

一個通過使大型語言模型能夠遞迴調用自身，將複雜問題分解為更簡單子問題的方式，來改進思維鏈（CoT）推理的系統。

## 概述

這個概念驗證系統通過以下方式增強大型語言模型推理能力：
1. 將初始查詢分解為多個子問題
2. 為每個子問題生成有針對性的提示
3. 遞迴解決複雜子問題
4. 將回應彙整成全面的最終答案

系統會追蹤整個思考過程，保存每一步驟以供審查和分析。

## 需求

- Python 3.7+
- 本地安裝並運行 Ollama
- Ollama 中已加載 `deepseek-r1:14b` 模型
- `argparse` 模組（Python 標準庫中已包含）

## 安裝

```bash
# 克隆儲存庫
git clone https://github.com/Gary0302/recursive_llm.git
cd recursive_llm

# 安裝依賴
pip install requests
```

## 使用方法

### 命令行界面

系統提供了便捷的命令行界面，具有多個選項：

```bash
python recursive_llm.py [-r 遞迴深度] [-p 提示詞] [-m 模型名稱] [-s]
```

參數：
- `-r`：最大遞迴深度（1-4，默認值：3）
- `-p`：自定義提示詞/查詢（默認值："What are the environmental and economic impacts of renewable energy adoption globally?"）
- `-m`：要使用的模型名稱（默認值："deepseek-r1:14b"）
- `-s`：啟用將思考過程保存到文件

示例：
```bash
# 使用默認設置
python recursive_llm.py

# 自定義查詢，深度為2
python recursive_llm.py -r 2 -p "解釋量子計算對網絡安全的影響"

# 使用不同的模型並保存思考過程
python recursive_llm.py -m "llama3:8b" -s

# 最大遞迴深度與自定義提示
python recursive_llm.py -r 4 -p "比較不同機器學習方法在時間序列分析中的應用" -s
```

### 作為模組

您還可以以程式方式使用系統：

```python
from recursive_llm import RecursiveLLM

# 初始化系統
system = RecursiveLLM(
    model_name="deepseek-r1:14b",
    max_recursion_depth=3,
    save_thoughts=True
)

# 處理查詢
query = "可再生能源在全球範圍內的採用對環境和經濟有哪些影響？"
result = system.process_query(query)

print(result)

# 保存思考過程
system.save_thought_process("thought_process.json")
```

## 工作原理

1. **問題分解**：大型語言模型將複雜查詢分解為2-5個子問題
2. **動態提示**：為每個子問題生成特定提示
3. **遞迴處理**：對於複雜的子問題，遞迴重複該過程
4. **解決方案彙整**：將所有答案組合成一個連貫的最終回應
5. **思考記錄**：記錄所有步驟和決策以保持透明度

## 自定義

- 調整 `max_recursion_depth` 以控制問題分解的深度（建議1-4）
- 修改 `assess_complexity()` 中的複雜度評估邏輯
- 通過設置不同的 `model_name` 來更改 Ollama 模型
- 通過 `save_thoughts` 啟用/禁用思考過程記錄

## 局限性

- 系統依賴於底層模型有效分解問題的能力
- 非常深層次的遞迴可能會引入不一致性或偏離原始問題
- 簡單的相似度度量可能並不總是能防止遞迴循環
- 性能將根據所使用的模型和查詢的複雜性而有所不同