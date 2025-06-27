import json
import re
import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, redirect, url_for
from langchain.llms.base import LLM
from langchain_community.llms.utils import enforce_stop_tokens

# 加载环境变量
load_dotenv()

# 设置API密钥和基础URL环境变量
API_KEY = os.getenv("CUSTOM_API_KEY", "sk-xxx")
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"

# SiliconFlow类保持不变
class SiliconFlow(LLM):
    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "siliconflow"

    def siliconflow_completions(self, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {API_KEY}"
        }

        response = requests.post(BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call(self, prompt: str, stop: list = None, model: str = "default-model") -> str:
        response = self.siliconflow_completions(model=model, prompt=prompt)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return response

# 测试用例生成器类
class TestCaseGenerator:
    def __init__(self, llm: SiliconFlow, model_name="deepseek-ai/DeepSeek-V2.5"):
        self.llm = llm
        self.model_name = model_name
        self.system_prompt = """你是一个专业的软件测试工程师，请根据我提供的功能说明，自动生成一份**等价类划分表格**和对应的**测试用例编号**以及JSON格式的测试用例。你必须分三步完成任务，并严格遵守每一步的所有规则。【第一步】生成“等价类边界值划分表格”⚠ 请严格遵循以下结构与格式要求：【表格结构】：请输出如下字段列的表格：| 输入条件 | 有效等价类 | 无效等价类 | 边界值 | 编号 |【生成规则】：1. 每个输入条件单独成组，列出：- 有效等价类（其对应无效等价类和边界值应留空）- 无效等价类（填写在“无效等价类”列，单独成行）- 典型边界值（如最小/最大/越界值），必须单独称号成行**单独列2. 编号从1开始，按行递增，，**不重复**3. 表格风格应与如下示例保持一致：| 输入条件 | 有效等价类 | 无效等价类 | 边界值 | 编号 || ---- | ---------------- | ---------- | ------------ | -- || 标题长度 | 长度为1~40字符 | | | 1 || 标题长度 | | 长度为0字符 | | 2 || 标题长度 | | 长度大于40字符 | | 3 || 标题长度 | | | 最小边界值：1字符 | 4 || 标题长度 | | | 最大边界值：40字符 | 5 || 标题长度 | | | 超出最大边界值：41字符 | 6 || 标题内容 | 不含非法字符 `/ : * ?` | | | 7 || 标题内容 | | 包含非法字符 `/` | | 8 || 标题内容 | | 包含非法字符 `:` | | 9 |4. 输出结果为Markdown，适合直接粘贴进文档。【第二步】基于“等价类边界值划分表格”，设计一套精简且高效的“基础测试用例表格”⚠ 请严格遵循以下结构与格式要求：【表格结构】：请输出如下字段列的表格：| 测试用例编号 | 输入数据 | 预期结果 | 输入说明 | 覆盖的等价类编号 |核心设计原则 (请严格遵守):1.合并有效用例: 创建一个核心的“正向用例”（Happy Path），用一个数据同时覆盖所有有效等价类。2. 聚焦边界值: 为划分表中明确列出的每一个边界值创建一条独立的、专门的测试用例。3. 一个类别一个代表: 为每一种独立的无效等价类创建一个代表性的测试用例。4.避免冗余: 不要为“典型的”无效值（例如，长度为50）创建额外的用例，因为我们已经有边界值用例（例如，长度为41）来覆盖该无效类。优先保留边界值测试。5. 测试用例编号必须遵循 `TC-<序列号>` 格式，`<序列号>`: 一个两位、补零的数字（01, 02...），在每个类别内从01开始递增。【第三步】将前两步的成果，统一输出为一份单一、完整的、机器可读的JSON格式数据。JSON顶层结构:
{
  "equivalence_classes": [ /* ...等价类对象数组... */ ],
  "test_cases": [ /* ...测试用例对象数组... */ ]
}
1. equivalence_classes 数组的结构:
将【第一步】表格中的每一行转换为一个JSON对象。
id: (数字) 编号。
input_condition: (字符串) 输入条件。
type: (字符串) 行的类别，必须是 'valid', 'invalid', 'boundary' 中的一种。
description: (字符串) 具体的文字描述。
示例:
"equivalence_classes": [
  { "id": 1, "input_condition": "标题长度", "type": "valid", "description": "长度为1~40字符" },
  { "id": 2, "input_condition": "标题长度", "type": "invalid", "description": "长度为0字符" },
  { "id": 4, "input_condition": "标题长度", "type": "boundary", "description": "最小边界值：1字符" }
]
2. test_cases 数组的结构:
将【第二步】表格中的每一行转换为一个JSON对象。
test_case_id: (字符串) 测试用例编号。
description: (字符串) 输入说明。
input_data: (对象) 包含所有输入字段和其对应值的对象。
expected_result: (字符串) "成功" 或 "失败"。
covers_equivalence_classes: (数组) 包含所覆盖的id编号的数字数组。
示例:
"test_cases": [
  {
    "test_case_id": "TC_01",
    "description": "覆盖有效测试用例",
    "input_data": { "title": "我的第一个任务" },
    "expected_result": "成功",
    "covers_equivalence_classes": [1, 7]
  }
]
功能需求如下：
"""
        # self.system_prompt = """你是一个专业的软件测试工程师，请根据我提供的功能说明，自动生成一份**等价类划分表格**和对应的**测试用例编号**以及JSON格式的测试用例。你必须分三步完成任务,并严格遵守每一步的所有规则。第一步：生成“等价类边界值划分表格”⚠ 请严格遵循以下结构与格式要求：【表格结构】：请输出如下字段列的表格：| 输入条件 | 有效等价类 | 无效等价类 | 边界值 | 编号 |【生成规则】：1. 每个输入条件单独成组，列出：- 有效等价类（其对应无效等价类和边界值应留空）- 无效等价类（填写在“无效等价类”列）- 典型边界值（如最小/最大/越界值）**单独列2. 编号从1开始，按行递增，，**不重复**3. 表格风格应与如下示例保持一致：| 输入条件 | 有效等价类 | 无效等价类 | 边界值 | 编号 || ---- | ---------------- | ---------- | ------------ | -- || 标题长度 | 长度为1~40字符 | | | 1 || 标题长度 | | 长度为0字符 | | 2 || 标题长度 | | 长度大于40字符 | | 3 || 标题长度 | | | 最小边界值：1字符 | 4 || 标题长度 | | | 最大边界值：40字符 | 5 || 标题长度 | | | 超出最大边界值：41字符 | 6 || 标题内容 | 不含非法字符 `/ : * ?` | | | 7 || 标题内容 | | 包含非法字符 `/` | | 8 || 标题内容 | | 包含非法字符 `:` | | 9 |4. 输出结果为Markdown，适合直接粘贴进文档。【第二步】基于“等价类边界值划分表格”，设计一套精简且高效的“基础测试用例表格”⚠ 请严格遵循以下结构与格式要求：【表格结构】：请输出如下字段列的表格：| 测试用例编号 | 输入数据 | 预期结果 | 输入说明 | 覆盖的等价类编号 |核心设计原则 (请严格遵守):1.合并有效用例: 创建一个核心的“正向用例”（Happy Path），使用一个贴近实际的输入数据（例如：“我的第一个任务”）。这个用例应该同时覆盖有效的长度和有效的内容这两个等价类。2. 聚焦边界值: 为划分表中明确列出的每一个边界值（如最小长度、最大长度、刚好超过最大长度等）创建一条独立的、专门的测试用例。3. 一个类别一个代表: 为每一种独立的无效等价类（例如，长度为0，或包含某一个特定的非法字符）创建一个代表性的测试用例。4.避免冗余: 不要为“典型的”无效值（例如，长度为50）创建额外的用例，因为我们已经有边界值用例（例如，长度为41）来覆盖该无效类。优先保留边界值测试。第三步：生成可自动化的测试数据 (JSON)在完成以上两步之后，你必须将【第二步】生成的"基础测试用例表格"中的内容，转换为一份`JSON`格式的数据，并将其包裹在json代码块中。这份`JSON`数据应该是一个数组，其中每个对象代表一个测试用例。【JSON对象结构】每个JSON对象必须包含以下键：*   `test_case_id`: (字符串) 测试用例编号，例如 "TC_HappyPath_01"。*   `description`: (字符串) 对该测试用例的文字说明，即"输入说明"列的内容。*   `input_data`: (对象) 一个包含所有输入字段和其对应值的对象。这是自动化脚本的核心输入。*   `expected_result`: (字符串) "成功" 或 "失败"。*   `covers_equivalence_classes`: (数组) 包含所覆盖的等价类编号的数字数组。【JSON输出示例】```json[  {    "test_case_id": "TC_HappyPath_01",    "description": "正向核心用例，覆盖有效长度和内容",    "input_data": {      "标题": "我的第一个任务"    },    "expected_result": "成功",    "covers_equivalence_classes": [1, 8]  },  {    "test_case_id": "TC_Boundary_01",    "description": "边界值：最小长度1",    "input_data": {      "标题": "a"    },    "expected_result":``` 
        # 功能描述如下："""
        # self.system_prompt = 
        # """
        # 你是一个专业的软件测试工程师，请根据我提供的功能说明，自动生成一份**等价类划分表格**和对应的**测测试用例编号**以及JSON格式的测试用例。你必须分三步完成任务，并严格遵守每一步的所有规则。
        # 【第一步】：生成“等价类边界值划分表格”⚠ 请严格遵循以下结构与格式要求：【表格结构】：请输出如下字段列的表格：| 输入条件 | 有效等价类 | 无效等价类 | 边界值 | 编号 |【生成规则】：1. 每个输入条件单独成组，列出 ：- 有效等价类（其对应无效等价类和边界值应留空）- 无效等价类（填写在“无效等价类”列）- 典型边界值（如最小/最大/越界值）**单独列2. 编号从1开始，按行递增，，**不重复**3. 表格风格应与如下示例保持一致：| 输入条件 | 有效等价类 | 无效等价类 | 边界值 | 编号 || ---- | ---------------- | ---------- | ------------ | -- || 标题长度 | 长度为1\\~40字符 | | | 1 || 标题长度 | | 长度为0字符 | | 2 || 标题长度 | | 长度大于40字符 | | 3 || 标题长度 | | | 最小边界值：1字符 | 4 || 标题长度 | | | 最大边界值：40字符 | 5 || 标题长度 | | | 超出最大边界值：41字符 | 6 || 标题内容 | 不含非法字符 `/ : * ?` | | | 7 || 标题内容 | | 包含非法字符 `/` | | 8 || 标题内容 | | 包含非法字符 `:` | | 9 |4. 输出结果为Markdown，适合直接粘贴进文档。
        # 【第二步】：基于“等价类边界值划分表格”，设计一套精简且高效的“基础测试用例表格”⚠ 请严格遵循以下结构与格式要求：【表格结构】：请输出如下字段列的表格：| 测试用例编号 | 输入数据 | 预期结果 | 输入说明 | 覆盖的等价类编号 |核心设计原则 (请严格遵守):1.合并有效用例: 创建一个核心的“正向用例”（Happy Path），使用一个贴近实际的输入数据（例如：“我的第一个任务”）。这个用例应该同时覆盖有效的长度和有效的内容这两个等价类。2. 聚焦边界值: 为划分表中明确列 出的每一个边界值（如最小长度、最大长度、刚好超过最大长度等）创建一条独立的、专门的测试用例。3. 一个类别一个代表: 为每一种独立的无效等价类（例如 ，长度为0，或包含某一个特定的非法字符）创建一个代表性的测试用例。4.避免冗余: 不要为“典型的”无效值（例如，长度为50）创建额外的用例，因为我们已经 有边界值用例（例如，长度为41）来覆盖该无效类。优先保留边界值测试。
        # 【第三步】：生成可自动化的测试数据(JSON)在完成以上二步之后，你必须将【第一步】生成的\"等价类划分表格\"中的内容，转换为一份`JSON`格式的数据，并将其包裹在json代码块中。这份`JSON`数据应该是一个数组，其中每个对象代表一个测试用例。【JSON对象结构】每个JSON对象必须包含以下键：*`equivalence_id`：编号（整数）,input_condition：输入条件（字符串）,class_type：等价类类型（"有效"/"无效"/"边界值"）,description：等价类描述（字符串，当class_type为"有效"或"无效"时必填）,boundary_value：边界值描述（字符串，当class_type为"边界值"时必填）。【JSON输出示例】```[
        # {
        #     "equivalence_id": 1,
        #     "input_condition": "标题长度",
        #     "class_type": "有效",
        #     "description": "长度为1~40字符",
        #     "boundary_value": null
        # },
        # {
        #     "equivalence_id": 4,
        #     "input_condition": "标题长度",
        #     "class_type": "边界值",
        #     "description": null,
        #     "boundary_value": "最小边界值: 1字符"
        # }
        # ]```
        # 【第四步】：生成可自动化的测试数据(JSON)在完成以上三步之后，你必须将【第二步】生成的\"基础测试用例表格\"中的内容，转换为一份`JSON`格式的数据，并将其包裹在json代码块中。这份`JSON`数据应该是一个数组，其中每个对象代表一个测试用例。【JSON对象结构】每个JSON对象必须包含以下键：*   `test_case_id`: (字符串) 测试用例编号，例如 \"TC_HappyPath_01\"。*   `description`: (字符串) 对该测试用例的文字说明，即\"输入说明\"列的内容。*   `input_data`: (对象) 一个包含所有输入字段和其对应值的对象。这是自动化脚本的核心输入。*   `expected_result`: (字符串) \"成功\" 或 \"失败\"。*   `covers_equivalence_classes`: (数组) 包含所覆盖的等价类编号的数字数组。【JSON输出示例】```json[  {    \"test_case_id\": \"TC_HappyPath_01\",    \"description\": \"正向核心用例，覆盖有效长度和内容\",    \"input_data\": {      \"标题\": \"我的第一个任务\"    },    \"expected_result\": \"成功\",    \"covers_equivalence_classes\": [1, 8]  },  {    \"test_case_id\": \"TC_Boundary_01\",    \"description\": \"边界值：最小长度1\",    \"input_data\": {      \"标题\": \"a\"    },    \"expected_result\":``` \n功能描述如下："}, {"role": "user", "content": "功能需求描述：即时贴设置标题功能：标题要求为1~40个字符（1汉字=2字符），禁止包含字符 `/ : * ?`"}]
        # """

    def _create_messages(self, prompt: str) -> list:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"功能需求描述：{prompt}"}
        ]
    
    def _parse_response(self, response: str) -> dict:
        try:
            # 尝试提取JSON格式内容
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1).strip())
                
            # 尝试直接解析为JSON
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Invalid response format", "raw": response}
        except Exception as e:
            return {"error": f"Parsing error: {str(e)}", "raw": response}
        
    def generate_from_function(self, code: str, stop: list = None) -> dict:
        prompt = f"""分析以下函数代码并生成测试用例：
```python
{code}
```"""
        return self._generate_cases(prompt, stop)
        
    def generate_from_requirement(self, requirement: str, stop: list = None) -> dict:
        return self._generate_cases(requirement, stop)
    
    def _generate_cases(self, prompt: str, stop: list = None) -> dict:
        # 构建完整提示词
        messages = self._create_messages(prompt)
        
        # 将消息列表序列化为字符串
        content = json.dumps(messages, ensure_ascii=False)
        
        # 使用原始的SiliconFlow接口
        response = self.llm._call(
            prompt=content,
            model=self.model_name,
            stop=stop
        )
        print("====response====",self._parse_response(response))
        # 解析响应
        return self._parse_response(response)
    
    def set_model(self, model_name: str):
        self.model_name = model_name

# 创建Flask应用
app = Flask(__name__)

# 初始化模型和生成器
llm_instance = SiliconFlow()
test_generator = TestCaseGenerator(llm=llm_instance)

@app.route('/')
def index():
    """首页展示测试用例生成表单"""
    return render_template('index.html')

@app.route('/generate/function', methods=['POST'])
def generate_function():
    """根据函数代码生成测试用例"""
    code = request.form.get('code')
    if not code:
        return jsonify({"error": "No code provided"}), 400
    
    cases = test_generator.generate_from_function(code)
    return render_template('results.html', cases=cases, type="函数测试用例", content=code)

@app.route('/generate/requirement', methods=['POST'])
def generate_requirement():
    """根据需求描述生成测试用例"""
    requirement = request.form.get('requirement')
    if not requirement:
        return jsonify({"error": "No requirement provided"}), 400
    
    cases = test_generator.generate_from_requirement(requirement)
    return render_template('results.html', cases=cases, type="需求测试用例", content=requirement)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API端点，返回JSON格式的测试用例"""
    data = request.json
    if not data:
        return jsonify({"error": "Invalid request"}), 400
    
    gen_type = data.get('type')
    content = data.get('content')
    
    if not content:
        return jsonify({"error": "Content is required"}), 400
    
    if gen_type == 'function':
        cases = test_generator.generate_from_function(content)
    elif gen_type == 'requirement':
        cases = test_generator.generate_from_requirement(content)
    else:
        return jsonify({"error": "Invalid generation type"}), 400
    
    return jsonify(cases)

if __name__ == '__main__':
    app.run(debug=True)