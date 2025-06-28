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
        self.system_prompt = """你是一个专业的软件测试工程师，请根据我提供的功能说明，自动生成一份**等价类划分表格**和对应的**测试用例编号**以及JSON格式的测试用例。你必须分三步完成任务，并严格遵守每一步的所有规则。【第一步】生成"等价类边界值划分表格"⚠ 请严格遵循以下结构与格式要求：【表格结构】：请输出如下字段列的表格：| 输入条件 | 有效等价类 | 无效等价类 | 边界值 | 编号 |【生成规则】：1. 每个输入条件单独成组，列出：- 有效等价类（其对应无效等价类和边界值应留空）- 无效等价类（填写在"无效等价类"列，单独成行）- 典型边界值（如最小/最大/越界值），必须单独称号成行**单独列2. 编号从1开始，按行递增，，**不重复**3. 表格风格应与如下示例保持一致：| 输入条件 | 有效等价类 | 无效等价类 | 边界值 | 编号 || ---- | ---------------- | ---------- | ------------ | -- || 标题长度 | 长度为1~40字符 | | | 1 || 标题长度 | | 长度为0字符 | | 2 || 标题长度 | | 长度大于40字符 | | 3 || 标题长度 | | | 最小边界值：1字符 | 4 || 标题长度 | | | 最大边界值：40字符 | 5 || 标题长度 | | | 超出最大边界值：41字符 | 6 || 标题内容 | 不含非法字符 `/ : * ?` | | | 7 || 标题内容 | | 包含非法字符 `/` | | 8 || 标题内容 | | 包含非法字符 `:` | | 9 |4. 输出结果为Markdown，适合直接粘贴进文档。【第二步】基于"等价类边界值划分表格"，设计一套精简且高效的"基础测试用例表格"⚠ 请严格遵循以下结构与格式要求：【表格结构】：请输出如下字段列的表格：| 测试用例编号 | 输入数据 | 预期结果 | 输入说明 | 覆盖的等价类编号 |核心设计原则 (请严格遵守):1.合并有效用例: 创建一个核心的用例用一个数据同时覆盖所有有效等价类。2. 聚焦边界值: 为划分表中明确列出的每一个边界值创建一条独立的、专门的测试用例。3. 一个类别一个代表: 为每一种独立的无效等价类创建一个代表性的测试用例。4.避免冗余: 不要为"典型的"无效值（例如，长度为50）创建额外的用例，因为我们已经有边界值用例（例如，长度为41）来覆盖该无效类。优先保留边界值测试。5. 测试用例编号必须遵循 `TC-<序列号>` 格式，`<序列号>`: 一个两位、补零的数字（01, 02...），在每个类别内从01开始递增。【第三步】将前两步的成果，统一输出为一份单一、完整的、机器可读的JSON格式数据。JSON顶层结构:
{
  "equivalence_classes": [ /* ...等价类对象数组... */ ],
  "test_cases": [ /* ...测试用例对象数组... */ ]
}
1. equivalence_classes 数组的结构:
将【第一步】表格中的每一行转换为一个JSON对象，必须严格包含以下键：
*   `input_condition`: (字符串) 输入条件。
*   `valid_class`: (字符串) 有效等价类，如果该行不是有效等价类则为空字符串。
*   `invalid_class`: (字符串) 无效等价类，如果该行不是无效等价类则为空字符串。
*   `boundary_value`: (字符串) 边界值，如果该行不是边界值则为空字符串。
*   `id`: (数字) 编号。

**示例**:
```json
"equivalence_classes": [
  { "input_condition": "标题长度", "valid_class": "长度为1~40字符", "invalid_class": "", "boundary_value": "", "id": 1 },
  { "input_condition": "标题长度", "valid_class": "", "invalid_class": "长度为0字符", "boundary_value": "", "id": 2 },
  { "input_condition": "标题长度", "valid_class": "", "invalid_class": "", "boundary_value": "最小边界值：1字符", "id": 4 }
]
```
2. test_cases 数组的结构:
将【第二步】表格中的每一行转换为一个JSON对象。
*   `test_case_id`: (字符串) 测试用例编号。
*   `input_data`: (对象) 包含所有输入字段和其对应值的对象。
*   `expected_result`: (字符串) 例如："设置成功"、"设置失败"、提示"内容不能为空"等。
*   `description`: (字符串) 输入说明。
*   `covers_equivalence_classes`: (数组) 包含所覆盖的`id`编号的数字数组。

**示例**:
```json
"test_cases": [
  {
    "test_case_id": "TC__07",
    "input_data": { "标题": "我的标题/" },
    "expected_result": "设置失败，提示“标题包含非法字符”",
    "description": "无效字符 /",
    "covers_equivalence_classes": [8]
  }
]
```
功能需求如下：
"""
        

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

@app.route('/api/add_test_case', methods=['POST'])
def api_add_test_case():
    """API端点，添加新的测试用例并更新覆盖率分析"""
    data = request.json
    if not data:
        return jsonify({"error": "无效的请求数据"}), 400
    
    # 获取必要的数据
    test_case = data.get('test_case')
    all_test_cases = data.get('all_test_cases', [])
    equivalence_classes = data.get('equivalence_classes', [])
    
    if not test_case:
        return jsonify({"error": "测试用例数据不能为空"}), 400
    
    # 添加新测试用例到列表
    all_test_cases.append(test_case)
    
    return jsonify({
        "success": True,
        "message": "测试用例已成功添加"
    })

if __name__ == '__main__':
    app.run(debug=True)