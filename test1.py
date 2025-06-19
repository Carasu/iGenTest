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
        self.system_prompt = """你是一名专业测试工程师。请为以下需求生成结构化的测试用例：
        1. 使用JSON格式输出
        2. 每个测试用例包含字段：
           - "name": 测试场景描述
           - "steps": 操作步骤列表
           - "expected": 预期结果
        3. 覆盖正向、负向和边界值情况"""
    
    def _create_messages(self, prompt: str) -> list:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"需求描述：{prompt}"}
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