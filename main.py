import json
import os
import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from judge.LLM import LLM
    from judge.playwright_html_tester import PlaywrightHTMLTester
except ImportError:
    # Fallback when running directly or judge is not a package
    from LLM import LLM
    from playwright_html_tester import PlaywrightHTMLTester

class HTMLErrorDetector:
    """三层HTML错误检测器（聚焦错误识别，不打分）"""

    def __init__(self, llm_model: Optional[LLM] = None, vlm_model: Optional[LLM] = None, rl_mode: bool = False, strict_mode: bool = False, static_mode: bool = False):
        """
        初始化检测器

        参数:
            llm_model: 用于文本分析的LLM实例
            vlm_model: 用于视觉分析的VLM实例
            rl_mode: 是否为RL训练模式（快速模式，跳过昂贵的LLM/VLM检查）
            strict_mode: 是否启用严格模式（更严格的扣分标准和Prompt）
            static_mode: 是否启用静态语法检查（Layer 1 仅用代码分析，Layer 2/3 正常调用 LLM/VLM）
        """
        self.rl_mode = rl_mode
        self.strict_mode = strict_mode
        self.static_mode = static_mode
        
        # 修改默认模型名称，适配本地部署
        default_model_name = "auto" # 自动发现模型
        
        self.llm = llm_model if llm_model is not None else LLM(default_model_name)
        # VLM 如果本地没有部署，可能需要注意。如果本地也是同一个端口提供多模态，则复用 LLM 类
        self.vlm = vlm_model if vlm_model is not None else LLM(default_model_name)
        
        if self.static_mode:
            print("🚀 已启用静态语法检查 (Static Mode): Layer 1 仅代码分析，其他层正常调用 LLM/VLM")

    async def detect_async(self, html_file_path: str,
               playwright_report: Optional[Dict[str, Any]] = None,
               screenshots: Optional[List[Dict[str, Any]]] = None,
               high_quality: bool = False,
               weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        执行三层错误检测 (异步版)
        """
        print(f"🚀 开始HTML错误检测 (RL模式: {self.rl_mode}, 高清模式: {high_quality}, 严苛模式: {self.strict_mode}, 静态模式: {self.static_mode})...")
        
        # 默认权重
        # Archival weights used to produce the released experiment summaries.
        default_weights = {
            "layer1": 0.20,
            "layer2": 0.25,
            "layer3": 0.25,
            "layer4": 0.30
        }
        if weights:
            default_weights.update(weights)
        self.current_weights = default_weights

        try:
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except FileNotFoundError:
            error_msg = f"文件未找到: {html_file_path}"
            print(f"❌ 错误: {error_msg}")
            return self._create_error_report(html_file_path, error_msg, "file_not_found")
        except Exception as e:
            error_msg = f"读取文件失败: {str(e)}"
            print(f"❌ 错误: {error_msg}")
            return self._create_error_report(html_file_path, error_msg, "file_read_error")

        try:
            tasks = []
            
            # Layer 1
            if self.static_mode:
                tasks.append(self._detect_layer1_syntax_errors_static(html_content))
            else:
                tasks.append(self._detect_layer1_syntax_errors_async(html_content, playwright_report))
            
            # Layer 2
            # 物理/逻辑检测始终调用 LLM，除非被显式禁用（目前没有 Layer 2 的独立开关）
            # static_mode 仅影响 Layer 1
            tasks.append(self._detect_layer2_physics_errors_async(html_content))
            
            # Layer 3
            # 视觉检测始终调用 VLM，除非 RL 模式优化或被禁用
            # 优化截图列表：限制数量以避免 VLM 超时
            optimized_screenshots = screenshots
            if screenshots and len(screenshots) > 5:
                print(f"⚠️ 截图数量过多 ({len(screenshots)}), 进行下采样至 5 张以优化性能...")
                # 保留首尾，中间随机取3张
                first = screenshots[0]
                last = screenshots[-1]
                middle = screenshots[1:-1]
                import random
                sampled_middle = random.sample(middle, min(len(middle), 3))
                # 保持时间顺序
                sampled_middle.sort(key=lambda x: x.get('timestamp', 0))
                optimized_screenshots = [first] + sampled_middle + [last]

            tasks.append(self._detect_layer3_visual_errors_async(optimized_screenshots, high_quality=high_quality))

            # 执行并行任务
            print("⏳ 正在并行执行 Layer 1-3 检测...")
            results = await asyncio.gather(*tasks)
            layer1_result, layer2_result, layer3_result = results

            # Layer 4 (Runtime)
            print("LAYER 4: 运行与交互能力检测...")
            layer4_result = self._detect_layer4_runtime_interaction(playwright_report)

            # 整合报告
            print("📊 整合检测报告...")
            comprehensive_report = self._generate_comprehensive_report(
                html_file_path, layer1_result, layer2_result, layer3_result, layer4_result, weights=self.current_weights
            )

            print("✅ 检测完成！")
            return comprehensive_report

        except Exception as e:
            error_msg = f"检测过程中出现异常: {str(e)}"
            print(f"❌ 错误: {error_msg}")
            return self._create_error_report(html_file_path, error_msg, "detection_error")

    async def _detect_layer1_syntax_errors_static(self, html_content: str) -> Dict[str, Any]:
        """第一层：静态语法检测 (无LLM)"""
        print("LAYER 1 (Static): 静态语法分析...")
        errors = []
        
        # 1. HTMLParser 基础检查
        try:
            from html.parser import HTMLParser
            class SyntaxChecker(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.errors = []
                    # 常见的自闭合标签，不需要 end tag
                    self.void_elements = {
                        'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 
                        'link', 'meta', 'param', 'source', 'track', 'wbr'
                    }
                    self.stack = []
                    
                def handle_starttag(self, tag, attrs):
                    if tag not in self.void_elements:
                        self.stack.append(tag)
                        
                def handle_endtag(self, tag):
                    if tag in self.void_elements:
                        return # 忽略自闭合标签的闭合（虽然不规范但不算错）
                        
                    if not self.stack:
                        self.errors.append(f"发现多余的闭合标签: </{tag}>")
                        return
                        
                    if self.stack[-1] == tag:
                        self.stack.pop()
                    else:
                        # 尝试寻找匹配的标签
                        if tag in self.stack:
                            # 栈中存在，说明中间有未闭合的标签
                            while self.stack and self.stack[-1] != tag:
                                unclosed = self.stack.pop()
                                self.errors.append(f"标签 <{unclosed}> 未闭合 (被 </{tag}> 中断)")
                            if self.stack:
                                self.stack.pop() # 弹出目标标签
                        else:
                            self.errors.append(f"发现多余的闭合标签: </{tag}>")
                            
                def handle_error(self, message):
                    self.errors.append(f"解析错误: {message}")

            parser = SyntaxChecker()
            parser.feed(html_content)
            # 检查剩余栈
            while parser.stack:
                unclosed = parser.stack.pop()
                # 忽略 html, body 等顶层标签，浏览器容错性强
                if unclosed not in ['html', 'body']:
                    errors.append(f"标签 <{unclosed}> 未闭合")
            
            if parser.errors:
                for e in parser.errors:
                    errors.append({"description": e, "severity": "中等", "category": "HTML语法错误"})
                    
        except Exception as e:
            errors.append({"description": f"HTML解析异常: {e}", "severity": "严重", "category": "解析错误"})

        # 2. 简单的 CSS/JS 关键词检查 (正则)
        import re
        # 检查未闭合的大括号 (非常粗略)
        open_braces = len(re.findall(r'\{', html_content))
        close_braces = len(re.findall(r'\}', html_content))
        if abs(open_braces - close_braces) > 5: # 允许少量误差
             errors.append({"description": f"大括号数量严重不匹配 ({{:{open_braces}, }}:{close_braces})", "severity": "严重", "category": "语法结构错误"})

        has_errors = len(errors) > 0
        
        # 扣分逻辑
        if self.strict_mode:
            # 严格模式：基础分扣更多，且有惩罚系数
            deduction_per_error = 25 # 严苛模式每个错误扣25分
            score = max(0, 100 - len(errors) * deduction_per_error)
        else:
            # 普通模式
            deduction_per_error = 10
            score = max(0, 100 - len(errors) * deduction_per_error)
        
        return {
            "has_errors": has_errors,
            "error_count": len(errors),
            "score": score,
            "score_breakdown": {},
            "errors": errors,
            "summary": "静态语法检查完成"
        }

    async def _detect_layer2_physics_errors_static(self) -> Dict[str, Any]:
        """第二层：静态跳过"""
        return {
            "has_errors": False,
            "error_count": 0,
            "score": 100,
            "summary": "静态模式：跳过物理逻辑检测"
        }

    async def _detect_layer3_visual_errors_static(self) -> Dict[str, Any]:
        """第三层：静态跳过"""
        return {
            "has_errors": False,
            "error_count": 0,
            "score": 100,
            "summary": "静态模式：跳过视觉检测"
        }

    
    def detect(self, html_file_path: str,
               playwright_report: Optional[Dict[str, Any]] = None,
               screenshots: Optional[List[Dict[str, Any]]] = None,
               high_quality: bool = False,
               weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """同步入口 (为了兼容性)"""
        return asyncio.run(self.detect_async(html_file_path, playwright_report, screenshots, high_quality, weights))

    async def _detect_layer1_syntax_errors_async(self, html_content: str, 
                                    playwright_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """第一层：语法与结构错误检测 (异步包装)"""
        print("LAYER 1: 语法与结构错误检测...")
        
        # 在RL模式下，可以使用简化 Prompt
        simple_output = self.rl_mode
        
        # 构造 Prompt
        playwright_summary = ""
        if playwright_report:
            playwright_summary = f"""
**Playwright自动化测试报告:**
```json
{json.dumps(playwright_report, ensure_ascii=False, indent=2)}
```
"""
        
        if simple_output:
            prompt = f"""
你是一名资深的Web开发专家。请检测以下HTML代码的**语法和结构错误**。

{playwright_summary}

**HTML源码:**
```html
{html_content}
```

**输出要求：**
仅输出JSON对象，包含 score (0-100), has_errors (bool), error_count (int), errors (list of objects with 'description')。不要任何解释。
"""
        else:
            if self.strict_mode:
                role_desc = "你是一名**极其严苛**的代码审计专家（Code Auditor）。你的任务是吹毛求疵地找出代码中的每一个微小缺陷。"
                scoring_criteria = """
**评分标准（总分100分，严格扣分）：**
- 无错误：100分
- 每个轻微错误（如代码风格、非关键属性缺失）：扣10分
- 每个中等错误（如CSS选择器错误、潜在逻辑隐患）：扣25分
- 每个严重错误（如语法错误、JS报错、功能失效）：扣50分（直接不及格）
- 最低分：0分
"""
            else:
                role_desc = "你是一名资深的Web开发专家，请检测以下HTML代码中的**语法和结构错误**。"
                scoring_criteria = """
**评分标准（总分100分）：**
- 无错误：100分
- 每个轻微错误：扣5分
- 每个中等错误：扣15分
- 每个严重错误：扣30分
- 最低分：0分
"""

            prompt = f"""
{role_desc}

{playwright_summary}

**HTML源码:**
```html
{html_content}
```

**检测重点：**
1. **HTML语法错误（关键性）：**
   - 标签是否正确闭合（**忽略 <br>, <img>, <input>, <meta>, <link>, <hr> 等自闭合标签**）
   - 关键属性是否正确书写
   - 文档结构是否完整（doctype、head、body等）
   - **注意：不要对注释、空格、换行符或不影响运行的风格问题进行报错。**

2. **JavaScript错误（关键性）：**
   - 语法错误（括号不匹配、分号缺失等）
   - 运行时错误（变量未定义、函数调用错误等）
   - 逻辑错误（死循环、无限递归等）

3. **CSS错误（关键性）：**
   - 选择器错误
   - 属性名或值错误
   - 样式冲突导致布局崩坏

4. **功能性错误：**
   - 按钮点击无效果
   - 表单提交失败
   - 交互控件不工作

**输出要求：**
**仅输出JSON对象，不要添加任何解释文字或markdown标记！**

{scoring_criteria}

输出格式：
```json
{{
    "has_errors": <true/false>,
    "error_count": <整数>,
    "score": <0-100的整数>,
    "score_breakdown": {{
        "total_deductions": <扣分总数>,
        "minor_errors": <轻微错误数量>,
        "moderate_errors": <中等错误数量>,
        "severe_errors": <严重错误数量>
    }},
    "errors": [
        {{
            "category": "HTML语法错误/JavaScript错误/CSS错误/功能性错误",
            "severity": "严重/中等/轻微",
            "location": "错误位置描述（如：第XX行，函数XXX内）",
            "description": "详细的错误描述",
            "code_snippet": "相关代码片段（如果适用）"
        }}
    ],
    "summary": "整体错误情况总结"
}}
```

**重要说明：**
1. **对于 `<br>`, `<img>`, `<input>` 等标准自闭合标签，不要报错说未闭合！**
2. **不要把 Playwright 报告中的 "expected </3>" 这类奇怪的内部解析错误当作 HTML 源码错误，请以提供的 HTML 源码为准进行分析。**
3. **直接输出JSON对象，不要有"根据检测"、"我发现了"等前缀文字！**
"""

        try:
            # 异步调用 LLM (假设 LLM 类支持 async __call__ 或我们需要在线程池中运行)
            # 由于 LLM 类看起来是同步的 (self.llm(prompt))，我们使用 run_in_executor
            loop = asyncio.get_event_loop()
            llm_response = await loop.run_in_executor(None, lambda: self.llm(prompt))
            
            if llm_response is None:
                return {"has_errors": True, "error_count": 1, "errors": [{"description": "LLM返回None"}]}
            
            response_str = llm_response['content'] if isinstance(llm_response, dict) and 'content' in llm_response else llm_response
            result = self._parse_json_response(response_str)
            
            if not isinstance(result, dict):
                return {"has_errors": True, "error_count": 1, "score": 0, "errors": [{"description": "解析结果格式错误"}]}
            
            return {
                "has_errors": result.get("has_errors", False),
                "error_count": result.get("error_count", 0),
                "score": result.get("score", 100 if not result.get("has_errors", False) else 0),
                "score_breakdown": result.get("score_breakdown", {}),
                "errors": result.get("errors", []),
                "summary": result.get("summary", "")
            }
        except Exception as e:
            print(f"⚠️ Layer 1 检测出错: {e}")
            return {"has_errors": True, "error_count": 1, "score": 0, "errors": [{"description": f"检测异常: {str(e)}"}], "summary": "检测过程出错"}

    async def _detect_layer2_physics_errors_async(self, html_content: str) -> Dict[str, Any]:
        """第二层：数学物理原则错误检测 (异步包装)"""
        print("LAYER 2: 数学物理原则错误检测...")
        
        simple_output = self.rl_mode
        
        if simple_output:
            prompt = f"""
你是一名顶尖的数学、物理教育专家。请检测以下HTML代码中是否存在**违反数学定理或物理定律的错误**。

**重要说明：如果HTML内容不涉及数学或物理（例如是文学、历史、简单的信息展示等），请直接返回满分（100分），has_errors=false，不要强行寻找错误。**

**HTML源码:**
```html
{html_content}
```

**输出要求：**
仅输出JSON对象，包含 score (0-100), has_errors (bool), error_count (int), errors (list of objects with 'description')。不要任何解释。
"""
        else:
            if self.strict_mode:
                role_desc = "你是一名**极其严苛**的物理/数学教育专家。对于任何轻微的科学不准确、公式书写不规范或逻辑漏洞，都要予以严厉指出。"
                scoring_criteria = """
**评分标准（总分100分，严格扣分）：**
- 无错误：100分
- 每个轻微错误（如单位符号不规范）：扣15分
- 每个中等错误（如数值精度不够、轻微逻辑瑕疵）：扣40分
- 每个严重错误（违反物理定律/数学定理、公式错误）：扣100分（直接零分）
- 最低分：0分
"""
            else:
                role_desc = "你是一名顶尖的数学、物理教育专家，请检测以下HTML代码中是否存在**违反数学定理或物理定律的错误**。"
                scoring_criteria = """
**评分标准（总分100分）：**
- 无错误：100分
- 每个轻微错误：扣8分
- 每个中等错误：扣20分
- 每个严重错误（违反物理定律/数学定理）：扣40分
- 最低分：0分
"""

            prompt = f"""
{role_desc}

**HTML源码:**
```html
{html_content}
```

**检测重点（按优先级）：**

1. **核心物理定律违反：**
   - 力矩平衡：天平/杠杆问题中，是否满足 F1×L1 = F2×L2
   - 能量守恒：能量转换过程是否守恒
   - 牛顿定律：力与运动的关系是否正确
   - 动量守恒：碰撞问题中动量是否守恒

2. **数学计算错误：**
   - 公式书写是否正确
   - 数值计算是否准确
   - 函数图像与公式是否匹配
   - 几何关系是否正确

3. **单位与量纲错误：**
   - 单位是否正确使用
   - 量纲是否一致
   - 单位换算是否正确

4. **逻辑一致性错误：**
   - 代码中的物理量关系是否自洽
   - 显示的数值与物理状态是否匹配
   - 参数范围是否合理

{scoring_criteria}

**输出要求：**
**请仅输出JSON对象，不要添加任何解释文字或markdown标记！**

输出格式示例：
```json
{{
    "has_errors": false,
    "error_count": 0,
    "score": 100,
    "score_breakdown": {{
        "total_deductions": 0,
        "minor_errors": 0,
        "moderate_errors": 0,
        "severe_errors": 0
    }},
    "errors": [],
    "educational_impact": "",
    "summary": "未发现明显错误"
}}
```

**重要说明：**
1. **如果HTML内容不涉及数学或物理（例如是文学、历史、简单的信息展示等），请直接返回满分（100分），has_errors=false，不要强行寻找错误。**
2. **直接输出JSON对象，不要有"根据检测"、"我发现了"等前缀文字！**
"""

        try:
            loop = asyncio.get_event_loop()
            llm_response = await loop.run_in_executor(None, lambda: self.llm(prompt))
            
            if llm_response is None:
                return {"has_errors": True, "error_count": 1, "errors": [{"description": "LLM返回None"}]}
            
            response_str = llm_response['content'] if isinstance(llm_response, dict) and 'content' in llm_response else llm_response
            result = self._parse_json_response(response_str)
            
            if not isinstance(result, dict):
                return {"has_errors": True, "error_count": 1, "score": 0, "errors": [{"description": "解析结果格式错误"}]}
            
            return {
                "has_errors": result.get("has_errors", False),
                "error_count": result.get("error_count", 0),
                "score": result.get("score", 100 if not result.get("has_errors", False) else 0),
                "score_breakdown": result.get("score_breakdown", {}),
                "errors": result.get("errors", []),
                "educational_impact": result.get("educational_impact", ""),
                "summary": result.get("summary", "")
            }
        except Exception as e:
            print(f"⚠️ Layer 2 检测出错: {e}")
            return {"has_errors": True, "error_count": 1, "score": 0, "errors": [{"description": f"检测异常: {str(e)}"}], "summary": "检测过程出错"}

    async def _detect_layer3_visual_errors_async(self, screenshots: Optional[List[Dict[str, Any]]] = None, high_quality: bool = False) -> Dict[str, Any]:
        """第三层：视觉物理一致性检测 (异步包装)"""
        print("LAYER 3: 视觉物理一致性检测...")

        valid_screenshots: List[Dict[str, Any]] = []
        if isinstance(screenshots, list):
            for s in screenshots:
                p = s.get('path') or s.get('screenshot') or s.get('file')
                desc = s.get('description', s.get('desc', ''))
                if isinstance(p, str) and os.path.exists(p):
                    valid_screenshots.append({'path': p, 'description': desc})

        if not valid_screenshots:
            print("⚠️ Layer 3: 无有效截图，跳过视觉检测")
            # 如果没有截图，说明页面可能无法渲染，Layer 3 得分应为 0
            return {
                "has_errors": True, 
                "error_count": 1, 
                "score": 0, 
                "errors": [{"description": "无有效截图，可能是页面渲染失败"}], 
                "summary": "无截图可供检测"
            }

        # 在RL模式下，只选取首尾截图，减少Token消耗
        if self.rl_mode and len(valid_screenshots) > 2:
             print(f"⚠️ Layer 3 (RL模式): 仅选取首尾截图进行检测 (原{len(valid_screenshots)}张)")
             valid_screenshots = [valid_screenshots[0], valid_screenshots[-1]]

        image_paths = [s['path'] for s in valid_screenshots]
        
        simple_output = self.rl_mode
        
        if simple_output:
            prompt_text = """
你是一名顶尖的物理、数学教育专家。请仔细观察这些HTML页面截图，**专注于识别违反物理定律或数学原则的视觉呈现错误**。

**重要说明：如果页面内容不涉及数学或物理（例如是文学、历史、简单的信息展示等），请直接返回满分（100分），has_errors=false，不要强行寻找错误。**

**输出要求：**
请直接输出一个 **JSON数组**，其中每个元素对应一张截图。
每个元素包含: score (0-100), has_errors (bool), errors (list of objects with 'description')。
**不要输出Markdown标记，不要任何解释，直接以 [ 开头。**
"""
        else:
            if self.strict_mode:
                role_desc = "你是一名**极其严苛**的物理/数学教育专家。请仔细观察这些HTML页面截图，**对任何违反物理定律、数学原则、视觉呈现不专业或页面过于静态（缺乏交互/动态演示）的地方进行严厉批评**。"
                scoring_criteria = """
**评分标准（每张截图单独评分，总分100分，严格模式）：**
- 无错误且视觉生动：100分
- 页面过于静态/缺乏必要的动态演示：扣40分
- 每个轻微错误（如视觉对齐微瑕）：扣20分
- 每个中等错误（如图像模糊、逻辑不清）：扣50分
- 每个严重错误（严重违反物理定律、明显渲染错误）：扣100分
- 最低分：0分
"""
            else:
                role_desc = "你是一名顶尖的物理、数学教育专家。请仔细观察这些HTML页面截图，**专注于识别违反物理定律或数学原则的视觉呈现错误**。"
                scoring_criteria = """
**评分标准（每张截图单独评分，总分100分）：**
- 优秀交互与动态（加分项）：如果是交互式演示且有清晰的动画/状态变化，给满分。
- 无错误但静态：扣15分（如果题目要求是演示/模拟器，但页面是静态的）。
- 每个轻微错误：扣10分
- 每个中等错误：扣25分
- 每个严重错误（严重违反物理定律）：扣50分
- 最低分：0分
"""

            prompt_text = f"""
{role_desc}

**重要说明：如果页面内容不涉及数学或物理（例如是文学、历史、简单的信息展示等），请直接返回满分（100分），has_errors=false，不要强行寻找错误。但如果题目要求生成模拟器或交互式演示，而页面是静态的，请务必扣分。**

## 检测重点（按优先级）：
1. **页面是否过于静态（缺乏应有的动态/交互效果）** - **特别关注：** 是否有动画效果？是否有状态变化？如果是多页内容，是否仅仅是静态堆砌而没有交互切换？
2. 天平/杠杆类错误
3. 运动学错误
4. 几何/函数图像错误
5. 能量/电路类错误
6. 数值逻辑错误

## 输出要求：
请直接输出一个 **JSON数组**，其中每个元素对应一张截图。

{scoring_criteria}

**格式示例：**
```json
[
    {{
        "screenshot_index": 0,
        "screenshot_description": "截图描述",
        "has_errors": false,
        "score": 100,
        "score_breakdown": {{
            "total_deductions": 0,
            "minor_errors": 0,
            "moderate_errors": 0,
            "severe_errors": 0
        }},
        "errors": [],
        "summary": "无明显错误"
    }}
]
```

**重要：直接输出JSON数组，不要包含 ```json ... ``` 标记，不要有任何前缀文字！**
"""

        try:
            # 分批检测 (异步)
            # 简化为一次性调用 (如果图片不多) 或者依然分批
            # 这里简单起见，使用线程池调用同步 VLM
            loop = asyncio.get_event_loop()
            
            # 增加随机延时，错峰请求
            import random
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # 构造 batch prompt (简单处理，假设一次发完或者简单分批)
            # 为了简化异步逻辑，这里暂时还是同步分批，但在线程中运行
            
            def run_vlm_batch():
                batch_size = 3
                all_results = []
                for start in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[start:start + batch_size]
                    batch_prompt = (
                        prompt_text
                        + f"\n\n本轮有{len(batch_paths)}张截图（索引 {start}-{start + len(batch_paths) - 1}）。"
                    )
                    # 减少 max_tokens 如果是 simple_output
                    max_tokens = 2000 if simple_output else 12000
                    response_str = self.vlm(batch_prompt, image_paths=batch_paths, max_tokens=max_tokens, temperature=0.2, high_quality=high_quality)
                    
                    if response_str is None:
                        for i in range(len(batch_paths)):
                            all_results.append(self._make_placeholder_error(start + i, "VLM返回None"))
                        continue

                    resp = response_str['content'] if isinstance(response_str, dict) and 'content' in response_str else response_str
                    if resp is None:
                         for i in range(len(batch_paths)):
                            all_results.append(self._make_placeholder_error(start + i, "VLM content为None"))
                         continue

                    parsed = self._parse_json_response(resp)
                    if isinstance(parsed, list):
                        for i, item in enumerate(parsed):
                            all_results.append(self._standardize_visual_error(item, start + i))
                        if len(parsed) < len(batch_paths):
                            for i in range(len(parsed), len(batch_paths)):
                                all_results.append(self._make_placeholder_error(start + i, "VLM未返回该截图结果"))
                    elif isinstance(parsed, dict):
                        all_results.append(self._standardize_visual_error(parsed, start))
                        if len(batch_paths) > 1:
                            for i in range(1, len(batch_paths)):
                                all_results.append(self._make_placeholder_error(start + i, "VLM仅返回单个结果"))
                return all_results

            all_results = await loop.run_in_executor(None, run_vlm_batch)

            # 汇总结果
            total_errors = sum(len(r.get('errors', [])) for r in all_results)
            has_any_error = any(r.get('has_errors', False) for r in all_results)
            scores = [r.get('score', 100) for r in all_results]
            average_score = sum(scores) / len(scores) if scores else 0

            return {
                "has_errors": has_any_error,
                "error_count": total_errors,
                "score": round(average_score, 2),
                "score_breakdown": {
                    "average_score": round(average_score, 2),
                    "min_score": min(scores) if scores else 0,
                    "max_score": max(scores) if scores else 0,
                    "screenshots_with_errors": sum(1 for r in all_results if r.get('has_errors', False))
                },
                "screenshot_results": all_results,
                "summary": f"检测了{len(all_results)}张截图，平均分{round(average_score, 2)}分，发现{total_errors}个视觉物理错误" if has_any_error else f"检测了{len(all_results)}张截图，平均分{round(average_score, 2)}分，未发现视觉物理错误"
            }

        except Exception as e:
            print(f"⚠️ Layer 3 检测异常: {e}")
            return {"has_errors": True, "error_count": 1, "score": 0, "errors": [{"description": f"检测异常: {str(e)}"}], "summary": "检测过程出错"}

    def _standardize_visual_error(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """标准化视觉错误检测结果"""
        if not isinstance(item, dict):
            return self._make_placeholder_error(idx, "返回格式错误")
        
        return {
            "screenshot_index": item.get("screenshot_index", idx),
            "screenshot_description": item.get("screenshot_description", ""),
            "has_errors": item.get("has_errors", False),
            "score": item.get("score", 100 if not item.get("has_errors", False) else 0),
            "score_breakdown": item.get("score_breakdown", {}),
            "errors": item.get("errors", []),
            "summary": item.get("summary", "")
        }

    def _make_placeholder_error(self, idx: int, reason: str) -> Dict[str, Any]:
        """创建占位错误对象"""
        return {
            "screenshot_index": idx,
            "screenshot_description": "",
            "has_errors": True,
            "score": 0,
            "errors": [{"description": reason}],
            "summary": reason
        }

    def _detect_layer4_runtime_interaction(self, playwright_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """第四层：运行与交互能力检测"""
        if not isinstance(playwright_report, dict):
            return {
                "has_errors": False,
                "error_count": 0,
                "score": 60,
                "score_breakdown": {
                    "interaction_success_rate": 0,
                    "ineffective_interactions": 0,
                    "js_error_count": 0,
                    "console_error_count": 0,
                    "console_warning_count": 0,
                    "render_error_count": 0,
                    "screenshots_with_errors": 0
                },
                "errors": [],
                "summary": "无运行数据，采用静态回退"
            }

        summary = playwright_report.get("summary", {}) or {}
        runtime_errors = playwright_report.get("runtime_errors", {}) or {}
        screenshots = playwright_report.get("screenshots", []) or []

        interaction_success_rate = float(summary.get("interaction_success_rate", 0) or 0.0)
        total_interactions = int(summary.get("total_interactions", 0) or 0)
        ineffective_interactions = int(summary.get("ineffective_interactions", 0) or 0)
        failed_interactions = int(summary.get("failed_interactions", 0) or 0)
        no_change_interactions = int(summary.get("no_change_interactions", 0) or 0)
        no_effect_interactions = int(summary.get("no_effect_interactions", 0) or 0)
        js_error_count = int(summary.get("js_error_count", runtime_errors.get("page_errors") and len(runtime_errors.get("page_errors", [])) or 0) or 0)
        console_error_count = int(summary.get("console_error_count", runtime_errors.get("console_errors") and len([e for e in runtime_errors.get("console_errors", []) if e.get("type") == "error"]) or 0) or 0)
        console_warning_count = int(summary.get("console_warning_count", runtime_errors.get("console_errors") and len([e for e in runtime_errors.get("console_errors", []) if e.get("type") == "warning"]) or 0) or 0)
        render_error_count = int(summary.get("render_error_count", runtime_errors.get("render_errors") and len(runtime_errors.get("render_errors", [])) or 0) or 0)
        screenshots_with_errors = int(summary.get("screenshots_with_errors", 0) or 0)
        network_error_count = int(summary.get("network_error_count", 0) or 0)

        content_lengths: List[int] = []
        try:
            for s in screenshots:
                try:
                    content_lengths.append(int((s or {}).get("content_length", 0) or 0))
                except Exception:
                    content_lengths.append(0)
        except Exception:
            content_lengths = []
            
        if not content_lengths and self.rl_mode and render_error_count == 0:
            # RL模式下，如果没有截图且无渲染错误，假设内容长度正常
            initial_content_length = 1000
            avg_content_length = 1000.0
            max_content_length = 1000
            min_content_length = 1000
        else:
            initial_content_length = content_lengths[0] if content_lengths else 0
            avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0.0
            max_content_length = max(content_lengths) if content_lengths else 0
            min_content_length = min(content_lengths) if content_lengths else 0

        score = 100.0
        score -= (100.0 - interaction_success_rate) * 0.8
        score -= min(ineffective_interactions * 1 + no_effect_interactions * 1 + failed_interactions * 1 + no_change_interactions * 1, 20)
        score -= min(js_error_count * 12, 36)
        score -= min(console_error_count * 5, 20)
        score -= min(console_warning_count * 1, 10)
        score -= min(render_error_count * 20, 40)
        score -= min(screenshots_with_errors * 3, 15)
        blank_grace_condition = (initial_content_length < 10 and max_content_length >= 30)
        if avg_content_length < 10 and (render_error_count > 0 or interaction_success_rate == 0) and not blank_grace_condition:
            score -= 30
        
        # Strict Mode: Penalize Static Pages
        # 如果 total_interactions 很少，且页面内容长度没有显著变化，视为静态页面
        is_static_behavior = False
        
        # 启发式判断：交互次数少，或者交互后页面完全没变（content_length 变化极小）
        # 这里需要更复杂的逻辑，但基于现有数据：
        # 如果 no_change_interactions 占比很高，说明点击无效
        if total_interactions > 0:
            effective_rate = (total_interactions - no_change_interactions - no_effect_interactions - failed_interactions) / total_interactions
            if effective_rate < 0.2: # 有效交互低于 20%
                 is_static_behavior = True
        elif total_interactions == 0 and initial_content_length > 100:
             # 有内容但没交互点，纯展示页
             is_static_behavior = True

        if self.strict_mode:
            if is_static_behavior:
                 score -= 40
            elif no_change_interactions + no_effect_interactions > total_interactions * 0.8:
                 score -= 40
        else:
            # 普通模式下，也对静态行为进行一定惩罚，但更轻
            if is_static_behavior:
                score -= 15 # 静态页面扣 15 分

        score = max(0.0, round(score, 2))

        error_items: List[Dict[str, Any]] = []
        
        if is_static_behavior:
             error_items.append({
                "category": "交互性不足",
                "severity": "严重" if self.strict_mode else "中等",
                "description": "页面缺乏有效交互，无法实现翻页或状态切换 (Static Penalty)",
                "code_snippet": ""
            })

        if self.strict_mode:
            if no_change_interactions + no_effect_interactions > total_interactions * 0.8 and not is_static_behavior:
                 error_items.append({
                    "category": "交互无效",
                    "severity": "严重",
                    "description": "大部分交互无明显视觉反馈，页面缺乏动态性 (Strict Mode Penalty)",
                    "code_snippet": ""
                })

        if js_error_count > 0:
            error_items.append({
                "category": "JavaScript运行时错误",
                "severity": "严重" if js_error_count >= 3 else "中等",
                "description": f"检测到{js_error_count}个JS错误",
                "code_snippet": ""
            })
        if render_error_count > 0:
            error_items.append({
                "category": "渲染错误",
                "severity": "严重",
                "description": f"检测到{render_error_count}个渲染失败",
                "code_snippet": ""
            })
        if console_error_count > 0:
            error_items.append({
                "category": "控制台错误",
                "severity": "中等",
                "description": f"检测到{console_error_count}个控制台错误",
                "code_snippet": ""
            })
        if no_effect_interactions > 0 or failed_interactions > 0 or no_change_interactions > 0:
            total_ineff = no_effect_interactions + failed_interactions + no_change_interactions
            error_items.append({
                "category": "无效交互",
                "severity": "中等" if total_ineff <= 5 else "严重",
                "description": f"无效交互{total_ineff}次",
                "code_snippet": ""
            })
        if screenshots_with_errors > 0:
            error_items.append({
                "category": "截图错误标记",
                "severity": "轻微",
                "description": f"{screenshots_with_errors}张截图检测到页面错误",
                "code_snippet": ""
            })
        if avg_content_length < 10 and (render_error_count > 0 or interaction_success_rate == 0) and not blank_grace_condition:
            error_items.append({
                "category": "内容过少",
                "severity": "严重",
                "description": "页面内容过少或空白",
                "code_snippet": ""
            })

        error_count = js_error_count + render_error_count + console_error_count + screenshots_with_errors + no_effect_interactions + failed_interactions + no_change_interactions + (1 if (avg_content_length < 10 and (render_error_count > 0 or interaction_success_rate == 0) and not blank_grace_condition) else 0)
        has_errors = error_count > 0 or interaction_success_rate < 100.0

        return {
            "has_errors": has_errors,
            "error_count": int(error_count),
            "score": score,
            "score_breakdown": {
                "interaction_success_rate": round(interaction_success_rate, 2),
                "total_interactions": total_interactions,
                "ineffective_interactions": ineffective_interactions,
                "failed_interactions": failed_interactions,
                "no_change_interactions": no_change_interactions,
                "no_effect_interactions": no_effect_interactions,
                "js_error_count": js_error_count,
                "console_error_count": console_error_count,
                "console_warning_count": console_warning_count,
                "render_error_count": render_error_count,
                "screenshots_with_errors": screenshots_with_errors,
                "network_error_count": network_error_count,
                "initial_content_length": initial_content_length,
                "avg_content_length": round(avg_content_length, 2),
                "max_content_length": max_content_length,
                "min_content_length": min_content_length
            },
            "errors": error_items,
            "summary": f"交互成功率{round(interaction_success_rate, 2)}%，运行问题{int(error_count)}项，得分{score}"
        }

    def _generate_comprehensive_report(self, html_file: str, layer1: Dict, layer2: Dict, layer3: Dict, layer4: Dict, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """整合三层检测报告"""
        
        total_error_count = (
            layer1.get('error_count', 0) + 
            layer2.get('error_count', 0) + 
            layer3.get('error_count', 0) +
            layer4.get('error_count', 0)
        )
        
        has_any_error = (
            layer1.get('has_errors', False) or 
            layer2.get('has_errors', False) or 
            layer3.get('has_errors', False) or 
            layer4.get('has_errors', False)
        )
        
        # 获取各层得分
        layer1_score = layer1.get('score', 0)
        layer2_score = layer2.get('score', 0)
        layer3_score = layer3.get('score', 0)
        layer4_score = layer4.get('score', 0)
        
        # 计算加权总分
        default_weights = {
            "layer1": 0.20,
            "layer2": 0.25,
            "layer3": 0.25,
            "layer4": 0.30
        }
        
        if weights:
            default_weights.update(weights)
        weights = default_weights
        
        weighted_score = (
            layer1_score * weights["layer1"] + 
            layer2_score * weights["layer2"] + 
            layer3_score * weights["layer3"] +
            layer4_score * weights["layer4"]
        )

        # 运行时得分门控机制 (Runtime Score Gating)
        # 如果Layer 4 (运行与交互) 得分过低，说明页面不可用，强制压低总分
        if layer4_score < 20:
            # 严重故障（如白屏、无法加载），最高不超过20分
            if weighted_score > 20:
                print(f"⚠️ Layer 4 得分过低 ({layer4_score})，触发严重故障门控，总分由 {weighted_score} 降至 20")
                weighted_score = 20.0
        elif layer4_score < 40:
            # 重大功能缺陷，最高不超过40分
            if weighted_score > 40:
                print(f"⚠️ Layer 4 得分过低 ({layer4_score})，触发功能缺陷门控，总分由 {weighted_score} 降至 40")
                weighted_score = 40.0
        elif layer4_score < 60:
            # 不及格，最高不超过59
            if weighted_score >= 60:
                print(f"⚠️ Layer 4 不及格 ({layer4_score})，触发及格线门控，总分由 {weighted_score} 降至 59")
                weighted_score = 59.0

        # 生成改进建议
        suggestions = []
        
        if layer1.get('has_errors'):
            suggestions.append({
                "layer": "第一层（语法结构）",
                "priority": "高",
                "current_score": layer1_score,
                "suggestion": "请优先修复HTML/JavaScript/CSS的语法错误，这些错误会导致页面无法正常运行"
            })
        
        if layer2.get('has_errors'):
            critical_physics = any(
                err.get('severity') == '严重' 
                for err in layer2.get('errors', [])
            )
            suggestions.append({
                "layer": "第二层（物理数学原则）",
                "priority": "极高" if critical_physics else "高",
                "current_score": layer2_score,
                "suggestion": "代码中存在违反物理/数学原则的错误，这会误导学生对基础概念的理解，必须修正"
            })
        
        if layer3.get('has_errors'):
            suggestions.append({
                "layer": "第三层（视觉物理一致性）",
                "priority": "高",
                "current_score": layer3_score,
                "suggestion": "视觉呈现与物理原则不一致，学生会看到错误的演示效果，请检查绘图逻辑"
            })
        if layer4.get('has_errors'):
            suggestions.append({
                "layer": "第四层（运行与交互能力）",
                "priority": "极高" if (layer4.get('score', 0) < 60) else "高",
                "current_score": layer4_score,
                "suggestion": "页面存在运行或交互问题（JS错误、无效交互、渲染失败等），会直接导致教学不可用，请优先修复"
            })

        # 生成评级
        if weighted_score >= 90:
            grade = "优秀"
        elif weighted_score >= 80:
            grade = "良好"
        elif weighted_score >= 70:
            grade = "中等"
        elif weighted_score >= 60:
            grade = "及格"
        else:
            grade = "不及格"

        return {
            "检测元信息": {
                "HTML文件": html_file,
                "检测时间": datetime.now().isoformat(),
                "检测器版本": "2.1-错误检测+评分"
            },
            "检测结果总览": {
                "存在错误": has_any_error,
                "错误总数": total_error_count,
                "综合得分": round(weighted_score, 2),
                "评级": grade,
                "各层错误分布": {
                    "第一层_语法结构": layer1.get('error_count', 0),
                    "第二层_物理数学": layer2.get('error_count', 0),
                    "第三层_视觉一致性": layer3.get('error_count', 0),
                    "第四层_运行与交互能力": layer4.get('error_count', 0)
                },
                "各层得分": {
                    "第一层_语法结构": layer1_score,
                    "第二层_物理数学": layer2_score,
                    "第三层_视觉一致性": layer3_score,
                    "第四层_运行与交互能力": layer4_score
                },
                "权重配置": weights
            },
            "第一层_语法结构检测": layer1,
            "第二层_物理数学检测": layer2,
            "第三层_视觉一致性检测": layer3,
            "第四层_运行与交互能力检测": layer4,
            "改进建议": suggestions,
            "结论": f"综合得分{round(weighted_score, 2)}分（{grade}），{'发现错误，需要修正' if has_any_error else '未发现明显错误，通过检测'}"
        }

    def _create_error_report(self, html_file: str, error_message: str, error_type: str) -> Dict[str, Any]:
        """创建错误报告"""
        return {
            "状态": "error",
            "错误类型": error_type,
            "错误消息": error_message,
            "检测元信息": {
                "HTML文件": html_file,
                "检测时间": datetime.now().isoformat()
            },
            "检测结果总览": {
                "存在错误": True,
                "错误总数": 1,
                "综合得分": 0,
                "评级": "检测失败",
                "各层错误分布": {
                    "第一层_语法结构": 0,
                    "第二层_物理数学": 0,
                    "第三层_视觉一致性": 0
                },
                "各层得分": {
                    "第一层_语法结构": 0,
                    "第二层_物理数学": 0,
                    "第三层_视觉一致性": 0
                }
            },
            "结论": f"检测失败: {error_message}"
        }

    def _parse_json_response(self, response: str) -> Any:
        """从LLM响应中解析JSON"""
        if not response:
            print(f"⚠️ JSON解析失败: 响应为空")
            return {"has_errors": True, "errors": [{"description": "Empty response"}]}
        
        try:
            # 第一步：处理Markdown代码块
            cleaned_response = str(response).strip()
            
            # 移除前导的解释文字（如："根据检测..."）
            # 查找第一个 { 或 [ 的位置
            first_brace = cleaned_response.find('{')
            first_bracket = cleaned_response.find('[')
            
            # 确定JSON开始位置
            json_start = -1
            if first_brace != -1 and first_bracket != -1:
                json_start = min(first_brace, first_bracket)
            elif first_brace != -1:
                json_start = first_brace
            elif first_bracket != -1:
                json_start = first_bracket
            
            # 如果找到JSON开始位置，且前面有文字，则跳过前面的文字
            if json_start > 0:
                # 检查前面是否有```json标记
                prefix = cleaned_response[:json_start]
                if '```json' in prefix or '```' in prefix:
                    # 找到```json或```后的第一个{或[
                    cleaned_response = cleaned_response[json_start:]
                else:
                    # 直接从第一个{或[开始
                    cleaned_response = cleaned_response[json_start:]
            
            # 移除尾部的```或其他标记
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3].rstrip()
            
            # 尝试直接解析
            return json.loads(cleaned_response)
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON解析失败: {e}")
            print(f"原始响应前500字符: {response[:500]}")
            
            try:
                text = str(response)
                
                # 修复常见错误
                text = text.replace(',}', '}').replace(',]', ']')
                
                # 尝试提取数组
                start = text.find('[')
                end = text.rfind(']')
                if start != -1 and end != -1 and end > start:
                    candidate = text[start:end+1]
                    try:
                        if not candidate.rstrip().endswith(']'):
                            last_complete = candidate.rfind('},')
                            if last_complete != -1:
                                candidate = candidate[:last_complete+1] + ']'
                        
                        parsed = json.loads(candidate)
                        print(f"✅ 成功从数组中提取JSON")
                        return parsed
                    except Exception:
                        pass
                
                # 尝试提取对象
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    candidate = text[start:end+1]
                    try:
                        parsed = json.loads(candidate)
                        print(f"✅ 成功从对象中提取JSON")
                        return parsed
                    except Exception:
                        pass
                
            except Exception as extract_error:
                print(f"⚠️ JSON提取失败: {extract_error}")
                pass
            
            return {
                "has_errors": True,
                "errors": [{"description": "Failed to parse JSON response"}]
            }

    async def batch_detect_async(self, html_files: List[str], base_output_dir: Optional[str] = None, concurrency: int = 4, skip_existing: bool = False) -> Dict[str, Any]:
        """异步批量检测多个HTML文件，支持并发和跳过已检测文件"""
        print(f"🚀 开始批量检测 {len(html_files)} 个HTML文件 (并发: {concurrency}, 跳过已存: {skip_existing})...")
        
        # 依赖检查
        try:
            import playwright
        except ImportError:
            print("❌ 错误: 未安装 playwright，无法进行截图和交互检测。")
            print("请运行: npm install @playwright/test && npx playwright install chromium")
            return {'status': 'error', 'message': 'Playwright not installed'}

        batch_results = {}
        successful_detections = 0
        failed_detections = 0
        total_errors_found = 0
        scores_list = []
        skipped_count = 0
        
        # 确定截图模式
        screenshot_mode = 'key_frames' if self.rl_mode else 'all'
        
        # 使用信号量控制并发
        sem = asyncio.Semaphore(concurrency)
        
        async def process_single_file(idx: int, html_file: str):
            nonlocal successful_detections, failed_detections, total_errors_found, skipped_count
            
            async with sem:
                try:
                    html_dir = Path(html_file).parent
                    root = Path(base_output_dir) if base_output_dir else html_dir
                    output_dir_for_save = root / Path(html_file).stem
                    
                    # 检查是否跳过
                    if skip_existing:
                        # 简单的检查：看目录下是否有 error_detection_report_*.json
                        if output_dir_for_save.exists():
                            reports = list(output_dir_for_save.glob("error_detection_report_*.json"))
                            if reports:
                                print(f"⏭️  跳过已检测文件 {idx}/{len(html_files)}: {os.path.basename(html_file)}")
                                skipped_count += 1
                                return None
                    
                    print(f"📄 [{idx}/{len(html_files)}] 正在检测: {os.path.basename(html_file)}")
                    output_dir_for_save.mkdir(parents=True, exist_ok=True)

                    # 生成Playwright截图和报告
                    playwright_report = None
                    screenshots_for_detect = None
                    
                    try:
                        tester = PlaywrightHTMLTester(
                            html_file_path=html_file, 
                            output_dir=str(output_dir_for_save),
                            screenshot_mode=screenshot_mode
                        )
                        await tester.run_complete_test()
                        
                        # 读取报告
                        report_path = output_dir_for_save / 'playwright_test_report.json'
                        if report_path.exists():
                            with open(report_path, 'r', encoding='utf-8') as f:
                                playwright_report = json.load(f)
                            
                            # 提取截图
                            ss = playwright_report.get('screenshots')
                            if isinstance(ss, list):
                                screenshots_for_detect = []
                                for s in ss:
                                    p = s.get('path')
                                    desc = s.get('description', '')
                                    if isinstance(p, str) and os.path.exists(p):
                                        screenshots_for_detect.append({'path': p, 'description': desc})
                    
                    except Exception as e:
                        print(f"⚠️ [{os.path.basename(html_file)}] Playwright测试失败: {e}")

                    # 执行三层检测 (异步)
                    detection_result = await self.detect_async(
                        html_file,
                        playwright_report=playwright_report,
                        screenshots=screenshots_for_detect
                    )

                    # 检查是否是系统错误
                    is_system_error = detection_result.get('状态') == 'error'
                    
                    if is_system_error:
                        result_entry = {
                            'status': 'failed',
                            'error': detection_result.get('错误消息', 'Unknown error'),
                            'error_type': detection_result.get('错误类型', 'unknown')
                        }
                        failed_detections += 1
                        print(f"❌ [{os.path.basename(html_file)}] 检测失败")
                        return (html_file, result_entry)
                    else:
                        # 保存报告
                        self.save_detection_report(detection_result, str(output_dir_for_save), html_file)
                        
                        # 统计错误和分数
                        overview = detection_result.get('检测结果总览', {})
                        error_count = overview.get('错误总数', 0)
                        score = overview.get('综合得分', 0)
                        total_errors_found += error_count
                        scores_list.append(score)
                        
                        result_entry = {
                            'status': 'success',
                            'detection': detection_result,
                            'error_count': error_count,
                            'score': score,
                            'output_dir': str(output_dir_for_save)
                        }
                        successful_detections += 1
                        
                        if error_count > 0:
                            print(f"⚠️  [{os.path.basename(html_file)}] 完成，得分 {score}，发现 {error_count} 个错误")
                        else:
                            print(f"✅ [{os.path.basename(html_file)}] 完成，得分 {score}，无错误")
                        
                        return (html_file, result_entry)
                        
                except Exception as e:
                    print(f"❌ [{os.path.basename(html_file)}] 处理异常: {e}")
                    failed_detections += 1
                    return (html_file, {'status': 'failed', 'error': str(e)})

        # 创建所有任务
        tasks = [process_single_file(i, f) for i, f in enumerate(html_files, 1)]
        results = await asyncio.gather(*tasks)
        
        # 收集结果
        for res in results:
            if res:
                batch_results[res[0]] = res[1]
        
        # 计算平均分
        average_score = sum(scores_list) / len(scores_list) if scores_list else 0
        
        print(f"\n{'='*60}")
        print(f"📊 批量检测完成!")
        print(f"✅ 成功: {successful_detections}, ❌ 失败: {failed_detections}, ⏭️  跳过: {skipped_count}")
        print(f"⚠️  错误总数: {total_errors_found}")
        print(f"📈 平均得分: {round(average_score, 2)} 分")
        if scores_list:
            print(f"📊 得分范围: {min(scores_list)} - {max(scores_list)} 分")
        print(f"{'='*60}")
        
        return {
            'batch_meta': {
                'total_files': len(html_files),
                'successful_detections': successful_detections,
                'failed_detections': failed_detections,
                'skipped_files': skipped_count,
                'total_errors_found': total_errors_found,
                'average_score': round(average_score, 2),
                'min_score': min(scores_list) if scores_list else 0,
                'max_score': max(scores_list) if scores_list else 0,
                'detection_time': datetime.now().isoformat()
            },
            'individual_results': batch_results
        }

    def save_detection_report(self, detection_result: Dict[str, Any], output_dir: str, html_file: str):
        """保存检测报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = os.path.splitext(os.path.basename(html_file))[0]
        report_filename = f"error_detection_report_{html_filename}_{timestamp}.json"
        report_path = os.path.join(output_dir, report_filename)
        
        report_data = {
            'html_file': html_file,
            'detection_time': datetime.now().isoformat(),
            'detection_result': detection_result
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 检测报告已保存到: {report_path}")
        return report_path

    @staticmethod
    def find_html_files(directory: str) -> List[str]:
        """在指定目录中查找HTML文件"""
        html_files: List[str] = []
        directory_path = Path(directory)

        if not directory_path.exists():
            print(f"❌ 错误: 目录不存在 {directory}")
            return html_files

        group_dirs = [d for d in directory_path.iterdir() 
                     if d.is_dir() and d.name.lower().startswith('group')]
        
        if group_dirs:
            for g in group_dirs:
                for html_file in g.glob("**/*.html"):
                    html_files.append(str(html_file.resolve()))
            print(f"📁 在 {len(group_dirs)} 个 group_* 子目录中找到 {len(html_files)} 个HTML文件")
        else:
            for html_file in directory_path.glob("**/*.html"):
                html_files.append(str(html_file.resolve()))
            print(f"📁 递归找到 {len(html_files)} 个HTML文件")

        return html_files

    @staticmethod
    def find_html_files_limited_subdirs(directory: str, max_groups: int) -> List[str]:
        """在指定目录中仅处理前N个子文件夹中的HTML文件"""
        html_files: List[str] = []
        directory_path = Path(directory)

        if not directory_path.exists():
            print(f"❌ 错误: 目录不存在 {directory}")
            return html_files

        subdirs = [d for d in directory_path.iterdir() if d.is_dir()]
        if not subdirs:
            print(f"ℹ️  目录下无子文件夹，回退为递归查找所有HTML文件")
            return HTMLErrorDetector.find_html_files(directory)

        selected = sorted(subdirs, key=lambda p: p.name)[:max_groups]
        print(f"📁 选择 {len(selected)} / {len(subdirs)} 个子文件夹进行检测")

        for sd in selected:
            for html_file in sd.glob("**/*.html"):
                html_files.append(str(html_file.resolve()))

        print(f"🧾 在选中的 {len(selected)} 个子文件夹中共找到 {len(html_files)} 个HTML文件")
        return html_files


async def amain():
    """主函数"""
    parser = argparse.ArgumentParser(description="三层HTML错误检测系统（错误检测专用）")
    parser.add_argument("--html_input", nargs="?", default="",
                       help="要检测的HTML文件或目录路径")
    parser.add_argument("--max_groups", "-n", type=int, default=0,
                        help="本次运行最多检测的子文件夹数量（0表示不限制）")
    parser.add_argument("--output-dir", "-o", default="",
                        help="检测报告的输出目录（默认与输入文件同目录）")
    parser.add_argument("--concurrency", "-c", type=int, default=4,
                        help="批量检测时的并发数（默认4）")
    parser.add_argument("--skip-existing", "-s", action="store_true",
                        help="跳过已存在检测报告的文件")
    parser.add_argument("--rl_mode", action="store_true",
                        help="启用RL快速训练模式（跳过LLM/VLM检查，仅进行静态和运行时交互检测）")
    args = parser.parse_args()

    if not args.html_input:
        print("❌ 错误: 请提供要检测的HTML文件或目录路径")
        print(f"使用方法: python {sys.argv[0]} <文件/目录路径> [--max_groups N]")
        sys.exit(1)

    html_input = args.html_input
    input_path = Path(html_input)
    
    if not input_path.exists():
        print(f"❌ 错误: 路径不存在 {html_input}")
        sys.exit(1)
    
    # 初始化检测器
    detector = HTMLErrorDetector(rl_mode=args.rl_mode)
    
    if input_path.is_file():
        # 单文件检测
        print(f"🚀 单文件检测模式: {html_input}")
        
        html_dir = input_path.parent
        
        # 确定输出目录
        output_root = args.output_dir if args.output_dir else str(html_dir)
        output_dir = str(Path(output_root) / input_path.stem)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 生成Playwright报告和截图
        playwright_report = None
        screenshots = None
        
        try:
            screenshot_mode = 'key_frames' if args.rl_mode else 'all'
            tester = PlaywrightHTMLTester(
                html_file_path=str(input_path), 
                output_dir=output_dir,
                screenshot_mode=screenshot_mode
            )
            await tester.run_complete_test()
            
            report_path = Path(output_dir) / 'playwright_test_report.json'
            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    playwright_report = json.load(f)
                
                ss = playwright_report.get('screenshots')
                if isinstance(ss, list):
                    screenshots = []
                    for s in ss:
                        p = s.get('path')
                        desc = s.get('description', '')
                        if isinstance(p, str) and os.path.exists(p):
                            screenshots.append({'path': p, 'description': desc})
        except Exception as e:
            print(f"⚠️ Playwright测试失败: {e}")
        
        # 执行检测
        result = await detector.detect_async(
            str(input_path),
            playwright_report=playwright_report,
            screenshots=screenshots
        )
        
        # 保存报告
        detector.save_detection_report(result, output_dir, str(input_path))
        
        # 打印结果摘要
        print("\n" + "=" * 80)
        print("📊 检测结果摘要")
        print("=" * 80)
        
        overview = result.get('检测结果总览', {})
        print(f"综合得分: {overview.get('综合得分', 0)} 分")
        print(f"评级: {overview.get('评级', '未知')}")
        print(f"存在错误: {'是' if overview.get('存在错误') else '否'}")
        print(f"错误总数: {overview.get('错误总数', 0)}")
        
        print(f"\n各层得分:")
        scores = overview.get('各层得分', {})
        print(f"  - 第一层（语法结构）: {scores.get('第一层_语法结构', 0)} 分")
        print(f"  - 第二层（物理数学）: {scores.get('第二层_物理数学', 0)} 分")
        print(f"  - 第三层（视觉一致性）: {scores.get('第三层_视觉一致性', 0)} 分")
        
        dist = overview.get('各层错误分布', {})
        print(f"\n各层错误分布:")
        print(f"  - 第一层（语法结构）: {dist.get('第一层_语法结构', 0)} 个")
        print(f"  - 第二层（物理数学）: {dist.get('第二层_物理数学', 0)} 个")
        print(f"  - 第三层（视觉一致性）: {dist.get('第三层_视觉一致性', 0)} 个")
        
        print(f"\n结论: {result.get('结论', '')}")
        print("=" * 80)
        
    else:
        # 批量检测
        directory = str(input_path)
        print(f"🚀 批量检测模式: {directory}")
        
        if args.max_groups and args.max_groups > 0:
            html_files = HTMLErrorDetector.find_html_files_limited_subdirs(directory, args.max_groups)
        else:
            html_files = HTMLErrorDetector.find_html_files(directory)
        
        if not html_files:
            print(f"❌ 在目录 {directory} 中未找到HTML文件")
            return
        
        # 执行批量检测
        batch_report = await detector.batch_detect_async(
            html_files, 
            base_output_dir=args.output_dir if args.output_dir else None,
            concurrency=args.concurrency,
            skip_existing=args.skip_existing
        )
        
        # 打印批量检测摘要
        print("\n" + "=" * 80)
        print("📊 批量检测摘要")
        print("=" * 80)
        
        meta = batch_report.get('batch_meta', {})
        print(f"📁 总文件数: {meta.get('total_files', 0)}")
        print(f"✅ 成功检测: {meta.get('successful_detections', 0)}")
        print(f"❌ 检测失败: {meta.get('failed_detections', 0)}")
        print(f"⚠️  发现错误总数: {meta.get('total_errors_found', 0)}")
        print(f"📈 平均得分: {meta.get('average_score', 0)} 分")
        print(f"📊 得分范围: {meta.get('min_score', 0)} - {meta.get('max_score', 0)} 分")

        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(amain())
# Example: python main.py --html_input examples/generated --skip-existing
