#!/usr/bin/env python3
"""
融合版 Playwright HTML 测试器 - 改进版
- 增强错误检测：交互失败记录、JS运行时错误监听、交互有效性验证
- 提高语法检查严格度：针对教育内容的关键项
"""

import os
import re
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# 可选依赖：Playwright
PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.async_api import async_playwright, Page
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    async_playwright = None
    Page = None

# 可选依赖：BeautifulSoup
BS4_AVAILABLE = False
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except Exception:
    BeautifulSoup = None


class HTMLSyntaxChecker:
    """HTML语法与可访问性检查器（综合静态检查，BeautifulSoup可选）"""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def check_html_file(self, file_path: str) -> Dict[str, Any]:
        self.errors, self.warnings = [], []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"无法读取文件: {e}")
            return self._generate_report(file_path, '', [], [], [])

        # 检测是否为教育内容
        is_educational = self._is_educational_content(content)

        # 基础结构与声明
        self._check_doctype(content)
        self._check_html_structure(content)
        self._check_tag_closure(content)
        self._check_required_tags(content, is_educational)

        # 额外可访问性建议
        self._check_accessibility_hints(content, is_educational)

        # ✅ 新增：检测无效的按钮和控件
        self._check_ineffective_controls(content)

        # BeautifulSoup 深度检查（如果可用）
        img_tags: List[str] = re.findall(r"<img[^>]*>", content, flags=re.IGNORECASE)
        inputs: List[str] = re.findall(r"<input[^>]*>", content, flags=re.IGNORECASE)
        labels: List[str] = re.findall(r"<label[^>]*>", content, flags=re.IGNORECASE)
        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(content, 'html.parser')
                self._check_with_beautifulsoup(soup)
            except Exception as e:
                self.errors.append(f"HTML解析错误: {e}")

        return self._generate_report(file_path, content, img_tags, inputs, labels)

    def _is_educational_content(self, content: str) -> bool:
        """检测是否为教育内容（数学/物理/科学）"""
        indicators = [
            'katex', 'mathjax', 'math-tex', 'math.js',  # 数学渲染库
            r'\\[(\s*', r'\\]', r'\\\(', r'\\\)',  # LaTeX
            'canvas', 'svg', 'd3.js', 'plotly', 'chart.js',  # 可视化
            'equation', 'formula', 'theorem', '公式', '方程', '函数', '定理',
            'physics', 'mathematics', '物理', '数学'
        ]
        content_lower = content.lower()
        return any(ind.lower() in content_lower for ind in indicators)

    def _check_doctype(self, content: str):
        if not re.search(r'<!DOCTYPE\s+html', content, re.IGNORECASE):
            self.warnings.append("缺少<!DOCTYPE html>声明")

    def _check_html_structure(self, content: str):
        if not re.search(r'<html[^>]*>', content, re.IGNORECASE):
            self.errors.append("缺少<html>标签")
        if not re.search(r'<head[^>]*>', content, re.IGNORECASE):
            self.errors.append("缺少<head>标签")  # 提升为错误
        if not re.search(r'<body[^>]*>', content, re.IGNORECASE):
            self.errors.append("缺少<body>标签")  # 提升为错误

    def _check_tag_closure(self, content: str):
        self_closing = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
                        'link', 'meta', 'param', 'source', 'track', 'wbr'}
        tags = re.findall(r'<(/?)(\w+)[^>]*>', content, re.IGNORECASE)
        stack: List[str] = []
        for is_closing, name in tags:
            name = name.lower()
            if name in self_closing:
                continue
            if is_closing:
                if not stack:
                    self.errors.append(f"多余的闭合标签: </{name}> - 可能导致渲染错误")
                elif stack[-1] != name:
                    self.errors.append(f"标签不匹配: 期望</{stack[-1]}>，但找到</{name}> - 严重渲染错误")
                else:
                    stack.pop()
            else:
                stack.append(name)
        for tag in stack:
            self.errors.append(f"未闭合的标签: <{tag}> - 可能导致页面布局错乱")

        # ✅ 新增：检测常见的渲染错误模式
        self._check_render_breaking_patterns(content)

    def _check_required_tags(self, content: str, is_educational: bool):
        """检查必需标签，教育内容更严格"""
        if not re.search(r'<title[^>]*>', content, re.IGNORECASE):
            if is_educational:
                self.errors.append("教育内容缺少<title>标签")
            else:
                self.warnings.append("建议添加<title>标签")
        
        if re.search(r"<html[^>]*lang=", content, flags=re.IGNORECASE) is None:
            self.warnings.append('建议在<html>上设置lang属性以提升可访问性')
        
        if 'meta name="viewport"' not in content.lower():
            if is_educational:
                self.errors.append('教育内容缺少<meta name="viewport">，可能影响移动端显示')
            else:
                self.warnings.append('缺少<meta name="viewport">，在移动端可能体验欠佳')
        
        if 'meta charset' not in content.lower():
            self.warnings.append('缺少<meta charset>，可能导致字符编码问题')

    def _check_accessibility_hints(self, content: str, is_educational: bool):
        """可访问性检查，教育内容更严格"""
        img_tags = re.findall(r"<img[^>]*>", content, flags=re.IGNORECASE)
        img_without_alt = [t for t in img_tags if re.search(r"alt=", t, flags=re.IGNORECASE) is None]
        if img_without_alt:
            if is_educational:
                self.errors.append(f"教育内容中有{len(img_without_alt)}个<img>缺少alt属性（图表/公式必须有描述）")
            else:
                self.warnings.append(f"{len(img_without_alt)}个<img>缺少alt属性")

        inputs = re.findall(r"<input[^>]*>", content, flags=re.IGNORECASE)
        labels = re.findall(r"<label[^>]*>", content, flags=re.IGNORECASE)
        if inputs and not labels:
            self.warnings.append('存在<input>但缺少<label>，可访问性欠佳')

        if '<style' in content.lower() and '<link' not in content.lower():
            self.warnings.append('使用了内联<style>，考虑抽离为外部CSS以便缓存和维护')

    def _check_render_breaking_patterns(self, content: str):
        """检测可能导致渲染失败的常见模式"""
        # 1. 检测<script>标签中的语法错误（简单检测）
        script_blocks = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL | re.IGNORECASE)
        for i, script in enumerate(script_blocks):
            # 检测常见的JS语法错误
            if 'function(' in script or 'function (' in script:
                # 检测函数括号不匹配
                open_parens = script.count('(')
                close_parens = script.count(')')
                open_braces = script.count('{')
                close_braces = script.count('}')
                if open_parens != close_parens:
                    self.errors.append(f"脚本块{i+1}中括号不匹配 - 可能导致JS错误")
                if open_braces != close_braces:
                    self.errors.append(f"脚本块{i+1}中花括号不匹配 - 可能导致JS错误")

        # 2. 检测<style>标签中的CSS错误（简单检测）
        style_blocks = re.findall(r'<style[^>]*>(.*?)</style>', content, re.DOTALL | re.IGNORECASE)
        for i, style in enumerate(style_blocks):
            open_braces = style.count('{')
            close_braces = style.count('}')
            if open_braces != close_braces:
                self.errors.append(f"样式块{i+1}中花括号不匹配 - 可能导致CSS解析错误")

        # 3. 检测空的body标签（内容过少）
        body_match = re.search(r'<body[^>]*>(.*?)</body>', content, re.DOTALL | re.IGNORECASE)
        if body_match:
            body_content = body_match.group(1).strip()
            # 移除script和style标签后检查
            body_without_script_style = re.sub(r'<script[^>]*>.*?</script>', '', body_content, flags=re.DOTALL | re.IGNORECASE)
            body_without_script_style = re.sub(r'<style[^>]*>.*?</style>', '', body_without_script_style, flags=re.DOTALL | re.IGNORECASE)
            body_without_script_style = re.sub(r'<!--.*?-->', '', body_without_script_style, flags=re.DOTALL)
            body_text = re.sub(r'<[^>]+>', '', body_without_script_style).strip()

            if len(body_text) < 10:
                self.errors.append(f"body标签内容过少（仅{len(body_text)}字符）- 可能是空白页面或渲染失败")

        # 4. 检测无效的属性值
        if re.search(r'<\w+[^>]*\s+\w+=""[^>]*>', content):
            self.warnings.append("发现空的属性值，可能导致功能异常")

        # 5. 检测重复的id属性
        id_pattern = r'id\s*=\s*["\']([^"\']+)["\']'
        ids = re.findall(id_pattern, content, re.IGNORECASE)
        duplicate_ids = [id_val for id_val in set(ids) if ids.count(id_val) > 1]
        for dup_id in duplicate_ids:
            self.errors.append(f"重复的ID: '{dup_id}' - 违反HTML规范，可能导致JS选择器失败")

    def _check_ineffective_controls(self, content: str):
        """检测无实质性功能的按钮和控件"""
        # 1. 提取所有脚本内容
        script_content = '\n'.join(re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL | re.IGNORECASE))

        # 2. 查找所有按钮
        button_pattern = r'<button([^>]*)>(.*?)</button>'
        buttons = re.findall(button_pattern, content, re.DOTALL | re.IGNORECASE)

        for i, (attrs, text) in enumerate(buttons):
            button_id = None
            has_onclick = False
            has_type_submit = False

            # 检查按钮属性
            id_match = re.search(r'id\s*=\s*["\']([^"\']+)["\']', attrs, re.IGNORECASE)
            if id_match:
                button_id = id_match.group(1)

            if re.search(r'onclick\s*=\s*["\'][^"\']*["\']', attrs, re.IGNORECASE):
                onclick_value = re.search(r'onclick\s*=\s*["\']([^"\']*)["\']', attrs, re.IGNORECASE)
                if onclick_value and onclick_value.group(1).strip():
                    has_onclick = True
                else:
                    # 空的onclick
                    self.errors.append(f"按钮{i+1}有空的onclick属性 - 不会有任何效果")

            if re.search(r'type\s*=\s*["\']submit["\']', attrs, re.IGNORECASE):
                has_type_submit = True

            # 检查是否在脚本中被引用
            referenced_in_script = False
            if button_id:
                # 检查是否通过getElementById等方式引用
                if (f"getElementById('{button_id}')" in script_content or
                    f'getElementById("{button_id}")' in script_content or
                    f"querySelector('#" in script_content and button_id in script_content or
                    f'addEventListener' in script_content):
                    referenced_in_script = True

            # 判断按钮是否有功能
            if not has_onclick and not has_type_submit and not referenced_in_script:
                button_desc = f"按钮{i+1}"
                if button_id:
                    button_desc += f" (id='{button_id}')"
                button_text = text.strip()[:30]
                if button_text:
                    button_desc += f" - '{button_text}'"
                self.warnings.append(f"{button_desc} 似乎没有绑定任何事件处理器 - 可能无实质性功能")

        # 3. 检查input[type="button"]
        input_button_pattern = r'<input([^>]*type\s*=\s*["\']button["\'][^>]*)>'
        input_buttons = re.findall(input_button_pattern, content, re.IGNORECASE)

        for i, attrs in enumerate(input_buttons):
            has_onclick = 'onclick' in attrs.lower()
            button_id = None
            id_match = re.search(r'id\s*=\s*["\']([^"\']+)["\']', attrs, re.IGNORECASE)
            if id_match:
                button_id = id_match.group(1)

            referenced_in_script = False
            if button_id and (f"getElementById('{button_id}')" in script_content or
                            f'getElementById("{button_id}")' in script_content):
                referenced_in_script = True

            if not has_onclick and not referenced_in_script:
                desc = f"input按钮{i+1}"
                if button_id:
                    desc += f" (id='{button_id}')"
                self.warnings.append(f"{desc} 没有绑定任何事件处理器 - 可能无实质性功能")

    def _check_with_beautifulsoup(self, soup: 'BeautifulSoup'):
        ids = [el.get('id') for el in soup.find_all(attrs={'id': True})]
        duplicate_ids = [i for i in set(ids) if ids.count(i) > 1]
        for dup in duplicate_ids:
            self.errors.append(f"重复的ID: {dup}")

    def _generate_report(self, file_path: str, content: str, img_tags: List[str], 
                        inputs: List[str], labels: List[str]) -> Dict[str, Any]:
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'meta': {
                'html_file': file_path,
                'img_count': len(img_tags),
                'input_count': len(inputs),
                'label_count': len(labels),
                'is_educational': self._is_educational_content(content) if content else False
            }
        }


class PlaywrightHTMLTester:
    """Playwright HTML自动化测试器（带静态回退）- 改进版"""

    def __init__(self, html_file_path: str, output_dir: Optional[str] = None, 
                 headless: bool = True, static_only: bool = False, screenshot_mode: str = 'all',
                 offline_mode: bool = False, local_assets_dir: Optional[str] = None):
        self.html_file_path = Path(html_file_path)
        self.headless = headless
        self.static_only = static_only or (not PLAYWRIGHT_AVAILABLE)
        self.screenshot_mode = screenshot_mode  # 'all', 'errors_only', 'key_frames', 'none'
        self.offline_mode = offline_mode
        self.local_assets_dir = Path(local_assets_dir) if local_assets_dir else Path(__file__).parent / 'assets'
        self.local_assets_dir.mkdir(exist_ok=True, parents=True)

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            base = self.html_file_path.stem
            self.output_dir = Path(f"test_outputs/{base}_{ts}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.screenshot_counter = 0
        self.interaction_log: List[Dict[str, Any]] = []
        self.screenshots_info: List[Dict[str, Any]] = []

        # 新增：错误追踪
        self.console_errors: List[Dict[str, Any]] = []
        self.page_errors: List[Dict[str, Any]] = []
        self.render_errors: List[Dict[str, Any]] = []  # 渲染失败错误
        self._console_error_buckets: Dict[str, Dict[str, Any]] = {}
        self._console_warning_buckets: Dict[str, Dict[str, Any]] = {}
        self.request_count: int = 0
        self.response_count: int = 0
        self.failed_request_count: int = 0
        self.browser_instance = None

        # 日志
        self.logger = self._setup_logger()

        # 检测器
        self.syntax_checker = HTMLSyntaxChecker()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('PlaywrightHTMLTester')
        logger.setLevel(logging.INFO)
        if logger.handlers:
            return logger
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        file_handler = logging.FileHandler(self.output_dir / 'test_log.txt', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(fmt)
        file_handler.setFormatter(fmt)
        logger.addHandler(console)
        logger.addHandler(file_handler)
        return logger

    async def run_complete_test(self) -> None:
        self.logger.info("开始HTML测试流程")

        # 步骤1：静态语法检查
        syntax_report = self.syntax_checker.check_html_file(str(self.html_file_path))
        self._log_syntax_report(syntax_report)

        # 步骤2：真实交互与截图（可选）
        try:
            if not self.static_only:
                await self._run_interaction_tests()
            else:
                self.logger.info("已启用 --static-only，跳过浏览器交互")
        except Exception as e:
            # ✅ 修改：捕获致命错误，不让程序直接退出，保证能走到步骤3
            self.logger.error(f"❌ 浏览器交互测试发生致命错误: {e}")
            self.page_errors.append({'error': f"Browser Crashed: {str(e)}", 'timestamp': datetime.now().isoformat()})

        # 步骤3：生成最终报告 (无论前面是否成功，都要执行)
        try:
            self._generate_final_report(syntax_report)
            self.logger.info(f"测试流程完成，结果保存在: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"❌ 生成报告失败: {e}")

    def _log_syntax_report(self, report: Dict[str, Any]):
        if report.get('errors'):
            self.logger.error(f"发现 {len(report['errors'])} 个语法错误:")
            for e in report['errors']:
                self.logger.error(f"  ❌ {e}")
        else:
            self.logger.info("✅ 未发现语法错误")
        if report.get('warnings'):
            self.logger.warning(f"发现 {len(report['warnings'])} 个警告:")
            for w in report['warnings']:
                self.logger.warning(f"  ⚠️  {w}")
        else:
            self.logger.info("✅ 未发现警告")

    async def cleanup(self):
        if self.browser_instance:
            try:
                self.logger.info(f"🧹 [Tester] 正在强制关闭遗留浏览器...")
                await self.browser_instance.close()
            except Exception:
                pass
            self.browser_instance = None

    async def _run_interaction_tests(self):
        if not PLAYWRIGHT_AVAILABLE:
            return
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            self.browser_instance = browser
            context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
            
            # 【关键优化】设置 Context 级别的默认超时 (20秒)
            # 这会限制所有后续操作（如 click, wait_for_selector）的最大等待时间
            context.set_default_timeout(20000)

            page = await context.new_page()
            # 【关键优化】页面加载超时 (20秒)
            page.set_default_timeout(20000)
            page.set_default_navigation_timeout(20000)
            
            # ✅ 新增：监听控制台错误
            page.on('console', self._handle_console_message)

            # ✅ 新增：监听页面错误
            page.on('pageerror', self._handle_page_error)
            page.on('request', lambda req: setattr(self, 'request_count', self.request_count + 1))
            page.on('response', lambda res: setattr(self, 'response_count', self.response_count + 1))
            page.on('requestfailed', lambda req: setattr(self, 'failed_request_count', self.failed_request_count + 1))
            
            # ✅ 新增：离线资源拦截
            if self.offline_mode:
                await page.route("**/*", self._handle_offline_route)

            try:
                await page.add_init_script("""
                (()=>{
                  try{
                    window.__mutationCount = 0;
                    const obs = new MutationObserver(()=>{window.__mutationCount++});
                    obs.observe(document.documentElement,{childList:true,subtree:true,attributes:true});
                  }catch(e){}
                })();
                """)
            except Exception:
                pass
            
            try:
                file_url = f"file://{self.html_file_path.absolute()}"
                self.logger.info(f"正在加载HTML文件: {file_url}")
                try:
                    # 缩短超时时间到 15秒 (对于本地生成文件足够了)
                    await page.goto(file_url, wait_until='load', timeout=15000)
                    self.logger.info("页面加载成功，等待网络空闲...")
                    await page.wait_for_load_state('networkidle', timeout=5000)
                except Exception as load_error:
                    self.logger.warning(f"页面加载或网络等待超时: {load_error}")
                    try:
                        await page.wait_for_load_state('domcontentloaded', timeout=5000)
                        self.logger.info("DOM内容已加载，继续测试")
                    except Exception as dom_error:
                        self.logger.error(f"DOM加载失败: {dom_error}")
                        await browser.close()
                        return

                try:
                    await self._ensure_content_ready(page)
                    # 无论是否有交互，必须截一张初始图作为基准
                    await self._take_screenshot(page, "initial_load", "页面初始加载状态")
                except Exception as se:
                    self.logger.warning(f"初始截图失败: {se}")

                try:
                    await self._test_interactive_elements(page)
                except Exception as ie:
                    self.logger.warning(f"交互元素测试失败: {ie}")
            except Exception as e:
                self.logger.error(f"交互测试过程中发生错误: {e}")
            finally:
                await browser.close()

    async def _handle_offline_route(self, route):
        """处理离线模式下的网络请求 (增强版)"""
        req = route.request
        url = req.url.lower()
        
        # 1. 允许本地文件
        if url.startswith('file:') or url.startswith('data:'):
            await route.continue_()
            return

        # ================= [新增 Mock 逻辑] =================
        # 针对 Plotly (防止 Plotly is not defined)
        if 'plotly' in url:
            mock_js = "window.Plotly = {newPlot: function(){ return Promise.resolve(); }, react: function(){}, purge: function(){}, Plots: {resize: function(){}}};"
            await route.fulfill(status=200, content_type='application/javascript', body=mock_js)
            # self.logger.info(f"🛡️ Mocked Plotly: {url}") # 可选日志
            return

        # 针对 MathJax
        if 'mathjax' in url:
            mock_js = "window.MathJax = {typeset: function(){}, typesetPromise: () => Promise.resolve()};"
            await route.fulfill(status=200, content_type='application/javascript', body=mock_js)
            return

        # 针对 jQuery
        if 'jquery' in url or '/jq' in url:
             mock_js = "window.$ = window.jQuery = function(){ return {ready: function(fn){fn()}, click: function(){}, on: function(){}} };"
             await route.fulfill(status=200, content_type='application/javascript', body=mock_js)
             return
        
        # 针对 ECharts (防止图表报错)
        if 'echarts' in url:
            mock_js = "window.echarts = {init: ()=>({setOption:()=>{}, resize:()=>{}, on:()=>{}})};"
            await route.fulfill(status=200, content_type='application/javascript', body=mock_js)
            return
        # ===================================================

        # 2. 尝试读取本地资源 (保持原有逻辑)
        filename = url.split('/')[-1].split('?')[0]
        local_file = self.local_assets_dir / filename
        
        if local_file.exists():
            try:
                ct = 'application/javascript'
                if filename.endswith('.css'): ct = 'text/css'
                elif filename.endswith('.json'): ct = 'application/json'
                elif filename.endswith('.png'): ct = 'image/png'
                elif filename.endswith('.jpg'): ct = 'image/jpeg'
                elif filename.endswith('.svg'): ct = 'image/svg+xml'
                
                with open(local_file, 'rb') as f:
                    await route.fulfill(status=200, content_type=ct, body=f.read())
                self.logger.info(f"🔄 离线重定向: {url} -> {local_file.name}")
                return
            except Exception:
                pass
        
        # 3. 其他外部资源一律返回空，防止超时等待
        await route.fulfill(status=200, body=b'')

    def _handle_console_message(self, msg):
        """处理控制台消息"""
        if msg.type in ['error', 'warning']:
            key = (msg.text or '').strip()
            bucket = self._console_error_buckets if msg.type == 'error' else self._console_warning_buckets
            if key not in bucket:
                bucket[key] = {
                    'type': msg.type,
                    'text': msg.text,
                    'location': msg.location if hasattr(msg, 'location') else None,
                    'timestamp': datetime.now().isoformat(),
                    'count': 1
                }
                self.console_errors.append(bucket[key])
                if msg.type == 'error':
                    self.logger.error(f"🔴 浏览器控制台错误: {msg.text}")
                else:
                    self.logger.warning(f"🟡 浏览器控制台警告: {msg.text}")
            else:
                bucket[key]['count'] = int(bucket[key]['count']) + 1

    def _handle_page_error(self, exception):
        """处理页面JavaScript错误"""
        self.page_errors.append({
            'error': str(exception),
            'timestamp': datetime.now().isoformat()
        })
        self.logger.error(f"💥 JavaScript运行时错误: {exception}")

    async def _test_interactive_elements(self, page: 'Page'):
        selectors: List[Tuple[str, str]] = [
            ('button', 'button'),
            ('input[type="button"]', 'input按钮'),
            ('input[type="submit"]', '提交按钮'),
            ('input[type="reset"]', '重置按钮'),
            ('input[type="text"]', '文本输入框'),
            ('input[type="email"]', '邮箱输入框'),
            ('input[type="password"]', '密码输入框'),
            ('input[type="number"]', '数字输入框'),
            ('input[type="range"]', '滑块'),
            ('input[type="checkbox"]', '复选框'),
            ('input[type="radio"]', '单选框'),
            ('select', '下拉选择框'),
            ('textarea', '文本区域'),
            ('a[href]', '链接'),
            ('[onclick]', '点击事件元素'),
            ('[onchange]', '变化事件元素'),
            # ✅ 新增：更多交互元素
            ('[contenteditable]', '可编辑内容'),
            ('canvas', 'Canvas画布'),
            ('details', '折叠详情'),
            ('[role="button"]', 'ARIA按钮'),
        ]
        total_attempted = 0
        total_succeeded = 0
        for selector, label in selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    self.logger.info(f"发现 {len(elements)} 个{label}")
                    for i, el in enumerate(elements):
                        total_attempted += 1
                        success = await self._interact_with_element(page, el, label, i)
                        if success:
                            total_succeeded += 1
                        await page.wait_for_timeout(300)
            except Exception as e:
                self.logger.error(f"处理{label}时发生错误: {e}")
        
        success_rate = (total_succeeded / total_attempted * 100) if total_attempted > 0 else 0
        self.logger.info(f"总共尝试 {total_attempted} 次交互，成功 {total_succeeded} 次 ({success_rate:.1f}%)")

    async def _interact_with_element(self, page: 'Page', element, element_type: str, index: int) -> bool:
        """与元素交互，返回是否成功"""
        try:
            tag = await element.evaluate('el => el.tagName.toLowerCase()')
            element_id = await element.get_attribute('id') or f"element_{index}"
            text = await element.text_content() or ""
            await element.scroll_into_view_if_needed(timeout=2000)
            await page.wait_for_timeout(200)
            await element.wait_for_element_state('visible', timeout=2000)
            desc = f"{element_type}_{index}_{element_id}"

            if tag == 'button' or (tag == 'input' and (await element.get_attribute('type')) in ['button', 'submit', 'reset']):
                return await self._click_element(page, element, desc, text)
            elif tag == 'input':
                input_type = await element.get_attribute('type') or 'text'
                return await self._handle_input_element(page, element, input_type, desc)
            elif tag == 'select':
                return await self._handle_select_element(page, element, desc)
            elif tag == 'textarea':
                return await self._handle_textarea_element(page, element, desc)
            elif tag == 'canvas':
                return await self._handle_canvas_element(page, element, desc)
            elif tag == 'details':
                return await self._handle_details_element(page, element, desc)
            elif tag == 'a':
                href = await element.get_attribute('href')
                if href and not href.startswith(('http', 'mailto:', 'tel:')):
                    return await self._click_element(page, element, desc, text)
            else:
                return await self._click_element(page, element, desc, text)
        except Exception as e:
            self.logger.warning(f"与元素交互时发生错误: {e}")
            # ✅ 记录失败的交互
            self.interaction_log.append({
                'action': 'interact',
                'element': f"{element_type}_{index}",
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    async def _click_element(self, page: 'Page', element, desc: str, text: str = "") -> bool:
        """点击元素，返回是否成功（验证点击是否有实质性效果）"""
        try:
            disabled = await element.get_attribute('disabled')
            pre_filled = 0
            if not disabled:
                try:
                    pre_filled = await self._prepare_form_context(page, element, False)
                except Exception:
                    pre_filled = 0
            # ✅ 点击前：记录页面状态
            before_snapshot = await self._capture_page_state(page)

            await element.click(timeout=2000)
            await page.wait_for_timeout(500)  # 等待可能的DOM更新

            # ✅ 点击后：检查页面是否有变化
            after_snapshot = await self._capture_page_state(page)
            has_effect = self._compare_page_states(before_snapshot, after_snapshot)

            if has_effect:
                self.logger.info(f"✅ 点击元素: {desc} - {text[:50]} (已验证有效果)")
                await self._take_screenshot(page, f"click_{desc}", f"点击{desc}")
                self.interaction_log.append({
                    'action': 'click',
                    'element': desc,
                    'text': text,
                    'status': 'success',
                    'has_effect': True,
                    'precondition_applied': pre_filled > 0,
                    'precondition_fill_count': pre_filled,
                    'timestamp': datetime.now().isoformat()
                })
                return True
            else:
                retry_filled = 0
                try:
                    retry_filled = await self._prepare_form_context(page, element, True)
                except Exception:
                    retry_filled = 0
                if retry_filled > 0:
                    before_snapshot2 = await self._capture_page_state(page)
                    try:
                        await element.click(timeout=10000)
                        await page.wait_for_timeout(600)
                    except Exception:
                        pass
                    after_snapshot2 = await self._capture_page_state(page)
                    has_effect2 = self._compare_page_states(before_snapshot2, after_snapshot2)
                    if has_effect2:
                        self.logger.info(f"✅ 点击元素: {desc} (满足前置条件后生效)")
                        await self._take_screenshot(page, f"click_{desc}", f"点击{desc}")
                        self.interaction_log.append({
                            'action': 'click',
                            'element': desc,
                            'text': text,
                            'status': 'success',
                            'has_effect': True,
                            'precondition_applied': True,
                            'precondition_fill_count': pre_filled + retry_filled,
                            'timestamp': datetime.now().isoformat()
                        })
                        return True
                self.logger.warning(f"⚠️  点击元素 {desc} 无实质性效果 (DOM/内容无变化)")
                await self._take_screenshot(page, f"click_{desc}_no_effect", f"点击{desc}无效果")
                self.interaction_log.append({
                    'action': 'click',
                    'element': desc,
                    'text': text,
                    'status': 'no_effect',
                    'has_effect': False,
                    'precondition_applied': (pre_filled + retry_filled) > 0,
                    'precondition_fill_count': pre_filled + retry_filled,
                    'timestamp': datetime.now().isoformat()
                })
                return False
        except Exception as e:
            self.logger.warning(f"❌ 无法点击元素 {desc}: {e}")
            self.interaction_log.append({
                'action': 'click',
                'element': desc,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    async def _handle_input_element(self, page: 'Page', element, input_type: str, desc: str) -> bool:
        """处理输入元素，返回是否成功"""
        try:
            if input_type in ['text', 'email', 'password', 'search', 'url']:
                val = self._get_test_value_for_input_type(input_type)
                await element.fill(val, timeout=10000)
                self.logger.info(f"✅ 填充输入框: {desc} = {val}")
                await self._take_screenshot(page, f"input_{desc}", f"填充{desc}")
                status = 'success'
            elif input_type == 'number':
                await element.fill('123', timeout=10000)
                self.logger.info(f"✅ 填充数字输入框: {desc} = 123")
                await self._take_screenshot(page, f"input_{desc}", f"填充{desc}")
                status = 'success'
            elif input_type == 'range':
                # ✅ 改进：验证滑块是否真的改变了值
                status = await self._robust_adjust_range(page, element, desc)
                await self._take_screenshot(page, f"range_{desc}", f"调整{desc}")
            elif input_type in ['checkbox', 'radio']:
                await element.check(timeout=10000)
                self.logger.info(f"✅ 选中: {desc}")
                await self._take_screenshot(page, f"check_{desc}", f"选中{desc}")
                status = 'success'
            else:
                status = 'skipped'
            
            self.interaction_log.append({
                'action': f'input_{input_type}',
                'element': desc,
                'status': status,
                'timestamp': datetime.now().isoformat()
            })
            return status == 'success'
        except Exception as e:
            self.logger.warning(f"❌ 无法操作输入元素 {desc}: {e}")
            self.interaction_log.append({
                'action': f'input_{input_type}',
                'element': desc,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    async def _robust_adjust_range(self, page: 'Page', element, desc: str) -> str:
        """调整滑块并验证是否生效，返回状态"""
        try:
            # ✅ 获取初始值
            initial_value = await element.evaluate('el => el.value')
            
            # 读取滑块参数
            params = await element.evaluate('''el => ({
                min: Number(el.min || 0), 
                max: Number(el.max || 100), 
                step: Number(el.step || 1), 
                value: Number(el.value || 0)
            })''')
            min_v = params.get('min', 0)
            max_v = params.get('max', 100)
            step = params.get('step', 1)

            # 目标值：尽量与当前值不同（靠近 75% 处）
            target = min_v + (max_v - min_v) * 0.75
            # 对齐到步长
            try:
                if step > 0:
                    target = round(target / step) * step
            except Exception:
                pass

            # 方式1：直接设置并触发事件
            try:
                await element.evaluate('''(el, v) => {
                    el.value = String(v);
                    el.dispatchEvent(new Event('input', {bubbles: true}));
                    el.dispatchEvent(new Event('change', {bubbles: true}));
                }''', target)
            except Exception as e:
                self.logger.debug(f"直接设值失败: {e}")

            # 方式2：键盘递增
            try:
                await element.focus()
                for _ in range(5):
                    await page.keyboard.press('ArrowRight')
                    await page.wait_for_timeout(80)
            except Exception as e:
                self.logger.debug(f"键盘递增失败: {e}")

            # 方式3：真实拖拽
            try:
                bbox = await element.bounding_box()
                if bbox:
                    start_x = bbox['x'] + bbox['width'] * 0.25
                    end_x = bbox['x'] + bbox['width'] * 0.80
                    y = bbox['y'] + bbox['height'] / 2
                    await page.mouse.move(start_x, y)
                    await page.mouse.down()
                    await page.mouse.move(end_x, y, steps=6)
                    await page.mouse.up()
            except Exception as e:
                self.logger.debug(f"拖拽失败: {e}")

            # ✅ 等待事件处理完成
            await page.wait_for_timeout(500)
            
            # ✅ 验证值是否改变
            final_value = await element.evaluate('el => el.value')
            
            if str(initial_value) != str(final_value):
                self.logger.info(f"✅ 滑块值已改变: {desc} ({initial_value} → {final_value})")
                return 'success'
            else:
                self.logger.warning(f"⚠️  滑块值未改变: {desc} (仍为 {initial_value})")
                return 'no_change'
                
        except Exception as e:
            self.logger.warning(f"❌ 调整滑块失败 {desc}: {e}")
            return 'failed'

    async def _handle_select_element(self, page: 'Page', element, desc: str) -> bool:
        """处理下拉选择框，返回是否成功"""
        try:
            options = await element.query_selector_all('option')
            if len(options) > 1:
                option_value = await options[1].get_attribute('value')
                if option_value:
                    await element.select_option(option_value, timeout=10000)
                else:
                    await element.select_option(index=1, timeout=10000)
                self.logger.info(f"✅ 选择下拉选项: {desc}")
                await self._take_screenshot(page, f"select_{desc}", f"选择{desc}")
                self.interaction_log.append({
                    'action': 'select',
                    'element': desc,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                })
                return True
        except Exception as e:
            self.logger.warning(f"❌ 无法操作选择框 {desc}: {e}")
            self.interaction_log.append({
                'action': 'select',
                'element': desc,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    async def _handle_textarea_element(self, page: 'Page', element, desc: str) -> bool:
        """处理文本区域，返回是否成功"""
        try:
            txt = "这是测试文本内容。\n支持多行输入。"
            await element.fill(txt, timeout=10000)
            self.logger.info(f"✅ 填充文本区域: {desc}")
            await self._take_screenshot(page, f"textarea_{desc}", f"填充{desc}")
            self.interaction_log.append({
                'action': 'textarea',
                'element': desc,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            self.logger.warning(f"❌ 无法操作文本区域 {desc}: {e}")
            self.interaction_log.append({
                'action': 'textarea',
                'element': desc,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    async def _handle_canvas_element(self, page: 'Page', element, desc: str) -> bool:
        """处理Canvas元素"""
        try:
            # 简单点击Canvas
            await element.click(timeout=10000)
            self.logger.info(f"✅ 点击Canvas: {desc}")
            await self._take_screenshot(page, f"canvas_{desc}", f"Canvas交互{desc}")
            self.interaction_log.append({
                'action': 'canvas_click',
                'element': desc,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            self.logger.warning(f"❌ 无法操作Canvas {desc}: {e}")
            self.interaction_log.append({
                'action': 'canvas_click',
                'element': desc,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    async def _handle_details_element(self, page: 'Page', element, desc: str) -> bool:
        """处理折叠详情元素"""
        try:
            await element.click(timeout=10000)
            self.logger.info(f"✅ 展开/折叠详情: {desc}")
            await self._take_screenshot(page, f"details_{desc}", f"详情{desc}")
            self.interaction_log.append({
                'action': 'details_toggle',
                'element': desc,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            self.logger.warning(f"❌ 无法操作详情元素 {desc}: {e}")
            self.interaction_log.append({
                'action': 'details_toggle',
                'element': desc,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False

    async def _capture_page_state(self, page: 'Page') -> Dict[str, Any]:
        """捕获页面当前状态快照"""
        try:
            # 捕获多个维度的状态
            state = {
                'body_text': await page.evaluate('document.body.innerText') or '',
                'body_html_length': len(await page.evaluate('document.body.innerHTML') or ''),
                'visible_elements_count': await page.evaluate('''() => {
                    const all = document.querySelectorAll('*');
                    return Array.from(all).filter(el => {
                        const style = window.getComputedStyle(el);
                        return style.display !== 'none' && style.visibility !== 'hidden';
                    }).length;
                }'''),
                'canvas_data': await page.evaluate('''() => {
                    const canvases = document.querySelectorAll('canvas');
                    return Array.from(canvases).map(c => {
                        try { return c.toDataURL(); } catch(e) { return ''; }
                    });
                }'''),
                'mutation_count': await page.evaluate('window.__mutationCount || 0')
            }
            return state
        except Exception as e:
            self.logger.debug(f"捕获页面状态失败: {e}")
            return {}

    def _compare_page_states(self, before: Dict[str, Any], after: Dict[str, Any]) -> bool:
        """比较两个页面状态，判断是否有实质性变化"""
        if not before or not after:
            return False

        # 1. 文本内容变化
        text_before = before.get('body_text', '').strip()
        text_after = after.get('body_text', '').strip()
        if text_before != text_after:
            return True

        # 2. HTML长度显著变化（超过5%）
        html_len_before = before.get('body_html_length', 0)
        html_len_after = after.get('body_html_length', 0)
        if html_len_before > 0:
            change_ratio = abs(html_len_after - html_len_before) / html_len_before
            if change_ratio > 0.05:  # 超过5%变化
                return True

        # 3. 可见元素数量变化
        visible_before = before.get('visible_elements_count', 0)
        visible_after = after.get('visible_elements_count', 0)
        if visible_before != visible_after:
            return True

        # 4. Canvas内容变化（如绘图更新）
        canvas_before = before.get('canvas_data', [])
        canvas_after = after.get('canvas_data', [])
        if canvas_before != canvas_after:
            return True

        if int(after.get('mutation_count', 0)) > int(before.get('mutation_count', 0)):
            return True

        return False

    async def _ensure_content_ready(self, page: 'Page') -> None:
        try:
            attempts = [1000, 2000, 4000]
            for idx, delay in enumerate(attempts):
                try:
                    length = await page.evaluate('document.body && document.body.innerText ? document.body.innerText.length : 0')
                except Exception:
                    length = 0
                if length >= 30:
                    break
                try:
                    await page.wait_for_load_state('networkidle', timeout=max(500, delay))
                except Exception:
                    pass
                await page.wait_for_timeout(delay)
                try:
                    length2 = await page.evaluate('document.body && document.body.innerText ? document.body.innerText.length : 0')
                except Exception:
                    length2 = 0
                if length2 >= 30:
                    break
                try:
                    await self._take_screenshot(page, f"initial_load_retry_{idx+1}", f"初始加载重试{idx+1}")
                except Exception:
                    pass
        except Exception:
            pass

    async def _prepare_form_context(self, page: 'Page', element, force: bool) -> int:
        try:
            js = r"""
            (el, force)=>{
              const form = el.closest('form');
              const root = form || document;
              let count=0;

              const fillTextByConstraint = (elem, defVal)=>{
                const minLen = parseInt(elem.getAttribute('minlength')||'0')||0;
                const maxLen = parseInt(elem.getAttribute('maxlength')||'0')||0;
                const pattern = elem.getAttribute('pattern')||'';
                let val = defVal || '';
                if(pattern){
                  if(/https?:\/\//.test(pattern)){
                    val = 'https://example.com';
                  }else if(/@/.test(pattern)||/\S+@\S+\.\S+/.test(pattern)){
                    val = 'test@example.com';
                  }else if(/\\d|\[0-9\]/.test(pattern)){
                    const len = Math.max(minLen||3,3);
                    val = '0'.repeat(Math.min(len, maxLen || len));
                  }else if(/[A-Za-z]/.test(pattern)||/\\w/.test(pattern)){
                    const base = 'TestValueXYZabc';
                    const need = Math.max(minLen||5,5);
                    val = base.slice(0, Math.min(need, maxLen || base.length));
                  }
                }
                if(!val) val = defVal || '测试文本';
                if(maxLen && val.length>maxLen) val = val.slice(0,maxLen);
                if(val.length < minLen) val = val + 'x'.repeat(minLen - val.length);
                return val;
              };

              const fillNum = (min,max,step)=>{
                let v = Number.isFinite(min)?min:0;
                if(Number.isFinite(max)) v = Math.min(v+1,max);
                if(Number.isFinite(step)&&step>0){ v = Math.round(v/step)*step; }
                return String(v);
              };

              const applyFill = (i)=>{
                const tag = i.tagName.toLowerCase();
                if(tag==='input'){
                  const tp = (i.getAttribute('type')||'text').toLowerCase();
                  if(['text','password','search'].includes(tp)){
                    const v = fillTextByConstraint(i, i.value);
                    i.value = v;
                    i.dispatchEvent(new Event('input',{bubbles:true}));
                    i.dispatchEvent(new Event('change',{bubbles:true}));
                    return 1;
                  }else if(tp==='email'){
                    i.value = 'test@example.com';
                    i.dispatchEvent(new Event('input',{bubbles:true}));
                    i.dispatchEvent(new Event('change',{bubbles:true}));
                    return 1;
                  }else if(tp==='url'){
                    i.value = 'https://example.com';
                    i.dispatchEvent(new Event('input',{bubbles:true}));
                    i.dispatchEvent(new Event('change',{bubbles:true}));
                    return 1;
                  }else if(tp==='number' || tp==='range'){
                    const min = Number(i.min);
                    const max = Number(i.max);
                    const step = Number(i.step);
                    i.value = fillNum(min,max,step);
                    i.dispatchEvent(new Event('input',{bubbles:true}));
                    i.dispatchEvent(new Event('change',{bubbles:true}));
                    return 1;
                  }else if(tp==='checkbox' || tp==='radio'){
                    if(!i.checked){ i.checked = true; i.dispatchEvent(new Event('change',{bubbles:true})); }
                    return 1;
                  }else{
                    const v = fillTextByConstraint(i, i.value);
                    i.value = v;
                    i.dispatchEvent(new Event('input',{bubbles:true}));
                    i.dispatchEvent(new Event('change',{bubbles:true}));
                    return 1;
                  }
                }else if(tag==='select'){
                  const opts = Array.from(i.querySelectorAll('option'));
                  if(opts.length>0){
                    const opt = opts[Math.min(1,opts.length-1)];
                    if(opt){ i.value = opt.value; i.dispatchEvent(new Event('change',{bubbles:true})); return 1; }
                  }
                }else if(tag==='textarea'){
                  const v = fillTextByConstraint(i, i.value);
                  i.value = v;
                  i.dispatchEvent(new Event('input',{bubbles:true}));
                  i.dispatchEvent(new Event('change',{bubbles:true}));
                  return 1;
                }
                return 0;
              };

              // 解析按钮依赖：从 inline onclick / 函数源码中解析选择器与ID
              const selectors = [];
              const onclickAttr = el.getAttribute('onclick')||'';
              const onclickSrc = onclickAttr || (el.onclick && el.onclick.toString()) || '';
              const dsReq = el.getAttribute('data-requires')||'';
              const dsSel = el.getAttribute('data-requires-selectors')||'';
              const addSel = s=>{ if(s && !selectors.includes(s)) selectors.push(s); };
              (onclickSrc.match(/getElementById\(['\"]([^'\"]+)['\"]\)/g)||[]).forEach(m=>{ const id=m.replace(/.*getElementById\(['\"]/,'').replace(/['\"]\).*/,''); addSel('#'+id); });
              (onclickSrc.match(/querySelector(All)?\(['\"]([^'\"]+)['\"]\)/g)||[]).forEach(m=>{ const sel=m.replace(/.*querySelector(All)?\(['\"]/,'').replace(/['\"]\).*/,''); addSel(sel); });
              dsReq.split(',').map(s=>s.trim()).forEach(id=>{ if(id) addSel('#'+id); });
              dsSel.split(',').map(s=>s.trim()).forEach(sel=>{ if(sel) addSel(sel); });

              // 优先填充解析到的依赖字段
              for(const sel of selectors){
                try{
                  const targets = Array.from(root.querySelectorAll(sel));
                  for(const t of targets){
                    if(t.disabled) continue;
                    count += applyFill(t);
                  }
                }catch(e){}
              }

              // 其余必填项与强制填充
              const inputs = Array.from(root.querySelectorAll('input, select, textarea'));
              for(const i of inputs){
                const required = !!i.required;
                if(!(required || force)) continue;
                if(i.disabled) continue;
                count += applyFill(i);
              }
              return count;
            }
            """
            return await page.evaluate(js, element, force)
        except Exception:
            return 0

    def _get_test_value_for_input_type(self, input_type: str) -> str:
        return {
            'text': '测试文本',
            'email': 'test@example.com',
            'password': 'TestPassword123',
            'search': '搜索关键词',
            'url': 'https://example.com'
        }.get(input_type, '测试值')

    async def _take_screenshot(self, page: 'Page', filename: str, description: str):
        """截图并检查页面状态"""
        try:
            # 检查模式
            if self.screenshot_mode == 'none':
                return
            
            # ✅ 检查页面是否有错误元素
            page_has_errors = False
            try:
                error_elements = await page.query_selector_all('.error, .exception, #error')
                page_has_errors = len(error_elements) > 0
            except Exception:
                pass
            
            # ✅ 检查页面内容长度
            content_length = 0
            try:
                body_text = await page.evaluate('document.body.innerText')
                content_length = len(body_text.strip()) if body_text else 0
                if content_length < 10:
                    error_msg = f"页面内容为空或过少 (仅{content_length}字符)"
                    if self.screenshot_mode == 'errors_only' or self.screenshot_mode == 'all':
                         # 即使是errors_only，如果渲染失败也应该记录
                         self.logger.error(f"❌ 渲染失败: {filename} - {error_msg}")
                    self.render_errors.append({
                        'error': error_msg,
                        'screenshot': filename,
                        'content_length': content_length,
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                self.logger.warning(f"⚠️  无法检查页面内容: {e}")

            # 决定是否截图
            should_take = False
            if self.screenshot_mode == 'all':
                should_take = True
            elif self.screenshot_mode == 'errors_only':
                if page_has_errors or content_length < 10:
                    should_take = True
            elif self.screenshot_mode == 'key_frames':
                # 保留初始加载(initial_load) 和 最终状态(或者有错误的)
                # 我们通过 description 或 filename 来判断是否是关键帧
                # 简单起见：保留第一次截图，和最后一次(难以预知)，以及有错误的
                # 这里改为：保留 explicit 的 "initial_load" 和 任何 "final" 标记，以及有错误的
                if page_has_errors or content_length < 10:
                    should_take = True
                elif "initial_load" in filename or "click_" in filename or "final" in filename:
                    # 交互后的截图通常以 click_ 开头，也算关键帧
                    should_take = True

            if should_take:
                self.screenshot_counter += 1
                ss_name = f"{self.screenshot_counter:03d}_{filename}.png"
                ss_path = self.output_dir / ss_name
                await page.screenshot(path=str(ss_path), full_page=True)
                
                # ✅ 验证截图完整性
                if not self._verify_image_integrity(ss_path):
                    self.logger.warning(f"⚠️ 截图验证失败（文件损坏或为空），已删除: {ss_name}")
                    # 记录渲染错误，确保 Layer 4 也能感知到此问题
                    self.render_errors.append({
                        'error': f"截图生成失败（可能因渲染崩溃或内存溢出）: {ss_name}",
                        'screenshot': ss_name,
                        'timestamp': datetime.now().isoformat()
                    })
                    if ss_path.exists():
                        ss_path.unlink()
                    return

                self.logger.info(f"📸 截图保存: {ss_name} - {description}")
                
                self.screenshots_info.append({
                    'path': str(ss_path),
                    'filename': ss_name,
                    'description': description,
                    'timestamp': datetime.now().isoformat(),
                    'page_has_errors': page_has_errors,
                    'content_length': content_length
                })
        except Exception as e:
            self.logger.error(f"❌ 截图失败: {e}")

    def _verify_image_integrity(self, image_path: Path) -> bool:
        """验证生成的图片是否有效"""
        try:
            if not image_path.exists() or image_path.stat().st_size == 0:
                return False
            # 尝试用 PIL 打开验证（如果安装了 PIL）
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    img.verify()
                return True
            except ImportError:
                # 如果没有 PIL，仅检查文件大小
                return True
            except Exception:
                return False
        except Exception:
            return False

    def _generate_final_report(self, syntax_report: Dict[str, Any]):
        """生成最终报告，包含完整的错误统计"""
        # ✅ 计算交互统计
        total_interactions = len(self.interaction_log)
        successful_interactions = len([i for i in self.interaction_log if i.get('status') == 'success'])
        failed_interactions = len([i for i in self.interaction_log if i.get('status') == 'failed'])
        no_change_interactions = len([i for i in self.interaction_log if i.get('status') == 'no_change'])
        no_effect_interactions = len([i for i in self.interaction_log if i.get('status') == 'no_effect'])  # 新增：无效果交互

        # ✅ 修正：将无效果交互也算作失败
        effective_interactions = successful_interactions
        ineffective_interactions = failed_interactions + no_change_interactions + no_effect_interactions
        success_rate = (effective_interactions / total_interactions * 100) if total_interactions > 0 else 0

        unique_console_errors = [v for v in self._console_error_buckets.values()]
        unique_console_warnings = [v for v in self._console_warning_buckets.values()]
        net_err_keys = [k for k in self._console_error_buckets.keys() if 'ERR_NAME_NOT_RESOLVED' in k or 'net::ERR_' in k]
        network_error_count = len(net_err_keys)
        report = {
            'test_info': {
                'html_file': str(self.html_file_path),
                'output_dir': str(self.output_dir),
                'playwright_available': PLAYWRIGHT_AVAILABLE,
                'headless': self.headless,
                'static_only': self.static_only,
                'test_timestamp': datetime.now().isoformat()
            },
            'syntax_check': syntax_report,
            # ✅ 运行时错误
            'runtime_errors': {
                'console_errors': unique_console_errors,
                'page_errors': self.page_errors,
                'render_errors': self.render_errors,
                'has_js_errors': len(self.page_errors) > 0,
                'has_render_errors': len(self.render_errors) > 0,
                'console_error_count': len(unique_console_errors),
                'console_warning_count': len(unique_console_warnings),
                'network_error_count': network_error_count,
                'network_request_count': self.request_count,
                'network_response_count': self.response_count,
                'failed_request_count': self.failed_request_count
            },
            'screenshots': self.screenshots_info,
            'interactions': self.interaction_log,
            # ✅ 改进：更详细的统计
            'summary': {
                # 语法检查
                'has_syntax_errors': bool(syntax_report.get('errors')),
                'syntax_error_count': len(syntax_report.get('errors', [])),
                'syntax_warning_count': len(syntax_report.get('warnings', [])),

                # 运行时错误
                'has_js_errors': len(self.page_errors) > 0,
                'js_error_count': len(self.page_errors),
                'console_error_count': len(unique_console_errors),
                'console_warning_count': len(unique_console_warnings),
                'network_error_count': network_error_count,
                'network_request_count': self.request_count,
                'network_response_count': self.response_count,
                'failed_request_count': self.failed_request_count,

                # ✅ 渲染错误统计
                'has_render_errors': len(self.render_errors) > 0,
                'render_error_count': len(self.render_errors),

                # 交互统计
                'total_interactions': total_interactions,
                'successful_interactions': successful_interactions,
                'effective_interactions': effective_interactions,  # 新增：有效交互数
                'failed_interactions': failed_interactions,
                'no_change_interactions': no_change_interactions,
                'no_effect_interactions': no_effect_interactions,  # 新增：无效果交互数
                'ineffective_interactions': ineffective_interactions,  # 新增：总无效交互数
                'interaction_success_rate': round(success_rate, 2),

                # 截图统计
                'screenshot_count': len(self.screenshots_info),
                'screenshots_with_errors': len([s for s in self.screenshots_info if s.get('page_has_errors')]),

                # ✅ 整体质量评分（0-100）
                'overall_quality_score': self._calculate_quality_score(
                    syntax_report,
                    success_rate,
                    len(self.page_errors),
                    len(unique_console_errors),
                    len(self.render_errors),
                    network_error_count,
                    self.failed_request_count
                )
            }
        }
        
        # 保存报告
        report_path = self.output_dir / 'playwright_test_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # ✅ 打印摘要
        #self._print_summary(report)

    def _calculate_quality_score(self, syntax_report: Dict[str, Any],
                                 interaction_rate: float,
                                 js_error_count: int,
                                 console_error_count: int,
                                 render_error_count: int = 0,
                                 network_error_count: int = 0,
                                 failed_request_count: int = 0) -> float:
        """计算整体质量分数 (0-100)"""
        score = 100.0

        # 语法错误扣分（每个错误-10分）
        score -= len(syntax_report.get('errors', [])) * 10

        # 语法警告扣分（每个警告-2分）
        score -= len(syntax_report.get('warnings', [])) * 2

        # 交互失败率扣分（最多-30分）
        score -= (100 - interaction_rate) * 0.3

        # JS运行时错误扣分（每个错误-15分，最多-40分）
        score -= min(js_error_count * 15, 40)

        # 控制台错误扣分（每个错误-5分，最多-20分）
        score -= min(console_error_count * 5, 20)

        # 渲染错误扣分（每个错误-20分，这是严重问题）
        score -= min(render_error_count * 20, 40)
        score -= min(network_error_count * 1, 10)
        score -= min(failed_request_count * 1, 10)

        return max(0.0, round(score, 2))

    def _print_summary(self, report: Dict[str, Any]):
        """打印测试摘要"""
        summary = report['summary']
        
        print("\n" + "="*60)
        print("📊 HTML质量测试摘要")
        print("="*60)
        
        # 语法检查
        print(f"\n🔍 语法检查:")
        if summary['has_syntax_errors']:
            print(f"  ❌ 发现 {summary['syntax_error_count']} 个错误")
        else:
            print(f"  ✅ 无语法错误")
        print(f"  ⚠️  {summary['syntax_warning_count']} 个警告")
        
        # 运行时错误
        if not self.static_only:
            print(f"\n💻 运行时检查:")
            if summary['has_js_errors']:
                print(f"  ❌ JavaScript错误: {summary['js_error_count']} 个")
            else:
                print(f"  ✅ 无JavaScript错误")
            print(f"  🟡 控制台错误: {summary['console_error_count']} 个")
            print(f"  🟡 控制台警告: {summary['console_warning_count']} 个")
            
            # 交互统计
            print(f"\n🖱️  交互测试:")
            print(f"  总尝试: {summary['total_interactions']} 次")
            print(f"  成功: {summary['successful_interactions']} 次")
            print(f"  失败: {summary['failed_interactions']} 次")
            print(f"  无变化: {summary['no_change_interactions']} 次")
            print(f"  成功率: {summary['interaction_success_rate']:.1f}%")
            
            # 截图
            print(f"\n📸 截图:")
            print(f"  总数: {summary['screenshot_count']} 张")
            if summary['screenshots_with_errors'] > 0:
                print(f"  ⚠️  {summary['screenshots_with_errors']} 张截图检测到页面错误")
        
        # 整体评分
        score = summary['overall_quality_score']
        print(f"\n⭐ 整体质量评分: {score:.1f}/100")
        if score >= 90:
            grade = "优秀 ✨"
        elif score >= 80:
            grade = "良好 👍"
        elif score >= 70:
            grade = "合格 ✓"
        elif score >= 60:
            grade = "勉强及格 ⚠️"
        else:
            grade = "不合格 ❌"
        print(f"  等级: {grade}")
        
        print("="*60 + "\n")


def _build_cli_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Playwright HTML测试器（改进版）')
    parser.add_argument('html', help='待测试的HTML文件路径')
    parser.add_argument('--output-dir', help='输出目录，默认 test_outputs/<文件名>_<时间戳>')
    parser.add_argument('--headless', dest='headless', action='store_true', help='以无头模式运行浏览器（默认）')
    parser.add_argument('--no-headless', dest='headless', action='store_false', help='以有头模式运行浏览器')
    parser.set_defaults(headless=True)
    parser.add_argument('--static-only', action='store_true', help='仅执行静态检查，跳过浏览器交互')
    return parser


async def amain(args=None):
    parser = _build_cli_parser()
    ns = parser.parse_args(args=args)
    tester = PlaywrightHTMLTester(ns.html, ns.output_dir, headless=ns.headless, static_only=ns.static_only)
    await tester.run_complete_test()


def main():
    asyncio.run(amain())


if __name__ == '__main__':
    main()
