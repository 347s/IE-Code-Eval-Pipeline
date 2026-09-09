from openai import OpenAI
from datetime import datetime
import json
import os
import argparse
from typing import List, Dict, Any, Union
import random
import base64
from pathlib import Path
import re
import httpx
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
import time            # 建议添加：防止 time 未定义
# OpenAI-compatible endpoint configuration. Never commit real credentials.
HARDCODED_API_KEY = os.getenv("OPENAI_API_KEY")
HARDCODED_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


# ===== 模型最大 tokens 预设 =====
# 根据你的运行结果与常见参考预先定义，不再每次动态探测。
MODEL_MAX_TOKENS: Dict[str, int] = {
    'gpt-4o': 15000,
    'gpt-4': 8192,
    'claude-4-sonnet': 15000,  # 你的检测结果显示 65536 可用
    'InnoSpark': 32000,
    'deepseek-chat': 8192,      # 保守预设
    'gpt-5': 50000,             # 你的输出中的理论值
}

def resolve_max_tokens(model_name: str) -> int:
    """基于静态映射解析模型的最大 tokens，支持环境变量覆盖。"""
    # 可选：用环境变量覆盖，格式为 JSON，例如 {"claude-4-sonnet": 64000}
    env_json = os.getenv('MODEL_MAX_TOKENS_JSON')
    if env_json:
        try:
            override = json.loads(env_json)
            if isinstance(override, dict) and model_name in override:
                return int(override[model_name])
        except Exception:
            pass
    return MODEL_MAX_TOKENS.get(model_name, 2048)

# ===== 工具函数 =====
def detect_max_tokens(client, model_name: str, base_url: str = None) -> int:
    """
    动态检测API模型的最大token数
    通过二分查找法测试不同的max_tokens值来找到实际限制
    """
    import time
    
    print(f"🔍 正在检测 {model_name} 的最大token限制...")
    
    # 优先使用静态映射作为理论上限，避免不必要的探测
    theoretical_max = MODEL_MAX_TOKENS.get(model_name, 50000)
    
    # 二分查找范围
    min_tokens = 64
    max_tokens = theoretical_max
    
    def test_token_limit(tokens: int) -> bool:
        """测试指定token数是否可用。仅在请求成功并返回内容时认为可用。"""
        try:
            test_message = "请回复'测试成功'"
            response = client.chat.completions.create(
                model=model_name,
                messages=[{'role': 'user', 'content': test_message}],
                max_tokens=tokens,
                temperature=0.0
            )
            # 成功返回且包含内容，视为可用
            if hasattr(response, 'choices') and response.choices:
                msg = getattr(response.choices[0], 'message', None)
                content = getattr(msg, 'content', None) if msg else None
                return content is not None
            return False
        except Exception as e:
            # 任意异常都视为该 tokens 设置不可用
            print(f"⚠️  测试 {tokens} tokens 时请求失败: {e}")
            return False
    
    # 首先测试理论最大值是否可用（严格以成功响应为准）
    if test_token_limit(theoretical_max):
        print(f"✅ {model_name} 支持理论最大值 {theoretical_max} tokens")
        return theoretical_max
    
    # 二分查找实际最大值
    last_working = 0
    
    while min_tokens <= max_tokens:
        mid_tokens = (min_tokens + max_tokens) // 2
        print(f"🧪 测试 {mid_tokens} tokens...")
        
        if test_token_limit(mid_tokens):
            last_working = mid_tokens
            min_tokens = mid_tokens + 1
            print(f"✅ {mid_tokens} tokens 可用")
        else:
            max_tokens = mid_tokens - 1
            print(f"❌ {mid_tokens} tokens 不可用/可能超限")
        
        # 避免请求过于频繁
        time.sleep(0.5)
    
    # 若未找到任何可用值，提供安全回退值
    fallback = 2048
    final_value = last_working if last_working > 0 else fallback
    print(f"🎯 检测完成！{model_name} 的最大token数约为: {final_value}（{'回退' if last_working == 0 else '检测'}）")
    return final_value

def get_model_info(client, model_name: str) -> Dict[str, Any]:
    """
    获取模型的详细信息，包括最大token数
    """
    try:
        # 尝试获取模型列表（如果API支持）
        models = client.models.list()
        for model in models.data:
            if model.id == model_name:
                return {
                    'id': model.id,
                    'created': getattr(model, 'created', None),
                    'owned_by': getattr(model, 'owned_by', None),
                    'max_tokens': resolve_max_tokens(model_name)
                }
    except Exception as e:
        print(f"⚠️  无法获取模型列表: {e}")
    
    # 如果无法获取模型列表，直接检测token限制
    return {
        'id': model_name,
        'max_tokens': resolve_max_tokens(model_name)
    }

def encode_image(image_path: str, high_quality: bool = False) -> str:
    """
    将图片编码为base64格式，并进行压缩以减少Payload大小。
    
    参数:
        image_path: 图片路径
        high_quality: 是否开启高清模式。
                      False (默认): 强制缩放至 512px, JPEG quality=50 (RL训练/快速评测用)
                      True: 缩放至 1024px, JPEG quality=85 (基准评测/精细检查用)
    """
    print(f"🖼️ 正在处理图片: {image_path} (High Quality: {high_quality})")
    if PIL_AVAILABLE:
        try:
            with Image.open(image_path) as img:
                # 设置压缩参数
                if high_quality:
                    max_dim = 1024
                    quality = 85
                else:
                    max_dim = 512
                    quality = 50

                # Resize if too large
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim))
                
                # Convert to RGB if RGBA (to allow JPEG)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to buffer as JPEG with compression
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)
                encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
                print(f"✅ 图片压缩编码成功 ({len(encoded)} chars)")
                return encoded
        except Exception as e:
            print(f"⚠️ 图片压缩优化失败，尝试回退到原始读取: {e}")
            pass

    # 回退逻辑
    print(f"⚠️ 警告: 使用原始图片上传，可能导致 Context Window 溢出: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            print(f"✅ 图片原始编码成功 ({len(encoded)} chars)")
            return encoded
    except Exception as e:
        print(f"❌ 图片读取失败: {e}")
        return ""

def get_image_mime_type(image_path: str) -> str:
    """根据文件扩展名获取MIME类型"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/jpeg')

def format_multimodal_message(text: str = None, image_paths: List[str] = None, high_quality: bool = False) -> Dict[str, Any]:
    """格式化多模态消息"""
    content = []
    
    if text:
        content.append({
            "type": "text",
            "text": text
        })
    
    if image_paths:
        for image_path in image_paths:
            if os.path.exists(image_path):
                base64_image = encode_image(image_path, high_quality=high_quality)
                if base64_image:
                    mime_type = get_image_mime_type(image_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
                else:
                    print(f"⚠️ 警告: 图片编码失败或为空，将被跳过: {image_path}")
            else:
                print(f"⚠️ 警告: 图片文件未找到，将被跳过: {image_path}")

    return {
        "role": "user",
        "content": content
    }

 


# ===== 模型定义 =====
class LLM:
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, default_max_output_tokens: int = 8192, model_max_tokens_override: int = None):
        # Explicit arguments take precedence over environment variables.
        self.effective_api_key = api_key or HARDCODED_API_KEY or "EMPTY"
        self.effective_base_url = base_url if base_url else HARDCODED_BASE_URL
        
        # 移除全局 client 实例，改为每次调用临时创建
        # 这种短连接模式在高并发集群环境下最稳定，避免 Keep-Alive 死锁
        self.model_name = model_name
        print(model_name)
        self.max_retries = 5
        self.retry_delay = 5  # 重试间隔秒数
        # 默认输出上限（调用未显式传入 max_tokens 时采用）
        self.default_max_output_tokens = max(1, int(default_max_output_tokens)) if default_max_output_tokens else 8192
        # 模型上下文最大值可选覆盖（如需强制限制）
        self._max_tokens = int(model_max_tokens_override) if isinstance(model_max_tokens_override, int) and model_max_tokens_override > 0 else None

        # 启动时打印关键信息，便于排查（不打印密钥）
        try:
            print(f"🔧 LLM配置: model={self.model_name}, base_url={self.effective_base_url}")
            if not self.effective_api_key:
                print("⚠️ 未提供 API Key：请传入 api_key 或设置 OPENAI_API_KEY")
        except Exception:
            pass
            
        if self.model_name == 'auto':
            print("⏳ 模型名设为 'auto'，将在首次调用时自动解析...")

    def _create_temp_client(self):
        """创建一个临时的一次性客户端，强制禁用 Keep-Alive"""
        # 尝试从环境变量获取代理设置，以增强调试信息
        proxy_info = "System/Env"
        
        http_client = httpx.Client(
            # 彻底禁用连接池：max_keepalive_connections=0
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=1),
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=120.0, pool=5.0),
            trust_env=True  # 修改为 True，允许读取 HTTP_PROXY/HTTPS_PROXY 环境变量
        )
        return OpenAI(
            api_key=self.effective_api_key,
            base_url=self.effective_base_url,
            timeout=720.0,
            max_retries=0,
            http_client=http_client
        )

    @property
    def client(self):
        # 为了兼容已有代码调用（如获取模型列表），临时创建一个
        return self._create_temp_client()

    @staticmethod
    def list_available_models(api_key: str, base_url: str) -> List[str]:
        """静态方法：从指定 API 地址获取所有可用模型列表"""
        try:
            # 使用与类内部一致的稳定配置
            with httpx.Client(timeout=httpx.Timeout(20.0), trust_env=False) as http_client:
                client = OpenAI(
                    api_key=api_key or HARDCODED_API_KEY,
                    base_url=base_url or HARDCODED_BASE_URL,
                    http_client=http_client
                )
                models = client.models.list()
                return [m.id for m in models.data]
        except Exception as e:
            print(f"⚠️ 无法获取模型列表: {e}")
            return []
        
    def _resolve_model_name(self) -> str:
        """尝试从API获取可用的第一个模型ID"""
        try:
            print("🔍 正在自动发现可用模型...")
            models = self.client.models.list()
            if models.data:
                # 优先寻找包含 'gpt', 'claude', 'qwen', 'llama' 的模型
                candidates = [m.id for m in models.data]
                preferred = [
                    m for m in candidates 
                    if any(x in m.lower() for x in ['gpt', 'claude', 'qwen', 'llama', 'mistral', 'deepseek'])
                ]
                if preferred:
                    return preferred[0]
                return candidates[0]
        except Exception as e:
            print(f"⚠️ 无法自动获取模型列表: {e}")
        
        # 回退默认值
        print("⚠️ 自动发现失败，回退使用默认模型名: gpt-3.5-turbo")
        return "gpt-3.5-turbo"
    
    def get_max_tokens(self) -> int:
        """获取模型的最大token数"""
        if self._max_tokens is None:
            self._max_tokens = resolve_max_tokens(self.model_name)
        return self._max_tokens
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        return get_model_info(self.client, self.model_name)

    def __call__(
        self,
        messages,
        image_paths: Union[str, List[str]] = None,
        return_usage: bool = False,
        max_tokens: int = None,
        temperature: float = 0.7,
        high_quality: bool = False,  # 新增参数
        **kwargs  # 支持接收额外的参数，如 top_p
    ) -> Union[str, Dict[str, Any]]:
        import time
        start_time = time.time()

        # 标准化 image_paths 为列表
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # 格式化消息 - 支持多模态
        if image_paths or (isinstance(messages, dict) and 'image_paths' in messages):
            # 多模态输入
            if isinstance(messages, str):
                formatted_messages = [format_multimodal_message(text=messages, image_paths=image_paths, high_quality=high_quality)]
            elif isinstance(messages, dict):
                paths = messages.get('image_paths', image_paths)
                if isinstance(paths, str):
                    paths = [paths]
                formatted_messages = [format_multimodal_message(
                    text=messages.get('text', ''),
                    image_paths=paths,
                    high_quality=high_quality
                )]
            else:
                formatted_messages = [{'role': 'user', 'content': str(messages)}]
        else:
            # 原有的消息格式化逻辑
            if isinstance(messages, str):
                formatted_messages = [{'role': 'user', 'content': messages}]
            elif isinstance(messages, list):
                formatted_messages = []
                for msg in messages:
                    if isinstance(msg, dict):
                        formatted_messages.append(msg)
                    elif hasattr(msg, 'content'):
                        role = 'user'
                        if hasattr(msg, 'type'):
                            role = 'user' if getattr(msg, 'type', 'human') == 'human' else 'assistant'
                        formatted_messages.append({
                            'role': role,
                            'content': msg.content
                        })
            else:
                formatted_messages = [{'role': 'user', 'content': str(messages)}]

        # 重试机制（受限次数 + 针对多模态的安全回退）
        attempt = 0
        tried_text_fallback = False
        # 将输出 tokens 设为安全上限，避免超大值导致服务端报错
        # 默认输出上限由构造参数控制，可在调用时覆盖；不再使用环境变量硬封顶
        default_output_cap = self.default_max_output_tokens
        requested_max = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else default_output_cap

        # 简化输出上限：直接使用请求值或默认值，不做文本估算
        safe_max_tokens = max(1, int(requested_max))
        
        # 准备 API 调用参数 (吸收 llm2.py 的优点)
        api_kwargs = {
            'model': self.model_name,
            'messages': formatted_messages,
            'max_tokens': safe_max_tokens,
            'temperature': temperature,
        }
        
        # 仅当 kwargs 中有值时才合并，避免传递 None 导致报错
        if kwargs:
            clean_extra_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            api_kwargs.update(clean_extra_kwargs)

        while attempt < self.max_retries:
            try:
                # 关键修复：确保 model_name 不是 'auto'，如果是则先解析
                if self.model_name == 'auto':
                    # 使用当前临时 client 解析模型名
                    # 注意：_resolve_model_name 内部会调用 client.models.list()
                    # 这会触发一次 client 属性访问，从而创建一个临时 client
                    self.model_name = self._resolve_model_name()
                    # 更新 api_kwargs 中的 model 参数
                    api_kwargs['model'] = self.model_name
                    print(f"✅ [PID: {os.getpid()}] 自动发现并使用模型: {self.model_name}")

                # 使用准备好的 api_kwargs 进行调用
                # 每次访问 self.client 都会创建一个新的临时 client (短连接模式)
                print(f"📡 [Attempt {attempt+1}] Sending request to {self.model_name}...")
                response = self.client.chat.completions.create(**api_kwargs)
                print(f"📥 [Attempt {attempt+1}] Received response from {self.model_name}")

                end_time = time.time()
                execution_time = end_time - start_time

                usage_info = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0,
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0,
                    'total_tokens': getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') else 0,
                    'execution_time': execution_time,
                    'model': self.model_name
                }

                content = response.choices[0].message.content

                if return_usage:
                    return {
                        'content': content,
                        'usage': usage_info
                    }
                else:
                    return content

            except Exception as e:
                error_msg = str(e)
                attempt += 1

                # 提取可能的HTTP状态码
                m = re.search(r"Error code:\s*(\d{3})", error_msg)
                status_code = int(m.group(1)) if m else None

                # 详细的错误分类和处理
                if status_code in (400, 401, 403):
                    reason = {
                        400: "请求参数不合法（Bad Request）",
                        403: "无权限（Forbidden）",
                        401: "认证失败（Unauthorized）",
                        405: "客户端错误"
                    }.get(status_code, "客户端错误")
                    print(f"❌ 第{attempt}次尝试失败: {status_code} - {reason}，此类错误不可恢复，将停止重试。\n👉 请检查 API Key、base_url、模型名或请求参数。原始错误: {error_msg}")
                    raise
                elif status_code == 429:
                    print(f"❌ 第{attempt}次尝试失败: 429 Rate Limit - 速率限制")
                elif status_code and 500 <= status_code < 600:
                    # 5xx 认为是可恢复，尝试降配重试（减小 max_tokens）
                    print(f"❌ 第{attempt}次尝试失败: {status_code} 服务器错误 - 暂时不可用")
                    if safe_max_tokens > 512:
                        prev = safe_max_tokens
                        safe_max_tokens = max(512, prev // 2)
                        print(f"🔻 降配重试：将 max_tokens 从 {prev} 降到 {safe_max_tokens}")
                        continue
                elif "502" in error_msg:
                    print(f"❌ 第{attempt}次尝试失败: 502 Bad Gateway - 服务器暂时不可用")
                    if safe_max_tokens > 512:
                        prev = safe_max_tokens
                        safe_max_tokens = max(512, prev // 2)
                        print(f"🔻 降配重试：将 max_tokens 从 {prev} 降到 {safe_max_tokens}")
                        continue
                elif "timeout" in error_msg.lower():
                    print(f"❌ 第{attempt}次尝试失败: 请求超时")
                elif "connection" in error_msg.lower():
                    print(f"❌ 第{attempt}次尝试失败: 连接错误")
                else:
                    # 未知错误：可能是多模态消息格式不被后端支持（例如图片）
                    print(f"❌ 第{attempt}次尝试失败: {error_msg}")
                    if ("NoneType" in error_msg and "subscriptable" in error_msg) and not tried_text_fallback and (image_paths or (isinstance(messages, dict) and messages.get('image_paths'))):
                        # 针对多模态失败的安全回退：改用纯文本并附加图片文件名说明
                        image_list = image_paths or messages.get('image_paths') if isinstance(messages, dict) else image_paths
                        if isinstance(image_list, str):
                            image_list = [image_list]
                        image_names = ", ".join([Path(p).name for p in (image_list or [])])
                        if isinstance(messages, str):
                            fallback_text = messages + (f"\n[图像路径: {image_names}]" if image_names else "")
                        elif isinstance(messages, dict):
                            fallback_text = messages.get('text', '') + (f"\n[图像路径: {image_names}]" if image_names else "")
                        else:
                            fallback_text = str(messages) + (f"\n[图像路径: {image_names}]" if image_names else "")
                        formatted_messages = [{'role': 'user', 'content': fallback_text}]
                        tried_text_fallback = True
                        print("🔁 多模态请求可能不被支持，已切换为纯文本重试…")
                        # 继续下一次循环重试（不等待）
                        continue

                # 到达最大重试次数后停止
                if attempt >= self.max_retries:
                    print("⛔ 已达到最大重试次数，停止重试。")
                    raise

                # 指数退避等待时间，上限60秒（仅针对可恢复错误）
                wait_time = min(self.retry_delay * (2 ** (attempt - 1)), 60)
                print(f"⏳ {wait_time}秒后重试…")
                time.sleep(wait_time)

    def complete_long_output(
        self,
        messages: Union[str, List[Dict[str, Any]], Dict[str, Any]],
        image_paths: Union[str, List[str]] = None,
        per_call_tokens: int = None,
        max_rounds: int = 32,
        temperature: float = 0.1,
        stop_regex: str = None,
        joiner: str = "",
        return_usage: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        在网关对单次补全有上限的约束下，自动分段续写并拼接长输出。

        设计要点：
        - 单轮输出受稳定上限（默认使用实例默认值或 per_call_tokens 设置）。
        - 若本轮因长度截断（finish_reason=length）或接近上限，则继续下一轮。
        - 通过在消息历史中加入上一轮的 assistant 输出 + 明确的“继续输出”提示，避免重复。
        - 支持正则停止条件（例如匹配到文档结束标记）。
        - 返回值可选择包含聚合的 usage 统计。
        """

        import re as _re
        import time as _time

        # 解析每轮最大输出上限（不使用环境变量封顶）
        stable_cap = self.default_max_output_tokens
        per_call = int(per_call_tokens) if isinstance(per_call_tokens, int) and per_call_tokens > 0 else stable_cap
        per_call = max(256, per_call)  # 给出合理的下限，避免过小导致轮数过多

        # 标准化 image_paths 为列表
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # 复用 __call__ 中的格式化逻辑（复制关键路径以避免内部状态耦合）
        if image_paths or (isinstance(messages, dict) and 'image_paths' in messages):
            if isinstance(messages, str):
                formatted_messages = [format_multimodal_message(text=messages, image_paths=image_paths)]
            elif isinstance(messages, dict):
                paths = messages.get('image_paths', image_paths)
                if isinstance(paths, str):
                    paths = [paths]
                formatted_messages = [format_multimodal_message(
                    text=messages.get('text', ''),
                    image_paths=paths
                )]
            else:
                formatted_messages = [{'role': 'user', 'content': str(messages)}]
        else:
            if isinstance(messages, str):
                formatted_messages = [{'role': 'user', 'content': messages}]
            elif isinstance(messages, list):
                formatted_messages = []
                for msg in messages:
                    if isinstance(msg, dict):
                        formatted_messages.append(msg)
                    elif hasattr(msg, 'content'):
                        role = 'user'
                        if hasattr(msg, 'type'):
                            role = 'user' if getattr(msg, 'type', 'human') == 'human' else 'assistant'
                        formatted_messages.append({'role': role, 'content': msg.content})
            else:
                formatted_messages = [{'role': 'user', 'content': str(messages)}]

        aggregate_text_parts: List[str] = []
        aggregate_text_so_far: str = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        rounds = 0

        while rounds < max_rounds:
            rounds += 1
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages,
                    max_tokens=per_call,
                    temperature=temperature,
                )

                part = resp.choices[0].message.content or ""
                finish_reason = getattr(resp.choices[0], 'finish_reason', '')

                # usage 累计
                if hasattr(resp, 'usage'):
                    pt = getattr(resp.usage, 'prompt_tokens', 0)
                    ct = getattr(resp.usage, 'completion_tokens', 0)
                    total_prompt_tokens += pt or 0
                    total_completion_tokens += ct or 0

                aggregate_text_parts.append(part)
                # 累积文本用于跨分段的停止标记检测
                try:
                    aggregate_text_so_far += part
                except Exception:
                    pass

                # 停止条件：
                stop_hit = False
                if stop_regex:
                    try:
                        # 在当前分段与已累积的整体文本上均尝试匹配，确保跨分段标记也能被识别
                        if _re.search(stop_regex, part) or _re.search(stop_regex, aggregate_text_so_far):
                            stop_hit = True
                    except Exception:
                        # 无效正则忽略
                        pass

                # 若提供了 stop_regex，则严格依赖其命中作为停止条件；
                # 避免模型在中途以 finish_reason=stop 提前终止导致内容被截断（常见于长脚本）。
                close_to_cap = False
                try:
                    close_to_cap = (getattr(resp.usage, 'completion_tokens', 0) or 0) >= int(per_call * 0.95)
                except Exception:
                    close_to_cap = False

                if stop_regex:
                    if stop_hit:
                        break
                else:
                    # 未设置 stop_regex 时，依据 finish_reason 判断结束
                    if finish_reason and finish_reason != 'length' and not close_to_cap:
                        break

                # 构造续写的消息：加入上一轮的 assistant 输出，并追加一个“继续输出”的用户提示
                # 在提示中明确停止标记，减少重复与提前收尾
                continue_hint = '请继续输出未完成的内容，从上次截断处开始，不要重复已输出内容。'
                if stop_regex:
                    continue_hint += f"直至出现匹配标记：{stop_regex}。"
                formatted_messages = list(formatted_messages) + [
                    {'role': 'assistant', 'content': part},
                    {'role': 'user', 'content': continue_hint}
                ]

                # 小憩，避免打爆网关
                _time.sleep(min(self.retry_delay, 2))

            except Exception as e:
                # 可恢复错误：降配并继续；否则停止
                emsg = str(e)
                m = _re.search(r"Error code:\s*(\d{3})", emsg)
                status_code = int(m.group(1)) if m else None
                if status_code and 500 <= status_code < 600 or "502" in emsg:
                    # 降配重试
                    if per_call > 512:
                        prev = per_call
                        per_call = max(512, prev // 2)
                        print(f"🔻 长输出降配重试：将每轮 max_tokens 从 {prev} 调整到 {per_call}")
                        continue
                # 其他错误直接退出循环
                print(f"❌ 长输出在第{rounds}轮失败：{emsg}")
                break

        final_text = joiner.join(aggregate_text_parts)
        if return_usage:
            return {
                'content': final_text,
                'usage': {
                    'prompt_tokens': total_prompt_tokens,
                    'completion_tokens': total_completion_tokens,
                    'total_tokens': total_prompt_tokens + total_completion_tokens,
                    'rounds': rounds,
                    'model': self.model_name,
                }
            }
        return final_text

if __name__ == '__main__':

    model = LLM("claude-4-sonnet")
    response = model(prompt,max_tokens=16384, temperature=0.1)
    # 注意：下面不要再次以纯文本调用，否则会覆盖上面的图片请求响应
    # 如果需要对比纯文本与多模态表现，可另起变量：
    # text_only_resp = model(prompt)
    print("\n" + "="*50)
    print("🤖 模型响应:", response) 
    print("="*50)
