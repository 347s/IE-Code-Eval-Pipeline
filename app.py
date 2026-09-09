import streamlit as st
import json
import os
import pandas as pd
import glob
from datetime import datetime
from pathlib import Path
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import time
import logging
from typing import List, Dict, Any

import queue
import sys
import io
import asyncio
import nest_asyncio
try:
    nest_asyncio.apply()
except ValueError as e:
    # Handle cases like "Can't patch loop of type <class 'uvloop.Loop'>"
    # 降级为 debug 或直接忽略，避免刷屏
    pass
except Exception as e:
    logging.debug(f"nest_asyncio.apply() failed: {e}")

# Streamlit context handling for threads
# Note: We are moving away from relying on this for background logging, 
# but keep it imported if needed for other parts.
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Import local modules
import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting gracefully...')
    try:
        sys.exit(0)
    except Exception:
        os._exit(0)

try:
    signal.signal(signal.SIGINT, signal_handler)
except ValueError:
    # Ignored: signal only works in main thread. 
    # Streamlit runs scripts in a separate thread, so this might fail.
    pass

import importlib
try:
    import generator
    import main
    import LLM as LLM_module
    
    # Force reload to pick up changes during dev
    importlib.reload(generator)
    importlib.reload(main)
    importlib.reload(LLM_module)
    
    from generator import HTMLGenerator
    from main import HTMLErrorDetector
    from LLM import LLM
except ImportError as e:
    st.error(f"❌ 严重错误：无法导入本地模块，请确保 app.py 位于 judge 目录下。\n详细错误: {e}")
    st.stop()

# ==========================================
# Logging Setup to Streamlit
# ==========================================
class QueueLogHandler(logging.Handler):
    """
    Custom Logging Handler to redirect logs to a Queue.
    """
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            log_entry = self.format(record)
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_entry = f"[{timestamp}] {log_entry}"
            self.log_queue.put(formatted_entry)
        except Exception:
            self.handleError(record)

class StreamToQueue:
    """
    Redirects stdout/stderr to a queue, allowing print() statements to be captured.
    Also prints to original stdout/stderr so logs appear in the console.
    """
    def __init__(self, log_queue, original_stream):
        self.log_queue = log_queue
        self.original_stream = original_stream
        self.buffer = ""

    def write(self, message):
        # Write to original stream (console)
        self.original_stream.write(message)
        self.original_stream.flush() # Ensure it appears immediately

        if message.strip():  # Only capture non-empty messages
            timestamp = datetime.now().strftime("%H:%M:%S")
            # Split by newlines to handle multi-line prints
            for line in message.splitlines():
                if line.strip():
                    self.log_queue.put(f"[{timestamp}] {line}")

    def flush(self):
        self.original_stream.flush()

# Configure root logger to capture all logs
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Suppress Streamlit's "missing ScriptRunContext" warnings from background threads
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)

# ==========================================
# Page Config & Styling
# ==========================================
st.set_page_config(
    page_title="HTML 自动化评测平台 Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
    .stButton>button {
        background-color: #2ecc71; color: white; font-weight: bold; border-radius: 8px; padding: 0.5rem 1rem; border: none; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #27ae60; transform: scale(1.02); }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 5px solid #3498db; }
    .stExpander { border: 1px solid #e0e0e0; border-radius: 8px; }
    .stProgress > div > div > div > div { background-color: #3498db; }
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 HTML 交互式自动化评测平台 Pro")
st.markdown("---")

# ==========================================
# Config Persistence (Save/Load)
# ==========================================
CONFIG_FILE = "config_presets.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(config_data):
    try:
        # API keys are session-only and must never be written to disk.
        config_data = {
            key: ("" if "api_key" in key.lower() else value)
            for key, value in config_data.items()
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        st.toast("✅ 配置已保存！下次启动将自动加载。", icon="💾")
    except Exception as e:
        st.error(f"保存配置失败: {e}")

# Load existing config
saved_config = load_config()

# ==========================================
# Sidebar: Configuration
# ==========================================
with st.sidebar:
    st.title("⚙️ 系统配置 (System Config)")
    
    with st.expander("📚 指南与说明 (Tutorial & Guide)", expanded=False):
        st.markdown("""
        **适用人群：** 
        - 🐣 **新手**：跟随中文引导，傻瓜式操作。
        - 🧑‍💻 **极客**：参考英文参数名，精确调优。
        
        **操作步骤：**
        1. **配置模型 (Model Config)**：填入 Judge（评测者）和 Target（被测者）的 API 信息。
        2. **加载数据 (Load Data)**：上传 JSONL 文件或输入本地文件路径。
        3. **运行评测 (Run Eval)**：点击“开始并发评测”按钮。
        
        **配置保存：**
        修改参数后，点击侧边栏底部的 **"💾 保存当前配置"**，下次无需重新填写。
        """)
    
    # 1. Judge Model Configuration
    with st.expander("1. 评测模型配置 (Judge Model)", expanded=True):
        st.markdown("##### 👩‍⚖️ 评测官 / Judge Agent")
        
        # Mode Selection
        judge_mode = st.radio(
            "配置模式 (Configuration Mode)", 
            options=["统一模型 (Unified MLLM)", "分离模型 (Separate LLM & VLM)"],
            index=0 if saved_config.get("judge_mode", "unified") == "unified" else 1,
            help="统一模型：使用同一个多模态模型处理文本和图像。\n分离模型：分别配置处理文本的LLM和处理图像的VLM。"
        )
        is_unified = (judge_mode == "统一模型 (Unified MLLM)")
        
        # --- Unified / Text LLM Config ---
        config_title = "MLLM 配置 (Text & Vision)" if is_unified else "文本 LLM 配置 (Text Only)"
        st.markdown(f"###### {config_title}")
        
        judge_api_key = st.text_input("API Key", value=os.getenv("JUDGE_API_KEY", ""), type="password", help="[API Key] 仅在当前会话中使用，不会写入配置文件。")
        judge_base_url = st.text_input("Base URL", value=saved_config.get("judge_base_url", os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1")), help="[Base URL] AI 服务的接口地址。")
        
        # --- 动态模型选择逻辑 (Text/Unified) ---
        if 'judge_model_list' not in st.session_state:
            st.session_state['judge_model_list'] = []
        if 'use_manual_model_input' not in st.session_state:
            st.session_state['use_manual_model_input'] = True

        col_m1, col_m2 = st.columns([3, 1])
        with col_m2:
            if st.button("🔍 发现模型", key="btn_discover_text", help="尝试从 Base URL 获取可用模型列表"):
                with st.spinner("正在连接服务器..."):
                    try:
                        from LLM import LLM
                        models = LLM.list_available_models(judge_api_key, judge_base_url)
                        if models:
                            st.session_state['judge_model_list'] = models
                            st.session_state['use_manual_model_input'] = False
                            st.success(f"找到 {len(models)} 个模型")
                        else:
                            st.warning("未找到模型或连接失败")
                    except Exception as e:
                        st.error(f"连接错误: {e}")

        if not st.session_state['use_manual_model_input'] and st.session_state['judge_model_list']:
            selected_model = st.selectbox(
                "Model Name (选择模型)", 
                options=st.session_state['judge_model_list'],
                index=0,
                key="sel_model_text"
            )
            if st.button("✏️ 切换为手动输入", key="btn_manual_text"):
                st.session_state['use_manual_model_input'] = True
                st.rerun()
            judge_model_name = selected_model
        else:
            judge_model_name = st.text_input(
                "Model Name (手动输入)", 
                value=saved_config.get("judge_model_name", "auto"), 
                help="[Model Name] 用于打分的模型。输入 'auto' 可自动检测。",
                key="inp_model_text"
            )
            if st.session_state['judge_model_list']:
                if st.button("📋 切换回列表选择", key="btn_list_text"):
                    st.session_state['use_manual_model_input'] = False
                    st.rerun()
        
        # --- Separate VLM Config (Only if Separate Mode) ---
        vlm_api_key = None
        vlm_base_url = None
        vlm_model_name = None
        
        if not is_unified:
            st.markdown("---")
            st.markdown("###### 视觉 VLM 配置 (Vision Only)")
            
            vlm_api_key = st.text_input("VLM API Key", value=os.getenv("VLM_API_KEY", ""), type="password")
            vlm_base_url = st.text_input("VLM Base URL", value=saved_config.get("vlm_base_url", os.getenv("VLM_BASE_URL", "https://api.openai.com/v1")))
            
            # --- 动态模型选择逻辑 (VLM) ---
            if 'vlm_model_list' not in st.session_state:
                st.session_state['vlm_model_list'] = []
            if 'use_manual_vlm_input' not in st.session_state:
                st.session_state['use_manual_vlm_input'] = True

            col_v1, col_v2 = st.columns([3, 1])
            with col_v2:
                if st.button("🔍 发现VLM", key="btn_discover_vlm"):
                    with st.spinner("正在连接 VLM 服务器..."):
                        try:
                            from LLM import LLM
                            models = LLM.list_available_models(vlm_api_key, vlm_base_url)
                            if models:
                                st.session_state['vlm_model_list'] = models
                                st.session_state['use_manual_vlm_input'] = False
                                st.success(f"找到 {len(models)} 个模型")
                            else:
                                st.warning("未找到模型")
                        except Exception as e:
                            st.error(f"连接错误: {e}")

            if not st.session_state['use_manual_vlm_input'] and st.session_state['vlm_model_list']:
                vlm_selected = st.selectbox(
                    "VLM Model Name", 
                    options=st.session_state['vlm_model_list'],
                    index=0,
                    key="sel_model_vlm"
                )
                if st.button("✏️ 手动输入VLM", key="btn_manual_vlm"):
                    st.session_state['use_manual_vlm_input'] = True
                    st.rerun()
                vlm_model_name = vlm_selected
            else:
                vlm_model_name = st.text_input(
                    "VLM Model Name (手动输入)", 
                    value=saved_config.get("vlm_model_name", "auto"),
                    key="inp_model_vlm"
                )
                if st.session_state['vlm_model_list']:
                    if st.button("📋 列表选择VLM", key="btn_list_vlm"):
                        st.session_state['use_manual_vlm_input'] = False
                        st.rerun()

        st.markdown("---")
        rl_mode = st.toggle("🚀 极速模式 / RL Mode", value=saved_config.get("rl_mode", True), help="[RL Mode] ON: 跳过耗时视觉检测。OFF: 全量检测。")
        if rl_mode: st.caption("✅ 当前: **极速模式** (仅核心检测)")
        else: st.caption("🐢 当前: **全量模式** (含视觉检测)")

    # 2. Target Model Configuration
    with st.expander("2. 待测模型配置 (Target Model)", expanded=True):
        st.markdown("##### 🤖 考生 / Generator Agent")
        target_api_key = st.text_input("Target API Key", value=os.getenv("TARGET_API_KEY", ""), type="password")
        target_base_url = st.text_input("Target Base URL", value=saved_config.get("target_base_url", os.getenv("TARGET_BASE_URL", "http://localhost:8000/v1")))
        
        # --- 动态模型选择逻辑 (Target) ---
        if 'target_model_list' not in st.session_state:
            st.session_state['target_model_list'] = []
        if 'use_manual_target_input' not in st.session_state:
            st.session_state['use_manual_target_input'] = True

        col_t1, col_t2 = st.columns([3, 1])
        with col_t2:
            if st.button("🔍 发现模型", key="btn_discover_target", help="尝试从 Base URL 获取可用模型列表"):
                with st.spinner("正在连接服务器..."):
                    try:
                        from LLM import LLM
                        models = LLM.list_available_models(target_api_key, target_base_url)
                        if models:
                            st.session_state['target_model_list'] = models
                            st.session_state['use_manual_target_input'] = False
                            st.success(f"找到 {len(models)} 个模型")
                        else:
                            st.warning("未找到模型或连接失败")
                    except Exception as e:
                        st.error(f"连接错误: {e}")

        if not st.session_state['use_manual_target_input'] and st.session_state['target_model_list']:
            selected_target_model = st.selectbox(
                "Target Model Name (选择模型)", 
                options=st.session_state['target_model_list'],
                index=0,
                key="sel_model_target"
            )
            if st.button("✏️ 切换为手动输入", key="btn_manual_target"):
                st.session_state['use_manual_target_input'] = True
                st.rerun()
            target_model_name = selected_target_model
        else:
            target_model_name = st.text_input(
                "Target Model Name (手动输入)", 
                value=saved_config.get("target_model_name", "meta-llama/Llama-3-70b-instruct"),
                key="inp_model_target"
            )
            if st.session_state['target_model_list']:
                if st.button("📋 切换回列表选择", key="btn_list_target"):
                    st.session_state['use_manual_target_input'] = False
                    st.rerun()
        
        st.markdown("#### 🧠 角色设定 (System Prompt)")
        default_sys_prompt = "You are a helpful expert web developer. Please generate a single-file HTML5 application based on the user's request. Ensure the code is complete, functional, and self-contained (CSS and JS included). Do not use external CDNs if possible, or use reliable ones."
        target_system_prompt = st.text_area("System Prompt", value=saved_config.get("target_system_prompt", default_sys_prompt), height=150)

        st.markdown("#### 🎛️ 生成参数 (Generation Params)")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            target_temp = st.slider("温度 / Temperature", 0.0, 2.0, saved_config.get("target_temp", 0.7), 0.1)
            target_top_p = st.slider("核采样 / Top P", 0.0, 1.0, saved_config.get("target_top_p", 1.0), 0.05)
        with col_p2:
            target_max_tokens = st.number_input("最大长度 / Max Tokens", min_value=128, max_value=32000, value=saved_config.get("target_max_tokens", 4096), step=256)
            target_freq_penalty = st.slider("重复惩罚 / Freq Penalty", -2.0, 2.0, saved_config.get("target_freq_penalty", 0.0), 0.1)

    # 3. Advanced LLM Settings
    with st.expander("3. 高级连接配置 (Advanced LLM Settings)", expanded=False):
        llm_retries = st.number_input("最大重试次数 / Max Retries", min_value=0, max_value=10, value=saved_config.get("llm_retries", 3))
        llm_retry_delay = st.number_input("重试间隔 (秒) / Retry Delay", min_value=0.0, max_value=60.0, value=saved_config.get("llm_retry_delay", 2.0), step=0.5)
        llm_timeout = st.number_input("超时时间 (秒) / Timeout", min_value=10, max_value=600, value=saved_config.get("llm_timeout", 120), step=10)
        
        st.markdown("#### 👁️ 视觉与评分 (Vision & Scoring)")
        high_quality_mode = st.toggle("📷 高清视觉模式 (High Quality Vision)", value=saved_config.get("high_quality_mode", False), help="开启后将使用 1024px 高清截图进行视觉检测，会消耗更多 Token 但检测更精准。")
        strict_mode = st.toggle("🔥 严苛模式 (Strict Mode)", value=saved_config.get("strict_mode", False), help="开启后将启用更严格的评分标准和Prompt指令，对任何微小错误进行重扣分。")
        static_mode = st.toggle("⚡ 静态代码检测 (Static Syntax Check)", value=saved_config.get("static_mode", False), help="开启后，Layer 1 (语法) 将仅使用纯代码静态分析 (No LLM)，大幅提升速度但降低语义理解能力。")
        
        default_weights_json = json.dumps({
            "layer1": 0.20,
            "layer2": 0.25,
            "layer3": 0.25,
            "layer4": 0.30
        }, indent=2)
        custom_weights_str = st.text_area("⚖️ 评分权重 (Scoring Weights JSON)", value=saved_config.get("custom_weights", default_weights_json), height=150, help="自定义各层评分权重，总和建议为 1.0。")

    # 4. Output Paths
    with st.expander("4. 文件保存位置 (Output Paths)", expanded=False):
        default_base = os.path.join(os.getcwd(), "evaluation_outputs")
        output_base_dir = st.text_input("根输出目录 (Base Dir)", value=saved_config.get("output_base_dir", default_base))
        # 子目录将自动根据模型名称生成，不再暴露给用户配置
        
    # 5. Runtime Config
    with st.expander("5. 运行配置 (Runtime)", expanded=True):
        concurrency = st.slider("并发数 / Concurrency", min_value=1, max_value=32, value=saved_config.get("concurrency", 4))
        # Use checkbox for broader compatibility
        enable_resume = st.checkbox("🔄 断点续测 (Resume)", value=saved_config.get("enable_resume", True), help="如果发现已存在的报告和HTML文件，直接加载结果，跳过重新生成和评测。")

    # Save Button
    st.markdown("---")
    if st.button("💾 保存当前配置 (Save Config)"):
        current_config = {
            "judge_mode": "unified" if is_unified else "separate",
            "judge_api_key": judge_api_key,
            "judge_base_url": judge_base_url,
            "judge_model_name": judge_model_name,
            "vlm_api_key": vlm_api_key if not is_unified else "",
            "vlm_base_url": vlm_base_url if not is_unified else "",
            "vlm_model_name": vlm_model_name if not is_unified else "",
            "rl_mode": rl_mode,
            "target_api_key": target_api_key,
            "target_base_url": target_base_url,
            "target_model_name": target_model_name,
            "target_system_prompt": target_system_prompt,
            "target_temp": target_temp,
            "target_top_p": target_top_p,
            "target_max_tokens": target_max_tokens,
            "target_freq_penalty": target_freq_penalty,
            "llm_retries": llm_retries,
            "llm_retry_delay": llm_retry_delay,
            "llm_timeout": llm_timeout,
            "high_quality_mode": high_quality_mode,
            "strict_mode": strict_mode,
            "static_mode": static_mode,
            "custom_weights": custom_weights_str,
            "output_base_dir": output_base_dir,
            "concurrency": concurrency,
            "enable_resume": enable_resume
        }
        save_config(current_config)

# ==========================================
# Helper Functions
# ==========================================

async def process_single_item_async(item, idx, generator, detector, output_base_dir, gen_params, resume_mode=False, high_quality=False, weights=None):
    """
    Process a single benchmark item: Generate -> Evaluate (Async implementation)
    """
    result_entry = {
        "id": idx,
        "status": "failed",
        "error": "",
        "html_file": "",
        "report_file": "",
        "total_score": 0,
        "layer1_score": 0,
        "layer2_score": 0,
        "layer3_score": 0,
        "layer4_score": 0,
        "has_errors": True
    }
    
    prompt = item.get("generated_prompt", "")
    if not prompt:
        result_entry["error"] = "Missing prompt in input data"
        logging.error(f"Task {idx}: Missing prompt")
        return result_entry

    # Determine model-specific output directory
    model_name_safe = gen_params.get('target_model_name', 'default_model').replace('/', '_').replace('\\', '_').replace(' ', '_')
    model_output_dir = Path(output_base_dir) / model_name_safe
    model_html_dir = model_output_dir / "html"
    model_report_dir = model_output_dir / "reports"
    model_screenshot_dir = model_output_dir / "screenshots"
    
    # Create directories
    model_html_dir.mkdir(parents=True, exist_ok=True)
    model_report_dir.mkdir(parents=True, exist_ok=True)
    model_screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Resume Check
    if resume_mode:
        # Search for existing reports for this task index in the model specific dir
        existing_reports = glob.glob(str(model_report_dir / f"report_{idx}_*.json"))
        if existing_reports:
            # Sort by modification time, pick latest
            existing_reports.sort(key=os.path.getmtime, reverse=True)
            latest_report = existing_reports[0]
            
            try:
                with open(latest_report, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                
                # Try to find corresponding HTML file
                existing_htmls = glob.glob(str(model_html_dir / f"sample_{idx}_*.html"))
                html_file_path = existing_htmls[0] if existing_htmls else ""
                
                overview = report_data.get("检测结果总览", {})
                scores = overview.get("各层得分", {})
                
                result_entry.update({
                    "status": "success",
                    "error": "Resumed from existing report",
                    "html_file": html_file_path,
                    "report_file": latest_report,
                    "total_score": overview.get("综合得分", 0),
                    "layer1_score": scores.get("第一层_语法结构", 0),
                    "layer2_score": scores.get("第二层_物理数学", 0),
                    "layer3_score": scores.get("第三层_视觉一致性", 0),
                    "layer4_score": scores.get("第四层_运行与交互能力", 0),
                    "has_errors": overview.get("存在错误", False)
                })
                logging.info(f"⏭️ Task {idx}: Skipped (Resumed from {os.path.basename(latest_report)})")
                return result_entry
            except Exception as e:
                logging.warning(f"Task {idx}: Failed to resume from report: {e}")
                # Fallback to normal processing

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"🚀 Task {idx}: Starting generation...")
    
    # 1. Generate
    try:
        html_content = generator.generate(
            prompt, 
            system_prompt=gen_params.get('system_prompt'),
            temperature=gen_params.get('temperature', 0.7),
            max_tokens=gen_params.get('max_tokens', 4096),
            top_p=gen_params.get('top_p', 1.0),
            frequency_penalty=gen_params.get('frequency_penalty', 0.0)
        )
        if not html_content or len(html_content) < 50:
            error_msg = f"Generation too short ({len(html_content or '')} chars)"
            logging.error(f"Task {idx}: {error_msg}")
            result_entry["error"] = error_msg
            return result_entry
        
        file_name = f"sample_{idx}_{timestamp}.html"
        file_path = model_html_dir / file_name
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        result_entry["html_file"] = str(file_path)
        logging.info(f"✅ Task {idx}: Generation complete. Evaluating...")
        
    except Exception as e:
        logging.error(f"Task {idx}: Generation Error: {str(e)}")
        result_entry["error"] = f"Generation Error: {str(e)}"
        return result_entry

    # 1.5 Playwright Test
    playwright_report = None
    screenshots = []
    
    if globals().get("PLAYWRIGHT_AVAILABLE", False):
        try:
            # We need a separate output dir for screenshots
            pw_output_dir = model_screenshot_dir / f"task_{idx}_{timestamp}"
            pw_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use 'key_frames' mode for efficiency unless specified otherwise
            # Ensure PlaywrightHTMLTester is defined
            if 'PlaywrightHTMLTester' not in globals():
                from playwright_html_tester import PlaywrightHTMLTester
            
            tester = PlaywrightHTMLTester(
                html_file_path=str(file_path),
                output_dir=str(pw_output_dir),
                headless=True,
                screenshot_mode='key_frames' 
            )
            # Run async directly with timeout protection
            try:
                await asyncio.wait_for(tester.run_complete_test(), timeout=30.0)
            except asyncio.TimeoutError:
                logging.warning(f"Task {idx}: Playwright test timed out (30s).")
                if hasattr(tester, 'cleanup'):
                    await tester.cleanup()
            except Exception as e:
                logging.warning(f"Task {idx}: Playwright test error: {e}")
            
            # Load report
            pw_report_path = pw_output_dir / 'playwright_test_report.json'
            if pw_report_path.exists():
                with open(pw_report_path, 'r', encoding='utf-8') as f:
                    playwright_report = json.load(f)
                
                # Load screenshots info
                if playwright_report and 'screenshots' in playwright_report:
                    screenshots = playwright_report['screenshots']
            else:
                logging.warning(f"Task {idx}: Playwright report not found at {pw_report_path}")
                    
        except Exception as e:
            logging.warning(f"Task {idx}: Playwright testing failed: {e}")
    else:
        logging.info(f"Task {idx}: Playwright not available, skipping visual check.")

    # 2. Evaluate
    try:
        # 优化截图列表：限制数量以避免 VLM 超时
        optimized_screenshots = screenshots
        
        # RL模式下极致优化：只取第一张 (参考 RL 目录优化)
        if hasattr(detector, 'rl_mode') and detector.rl_mode and screenshots:
             optimized_screenshots = [screenshots[0]]
        elif screenshots and len(screenshots) > 5:
            logging.info(f"Task {idx}: 截图数量过多 ({len(screenshots)}), 进行下采样至 5 张...")
            # 保留首尾，中间随机取3张
            first = screenshots[0]
            last = screenshots[-1]
            middle = screenshots[1:-1]
            import random
            sampled_middle = random.sample(middle, min(len(middle), 3))
            # 保持时间顺序
            sampled_middle.sort(key=lambda x: x.get('timestamp', 0))
            optimized_screenshots = [first] + sampled_middle + [last]
            
        # Use detect_async instead of detect
        report = await detector.detect_async(
            str(file_path),
            playwright_report=playwright_report,
            screenshots=optimized_screenshots,
            high_quality=high_quality,
            weights=weights
        )
        
        report_file_name = f"report_{idx}_{timestamp}.json"
        report_path = model_report_dir / report_file_name
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        result_entry["report_file"] = str(report_path)
        
        overview = report.get("检测结果总览", {})
        scores = overview.get("各层得分", {})
        
        result_entry.update({
            "status": "success",
            "error": "",
            "total_score": overview.get("综合得分", 0),
            "layer1_score": scores.get("第一层_语法结构", 0),
            "layer2_score": scores.get("第二层_物理数学", 0),
            "layer3_score": scores.get("第三层_视觉一致性", 0),
            "layer4_score": scores.get("第四层_运行与交互能力", 0),
            "has_errors": overview.get("存在错误", False)
        })
        logging.info(f"🏆 Task {idx}: Done! Score: {result_entry['total_score']}")
        
    except Exception as e:
        logging.error(f"Task {idx}: Evaluation Error: {str(e)}")
        result_entry["error"] = f"Evaluation Error: {str(e)}"
    
    return result_entry

def process_single_item(item, idx, generator, detector, output_base_dir, gen_params, resume_mode=False, high_quality=False, weights=None):
    """
    Wrapper to run process_single_item_async in a fresh event loop
    """
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(process_single_item_async(
            item, idx, generator, detector, output_base_dir, gen_params, resume_mode, high_quality, weights
        ))
    except Exception as e:
        logging.error(f"Task {idx}: Async Execution Error: {e}")
        return {
            "id": idx,
            "status": "failed",
            "error": f"Async Execution Error: {e}",
            "total_score": 0
        }
    finally:
        try:
            loop.close()
        except Exception:
            pass

def plot_score_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x="total_score", bins=20, kde=True, color="#4c72b0", ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title("Distribution of Total Scores", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Total Score", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.set_xlim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    # Add mean line
    mean_score = df['total_score'].mean()
    ax.axvline(mean_score, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}')
    ax.legend()
    return fig

def plot_pass_rates(df, threshold=60):
    layers = ['layer1_score', 'layer2_score', 'layer3_score', 'layer4_score', 'total_score']
    labels = ['Syntax (L1)', 'Physics (L2)', 'Visual (L3)', 'Interaction (L4)', 'Overall']
    pass_rates = [len(df[df[layer] >= threshold]) / len(df) * 100 for layer in layers]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use a better color palette
    colors = sns.color_palette("viridis", len(layers))
    bars = ax.bar(labels, pass_rates, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_title(f"Pass Rates (Score >= {threshold})", fontsize=20, fontweight='bold', pad=20)
    ax.set_ylabel("Pass Rate (%)", fontsize=14)
    ax.set_ylim(0, 110) # Give some space for labels
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    return fig

def plot_radar_chart(df):
    categories = ['Syntax (L1)', 'Physics (L2)', 'Visual (L3)', 'Interact (L4)']
    scores = [df['layer1_score'].mean(), df['layer2_score'].mean(), df['layer3_score'].mean(), df['layer4_score'].mean()]
    scores = np.concatenate((scores, [scores[0]]))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += [angles[0]]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='#4c72b0', alpha=0.25)
    ax.plot(angles, scores, color='#4c72b0', linewidth=2)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="grey", size=10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_title("Average Score Profile", size=16, fontweight='bold', y=1.1)
    return fig

# ==========================================
# Main UI Content
# ==========================================

# ==========================================
# 0. Load Existing Results
# ==========================================
with st.expander("📂 0. 加载已有评测结果 (Load Existing Results)", expanded=False):
    summary_dir = Path(output_base_dir) / "summary"
    root_dir = Path(output_base_dir)
    
    found_files = []
    if summary_dir.exists():
        found_files.extend(list(summary_dir.glob("*.csv")))
    if root_dir.exists():
        found_files.extend(list(root_dir.glob("*.csv")))
        
    # Deduplicate by path
    found_files = list(set(found_files))
    
    if found_files:
        # Sort by modification time (newest first)
        found_files.sort(key=os.path.getmtime, reverse=True)
        
        # Display name with timestamp
        file_options = {}
        for f in found_files:
            mod_time = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
            label = f"{f.name}  [{mod_time}]"
            file_options[label] = f
            
        selected_file_label = st.selectbox("选择评测报告 (Select Report)", options=list(file_options.keys()))
        
        if st.button("📥 加载报告 (Load Report)"):
            try:
                file_path = file_options[selected_file_label]
                df = pd.read_csv(file_path)
                st.session_state['results_df'] = df
                st.session_state['loaded_report_name'] = file_path.name
                st.toast(f"✅ 已加载报告: {file_path.name}", icon="📊")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"❌ 加载失败: {e}")
    else:
        st.info(f"ℹ️ 在 `{output_base_dir}` 或其 `summary` 子目录下未找到 CSV 汇总报告。")

st.subheader("📁 1. 加载题目数据 (Load Benchmark Data)")

data_source = st.radio("选择数据来源 (Data Source)", ["📤 上传文件 (Upload)", "📂 本地路径 (Local Path)"], horizontal=True)

benchmark_data = []
loaded_file_name = ""

if data_source == "📤 上传文件 (Upload)":
    uploaded_file = st.file_uploader("拖拽 .jsonl 文件 (Drag & Drop)", type=["jsonl", "json"])
    if uploaded_file is not None:
        try:
            lines = uploaded_file.getvalue().decode("utf-8").splitlines()
            for line in lines:
                if line.strip():
                    benchmark_data.append(json.loads(line))
            loaded_file_name = uploaded_file.name
            st.success(f"✅ 成功加载 {len(benchmark_data)} 条测试数据")
        except Exception as e:
            st.error(f"❌ 解析文件失败: {e}")

else:  # Local Path
    local_path_input = st.text_input("输入 .jsonl 文件的绝对路径 (Absolute Path to .jsonl file)", 
                               value=saved_config.get("last_local_path", ""),
                               placeholder="examples/tasks.sample.jsonl")
    if local_path_input:
        if os.path.exists(local_path_input):
            try:
                with open(local_path_input, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            benchmark_data.append(json.loads(line))
                loaded_file_name = os.path.basename(local_path_input)
                st.success(f"✅ 成功加载 {len(benchmark_data)} 条测试数据")
                
                # Auto-save valid path
                if local_path_input != saved_config.get("last_local_path"):
                    saved_config["last_local_path"] = local_path_input
                    save_config(saved_config)
                    
            except Exception as e:
                st.error(f"❌ 读取文件失败: {e}")
        else:
            st.warning("⚠️ 文件不存在，请检查路径。")

if benchmark_data:
    with st.expander("👀 数据预览 (Data Preview - Top 3)"):
        st.json(benchmark_data[:3])

# ==========================================
# Execution Logic
# ==========================================

st.markdown("---")
st.subheader("🚀 2. 开始评测 (Run Evaluation)")
st.markdown("点击下方按钮开始任务。**右侧/下方将实时显示详细的运行日志**，请保持页面开启。")

# Layout: Top controls, Middle Logs
col_ctrl_1, col_ctrl_2, col_ctrl_3 = st.columns([1, 4, 1])

with col_ctrl_1:
    start_btn = st.button("🚀 立即开始并发评测 (Start)", disabled=not benchmark_data, help="点击开始批量生成和评测任务")

with col_ctrl_2:
    if st.button("🛑 停止评测 (Stop)", type="primary"):
        st.warning("正在停止...")
        # 1. Cancel futures
        if 'executor' in locals():
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logging.error(f"Error shutting down executor: {e}")
        
        # 2. Set global stop flag (if we had one, but we rely on st.stop() mostly)
        # 3. Terminate current process (nuclear option, but ensures stop)
        # Note: This kills the streamlit server too if not careful.
        # Better approach: Just use st.stop() which raises an exception.
        # But if threads are running, st.stop() only stops the main thread.
        
        # Force kill python process if needed? No, that kills the server.
        # We need to signal threads to stop.
        # Since we use futures, cancel_futures=True helps for pending.
        # For running threads, we can't easily kill them in Python without flags.
        # We'll rely on st.stop() and maybe a session state flag.
        st.session_state['stop_requested'] = True
        st.stop()

# Check for stop flag in main loop
if st.session_state.get('stop_requested', False):
    st.session_state['stop_requested'] = False # Reset
    st.stop()


with col_ctrl_3:
    # Check Playwright Status
    # Direct import check to be sure
    try:
        from playwright.sync_api import sync_playwright
        st.success("✅ Playwright 就绪", icon="🎭")
        PLAYWRIGHT_AVAILABLE = True
    except ImportError:
        st.error("❌ Playwright 库未安装", icon="⚠️")
        st.caption("Terminal: `pip install playwright`")
        PLAYWRIGHT_AVAILABLE = False
    except Exception as e:
        # Check if browsers are installed
        if "Executable doesn't exist" in str(e):
             st.error("❌ Playwright 浏览器未安装", icon="⚠️")
             st.caption("Terminal: `playwright install`")
        else:
             st.warning(f"⚠️ Playwright 状态未知: {e}")
        PLAYWRIGHT_AVAILABLE = False

# Persistent Progress Container (Defined OUTSIDE the button logic)
st.subheader("📊 评测进度 (Progress)")
progress_container = st.container()

# Placeholders inside the container
with progress_container:
    progress_bar = st.empty() # Use empty() first
    status_cols = st.columns(4)
    stat_done = status_cols[0].empty()
    stat_success = status_cols[1].empty()
    stat_fail = status_cols[2].empty()
    stat_avg_score = status_cols[3].empty()
    time_info = st.empty()
    model_status_placeholder = st.empty()

# Persistent Log Viewer (Moved below progress)
# log_container = st.empty() # Removed unused container
with st.expander("📜 运行日志 (Logs)", expanded=True):
    log_area = st.empty()
    if 'log_history' in st.session_state and st.session_state['log_history']:
         log_area.code("\n".join(st.session_state['log_history']), language="text")
    st.caption("💡 提示：只显示最近 500 行日志，完整日志请查看后台文件。")

st.markdown("---")
error_container = st.expander("🚨 错误汇总 (Errors)", expanded=True)

# ==========================================
# Background Execution Manager
# ==========================================
if 'bg_executor' not in st.session_state:
    st.session_state['bg_executor'] = None
if 'bg_futures' not in st.session_state:
    st.session_state['bg_futures'] = []
if 'bg_start_time' not in st.session_state:
    st.session_state['bg_start_time'] = None
if 'bg_log_queue' not in st.session_state:
    st.session_state['bg_log_queue'] = None
if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False

# ... (Previous code)

if start_btn or st.session_state['is_running']:
    # Initialize status placeholder early to avoid NameError and ensure visibility
    model_status = model_status_placeholder
    
    # Ensure UI widgets are active
    progress_bar_widget = progress_bar.progress(0)
    
    # 1. Initialize (Only on first click)
    if start_btn and not st.session_state['is_running']:
        # Clear previous logs
        st.session_state['log_history'] = []
        log_area.empty()

        # Validate paths
        Path(output_base_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup Queue
        log_queue = queue.Queue()
        st.session_state['bg_log_queue'] = log_queue # Persist queue
        
        # Setup Logger
        log_handler = QueueLogHandler(log_queue)
        logger.addHandler(log_handler)
        
        # Redirect stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        if not isinstance(original_stdout, StreamToQueue):
            sys.stdout = StreamToQueue(log_queue, original_stdout)
        if not isinstance(original_stderr, StreamToQueue):
            sys.stderr = StreamToQueue(log_queue, original_stderr)
            
        # Init Models & Start Executor
        model_status.info("🔄 正在连接模型... (Connecting to models...)")
        
        try:
            # ... Model Init Code (Same as before) ...
            generator = HTMLGenerator({
                "model_name": target_model_name,
                "api_key": target_api_key,
                "base_url": target_base_url
            })
            # Inject advanced settings
            generator.llm.max_retries = llm_retries
            generator.llm.retry_delay = llm_retry_delay
            generator.llm.timeout = llm_timeout
            
            judge_llm = LLM(
                model_name=judge_model_name,
                api_key=judge_api_key,
                base_url=judge_base_url
            )
            judge_llm.max_retries = llm_retries
            judge_llm.retry_delay = llm_retry_delay
            judge_llm.timeout = llm_timeout
            
            if judge_llm.model_name == 'auto':
                 logging.info("Pre-resolving judge model name...")
                 judge_llm.model_name = judge_llm._resolve_model_name()
            
            if is_unified:
                judge_vlm = judge_llm
                logging.info("Using Unified MLLM for both text and vision.")
            else:
                judge_vlm = LLM(
                    model_name=vlm_model_name,
                    api_key=vlm_api_key,
                    base_url=vlm_base_url
                )
                judge_vlm.max_retries = llm_retries
                judge_vlm.retry_delay = llm_retry_delay
                judge_vlm.timeout = llm_timeout
                
                if judge_vlm.model_name == 'auto':
                     logging.info("Pre-resolving VLM model name...")
                     judge_vlm.model_name = judge_vlm._resolve_model_name()
                logging.info("Using Separate VLM for vision tasks.")
                
            detector = HTMLErrorDetector(llm_model=judge_llm, vlm_model=judge_vlm, rl_mode=rl_mode, strict_mode=strict_mode, static_mode=static_mode)
            logging.info("Models initialized successfully.")
            model_status.success("✅ 模型连接成功！")
            
            # Prepare Parameters
            gen_params = {
                'target_model_name': target_model_name,
                'system_prompt': target_system_prompt,
                'temperature': target_temp,
                'max_tokens': target_max_tokens,
                'top_p': target_top_p,
                'frequency_penalty': target_freq_penalty
            }
            
            # Parse weights
            try:
                weights_config = json.loads(custom_weights_str)
            except Exception as e:
                logging.warning(f"Failed to parse custom weights: {e}")
                weights_config = None
            
            # Start Executor
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
            st.session_state['bg_executor'] = executor
            st.session_state['bg_start_time'] = time.time()
            st.session_state['is_running'] = True
            
            # Submit Tasks
            futures = []
            for i, item in enumerate(benchmark_data):
                def task_safe(itm, idx, gen, det, base_dir, g_params, resume, hq, wts):
                    return process_single_item(itm, idx, gen, det, base_dir, g_params, resume, hq, wts)
                
                future = executor.submit(
                    task_safe, 
                    item, 
                    i+1, 
                    generator, 
                    detector, 
                    output_base_dir, 
                    gen_params,
                    enable_resume,
                    high_quality_mode,
                    weights_config
                )
                futures.append(future)
            st.session_state['bg_futures'] = futures
            
        except Exception as e:
            st.error(f"Failed to start: {e}")
            st.session_state['is_running'] = False
            st.stop()

    # 2. Monitoring Loop (Runs on both first click AND refresh)
    if st.session_state['is_running']:
        log_queue = st.session_state['bg_log_queue']
        futures = st.session_state['bg_futures']
        start_time = st.session_state['bg_start_time']
        
        # Re-attach logger if lost during refresh (trickier part)
        # Since loggers are global, they might persist, but handlers might need checking
        # But StreamToQueue redirects sys.stdout, which is process-level.
        # So logs should still flow to the queue.
        
        completed_count = 0
        
        # Define helper for updating logs
        MAX_LOG_LINES = 500
        if 'log_history' not in st.session_state:
            st.session_state['log_history'] = []

        def update_ui_from_queue():
            """Helper to drain queue and update UI"""
            has_new = False
            temp_logs = []
            
            # 批量取出队列中的所有日志
            while not log_queue.empty():
                try:
                    # 使用 get_nowait 快速非阻塞获取
                    entry = log_queue.get_nowait()
                    temp_logs.append(entry)
                    # 限制每次最大更新量，防止单次渲染卡死
                    if len(temp_logs) > 200:
                        break
                except queue.Empty:
                    break
            
            if temp_logs:
                # 追加到 session state
                st.session_state['log_history'].extend(temp_logs)
                
                # 维护最大长度 - 只保留最新的 MAX_LOG_LINES 行
                if len(st.session_state['log_history']) > MAX_LOG_LINES:
                    st.session_state['log_history'] = st.session_state['log_history'][-MAX_LOG_LINES:]
                
                has_new = True
            
            if has_new:
                # 渲染日志
                log_text = "\n".join(st.session_state['log_history'])
                log_area.code(log_text, language="text")
        
        # Non-blocking check loop
        # We need to run this loop until done.
        # Using st.empty() to update UI
        
        try:
            while not all(f.done() for f in futures):
                if st.session_state.get('stop_requested', False):
                    # ... stop logic ...
                    pass
                
                # Update Logs
                update_ui_from_queue()
                
                # Update Progress
                current_done = sum(1 for f in futures if f.done())
                if current_done >= completed_count:
                    completed_count = current_done
                    progress = current_done / len(benchmark_data)
                    progress_bar_widget.progress(progress)
                    
                    stat_done.markdown(f"### 🏁 已完成: **{current_done} / {len(benchmark_data)}**")
                    
                    elapsed_time = time.time() - start_time
                    if current_done > 0:
                        rate = current_done / elapsed_time
                        if rate > 0:
                            remaining_tasks = len(benchmark_data) - current_done
                            remaining_seconds = remaining_tasks / rate
                            rem_min = int(remaining_seconds // 60)
                            rem_sec = int(remaining_seconds % 60)
                            time_info.markdown(f"**⏱️ 预计剩余时间:** {rem_min} 分 {rem_sec} 秒 (速度: {rate*60:.1f} 个/分)")
                
                time.sleep(0.5)
                # Important: In Streamlit, a while loop blocks the script execution.
                # If user refreshes, script restarts. We need to detect "is_running" and re-enter this loop.
            
            # Done
            st.session_state['is_running'] = False
            progress_bar_widget.progress(1.0)
            stat_done.markdown(f"### ✅ 已完成: **{len(benchmark_data)} / {len(benchmark_data)}**")
            
            # Process Results... (Collect futures results)
            results_list = []
            success_count = 0
            fail_count = 0
            total_score_sum = 0
            
            for i, future in enumerate(futures):
                idx = i 
                try:
                    result = future.result()
                    results_list.append(result)
                    
                    if result['status'] == 'success':
                        success_count += 1
                        total_score_sum += result['total_score']
                    else:
                        fail_count += 1
                        # 错误已经在日志中显示，这里不再弹窗
                except Exception as exc:
                    fail_count += 1
                    logging.error(f"Task {idx+1} Exception: {exc}")
            
            # Update final stats
            stat_success.metric("成功", success_count)
            stat_fail.metric("失败", fail_count)
            current_avg = total_score_sum / success_count if success_count > 0 else 0
            stat_avg_score.metric("均分", f"{current_avg:.1f}")
            
            time_info.success(f"✅ 评测完成！总耗时: {int(time.time() - start_time)} 秒")

            # Save Summary
            if results_list:
                results_list.sort(key=lambda x: x['id'])
                df = pd.DataFrame(results_list)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 安全获取 model name
                safe_model_name = target_model_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace(' ', '_')
                summary_output_dir = Path(output_base_dir) / "summary"
                summary_output_dir.mkdir(parents=True, exist_ok=True)
                
                csv_path = summary_output_dir / f"summary_{safe_model_name}_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                
                st.session_state['results_df'] = df
                st.session_state['show_success_msg'] = True
                st.session_state['saved_csv_path'] = str(csv_path)
            
        except Exception as e:
            st.error(f"Monitor loop error: {e}")
        time.sleep(1) # Show success message briefly
        model_status.empty() # Clear it
        
    # (Old code removed)

    # (Old executor code removed)
            
    # (Old monitor code removed)

# ==========================================
# Results & Visualization (Outside of start button logic, for persistence on reload)
# ==========================================

if 'results_df' in st.session_state:
    # Logic for re-rendering results when not clicking start button (e.g. refresh)
    if st.session_state.get('show_success_msg', False):
        st.success("🎉 评测完成！")
        if 'saved_csv_path' in st.session_state:
            st.info(f"💾 汇总报告已保存: `{st.session_state['saved_csv_path']}`")
        st.session_state['show_success_msg'] = False # Reset flag

    df = st.session_state['results_df']
    
    st.markdown("---")
    st.header("📊 3. 评测报告与可视化")

    
    total_count = len(df)
    if total_count > 0:
        pass_threshold = 60
        
        total_pass = len(df[df['total_score'] >= pass_threshold])
        l1_pass = len(df[df['layer1_score'] >= pass_threshold])
        l2_pass = len(df[df['layer2_score'] >= pass_threshold])
        l3_pass = len(df[df['layer3_score'] >= pass_threshold])
        l4_pass = len(df[df['layer4_score'] >= pass_threshold])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🏆 总合格率", f"{total_pass/total_count:.1%}", f"Avg: {df['total_score'].mean():.1f}")
        col2.metric("📝 语法 (L1)", f"{l1_pass/total_count:.1%}", f"Avg: {df['layer1_score'].mean():.1f}")
        col3.metric("⚛️ 物理 (L2)", f"{l2_pass/total_count:.1%}", f"Avg: {df['layer2_score'].mean():.1f}")
        col4.metric("👁️ 视觉 (L3)", f"{l3_pass/total_count:.1%}", f"Avg: {df['layer3_score'].mean():.1f}")
        col5.metric("🖱️ 交互 (L4)", f"{l4_pass/total_count:.1%}", f"Avg: {df['layer4_score'].mean():.1f}")
        
        st.subheader("� 图表 (Charts)")
        tab1, tab2, tab3 = st.tabs(["得分分布", "合格率对比", "能力雷达图"])
        
        with tab1:
            fig1 = plot_score_distribution(df)
            st.pyplot(fig1)
            buf1 = BytesIO()
            fig1.savefig(buf1, format="pdf", bbox_inches='tight')
            st.download_button("📥 下载 PDF", buf1.getvalue(), "score_dist.pdf", "application/pdf")
            
        with tab2:
            fig2 = plot_pass_rates(df, threshold=pass_threshold)
            st.pyplot(fig2)
            buf2 = BytesIO()
            fig2.savefig(buf2, format="pdf", bbox_inches='tight')
            st.download_button("📥 下载 PDF", buf2.getvalue(), "pass_rates.pdf", "application/pdf")
            
        with tab3:
            fig3 = plot_radar_chart(df)
            st.pyplot(fig3)
            buf3 = BytesIO()
            fig3.savefig(buf3, format="pdf", bbox_inches='tight')
            st.download_button("📥 下载 PDF", buf3.getvalue(), "radar_chart.pdf", "application/pdf")
            
        with st.expander("📋 查看详细数据表"):
            st.dataframe(df)

# ==========================================
# 4. Multi-Model Comparison
# ==========================================
import re

def parse_summary_filename(filename):
    """
    Parses filename like 'summary_model_name_20250128_123045.csv'
    Returns (model_name, timestamp_str, datetime_obj)
    """
    # Regex to match summary_{model_name}_{timestamp}.csv
    # Timestamp format: YYYYMMDD_HHMMSS (15 chars)
    # We look for the last pattern of _\d{8}_\d{6}
    match = re.search(r'summary_(.+)_(\d{8}_\d{6})\.csv', filename)
    if match:
        model_name = match.group(1)
        ts_str = match.group(2)
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            return model_name, ts_str, dt
        except ValueError:
            return None, None, None
    return None, None, None

def plot_comparison_bar(combined_df, highlight_models=None, model_order=None):
    """Grouped bar chart for model comparison with academic style (patterns/hatches)"""
    # Calculate means
    metrics = ['total_score', 'layer1_score', 'layer2_score', 'layer3_score', 'layer4_score']
    metric_labels = ['总分 (Total)', '语法 (L1)', '物理 (L2)', '视觉 (L3)', '交互 (L4)']
    
    # Calculate average scores per model per metric
    avg_scores = combined_df.groupby('Model', sort=False)[metrics].mean().reset_index()
    
    # Apply explicit model order if provided
    if model_order:
        avg_scores['Model'] = pd.Categorical(avg_scores['Model'], categories=model_order, ordered=True)
        avg_scores = avg_scores.sort_values('Model')
    
    # Melt for plotting
    df_melted = avg_scores.melt(id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Score')
    metric_map = dict(zip(metrics, metric_labels))
    df_melted['Metric'] = df_melted['Metric'].map(metric_map)
    
    # Unique models and metrics (preserve order from avg_scores)
    models = avg_scores['Model'].unique()
    metrics_list = metric_labels
    
    # Dynamic figure size based on number of models
    n_models = len(models)
    n_metrics = len(metrics_list)
    width = max(12, n_metrics * n_models * 0.8)
    fig, ax = plt.subplots(figsize=(width, 6))
    
    # Academic pastel colors (Match user reference: Blue, Pink, Purple, Green)
    # Reference: Level I (Blue), Level II (Purple), Level III (Teal), Level IV (Pink), Level V (Beige)
    colors = ['#C4D5E6', '#9FA0C3', '#8EC7C5', '#D2A1B3', '#E0D0B6']
    if n_models > len(colors):
        # Extend palette if needed
        import matplotlib.colors as mcolors
        extended_colors = list(mcolors.TABLEAU_COLORS.values())
        colors.extend(extended_colors)
    
    # Hatches for texture
    hatches = ['/', '\\', 'x', '.', 'o', '+', '*', '-', '|', 'O']
    
    # Bar width configuration
    bar_width = 0.8 / n_models
    x = np.arange(len(metrics_list))
    
    # Plot bars manually to control hatch and color perfectly
    for i, model in enumerate(models):
        model_data = df_melted[df_melted['Model'] == model]
        # Ensure order aligns with metrics_list
        scores = [model_data[model_data['Metric'] == m]['Score'].values[0] if not model_data[model_data['Metric'] == m].empty else 0 for m in metrics_list]
        
        offset = (i - n_models/2 + 0.5) * bar_width
        
        # Determine highlighting style
        if highlight_models and model not in highlight_models:
            bar_color = colors[i % len(colors)] # Keep original color
            bar_alpha = 0.3 # Reduced opacity
            bar_hatch = '' # Remove hatch for non-highlighted
            bar_linewidth = 0.5
        else:
            bar_color = colors[i % len(colors)]
            bar_alpha = 1.0
            bar_hatch = hatches[i % len(hatches)] * 2 # Double hatch density
            bar_linewidth = 1
            
        # Draw the bar with WHITE hatching (Academic Style)
        # 1. Main bar: Color fill + White Hatch + White Border
        bars = ax.bar(x + offset, scores, 
                      width=bar_width, 
                      label=model, 
                      color=bar_color, 
                      edgecolor='white', # White hatch and border to match reference style
                      linewidth=bar_linewidth,
                      alpha=bar_alpha,
                      hatch=bar_hatch)
                      
        # 2. (Optional) If we want a border, we would plot again with fill=False.
        # But standard academic pastel charts usually don't have black borders.
        
        # Add value labels on top
        for bar in bars:
            height = bar.get_height()
            # Only show labels for highlighted models if highlighting is active
            if not highlight_models or model in highlight_models:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')

    # Remove title
    # ax.set_title("Model Comparison - Average Scores", fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("平均分 (Score)", fontsize=14, fontfamily='Microsoft YaHei')
    ax.set_ylim(0, 115) # More space for legend
    ax.set_xticks(x)
    # Set Chinese font for X labels
    ax.set_xticklabels(metrics_list, fontsize=12, fontweight='bold', fontfamily='Microsoft YaHei')
    
    # Legend settings
    ax.legend(title='Model', bbox_to_anchor=(0.5, -0.15), loc='upper center', 
              ncol=min(n_models, 5), frameon=False, fontsize=11)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.subplots_adjust(bottom=0.2) # Make space for legend
    
    return fig

def plot_comparison_radar(combined_df, highlight_models=None, model_order=None, dynamic_scale=True):
    """Radar chart comparison"""
    categories = ['语法 (L1)', '物理 (L2)', '视觉 (L3)', '交互 (L4)']
    
    # Prepare data
    if model_order:
        models = model_order
    else:
        models = combined_df['Model'].unique()
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += [angles[0]]
    
    # Color palette
    colors = sns.color_palette("viridis", len(models))
    
    # Track min/max for dynamic scaling
    all_scores = []
    
    for i, model in enumerate(models):
        model_df = combined_df[combined_df['Model'] == model]
        scores = [
            model_df['layer1_score'].mean(), 
            model_df['layer2_score'].mean(), 
            model_df['layer3_score'].mean(), 
            model_df['layer4_score'].mean()
        ]
        all_scores.extend(scores)
        scores = np.concatenate((scores, [scores[0]]))
        
        # Highlight logic
        if highlight_models and model not in highlight_models:
            line_color = colors[i] # Keep original color
            line_alpha = 0.3 # Reduced opacity
            line_width = 1
            fill_alpha = 0.0
            zorder = 1
            marker_alpha = 0.3
        else:
            line_color = colors[i]
            line_alpha = 1.0
            line_width = 3 if highlight_models else 2
            fill_alpha = 0.05
            zorder = 2
            marker_alpha = 1.0
            
        ax.plot(angles, scores, linewidth=line_width, label=model, color=line_color, alpha=line_alpha, zorder=zorder, marker='o', markersize=4, markeredgecolor='white', markeredgewidth=0.5)
        # Only fill if opacity > 0
        if fill_alpha > 0:
            ax.fill(angles, scores, color=line_color, alpha=fill_alpha, zorder=zorder)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontfamily='Microsoft YaHei')
    
    # Dynamic Scale Logic
    if dynamic_scale and all_scores:
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        # Determine lower bound (round down to nearest 10, minus padding)
        # e.g. min=65 -> lower=50
        lower_bound = max(0, (int(min_score) // 10) * 10 - 10)
        
        # Determine upper bound
        upper_bound = min(100, (int(max_score) // 10) * 10 + 10)
        if upper_bound < 100 and max_score > 90:
             upper_bound = 100
             
        # Generate ticks
        # e.g. 50, 60, 70, 80, 90, 100
        ticks = list(range(lower_bound, upper_bound + 10, 10))
        # Remove first tick if it's the center to avoid clutter, unless it's 0
        if len(ticks) > 1 and ticks[0] == lower_bound:
             # ax.set_rlabel_position does not hide the center label automatically
             pass
             
        ax.set_ylim(lower_bound, upper_bound)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks], color="grey", size=9)
    else:
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], color="grey", size=9)
        ax.set_ylim(0, 100)

    # Remove title
    # ax.set_title("Model Capabilities Comparison", size=16, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    return fig

def plot_comparison_box(combined_df, highlight_models=None, model_order=None):
    """Box plot for score distribution comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create custom palette based on highlight
    if model_order:
        models = model_order
    else:
        models = combined_df['Model'].unique()
    
    default_palette = sns.color_palette("viridis", len(models))
    palette_dict = {}
    
    for i, model in enumerate(models):
        # Always use default palette but will control alpha later if needed
        # Or simpler: just map colors directly
        palette_dict[model] = default_palette[i]
            
    # Draw boxplot
    # Note: seaborn boxplot doesn't support alpha well directly on the artist creation
    # We will modify artists after plotting
    sns.boxplot(data=combined_df, x='Model', y='total_score', hue='Model', palette=palette_dict, legend=False, ax=ax, order=models)
    
    # Adjust alpha for non-highlighted boxes
    if highlight_models:
        # Iterate over boxes (Patch objects)
        # The order of artists in ax.artists usually matches the order of x-axis categories
        # But let's be careful. The x-axis order is determined by 'models' array usually if passed to order or if pandas categoricals used.
        # Here we rely on seaborn's default ordering which should match 'models' (unique appearance)
        
        # seaborn boxplot creates PathPatch for boxes.
        # The number of boxes = len(models)
        
        # Check if we have the right number of artists
        # Note: boxplot might create whiskers/caps/medians as Line2D, and boxes as PathPatch
        # ax.artists usually contains the boxes (Patch)
        
        # Let's map model name to index
        model_to_idx = {m: i for i, m in enumerate(models)}
        
        # Iterate through patches (the boxes)
        for i, patch in enumerate(ax.artists):
            if i < len(models):
                model_name = models[i]
                if model_name not in highlight_models:
                    # Set alpha to 0.3 for the box face
                    r, g, b, _ = patch.get_facecolor()
                    patch.set_facecolor((r, g, b, 0.3))
                    # Fade the edge color
                    patch.set_edgecolor((0, 0, 0, 0.3))
        
        # Handle lines (whiskers, caps, medians)
        # Seaborn usually draws 5-6 lines per box (2 whiskers, 2 caps, 1 median, maybe fliers)
        # We assume lines are ordered by category
        num_lines = len(ax.lines)
        if len(models) > 0:
            lines_per_box = num_lines // len(models)
            for i, line in enumerate(ax.lines):
                model_idx = i // lines_per_box
                if model_idx < len(models):
                    model_name = models[model_idx]
                    if model_name not in highlight_models:
                        line.set_alpha(0.3)

    # Handle stripplot transparency
    if highlight_models:
        # For stripplot, we can iterate over the PathCollection
        # But simpler to just replot stripplot with alpha control is hard per point group
        # Easier: Plot non-highlighted with low alpha, highlighted with high alpha
        
        # 1. Plot non-highlighted
        non_high_df = combined_df[~combined_df['Model'].isin(highlight_models)]
        if not non_high_df.empty:
            sns.stripplot(data=non_high_df, x='Model', y='total_score', color='black', alpha=0.1, size=4, ax=ax, order=models)
            
        # 2. Plot highlighted
        high_df = combined_df[combined_df['Model'].isin(highlight_models)]
        if not high_df.empty:
            sns.stripplot(data=high_df, x='Model', y='total_score', color='black', alpha=0.5, size=4, ax=ax, order=models)
            
    else:
        sns.stripplot(data=combined_df, x='Model', y='total_score', color='black', alpha=0.3, size=4, ax=ax, order=models)
    
    # Remove title
    # ax.set_title("Total Score Distribution by Model", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("总分 (Total Score)", fontsize=12, fontfamily='Microsoft YaHei')
    ax.set_xlabel("模型 (Model)", fontsize=12, fontfamily='Microsoft YaHei')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    # Rotate x labels if many models
    plt.xticks(rotation=45, ha='right')
    return fig

st.markdown("---")
st.header("📈 4. 多模型对比 (Multi-Model Comparison)")

with st.expander("📊 配置对比选项 (Configuration)", expanded=True):
    # Scan for files
    summary_dir = Path(output_base_dir) / "summary"
    root_dir = Path(output_base_dir)
    
    all_csvs = []
    if summary_dir.exists():
        all_csvs.extend(list(summary_dir.glob("summary_*.csv")))
    
    # Try root dir too just in case
    if root_dir.exists():
        all_csvs.extend(list(root_dir.glob("summary_*.csv")))
        
    all_csvs = list(set(all_csvs))
    
    if not all_csvs:
        st.info("尚未发现任何评测汇总报告 (CSV)。请先运行评测。")
    else:
        # Group by model
        model_map = {} # model_name -> list of (dt, filepath)
        
        for f in all_csvs:
            m_name, ts, dt = parse_summary_filename(f.name)
            if m_name and dt:
                if m_name not in model_map:
                    model_map[m_name] = []
                model_map[m_name].append((dt, f))
        
        # Select models
        # Sort available models to ensure stable order across runs (prevents widget reset)
        available_models = sorted(list(model_map.keys()))
        
        # 默认选中所有模型（不再限制为前5个）
        # We add a key to ensure state persistence
        selected_models = st.multiselect(
            "选择要对比的模型 (Select Models)", 
            available_models, 
            default=available_models,
            key="multi_model_selector"
        )
        
        # 新增高亮选择
        highlight_models = st.multiselect(
            "选择高亮模型 (Highlight Models - Optional)", 
            selected_models,
            default=[],
            help="选中模型将高亮显示，未选中模型将变灰。留空则显示所有。"
        )
        
        # 新增：手动调整显示顺序
        # 默认使用 selected_models 的顺序（这通常是用户添加的顺序，但 multiselect 可能会重排）
        # 我们提供一个单独的排序组件
        with st.expander("↕️ 调整模型显示顺序 (Sort Order)", expanded=False):
             st.caption("💡 提示：Streamlit 原生组件暂不支持拖拽。请在下方文本框中通过**剪切/粘贴**调整模型顺序（每行一个）。")
             # Create a string for the text area
             # We use a key based on selected models to reset if selection changes drastically? 
             # No, let's keep it simple. If user adds model, it appends.
             
             default_order_text = "\n".join(selected_models)
             
             # Trick: logic to preserve user edits if possible, but hard without session state management complexity.
             # Simple approach: Default to selected_models order.
             
             order_text = st.text_area(
                "编辑模型顺序 (Edit Order)",
                value=default_order_text,
                height=150,
                help="在此处手动调整顺序，每行一个模型名称。"
             )
             
             # Parse and validate
             if order_text:
                 input_models = [line.strip() for line in order_text.split('\n') if line.strip()]
                 # Filter to ensure only currently selected models are included
                 valid_order = [m for m in input_models if m in selected_models]
                 # Append any missing models (that might have been selected but user deleted from text area)
                 missing_models = [m for m in selected_models if m not in valid_order]
                 model_order = valid_order + missing_models
             else:
                 model_order = selected_models

        if selected_models:
            # For each model, pick the latest file
            # Or allow user to choose versions? For simplicity, pick latest.
            
            comparison_data = []
            
            for m in selected_models:
                # Sort by date desc
                runs = sorted(model_map[m], key=lambda x: x[0], reverse=True)
                latest_run = runs[0]
                filepath = latest_run[1]
                
                try:
                    df_temp = pd.read_csv(filepath)
                    df_temp['Model'] = m # Add model column
                    comparison_data.append(df_temp)
                except Exception as e:
                    st.error(f"Error reading {filepath.name}: {e}")
            
            if comparison_data:
                df_all = pd.concat(comparison_data, ignore_index=True)
                
                st.success(f"已加载 {len(selected_models)} 个模型的 {len(df_all)} 条数据")
                
                # Chart Type Selection
                chart_type = st.radio(
                    "选择图表类型 (Chart Type)", 
                    ["柱状图 (Bar Chart) - 均分对比", "雷达图 (Radar Chart) - 能力对比", "箱线图 (Box Plot) - 分布对比"],
                    horizontal=True
                )
                
                # Radar Chart specific options
                dynamic_radar_scale = True
                if "Radar Chart" in chart_type:
                    dynamic_radar_scale = st.checkbox("🔍 自动缩放雷达图坐标轴 (Zoom Y-Axis)", value=True, help="根据数据范围动态调整坐标轴起点，拉开模型间距。")

                st.subheader("🖼️ 可视化结果")
                
                if "Bar Chart" in chart_type:
                    fig = plot_comparison_bar(df_all, highlight_models, model_order)
                    filename_base = "model_comparison_bar"
                elif "Radar Chart" in chart_type:
                    fig = plot_comparison_radar(df_all, highlight_models, model_order, dynamic_scale=dynamic_radar_scale)
                    filename_base = "model_comparison_radar"
                else:
                    fig = plot_comparison_box(df_all, highlight_models, model_order)
                    filename_base = "model_comparison_box"
                
                st.pyplot(fig)
                
                # Download Buttons
                col_d1, col_d2 = st.columns([1, 5])
                with col_d1:
                     buf_png = BytesIO()
                     fig.savefig(buf_png, format="png", dpi=300, bbox_inches='tight')
                     st.download_button(
                         "📥 下载高清 PNG (300 DPI)",
                         buf_png.getvalue(),
                         f"{filename_base}.png",
                         "image/png"
                     )
                with col_d2:
                     buf_pdf = BytesIO()
                     fig.savefig(buf_pdf, format="pdf", bbox_inches='tight')
                     st.download_button(
                         "📥 下载矢量 PDF",
                         buf_pdf.getvalue(),
                         f"{filename_base}.pdf",
                         "application/pdf"
                     )
                    
                with st.expander("📥 导出对比数据 (Export Data)"):
                    st.dataframe(df_all)
                    csv = df_all.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 下载合并CSV (Download Merged CSV)",
                        csv,
                        "model_comparison.csv",
                        "text/csv",
                        key='download-csv'
                    )
            else:
                st.warning("未能加载有效数据。")
        else:
            st.info("请选择至少一个模型进行分析。")
