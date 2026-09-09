#!/bin/bash

echo ""
echo "========================================================"
echo "🚀 正在启动 Web 评测系统 (Streamlit App)..."
echo "========================================================"
echo ""

# 检查 streamlit 是否安装
if ! command -v streamlit &> /dev/null; then
    echo "❌ 错误: 未找到 streamlit。"
    echo "请先运行: pip install streamlit"
    exit 1
fi

# 启动应用
cd -- "$(dirname -- "$0")" || exit 1
python -m streamlit run app.py --server.address 127.0.0.1

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo ""
    echo "❌ 应用异常退出。"
    exit $exit_code
fi
