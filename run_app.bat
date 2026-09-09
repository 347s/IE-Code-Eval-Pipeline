@echo off
chcp 65001 >nul
setlocal

echo.
echo ========================================================
echo 🚀 正在启动 Web 评测系统 (Streamlit App)...
echo ========================================================
echo.

:: 检查 streamlit 是否安装
where streamlit >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到 streamlit。
    echo 请先运行: pip install streamlit
    pause
    exit /b 1
)

:: 启动应用
cd /d "%~dp0"
python -m streamlit run app.py --server.address 127.0.0.1

if %errorlevel% neq 0 (
    echo.
    echo ❌ 应用异常退出。
    pause
    exit /b %errorlevel%
)
