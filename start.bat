@echo off
chcp 65001 >nul 2>&1
echo 启动AI小说生成器...

cd /d "%~dp0"

if not exist venv (
    echo 创建虚拟环境...
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -r requirements.txt

echo 启动应用程序...
python main.py

pause
