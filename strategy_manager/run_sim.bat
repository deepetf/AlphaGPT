@echo off
REM =====================================================
REM CB 模拟盘 - Windows 任务计划脚本
REM 
REM 使用方法:
REM   1. 直接双击运行: 手动执行一次模拟
REM   2. 添加到任务计划程序:
REM      - 触发器: 每天 14:50
REM      - 操作: 启动程序 -> 选择此 .bat 文件
REM =====================================================

cd /d "c:\Trading\Projects\AlphaGPT\strategy_manager"

REM 激活虚拟环境 (如果有)
REM call venv\Scripts\activate.bat

REM 运行模拟
python run_sim.py --top-k 10 --take-profit 0.08

REM 保持窗口打开以便查看日志
pause
