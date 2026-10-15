# src/ai_test.py
# Week 10 作业
# 用法：python -m src.ai_test

import os
import logging

# ==========================================
# 1. 配置日志
# ==========================================
logging.basicConfig(
    filename='outputs/ai_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================================
# 2. 准备测试数据
# ==========================================
view_context = "多因子选股回测，股票池 AAPL/MSFT/GOOG/TSLA，选出 Top 3 组合"
metrics = {
    "累计收益": "12.5%",      # ← 改成你们仪表盘的真实数字
    "标普500收益": "9.8%",    # ← 改成你们仪表盘的真实数字
    "超额收益": "2.7%",       # ← 改成你们仪表盘的真实数字
    "选股数量": "3 只",
    "股票池": "AAPL, MSFT, GOOG, TSLA"
}

# ==========================================
# 3. 调用 AI 模块
# ==========================================
from ai_insights import generate_ai_insight

print(">>> 正在请求 AI 诊断报告...")
report = generate_ai_insight(view_context, metrics)

print("\n" + "="*40)
print("🤖 AI 洞察报告")
print("="*40)
print(report)

logging.info(f"AI 测试 | 视图: {view_context}")
logging.info(f"AI 输出:\n{report}")
print("\n📝 日志已写入 outputs/ai_test.log")
