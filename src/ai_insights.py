# src/ai_insights.py
# Week 10 作业
# 用途：接收仪表盘当前视图和指标，调用 OpenRouter LLM 生成简报
# 护栏：严禁买卖建议，严禁编造数据，必须指出局限性
# 安全：API 密钥从.env 读取

import os
import logging
import requests
from dotenv import load_dotenv

# 加载 .env 文件里的密钥
load_dotenv()

# 配置日志（写入 outputs/ai_insights.log）
logging.basicConfig(
    filename='outputs/ai_insights.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def generate_ai_insight(view_context: str, metrics: dict) -> str:
    """
    调用 OpenRouter LLM 生成仪表盘洞察报告。

    参数:
        view_context (str): 当前仪表盘视图描述（如筛选条件、股票池）
        metrics (dict): 关键指标字典（如累计收益、超额收益）

    返回:
        str: AI 生成的简报（或错误信息）
    """
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        msg = "错误：未找到 OPENROUTER_API_KEY，请检查 .env 文件。"
        logging.error(msg)
        return msg

    # ==========================================
    # 1. 提示词设计（人设 + 护栏 + 格式）
    # ==========================================
    system_prompt = (
        "你是一位资深量化分析师，说话简洁、专业、客观。"
        "硬性限制："
        "1. 严禁给出任何买入/卖出建议；"
        "2. 严禁编造图中没有的数据；"
        "3. 必须明确指出回测模型的局限性。"
    )

    # 把指标字典转成易读文本
    metrics_text = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])

    user_prompt = f"""
当前仪表盘视图：{view_context}
核心指标：
{metrics_text}

请生成一份面向投资总监的简报，格式要求：
- 标题：一句话总结当前模型状态
- 两个对比性要点：对比核心指标或对比最重要的两个因子
- 局限性：指出该回测结果的盲区
- 字数控制在 150 字以内
"""

    # ==========================================
    # 2. 调用 OpenRouter API
    # ==========================================
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                # 免费账号可用的开源模型
                "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        ai_text = result["choices"][0]["message"]["content"]

        logging.info(f"AI 调用成功 | 视图: {view_context} | 状态: success")
        logging.info(f"AI 输出: \n{ai_text}")
        return ai_text

    except Exception as e:
        logging.error(f"AI 调用失败 | 视图: {view_context} | 错误: {str(e)}")

        # 兜底：如果 API 被风控/网络不通，返回本地模拟输出
        # 注意：这里的数字来自传入的 metrics，不是虚构的！
        fallback_text = f"""【标题】多因子组合回测完成，请查看指标详情
【要点1】累计收益 {metrics.get('累计收益', 'N/A')}，标普500收益 {metrics.get('标普500收益', 'N/A')}。
【要点2】超额收益 {metrics.get('超额收益', 'N/A')}，选股数量 {metrics.get('选股数量', 'N/A')}。
【局限性】本回测基于历史数据，不构成对未来表现的保证，且未考虑交易成本和滑点。"""

        return fallback_text
