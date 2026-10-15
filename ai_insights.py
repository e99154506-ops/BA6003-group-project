# src/ai_insights.py
import os
import logging
import requests
from dotenv import load_dotenv

# 加载 .env 文件里的 API Key
load_dotenv()

# 设置日志，写入 outputs/ai_pipeline.log（PPT 第 17 页要求）
logging.basicConfig(
    filename='outputs/ai_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def generate_ai_insight(view_context: str, metrics: dict) -> str:
    """
    调用 OpenRouter LLM 生成模型诊断报告。
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logging.error("未找到 OPENROUTER_API_KEY")
        return "错误：未找到 OPENROUTER_API_KEY，请检查 .env 文件。"

    # 1. 提示词设计（PPT 第 16 页：人设 + 护栏 + 格式）
    system_prompt = (
        "你是一位资深量化分析师。你的回答必须简洁、专业、客观。"
        "硬性限制：严禁给出任何买入/卖出建议，严禁编造数据，必须明确指出模型的局限性。"
    )
    
    metrics_text = "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
    
    user_prompt = f"""
    当前仪表盘视图：{view_context}
    模型核心指标与因子权重：
    {metrics_text}
    
    请生成一份面向投资总监的简报，格式要求：
    - 标题：一句话总结当前模型状态
    - 两个对比性要点：对比核心指标（如收益 vs 回撤）或对比最重要的两个因子
    - 局限性：指出该回测结果的盲区
    字数控制在 150 字以内。
    """

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-flash-1.5",  # 或者换成老师 demo 里的模型
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
        )
        response.raise_for_status()
        result = response.json()
        ai_text = result["choices"][0]["message"]["content"]
        
        # 2. 记录日志（PPT 第 17 页：记录动作、状态、AI输出，绝对不记 Key）
        logging.info(f"AI 调用成功 | 视图: {view_context} | 状态: success")
        logging.info(f"AI 输出: {ai_text}")
        
        return ai_text
    except Exception as e:
        logging.error(f"AI 调用失败 | 视图: {view_context} | 错误: {str(e)}")
        return f"AI 服务调用失败: {str(e)}"
