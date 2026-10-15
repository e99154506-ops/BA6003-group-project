import os
import logging
import requests

# ==========================================
# 1. 检查环境变量（我们在终端 export 过的）
# ==========================================
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    print("❌ 错误：没找到环境变量 OPENROUTER_API_KEY！")
    print("请先在终端里执行：export OPENROUTER_API_KEY=\"你的密钥\"")
    exit()

# ==========================================
# 2. 准备日志
# ==========================================
logging.basicConfig(
    filename='outputs/ai_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================================
# 3. 模拟仪表盘数据
# ==========================================
view_context = "全样本因子回测，Fama-French 三因子 + 动量模型"
metrics = {
    "选股数量": "Top 20 只股票",
    "回测月份数": 60,
    "年化收益率": "12.5%",
    "最大回撤": "-8.3%",
    "夏普比率": "1.45"
}

# ==========================================
# 4. 拼装 Prompt（护栏 + 格式）
# ==========================================
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
- 两个对比性要点：对比核心指标或对比最重要的两个因子
- 局限性：指出该回测结果的盲区
字数控制在 150 字以内。
"""

# ==========================================
# 5. 调用 API（★关键：换成免费账号允许使用的开源模型★）
# ==========================================
print(">>> 正在请求 AI 诊断报告...")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

body = {
    # 关键：换成 Llama 3 免费模型，免费账号可用！
    "model": "meta-llama/llama-3-8b-instruct:free",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
}

try:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=body
    )
    response.raise_for_status()
    result = response.json()
    ai_text = result["choices"][0]["message"]["content"]
    
    print("\n" + "="*40)
    print("🤖 AI 洞察报告")
    print("="*40)
    print(ai_text)
    print("\n✅ 成功！这就是 AI 生成的报告。")
    
    logging.info(f"AI 调用成功 | 视图: {view_context} | 状态: success")
    logging.info(f"AI 输出: \n{ai_text}")

except Exception as e:
    print(f"\n❌ 出错了: {e}")
    logging.error(f"AI 调用失败 | 错误: {str(e)}")
