import os
import requests

# 在这里填入 API Key
# https://openrouter.ai/keys 免费注册拿一个
API_KEY = "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"   # <--- 把这行引号里的内容换成你自己的 Key


# 模拟的仪表盘数据
view_context = "全样本因子回测，Fama-French 三因子 + 动量模型"
metrics = {
    "选股数量": "Top 20 只股票",
    "回测月份数": 60,
    "年化收益率": "12.5%",
    "最大回撤": "-8.3%",
    "夏普比率": "1.45"
}


#提示词
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


# 调用 AI
print(">>> 正在请求 AI 诊断报告...")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

body = {
    "model": "meta-llama/llama-3.1-8b-instruct:free",   # 如果报错 model not found，换成 "meta-llama/llama-3.1-8b-instruct:free"
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
    
except Exception as e:
    print(f"\n❌ 出错了: {e}")
    print("请检查：1. 密钥是否填对  2. 模型名字是否正确  3. 网络是否通畅")
