import numpy as np

def performance_summary(portfolio_ret):
    cum = (1 + portfolio_ret).cumprod()
    ann_ret = portfolio_ret.mean() * 12
    ann_vol = portfolio_ret.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = ((cum / cum.cummax()) - 1).min()
    
    return {
        "年化收益": ann_ret,
        "年化波动": ann_vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "累计收益": cum.iloc[-1] - 1,
    }