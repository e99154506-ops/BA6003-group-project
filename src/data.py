# src/data.py
import time
import pandas as pd
import yfinance as yf
from pathlib import Path
from .config import TICKERS, START_DATE, END_DATE, PRICE_CSV, DATA_DIR

def download_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 已下载的
    done = set()
    frames = []

    if PRICE_CSV.exists():
        old = pd.read_csv(PRICE_CSV, parse_dates=["date"])
        done = set(old["symbol"].unique())
        frames.append(old)
        print(f"✅ 已缓存 {len(done)} 只，跳过")

    todo = [t for t in TICKERS if t not in done]
    print(f"📋 剩余需下载: {len(todo)} 只")

    if not todo:
        print("🎉 全部已下载完毕！")
        return

    BATCH = 5
    failed = []

    for i in range(0, len(todo), BATCH):
        batch = todo[i:i+BATCH]
        print(f"⬇️ 批次 {i//BATCH+1}/{(len(todo)-1)//BATCH+1}: {batch}")

        try:
            data = yf.download(
                batch, start=START_DATE, end=END_DATE,
                group_by="ticker", auto_adjust=False,
                progress=False, threads=False
            )
            for t in batch:
                try:
                    df = data[t].reset_index()
                    df["symbol"] = t
                    frames.append(df[["Date", "symbol", "Open", "High", "Low", "Close", "Volume"]])
                    done.add(t)
                except Exception:
                    failed.append(t)
        except Exception as e:
            print(f"  ❌ 批次失败: {e}")
            failed.extend(batch)

        time.sleep(5)

        # 每 5 批存一次中间结果（防中断丢数据）
        if (i // BATCH + 1) % 5 == 0 and frames:
            partial = pd.concat(frames, ignore_index=True)
            partial = partial.rename(columns={"Date": "date", "Close": "close"})
            partial["date"] = pd.to_datetime(partial["date"])
            partial.to_csv(PRICE_CSV, index=False)
            print(f"  💾 中间保存: {len(done)} 只")

    # 最终合并
    if frames:
        prices = pd.concat(frames, ignore_index=True)
        prices = prices.rename(columns={"Date": "date", "Close": "close"})
        prices["date"] = pd.to_datetime(prices["date"])
        prices.to_csv(PRICE_CSV, index=False)
        print(f"\n🎉 完成！共 {prices['symbol'].nunique()} 只, {len(prices)} 行")
        print(f"📁 保存至: {PRICE_CSV}")

    if failed:
        print(f"⚠️ 失败 {len(failed)} 只: {failed[:10]}")