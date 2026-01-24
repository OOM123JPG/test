import asyncio
import httpx
import time

async def send_req(i):
    async with httpx.AsyncClient(timeout=300) as client:
        start = time.time()
        resp = await client.post("http://127.0.0.1:8888/v1/completions", json={
            "prompt": f"Test prompt {i}", "max_tokens": 10
        })
        print(f"请求 {i} 完成，耗时: {time.time()-start:.2f}s")

async def main():
    # 模拟 4 个并发请求同时到达
    await asyncio.gather(*[send_req(i) for i in range(4)])

if __name__ == "__main__":
    asyncio.run(main())