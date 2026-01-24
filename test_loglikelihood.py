import requests
import json

def test_likelihood_alignment(base_url="http://localhost:8888/v1"):
    url = f"{base_url}/completions"
    
    # 测试用例：对比正确事实和错误事实
    # 理论上 "Paris" 的概率应该远高于 "London"
    test_cases = [
        {
            "correct": "The capital of France is Paris",
            "wrong": "The capital of France is London",
            "target": ["Paris", "London"]
        },
        {
            "correct": "1 + 1 = 2",
            "wrong": "1 + 1 = 5",
            "target": ["2", "5"]
        }
    ]

    for case in test_cases:
        print(f"\n--- Testing Case: {case['correct']} vs {case['wrong']} ---")
        
        results = []
        for prompt in [case['correct'], case['wrong']]:
            payload = {
                "model": "deepseek-v3-tucker",
                "prompt": prompt,
                "logprobs": 1,
                "echo": True,
                "temperature": 0
            }
            response = requests.post(url, json=payload).json()
            
            # 提取最后一位 Token 的概率
            tokens = response['choices'][0]['logprobs']['tokens']
            lps = response['choices'][0]['logprobs']['token_logprobs']
            
            last_token = tokens[-1]
            last_lp = lps[-1]
            results.append((last_token, last_lp, tokens, lps))

        (t_c, lp_c, tokens_c, _), (t_w, lp_w, _, _) = results

        # 打印详细对齐情况（观察 Token 和 Logprob 是否对应）
        print(f"Tokens 序列示例: {' | '.join(tokens_c)}")
        print(f"正确项 [{t_c}] Logprob: {lp_c:.4f}")
        print(f"错误项 [{t_w}] Logprob: {lp_w:.4f}")

        if lp_c > lp_w:
            print("✅ 趋势正确：模型认为正确答案的概率更高。")
        else:
            print("❌ 错误：正确答案概率反而更低，请检查 Index Shift 逻辑！")

        # 检查首位 Padding 过滤是否生效
        if tokens_c[0] == "<pad>":
            print("❌ 警告：首位发现 <pad>，API 的 Padding 过滤逻辑可能未生效。")
        else:
            print("✅ 过滤检查：首位不是 <pad>，过滤正常。")

if __name__ == "__main__":
    try:
        test_likelihood_alignment()
    except Exception as e:
        print(f"连接失败或报错: {e}")