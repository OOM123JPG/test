import requests
import json
import logging
from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils.function_utils import thread_safe

@register_model_api(name='distributed_deepseek')
class DistributedDeepSeekAdapter(ModelAPI):
    """适配器：调用 api_inference.py 提供的分布式推理服务"""
    
    def __init__(self, model_name: str,
                 base_url: str = "http://127.0.0.1:8888",
                 api_key: Optional[str] = None,
                 config: GenerateConfig = GenerateConfig(),
                 **kwargs) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        self.api_url = base_url.rstrip('/')
        self.chat_completions_url = f"{self.api_url}/v1/chat/completions"
        self.completions_url = f"{self.api_url}/v1/completions"
        self.session = requests.Session()
        self.timeout = kwargs.get('timeout', 300)
        self.max_workers = kwargs.get('max_workers', 8)

    def generate(self, input: Union[List[ChatMessage], str], 
                 config: GenerateConfig = None, 
                 **kwargs) -> ModelOutput:
        """
        改进后的生成方法：
        1. 自动识别 Log-likelihood (Completion) 模式与 Chat 模式
        2. 兼容解析 choices[0]['text'] 和 choices[0]['message']['content']
        """
        gen_config = config or self.config
        
        # 判断请求类型：EvalScope 在算概率时会传 logprobs，或者输入是纯字符串
        is_ll_task = kwargs.get('logprobs') is not None
        is_completion = isinstance(input, str)
        
        if is_ll_task or is_completion:
            # 走 /v1/completions 接口 (Log-likelihood 必须走这个)
            target_url = self.completions_url
            prompt = input if isinstance(input, str) else input[-1].content
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": gen_config.max_tokens or 1,
                "temperature": gen_config.temperature or 0.0,
                "logprobs": kwargs.get('logprobs', 1),
                "echo": kwargs.get('echo', True)
            }
        else:
            # 走 /v1/chat/completions 接口 (标准对话)
            target_url = self.chat_completions_url
            request_data = {
                "model": self.model_name,
                "messages": self._convert_chat_messages(input),
                "max_tokens": gen_config.max_tokens or 128,
                "temperature": gen_config.temperature or 0.7,
                "top_p": gen_config.top_p or 0.9
            }

        try:
            response = self.session.post(target_url, json=request_data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            # 安全性检查：确保 choices 存在且不为空
            if not result.get("choices"):
                logging.error(f"API 返回异常，没有 choices 字段: {result}")
                return ModelOutput(model=self.model_name, content="", error="Empty choices from API")
            
            choice = result["choices"][0]
            
            # 兼容性解析内容
            # 补全接口使用 'text'，聊天接口使用 'message'
            content = ""
            if "text" in choice:
                content = choice["text"]
            elif "message" in choice:
                content = choice["message"].get("content", "")
            
            # 返回 ModelOutput，并透传 raw_response 供 EvalScope 计算概率
            return ModelOutput(
                model=self.model_name,
                content=content,
                raw_response=result
            )
            
        except Exception as e:
            logging.error(f"[DistributedDeepSeekAdapter] 请求异常: {str(e)}")
            return ModelOutput(model=self.model_name, content="", error=str(e))

    def batch_generate(self, inputs: List[List[ChatMessage]], 
                      tools: List[List[ToolInfo]] = None,
                      tool_choices: List[ToolChoice] = None,
                      configs: List[GenerateConfig] = None,
                      **kwargs) -> List[ModelOutput]:
        """批处理生成：并发调用API"""
        
        if tools is None:
            tools = [[] for _ in inputs]
        if tool_choices is None:
            tool_choices = [None for _ in inputs]
        if configs is None:
            configs = [self.config for _ in inputs]
        
        logging.info(f"[batch_generate] 开始批处理，样本数: {len(inputs)}")
        
        results = [None] * len(inputs)
        
        # 使用线程池并发请求
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_index = {}
            for i, (input_msgs, input_tools, tool_choice, config) in enumerate(zip(inputs, tools, tool_choices, configs)):
                future = executor.submit(self.generate, input_msgs, input_tools, tool_choice, config)
                future_to_index[future] = i
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    logging.debug(f"[batch_generate] 样本 {index} 完成")
                except Exception as e:
                    logging.error(f"[batch_generate] 样本 {index} 失败: {e}")
                    results[index] = ModelOutput.from_content(
                        model=self.model_name,
                        content="",
                        error=str(e)
                    )
        
        successful_count = sum(1 for r in results if r and not r.error)
        logging.info(f"[batch_generate] 批处理完成，成功: {successful_count}/{len(inputs)}")
        
        return results

    def _convert_chat_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """转换ChatMessage为API期望的格式"""
        converted = []
        
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                # 已经是标准格式
                converted.append({
                    "role": msg.role,
                    "content": msg.content
                })
            elif hasattr(msg, 'text'):
                # 兼容旧格式，假设是用户消息
                converted.append({
                    "role": "user", 
                    "content": msg.text
                })
            else:
                # 其他格式，转为字符串
                converted.append({
                    "role": "user",
                    "content": str(msg)
                })
        
        return converted

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'session'):
            self.session.close()