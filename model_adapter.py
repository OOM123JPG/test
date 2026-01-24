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
        gen_config = config or self.config
        
        # 强制修正逻辑：如果任务包含 logprobs（由 EvalScope 内部传回），或者 max_tokens 为 1
        # 通常评测框架在算概率时会传入 logprobs
        is_ll_task = kwargs.get('logprobs') is not None or gen_config.max_tokens == 1
        
        if is_ll_task:
            target_url = self.completions_url
            prompt = input if isinstance(input, str) else input[-1].content
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": 1, 
                "logprobs": 1,
                "echo": True,
                "temperature": 0.0
            }
        else:
            target_url = self.chat_completions_url
            request_data = { "model": self.model_name, "messages": self._convert_chat_messages(input) }

        try:
            response = self.session.post(target_url, json=request_data, timeout=self.timeout)
            result = response.json()
            
            # --- 核心修复：防御空列表 ---
            choices = result.get("choices", [])
            if not choices:
                logging.error(f"API 返回了空 choices! 原始响应: {result}")
                return ModelOutput(model=self.model_name, content="", error="Empty choices")
            
            choice = choices[0]
            
            # 兼容性解析字段
            if "text" in choice:
                content = choice["text"]
            else:
                content = choice.get("message", {}).get("content", "")
            
            return ModelOutput(model=self.model_name, content=content, raw_response=result)
            
        except Exception as e:
            return ModelOutput(model=self.model_name, content="", error=str(e))
        
        
    def batch_generate(self, inputs: List[List[ChatMessage]], 
                    tools: List[List[ToolInfo]] = None,
                    tool_choices: List[ToolChoice] = None,
                    configs: List[GenerateConfig] = None,
                    **kwargs) -> List[ModelOutput]:
        """真正的批处理生成：一次API调用处理多个样本"""
        
        if tools is None:
            tools = [[] for _ in inputs]
        if tool_choices is None:
            tool_choices = [None for _ in inputs]
        if configs is None:
            configs = [self.config for _ in inputs]
        
        logging.info(f"[batch_generate] 开始真正的批处理，样本数: {len(inputs)}")
        
        # 检测任务类型
        first_config = configs[0]
        is_ll_batch = kwargs.get('logprobs') is not None or first_config.max_tokens == 1
        
        # 检查配置是否一致
        if not all(config.max_tokens == first_config.max_tokens and 
                config.temperature == first_config.temperature and
                config.top_p == first_config.top_p for config in configs):
            logging.warning("[batch_generate] 配置不一致，使用并发单请求模式")
            return self._fallback_batch_generate(inputs, tools, tool_choices, configs, **kwargs)
        
        try:
            if is_ll_batch:
                # ========== loglikelihood 任务：参数固定 ==========
                target_url = self.completions_url
                
                # 转换所有输入为prompts列表
                prompts = []
                for input_msgs in inputs:
                    prompt = input_msgs if isinstance(input_msgs, str) else input_msgs[-1].content
                    prompts.append(prompt)
                
                request_data = {
                    "model": self.model_name,
                    "prompt": prompts,
                    "max_tokens": 1,        # loglikelihood 固定参数
                    "logprobs": 1,          # loglikelihood 固定参数  
                    "echo": True,           # loglikelihood 固定参数
                    "temperature": 0.0      # loglikelihood 固定参数
                }
            else:
                # ========== 普通生成任务：使用用户配置 ==========
                target_url = self.completions_url  # 使用 completions 接口，支持批量
                
                # 转换所有输入为prompts列表
                prompts = []
                for input_msgs in inputs:
                    prompt = input_msgs if isinstance(input_msgs, str) else input_msgs[-1].content
                    prompts.append(prompt)
                
                request_data = {
                    "model": self.model_name,
                    "prompt": prompts,
                    "max_tokens": first_config.max_tokens,    # 使用用户配置
                    "temperature": first_config.temperature,  # 使用用户配置
                    "top_p": first_config.top_p              # 使用用户配置
                }
            
            # 发送批量请求
            response = self.session.post(target_url, json=request_data, timeout=self.timeout)
            result = response.json()
            
            # 解析批量响应
            choices = result.get("choices", [])
            results = []
            
            for i, choice in enumerate(choices):
                if "text" in choice:
                    content = choice["text"]
                else:
                    content = choice.get("message", {}).get("content", "")
                
                results.append(ModelOutput(
                    model=self.model_name,
                    content=content,
                    raw_response=result
                ))
            
            successful_count = len([r for r in results if r.content])
            logging.info(f"[batch_generate] 批量处理完成，成功: {successful_count}/{len(inputs)}")
            
            return results
            
        except Exception as e:
            logging.error(f"[batch_generate] 批量请求失败，回退到并发模式: {e}")
            return self._fallback_batch_generate(inputs, tools, tool_choices, configs, **kwargs)

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'session'):
            self.session.close()