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
                 base_url: str = "http://127.0.0.1:8888",  # 默认端口8888
                 api_key: Optional[str] = None,
                 config: GenerateConfig = GenerateConfig(),
                 **kwargs) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        
        # API服务地址
        self.api_url = base_url.rstrip('/')
        self.chat_completions_url = f"{self.api_url}/v1/chat/completions"
        self.completions_url = f"{self.api_url}/v1/completions"
        
        # HTTP客户端配置
        self.session = requests.Session()
        self.timeout = kwargs.get('timeout', 300)  # 默认5分钟超时
        
        # 并发控制
        self.max_workers = kwargs.get('max_workers', 8)
        
        logging.info(f"[DistributedDeepSeekAdapter] 初始化完成，API地址: {self.api_url}")

    def generate(self, input: List[ChatMessage], 
                 tools: List[ToolInfo] = None,
                 tool_choice: ToolChoice = None,
                 config: GenerateConfig = None, 
                 **kwargs) -> ModelOutput:
        """单条生成：调用 /v1/chat/completions"""
        
        # 使用传入的config或默认config
        gen_config = config or self.config
        
        # 转换消息格式
        messages = self._convert_chat_messages(input)
        
        # 构建请求数据
        request_data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": gen_config.max_tokens or 128,
            "temperature": gen_config.temperature if gen_config.temperature is not None else 0.7,
            "top_p": gen_config.top_p if gen_config.top_p is not None else 0.9
        }
        
        # 添加其他生成参数
        if hasattr(gen_config, 'repetition_penalty') and gen_config.repetition_penalty is not None:
            request_data["repetition_penalty"] = gen_config.repetition_penalty
        
        try:
            logging.debug(f"[generate] 发送请求: {json.dumps(request_data, ensure_ascii=False)[:200]}...")
            
            response = self.session.post(
                self.chat_completions_url,
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            logging.debug(f"[generate] 收到响应: {content[:100]}...")
            
            return ModelOutput.from_content(
                model=self.model_name,
                content=content
            )
            
        except requests.exceptions.RequestException as e:
            logging.error(f"[generate] API请求失败: {e}")
            return ModelOutput.from_content(
                model=self.model_name,
                content="",
                error=str(e)
            )

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