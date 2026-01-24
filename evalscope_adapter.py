import torch
import logging
from typing import List, Optional, Dict, Any, Union
from evalscope.api.dataset import Dataset
from evalscope.api.model import ModelAPI, GenerateConfig, ModelOutput
from evalscope.api.messages import ChatMessage
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.utils.function_utils import thread_safe

@register_model_api(name='compressed_deepseek')
class CompressedDeepSeekAdapter(ModelAPI):
    def __init__(self, model_name: str, model_obj: Any, tokenizer: Any, 
                 base_url: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 config: GenerateConfig = GenerateConfig(),
                 **kwargs) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        self.model = model_obj
        self.tokenizer = tokenizer
        self.model.eval()
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # ← 新增：标记支持批处理
        self.support_batch_generate = True
        logging.info(f"[CompressedDeepSeekAdapter] 初始化完成，支持批处理")

    # ← 保留 generate() 作为兼容性备选
    @thread_safe
    def generate(self, input: Union[List[ChatMessage], List[List[ChatMessage]]], 
                 tools: List[ToolInfo] = None,
                 tool_choice: ToolChoice = None,
                 config: GenerateConfig = GenerateConfig(), 
                 **kwargs) -> ModelOutput:
        """单条生成（EvalScope 默认调用）"""
        logging.warning("[generate] 调用了单条接口，应该使用批处理")
        
        # 转换为单条 prompt
        if isinstance(input, list):
            if len(input) > 0 and isinstance(input[0], list):
                prompt = self._process_messages(input[0])
            elif len(input) > 0 and isinstance(input[0], ChatMessage):
                prompt = self._process_messages(input)
            else:
                prompt = input[0] if input else ""
        else:
            prompt = str(input)
        
        # 调用单条生成
        inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        input_ids = inputs.input_ids.to("cpu")
        attention_mask = inputs.attention_mask.to("cpu")
        max_tokens = config.max_tokens or self.config.max_tokens or 128
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                temperature=1.0,
            )
        
        input_len = input_ids.shape[1]
        decoded = self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return ModelOutput.from_content(
            model=self.model_name,
            content=decoded
        )
    
    @thread_safe
    def batch_generate(self, inputs: Dataset, config: GenerateConfig = GenerateConfig()) -> List[ModelOutput]:
        """
        批处理接口 - EvalScope 会优先调用此方法（如果定义了）
        """
        logging.info(f"[batch_generate] 处理数据集，样本数: {len(inputs)}")
        
        results = []
        batch_size = 4
        
        for batch_idx in range(0, len(inputs), batch_size):
            batch_data = inputs[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            logging.info(f"  >> Batch {batch_num}: 处理 {len(batch_data)} 条样本")
            
            # 提取 prompt 列表
            prompts = []
            for item in batch_data:
                if isinstance(item, dict):
                    prompt = item.get('question') or item.get('query') or str(item)
                elif isinstance(item, ChatMessage):
                    prompt = item.content
                else:
                    prompt = str(item)
                prompts.append(prompt)
            
            # 批量 tokenize
            inputs_tokens = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            input_ids = inputs_tokens.input_ids.to("cpu")
            attention_mask = inputs_tokens.attention_mask.to("cpu")
            
            # 获取生成参数
            gen_config = config or self.config
            max_tokens = gen_config.max_tokens or 128
            
            logging.info(f"    - 批量生成: max_tokens={max_tokens}, batch_size={len(prompts)}")
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.0,
                    temperature=1.0,
                )
            
            # 批量解码
            input_len = input_ids.shape[1]
            for i in range(len(output_ids)):
                decoded = self.tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True)
                logging.info(f"    >> 样本 {batch_idx + i}: {decoded[:50]}...")
                results.append(ModelOutput.from_content(
                    model=self.model_name,
                    content=decoded
                ))
        
        logging.info(f"[batch_generate] ✓ 完成，共 {len(results)} 条结果")
        return results

    def _process_messages(self, messages: Union[List[ChatMessage], ChatMessage]) -> str:
        if isinstance(messages, ChatMessage):
            return messages.content
        elif isinstance(messages, list):
            return "\n".join([m.content for m in messages])
        else:
            return str(messages)
        