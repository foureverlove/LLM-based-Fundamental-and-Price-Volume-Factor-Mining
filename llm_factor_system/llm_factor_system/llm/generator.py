# ==================== llm/generator.py ====================
"""
LLM因子生成器
"""
import os
from typing import Dict, List, Optional, Tuple
import json
import re
from datetime import datetime

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        import openai

class LLMFactorGenerator:
    """LLM因子生成器"""
    
    def __init__(self, model: str = "gpt-4.1-mini", api_key: str = None,
                 api_base: str = None, temperature: float = 0.1, max_tokens: int = 2000):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base 
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # #region agent log
        import json
        log_path = r"c:\Users\qtx27\Desktop\llm_factor_system\.cursor\debug.log"
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"llm-init","hypothesisId":"H1","location":"generator.py:27","message":"LLM初始化参数","data":{"model":self.model,"has_api_key":bool(self.api_key),"api_key_prefix":self.api_key[:10]+"..." if self.api_key else None,"api_base":self.api_base,"langchain_available":LANGCHAIN_AVAILABLE},"timestamp":int(__import__('time').time()*1000)}) + "\n")
        except: pass
        # #endregion
        
        # 确保api_base不为None（如果使用代理API）
        if not self.api_base and self.api_key:
            # 如果没有指定base_url，根据环境变量或使用默认代理
            if os.getenv('CLOSEAI_BASE_URL'):
                self.api_base = os.getenv('CLOSEAI_BASE_URL')
            elif os.getenv('OPENAI_API_BASE'):
                self.api_base = os.getenv('OPENAI_API_BASE')
        
        # 初始化客户端（优先使用LangChain，兼容OpenAI SDK）
        if LANGCHAIN_AVAILABLE and self.api_key:
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"llm-init","hypothesisId":"H1","location":"generator.py:51","message":"使用LangChain初始化","data":{"api_base":self.api_base,"model":self.model},"timestamp":int(__import__('time').time()*1000)}) + "\n")
            except: pass
            # #endregion
            
            # LangChain ChatOpenAI 初始化
            # 根据框架总结，应该使用 openai_api_base 参数
            llm_kwargs = {
                "model": self.model,
                "temperature": self.temperature,
                "openai_api_key": self.api_key,
                "max_tokens": self.max_tokens
            }
            
            # 添加 base_url - LangChain 使用 openai_api_base 参数
            if self.api_base:
                llm_kwargs["openai_api_base"] = self.api_base
            
            self.llm = ChatOpenAI(**llm_kwargs)
            self.use_langchain = True
            
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"llm-init","hypothesisId":"H1","location":"generator.py:75","message":"LangChain初始化完成","data":{"used_kwargs":list(llm_kwargs.keys())},"timestamp":int(__import__('time').time()*1000)}) + "\n")
            except: pass
            # #endregion
        else:
            # 降级使用OpenAI SDK（确保导入 openai）
            import openai
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"llm-init","hypothesisId":"H1","location":"generator.py:60","message":"使用OpenAI SDK初始化","data":{"api_base":self.api_base,"fallback_base":("https://api.deepseek.com" if "deepseek" in self.model else None)},"timestamp":int(__import__('time').time()*1000)}) + "\n")
            except: pass
            # #endregion
            # 确保base_url正确设置
            final_base_url = self.api_base or ("https://api.deepseek.com" if "deepseek" in self.model else None)
            # 兼容不同 openai 客户端（openai-python v3+ 提供 OpenAI 类）
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=final_base_url
                )
            except Exception:
                # 兜底：旧版 openai 包通过 api_key 和 api_base 直接设置
                openai.api_key = self.api_key
                if final_base_url:
                    openai.api_base = final_base_url
                self.client = openai
            self.use_langchain = False
    
    def generate_price_factor(self, available_fields: List[str],
                            existing_factors: List[str] = None,
                            rag_context: List[str] = None) -> Tuple[str, str]:
        """
        生成量价因子
        
        Args:
            available_fields: 可用字段列表
            existing_factors: 现有因子表达式列表（用于避免重复）
            rag_context: RAG检索的上下文
            
        Returns:
            (因子表达式, 解释说明)
        """
        prompt = self._build_price_factor_prompt(available_fields, 
                                               existing_factors, rag_context)
        
        content = self._call_llm(prompt)
        return self._parse_factor_response(content)
    
    def generate_fundamental_factor(self, available_fields: List[str],
                                  operator_types: Dict[str, List[str]],
                                  existing_factors: List[str] = None) -> Tuple[str, str]:
        """生成基本面因子"""
        prompt = self._build_fundamental_factor_prompt(available_fields, 
                                                      operator_types, existing_factors)
        
        content = self._call_llm(prompt)
        return self._parse_factor_response(content)
    
    def improve_factor(self, original_expr: str, original_metrics: Dict,
                      available_fields: List[str], 
                      improvement_idea: str = None) -> Tuple[str, str]:
        """改进现有因子"""
        prompt = self._build_improvement_prompt(original_expr, original_metrics,
                                              available_fields, improvement_idea)
        
        content = self._call_llm(prompt)
        return self._parse_factor_response(content)
    
    def extract_idea(self, expr: str, performance: Dict) -> str:
        """从因子中提取改进想法"""
        prompt = self._build_idea_extraction_prompt(expr, performance)
        
        content = self._call_llm(prompt, temperature=0.3, max_tokens=500)
        return content
    
    def _call_llm(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """调用LLM API"""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if getattr(self, 'use_langchain', False):
            # 多版本兼容：优先使用消息接口
            try:
                from langchain.schema import HumanMessage
                # 使用 invoke 替代 __call__，避免 LangChainDeprecationWarning
                try:
                    resp = self.llm.invoke([HumanMessage(content=prompt)])
                except Exception:
                    # 部分旧版/不同实现使用 generate/predict
                    try:
                        resp = self.llm.predict(prompt)
                    except Exception:
                        try:
                            resp = self.llm(prompt)
                        except Exception as e:
                            return str(e)
            except Exception:
                # 如果无法导入 HumanMessage，退回到通用调用
                try:
                    resp = self.llm.invoke(prompt)
                except Exception:
                    try:
                        resp = self.llm.predict(prompt)
                    except Exception:
                        try:
                            resp = self.llm(prompt)
                        except Exception as e:
                            return str(e)

            # 解析不同返回形态
            if isinstance(resp, str):
                return resp
            if hasattr(resp, 'content'):
                return resp.content
            if hasattr(resp, 'generations'):
                gens = resp.generations
                if gens and len(gens) > 0:
                    first = gens[0]
                    # generations 可能是 list[list[Generation]] 或 list[Generation]
                    if isinstance(first, list) and len(first) > 0 and hasattr(first[0], 'text'):
                        return first[0].text
                    if hasattr(first, 'text'):
                        return first.text
            # 最后兜底
            return str(resp)
        else:
            # OpenAI SDK 调用（兼容 openai-python v3+ 或 旧版）
            try:
                # v3 客户端风格
                if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temp,
                        max_tokens=tokens
                    )
                else:
                    # 旧版 openai.Completion/create/chat completion
                    response = self.client.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temp,
                        max_tokens=tokens
                    )
            except Exception as e:
                return str(e)

            # 解析返回
            try:
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        return choice.message.content
                    if hasattr(choice, 'text'):
                        return choice.text
                # 兼容直接字典返回
                if isinstance(response, dict):
                    choices = response.get('choices')
                    if choices and len(choices) > 0:
                        ch = choices[0]
                        if 'message' in ch and 'content' in ch['message']:
                            return ch['message']['content']
                        if 'text' in ch:
                            return ch['text']
            except Exception:
                pass
            return str(response)
    
    def _build_price_factor_prompt(self, available_fields: List[str],
                                 existing_factors: List[str],
                                 rag_context: List[str]) -> str:
        """构建量价因子生成提示"""
        
        rag_section = ""
        if rag_context:
            rag_section = f"""
        ### 2. 参考因子模式
        以下是一些成功的因子结构，供参考但不要直接复制：
        {chr(10).join(f"- {factor}" for factor in rag_context[:3])}
        """
                
        existing_section = ""
        if existing_factors:
            existing_section = f"""

        ### 3. 避免重复
        请注意避免与以下已有因子过于相似：
        {chr(10).join(f"- {factor}" for factor in existing_factors[:5])}
        """
        
        prompt = f"""
        你是一个专业的量化研究员，精通股票数据的处理和因子构建。请基于以下信息生成一个新的量价因子：

        ### 1. 输入内容
        可用字段：[{', '.join(available_fields)}]
        可用算子：Slope, Mean, Std, EMA, Corr, Max, Min, Ref, Delta, Cov, Var

        {rag_section}
        {existing_section}

        ### 4. 因子表达式要求
        1. 必须保持量纲一致性，确保公式左右两侧单位匹配
        2. 必须使用以上提供的字段和算子
        3. 表达式必须可执行，使用正确的函数调用格式
        4. 避免使用未来函数

        ### 5. 输出格式（必须严格遵循）
        expression: <因子的Python表达式，使用上述算子>
        explanation: <用最简单的例子解释这个因子的运行原理，让非专业人士也能理解，100字以内>

        现在请生成一个新的量价因子：
        """
        
        return prompt
    
    def _build_fundamental_factor_prompt(self, available_fields: List[str],
                                       operator_types: Dict[str, List[str]],
                                       existing_factors: List[str]) -> str:
        """构建基本面因子生成提示"""
        
        # 构建算子描述
        operator_desc = []
        for op_type, ops in operator_types.items():
            operator_desc.append(f"{op_type}算子: {', '.join(ops)}")
        
        existing_section = ""
        if existing_factors:
            existing_section = f"""
        ### 3. 避免重复
        请注意避免与以下已有因子过于相似：
        {chr(10).join(f"- {factor}" for factor in existing_factors[:5])}
        """
        
        prompt = f"""
        你是一个专业的量化研究员，精通基本面数据处理和财务因子构建。请基于以下内容进行因子构建，并对因子进行解释：

        ### 1. 输入内容
        可用字段（基本面数据）：[{', '.join(available_fields)}]
        注意：基本面数据为季度频率，需要通过时间对齐匹配避免未来数据。

        ### 2. 可用算子库
        {chr(10).join(operator_desc)}

        {existing_section}

        ### 4. 因子表达式要求
        1. 必须使用以上提供的字段和算子
        2. 表达式必须有经济学含义，反映公司基本面变化
        3. 保持量纲一致性，确保公式具备合理的金融逻辑
        4. 避免使用未来函数

        ### 5. 因子表示形式
        输出一个数学表达式，可包含括号、运算符和函数调用

        ### 6. 输出格式（必须严格遵循）
        expression: <因子的Python表达式>
        explanation: <用最简单的例子解释这个因子的运行原理和经济学含义，让非专业人士也能理解，100字以内>

        请生成一个新的基本面因子：
        """
        
        return prompt
    
    def _build_improvement_prompt(self, original_expr: str, original_metrics: Dict,
                                available_fields: List[str], 
                                improvement_idea: str = None) -> str:
        """构建因子改进提示"""
        
        idea_section = ""
        if improvement_idea:
            idea_section = f"""
        ### 3. 改进思路参考
        {improvement_idea}
        """
                
        prompt = f"""
        你是一个经验丰富的量化策略优化专家，请对以下因子进行优化，提高其IC、ICIR、多空收益等指标：

        ### 1. 现有因子信息
        因子表达式: {original_expr}
        Rank IC: {original_metrics.get('rank_ic', 'N/A')}
        ICIR: {original_metrics.get('icir', 'N/A')}
        多头超额收益率: {original_metrics.get('long_excess_return', 'N/A')}
        多头夏普比率: {original_metrics.get('long_sharpe', 'N/A')}
        多头最大回撤: {original_metrics.get('long_max_drawdown', 'N/A')}

        ### 2. 可用资源
        可用字段: [{', '.join(available_fields)}]
        可用算子: Slope, Mean, Std, EMA, Corr, Max, Min, Ref, Delta, Cov, Var

        {idea_section}

        ### 4. 优化要求
        1. 只能基于已有字段和算子进行优化，确保方法正确
        2. 可借鉴成功因子结构，但不能直接复制
        3. 保持量纲一致性，确保优化后的表达式具备金融逻辑

        ### 5. 分析要求（请按步骤思考）
        请按照以下步骤进行分析：
        1. Optimization Strategy：阐述优化方向的理论依据
        2. Alpha Idea：解释优化后因子为何能提升收益
        3. Factor Interpretation：详细解释表达式中每个部分的金融含义

        ### 6. 输出格式（必须严格遵循）
        expression: <优化后的因子表达式>
        explanation: <解释因子原理，100字以内>

        请对上述因子进行优化：
        """
        
        return prompt
    
    def _build_idea_extraction_prompt(self, expr: str, performance: Dict) -> str:
        """构建想法提取提示"""
        prompt = f"""
        你是一个量化策略研究员，请从以下因子表现中提取可复用的改进思路（idea）：

        ### 1. 因子信息
        因子表达式：{expr}
        表现指标：
        - IC均值: {performance.get('ic_mean', 'N/A')}
        - 风险调整IC: {performance.get('icir', 'N/A')}
        - 多头超额收益率: {performance.get('long_excess_return', 'N/A')}
        - 多头夏普比率: {performance.get('long_sharpe', 'N/A')}

        ### 2. 提取要求
        1. 分析因子失效或有效的市场环境
        2. 提取可迁移的优化逻辑（如结构、参数、组合方式）
        3. 避免过度拟合，保持逻辑泛化性

        ### 3. 输出格式
        核心思路: <提炼出的核心思想>
        适用场景: <该思路适用的市场环境>
        潜在改进: <可能提升的指标类型>

        请提取改进思路：
        """
        
        return prompt
    
    def _parse_factor_response(self, content: str) -> Tuple[str, str]:
        """解析LLM响应"""
        expr_match = re.search(r'expression:\s*(.+)', content, re.IGNORECASE)
        exp_match = re.search(r'explanation:\s*(.+)', content, re.IGNORECASE)
        
        if expr_match and exp_match:
            expr = expr_match.group(1).strip()
            explanation = exp_match.group(1).strip()
            
            # 清理表达式
            expr = expr.replace('```python', '').replace('```', '').strip()
            
            return expr, explanation
        
        # 如果没匹配到格式，尝试提取表达式
        lines = content.strip().split('\n')
        for line in lines:
            if '(' in line and ')' in line and any(op in line for op in 
                                                  ['Slope', 'Mean', 'Std', 'EMA', 'Corr']):
                return line.strip(), content[:100] + "..."
        
        # 默认返回
        return "Mean(Close, 20)", "默认因子：20日均线"