# llm_adapters.py
# -*- coding: utf-8 -*-
import logging
from typing import Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import google.generativeai as genai
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from openai import OpenAI
import requests
import os
import socket
import time


def check_base_url(url: str) -> str:
    """
    处理base_url的规则：
    1. 如果url以#结尾，则移除#并直接使用用户提供的url
    2. 否则检查是否需要添加/v1后缀
    """
    import re
    url = url.strip()
    if not url:
        return url
        
    if url.endswith('#'):
        return url.rstrip('#')
        
    if not re.search(r'/v\d+$', url):
        if '/v1' not in url:
            url = url.rstrip('/') + '/v1'
    return url

def detect_and_setup_proxy():
    """
    检测并设置代理配置，主要针对Clash等代理工具
    """
    # 常见的代理端口配置
    proxy_configs = [
        {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"},  # Clash for Windows 默认HTTP代理
        {"http": "socks5://127.0.0.1:7891", "https": "socks5://127.0.0.1:7891"},  # Clash for Windows 默认SOCKS5代理
        {"http": "http://127.0.0.1:8080", "https": "http://127.0.0.1:8080"},  # 其他常见HTTP代理
        {"http": "http://127.0.0.1:1080", "https": "http://127.0.0.1:1080"},  # 其他常见代理
    ]
    
    # 检查系统环境变量中是否已经设置了代理
    if os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY'):
        logging.info("系统环境变量中已配置代理")
        return True
    
    # 尝试检测可用的代理
    for proxy_config in proxy_configs:
        try:
            # 解析代理地址和端口
            import re
            http_proxy = proxy_config["http"]
            if "://" in http_proxy:
                proxy_url = http_proxy.split("://")[1]
            else:
                proxy_url = http_proxy
            
            if ":" in proxy_url:
                host, port = proxy_url.split(":")
                port = int(port)
                
                # 检查端口是否开放
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    # 端口开放，设置代理
                    os.environ['HTTP_PROXY'] = proxy_config["http"]
                    os.environ['HTTPS_PROXY'] = proxy_config["https"]
                    os.environ['http_proxy'] = proxy_config["http"]  # 小写版本，某些库需要
                    os.environ['https_proxy'] = proxy_config["https"]
                    logging.info(f"成功检测并设置代理: {proxy_config['http']}")
                    return True
                    
        except Exception as e:
            logging.debug(f"检测代理 {proxy_config} 失败: {e}")
            continue
    
    logging.warning("未检测到可用的代理配置，可能需要手动设置")
    return False

def test_google_connectivity():
    """
    测试能否访问Google服务
    """
    try:
        import urllib.request
        # 尝试访问Google AI的endpoint
        response = urllib.request.urlopen('https://generativelanguage.googleapis.com', timeout=10)
        return response.getcode() == 200
    except Exception as e:
        logging.debug(f"Google连接测试失败: {e}")
        return False

class BaseLLMAdapter:
    """
    统一的 LLM 接口基类，为不同后端（OpenAI、Ollama、ML Studio、Gemini等）提供一致的方法签名。
    """
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement .invoke(prompt) method.")

class DeepSeekAdapter(BaseLLMAdapter):
    """
    适配官方/OpenAI兼容接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from DeepSeekAdapter.")
            return ""
        return response.content

class OpenAIAdapter(BaseLLMAdapter):
    """
    适配官方/OpenAI兼容接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from OpenAIAdapter.")
            return ""
        return response.content

class GeminiAdapter(BaseLLMAdapter):
    """
    适配 Google Gemini (Google Generative AI) 接口
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # 检测并设置代理（针对中国地区网络访问问题）
        try:
            # 首先测试直连
            if not test_google_connectivity():
                logging.info("直连Google服务失败，尝试检测并设置代理...")
                proxy_detected = detect_and_setup_proxy()
                if proxy_detected:
                    # 等待一下让代理设置生效
                    time.sleep(1)
                    if test_google_connectivity():
                        logging.info("✅ 代理设置成功，可以访问Google服务")
                    else:
                        logging.warning("⚠️ 代理已设置但仍无法访问Google服务，请检查代理配置")
                else:
                    logging.warning("⚠️ 未能自动检测到代理，如果在中国地区使用，请确保VPN/代理正常运行")
            else:
                logging.info("✅ 可以直接访问Google服务")
        except Exception as e:
            logging.warning(f"代理检测过程出错: {e}")
        
        # 配置API密钥
        genai.configure(api_key=self.api_key)
        
        # 如果提供了base_url且不为空，则配置自定义endpoint
        if base_url and base_url.strip():
            # 设置环境变量来使用自定义endpoint
            os.environ['GOOGLE_AI_STUDIO_API_ENDPOINT'] = base_url.strip()
        
        # 创建生成模型
        self._model = genai.GenerativeModel(model_name=self.model_name)

    def invoke(self, prompt: str) -> str:
        try:
            # 创建生成配置
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            # 生成内容
            response = self._model.generate_content(
                contents=prompt,
                generation_config=generation_config,
                request_options={"timeout": self.timeout} if self.timeout else None
            )
            
            if response and response.text:
                return response.text
            else:
                logging.warning("No text response from Gemini API.")
                return ""
                
        except Exception as e:
            error_msg = str(e)
            
            # 检查是否是网络连接问题
            if any(keyword in error_msg.lower() for keyword in 
                   ['connection', 'timeout', 'network', 'proxy', 'ssl', 'certificate']):
                logging.error(f"Gemini API 网络连接失败: {e}")
                logging.info("💡 如果您在中国地区，请确保:")
                logging.info("   1. VPN/代理服务正常运行（如Clash for Windows）")
                logging.info("   2. 代理端口7890(HTTP)或7891(SOCKS5)可访问")
                logging.info("   3. 可以在浏览器中正常访问Google")
                
                # 尝试重新检测代理
                if not test_google_connectivity():
                    logging.info("🔄 正在重新尝试代理检测...")
                    detect_and_setup_proxy()
            else:
                logging.error(f"Gemini API 调用失败: {e}")
            
            return ""

class AzureOpenAIAdapter(BaseLLMAdapter):
    """
    适配 Azure OpenAI 接口（使用 langchain.ChatOpenAI）
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        import re
        match = re.match(r'https://(.+?)/openai/deployments/(.+?)/chat/completions\?api-version=(.+)', base_url)
        if match:
            self.azure_endpoint = f"https://{match.group(1)}"
            self.azure_deployment = match.group(2)
            self.api_version = match.group(3)
        else:
            raise ValueError("Invalid Azure OpenAI base_url format")
        
        self.api_key = api_key
        self.model_name = self.azure_deployment
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_version=self.api_version,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from AzureOpenAIAdapter.")
            return ""
        return response.content

class OllamaAdapter(BaseLLMAdapter):
    """
    Ollama 同样有一个 OpenAI-like /v1/chat 接口，可直接使用 ChatOpenAI。
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        if self.api_key == '':
            self.api_key= 'ollama'

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        response = self._client.invoke(prompt)
        if not response:
            logging.warning("No response from OllamaAdapter.")
            return ""
        return response.content

class MLStudioAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.invoke(prompt)
            if not response:
                logging.warning("No response from MLStudioAdapter.")
                return ""
            return response.content
        except Exception as e:
            logging.error(f"ML Studio API 调用超时或失败: {e}")
            return ""

class AzureAIAdapter(BaseLLMAdapter):
    """
    适配 Azure AI Inference 接口，用于访问Azure AI服务部署的模型
    使用 azure-ai-inference 库进行API调用
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        import re
        # 匹配形如 https://xxx.services.ai.azure.com/models/chat/completions?api-version=xxx 的URL
        match = re.match(r'https://(.+?)\.services\.ai\.azure\.com(?:/models)?(?:/chat/completions)?(?:\?api-version=(.+))?', base_url)
        if match:
            # endpoint需要是形如 https://xxx.services.ai.azure.com/models 的格式
            self.endpoint = f"https://{match.group(1)}.services.ai.azure.com/models"
            # 如果URL中包含api-version参数，使用它；否则使用默认值
            self.api_version = match.group(2) if match.group(2) else "2024-05-01-preview"
        else:
            raise ValueError("Invalid Azure AI base_url format. Expected format: https://<endpoint>.services.ai.azure.com/models/chat/completions?api-version=xxx")
        
        self.base_url = self.endpoint  # 存储处理后的endpoint URL
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.complete(
                messages=[
                    SystemMessage("You are a helpful assistant."),
                    UserMessage(prompt)
                ]
            )
            if response and response.choices:
                return response.choices[0].message.content
            else:
                logging.warning("No response from AzureAIAdapter.")
                return ""
        except Exception as e:
            logging.error(f"Azure AI Inference API 调用失败: {e}")
            return ""

# 火山引擎实现
class VolcanoEngineAIAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout  # 添加超时配置
        )
    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是DeepSeek，是一个 AI 人工智能助手"},
                    {"role": "user", "content": prompt},
                ],
                timeout=self.timeout  # 添加超时参数
            )
            if not response:
                logging.warning("No response from DeepSeekAdapter.")
                return ""
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"火山引擎API调用超时或失败: {e}")
            return ""

class SiliconFlowAdapter(BaseLLMAdapter):
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout  # 添加超时配置
        )
    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是DeepSeek，是一个 AI 人工智能助手"},
                    {"role": "user", "content": prompt},
                ],
                timeout=self.timeout  # 添加超时参数
            )
            if not response:
                logging.warning("No response from DeepSeekAdapter.")
                return ""
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"硅基流动API调用超时或失败: {e}")
            return ""
# grok實現
class GrokAdapter(BaseLLMAdapter):
    """
    适配 xAI Grok API
    """
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, temperature: float = 0.7, timeout: Optional[int] = 600):
        self.base_url = check_base_url(base_url)
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

    def invoke(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are Grok, created by xAI."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            if response and response.choices:
                return response.choices[0].message.content
            else:
                logging.warning("No response from GrokAdapter.")
                return ""
        except Exception as e:
            logging.error(f"Grok API 调用失败: {e}")
            return ""

def set_manual_proxy(http_proxy: str, https_proxy: str = None):
    """
    手动设置代理
    
    Args:
        http_proxy: HTTP代理地址，例如: "http://127.0.0.1:7890" 或 "socks5://127.0.0.1:7891"
        https_proxy: HTTPS代理地址，如果不提供则使用与http_proxy相同的值
    """
    if https_proxy is None:
        https_proxy = http_proxy
    
    os.environ['HTTP_PROXY'] = http_proxy
    os.environ['HTTPS_PROXY'] = https_proxy
    os.environ['http_proxy'] = http_proxy  # 小写版本
    os.environ['https_proxy'] = https_proxy
    
    logging.info(f"手动设置代理: HTTP={http_proxy}, HTTPS={https_proxy}")
    
    # 测试连接
    if test_google_connectivity():
        logging.info("✅ 代理设置成功，可以访问Google服务")
        return True
    else:
        logging.warning("⚠️ 代理设置后仍无法访问Google服务")
        return False

def clear_proxy():
    """
    清除代理设置
    """
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    for var in proxy_vars:
        if var in os.environ:
            del os.environ[var]
    logging.info("已清除代理设置")

def test_gemini_connection(api_key: str):
    """
    测试Gemini API连接
    
    Args:
        api_key: Gemini API密钥
        
    Returns:
        bool: 连接是否成功
    """
    try:
        logging.info("🧪 正在测试Gemini API连接...")
        
        # 先测试网络连接
        if not test_google_connectivity():
            logging.warning("❌ 无法访问Google服务，请检查网络连接或代理设置")
            return False
            
        # 尝试创建一个简单的API调用
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello")
        
        if response and response.text:
            logging.info("✅ Gemini API连接成功！")
            return True
        else:
            logging.warning("⚠️ Gemini API响应为空")
            return False
            
    except Exception as e:
        logging.error(f"❌ Gemini API连接失败: {e}")
        
        # 提供一些解决建议
        if "API_KEY" in str(e).upper():
            logging.info("💡 请检查API密钥是否正确")
        elif any(keyword in str(e).lower() for keyword in ['connection', 'timeout', 'network']):
            logging.info("💡 网络连接问题，建议:")
            logging.info("   1. 确保Clash等代理正常运行")
            logging.info("   2. 尝试运行: set_manual_proxy('http://127.0.0.1:7890')")
            logging.info("   3. 检查防火墙设置")
        
        return False

def create_llm_adapter(
    interface_format: str,
    base_url: str,
    model_name: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int
) -> BaseLLMAdapter:
    """
    工厂函数：根据 interface_format 返回不同的适配器实例。
    """
    fmt = interface_format.strip().lower()
    if fmt == "deepseek":
        return DeepSeekAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "openai":
        return OpenAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "azure openai":
        return AzureOpenAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "azure ai":
        return AzureAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "ollama":
        return OllamaAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "ml studio":
        return MLStudioAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "gemini":
        return GeminiAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "阿里云百炼":
        return OpenAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "火山引擎":
        return VolcanoEngineAIAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "硅基流动":
        return SiliconFlowAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    elif fmt == "grok":
        return GrokAdapter(api_key, base_url, model_name, max_tokens, temperature, timeout)
    else:
        raise ValueError(f"Unknown interface_format: {interface_format}")
