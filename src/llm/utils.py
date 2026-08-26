from dashscope import Generation
from http import HTTPStatus
import time
from typing import List, Optional, Iterable, Generator
from src.core.config import settings

# OpenAI 兼容模式（DashScope compatible-mode）
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None

def qwen_chat(messages: list) -> Optional[str]:
    api_key = settings.LLM_API_KEY
    
    response = Generation.call(
        api_key=api_key,
        model=settings.LLM_MODEL,
        messages=messages,
        result_format="message",
    )
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content
    else:
        raise Exception(f"Qwen API Error: {response.code} - {response.message}")

def call_api_with_retry(messages: list, max_retries=3, initial_delay=1) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            response = qwen_chat(messages)
            if response:
                return response
        except Exception as e:
            error_str = str(e)
            # If safety check fails, retrying won't help. Fail fast.
            # Check for various safety error markers - "DataInspectionFailed" is the code, "inappropriate content" is in the message
            if any(marker in error_str for marker in ["DataInspectionFailed", "inappropriate content", "semantically_inappropriate"]):
                print(f"Safety/Inspection check triggered, skipping retries.")
                return None
            print(f"Attempt {attempt+1} failed: {error_str}")
        
        if attempt < max_retries - 1:
            wait_time = initial_delay * (2 ** attempt)
            time.sleep(wait_time)
    return None


def qwen_chat_stream(messages: list) -> Iterable[str]:
    """Qwen 流式输出。

    约定：
    - yield 的每一段都是“增量文本片段”，可直接拼接。
    - 若当前 dashscope 版本不支持流式，自动降级为一次性调用再做分片。
    """
    api_key = settings.LLM_API_KEY

    # dashscope 的流式参数在不同版本/不同模型上可能不一致。
    # 这里采取“尽量启用流式 + 兼容降级”的策略。
    try:
        stream_resp = Generation.call(
            api_key=api_key,
            model=settings.LLM_MODEL,
            messages=messages,
            result_format="message",
            stream=True,
        )

        # 有些版本返回可迭代对象；有些版本可能直接返回一次性响应。
        if hasattr(stream_resp, "__iter__") and not hasattr(stream_resp, "output"):
            for chunk in stream_resp:
                # chunk 结构在不同版本中可能不同，这里做防御式解析
                try:
                    if getattr(chunk, "status_code", None) not in (None, HTTPStatus.OK):
                        raise Exception(f"Qwen Stream API Error: {chunk.code} - {chunk.message}")

                    output = getattr(chunk, "output", None)
                    if not output:
                        continue

                    choices = getattr(output, "choices", None) or []
                    if not choices:
                        continue

                    message = getattr(choices[0], "message", None)
                    if not message:
                        continue

                    content = getattr(message, "content", None)
                    if content:
                        yield content
                except Exception:
                    # 如果解析失败，忽略该 chunk，继续让上层决定是否中断
                    continue
            return

        # 若不是可迭代流对象，尝试按一次性响应处理
        if getattr(stream_resp, "status_code", None) == HTTPStatus.OK:
            text = stream_resp.output.choices[0].message.content
            for ch in text:
                yield ch
            return
        raise Exception(f"Qwen API Error: {stream_resp.code} - {stream_resp.message}")

    except TypeError:
        # 老版本 SDK 可能不支持 stream 参数
        text = qwen_chat(messages) or ""
        # 降级：模拟流式切片
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            yield text[i:i+chunk_size]


def qwen_chat_stream_openai_compatible(messages: list) -> Iterable[str]:
    """使用 DashScope OpenAI 兼容模式的流式输出（官方推荐方式）。

    - base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    - 增量文本：chunk.choices[0].delta.content
    """
    if OpenAI is None:
        raise RuntimeError("openai package is not available")

    client = OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=messages,
        stream=True,
    )

    for chunk in completion:
        # 兼容不同 openai 版本的对象结构
        try:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if not delta:
                continue
            content = getattr(delta, "content", None)
            if content:
                yield content
        except Exception:
            continue


def call_api_stream_with_retry(
    messages: list,
    max_retries: int = 3,
    initial_delay: int = 1,
) -> Generator[str, None, None]:
    """带重试的流式输出生成器。

    注意：
    - 为了避免“重试导致重复内容”，仅在第一次成功拿到片段后不再重试。
    - 如果第一次就失败，会指数退避重试。
    """
    for attempt in range(max_retries):
        yielded_any = False
        try:
            # 优先使用官方 OpenAI 兼容模式流式；失败则回退到 dashscope SDK 版本
            stream_iter: Iterable[str]
            try:
                stream_iter = qwen_chat_stream_openai_compatible(messages)
            except Exception:
                stream_iter = qwen_chat_stream(messages)

            for part in stream_iter:
                yielded_any = True
                yield part
            if yielded_any:
                return
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {str(e)}")

        if attempt < max_retries - 1:
            wait_time = initial_delay * (2 ** attempt)
            time.sleep(wait_time)


def get_zodiac_from_hour(hour: int) -> str:
    """根据小时（0-23）返回对应的生肖名称"""
    # 23-01: 子鼠 (Rat)
    # 01-03: 丑牛 (Ox)
    # 03-05: 寅虎 (Tiger)
    # 05-07: 卯兔 (Rabbit)
    # 07-09: 辰龙 (Dragon)
    # 09-11: 巳蛇 (Snake)
    # 11-13: 午马 (Horse)
    # 13-15: 未羊 (Goat)
    # 15-17: 申猴 (Monkey)
    # 17-19: 酉鸡 (Rooster)
    # 19-21: 戌狗 (Dog)
    # 21-23: 亥猪 (Pig)
    
    # 简化查表
    # 23点算作子时开始 (23, 0) -> 0 -> 鼠
    if hour >= 23 or hour < 1:
        return "鼠"
    elif 1 <= hour < 3:
        return "牛"
    elif 3 <= hour < 5:
        return "虎"
    elif 5 <= hour < 7:
        return "兔"
    elif 7 <= hour < 9:
        return "龙"
    elif 9 <= hour < 11:
        return "蛇"
    elif 11 <= hour < 13:
        return "马"
    elif 13 <= hour < 15:
        return "羊"
    elif 15 <= hour < 17:
        return "猴"
    elif 17 <= hour < 19:
        return "鸡"
    elif 19 <= hour < 21:
        return "狗"
    elif 21 <= hour < 23:
        return "猪"
    return "未知"
