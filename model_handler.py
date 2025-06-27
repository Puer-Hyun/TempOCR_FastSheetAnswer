"""LLM API 호출을 처리하는 핸들러 모듈"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Type, NamedTuple
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig
from google.genai import types as genaitypes
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image

def setup_logger(name: str, level: int = None) -> logging.Logger:
    """통합 로거 설정"""
    logger = logging.getLogger(name)
    logger.handlers = []  # 기존 핸들러 제거

    # 로그 레벨 설정
    if level is None:
        # 환경변수에서 로그 레벨 읽기
        log_level_str = os.getenv("MODEL_HANDLER_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, log_level_str, logging.INFO)

    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger

def calculate_image_tokens(img_base64: str) -> int:
    """이미지의 토큰 수를 계산"""
    try:
        # base64 이미지를 PIL Image로 변환
        img_data = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_data))
        width, height = img.size
        
        # 이미지 크기에 따른 토큰 계산
        if width <= 384 and height <= 384:
            # 384x384 이하의 이미지는 258 토큰
            return 258
        else:
            # 더 큰 이미지는 768x768 타일로 나누어짐
            tiles_x = (width + 767) // 768  # 올림 나눗셈
            tiles_y = (height + 767) // 768
            total_tiles = tiles_x * tiles_y
            return total_tiles * 258
    except Exception as e:
        logging.error(f"이미지 토큰 계산 실패: {str(e)}")
        return 258  # 에러 발생 시 기본값 반환

class ModelResponse(NamedTuple):
    content: Any
    usage_metadata: Dict
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_hit: bool = False
    processing_time: float = 0.0

class ModelHandler:
    """LLM API 호출을 처리하는 핸들러"""

    def __init__(self):
        load_dotenv()
        
        self.project_id = "postmath"
        self.location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        
        # 비동기 클라이언트 초기화
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        ).aio
        
        self.logger = setup_logger("model_handler")
        
        # 일반 클라이언트도 초기화
        self.regular_client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )
        self.active_chats = {}

    async def count_tokens(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash-lite"
    ) -> Dict:
        """프롬프트의 토큰 수를 계산"""
        try:
            token_count = await self.client.models.count_tokens(
                model=model,
                contents=prompt
            )
            return {
                "total_tokens": token_count.total_tokens,
                "prompt_tokens": token_count.prompt_token_count,
                "candidates_tokens": token_count.candidates_token_count
            }
        except Exception as e:
            self.logger.error(f"토큰 카운팅 실패: {str(e)}")
            raise

    async def count_tokens_with_images(
        self,
        prompt: str,
        images: List[str],
        model: str = "gemini-2.0-flash-lite"
    ) -> Dict:
        """이미지와 텍스트를 포함한 프롬프트의 토큰 수를 계산"""
        try:
            # 멀티모달 입력 구성
            contents = []
            contents.append({"text": prompt})
            
            for img_base64 in images:
                contents.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_base64
                    }
                })
            
            # 토큰 수 계산
            token_count = await self.client.models.count_tokens(
                model=model,
                contents=contents
            )
            
            return {
                "total_tokens": token_count.total_tokens,
                "prompt_tokens": token_count.prompt_token_count,
                "candidates_tokens": token_count.candidates_token_count
            }
        except Exception as e:
            self.logger.error(f"이미지 토큰 카운팅 실패: {str(e)}")
            raise

    async def generate_response(
        self,
        prompt: str,
        system_instruction: str = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        stream: bool = False,
        response_model: Type[BaseModel] = None,
    ) -> ModelResponse:
        """응답 생성"""
        try:
            import time
            start_time = time.time()
            
            # 생성 설정
            config = GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
                response_mime_type="application/json" if response_model else None,
                response_schema=response_model if response_model else None
            )
            
            model = "gemini-2.0-flash-lite"
            
            if stream:
                # 스트리밍 응답 처리
                full_response = ""
                async for chunk in self.client.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config=config
                ):
                    if chunk.text:
                        full_response += chunk.text
                        print(chunk.text, end="", flush=True)
                processing_time = time.time() - start_time
                return ModelResponse(
                    content=full_response,
                    usage_metadata=chunk.usage_metadata,
                    model=model,
                    input_tokens=chunk.usage_metadata.prompt_token_count,
                    output_tokens=chunk.usage_metadata.candidates_token_count,
                    total_tokens=chunk.usage_metadata.total_token_count,
                    processing_time=processing_time
                )
            else:
                # 일반 응답 처리
                response = await self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )
                
                processing_time = time.time() - start_time
                
                if response_model:
                    try:
                        text = response.text
                        if not text:
                            self.logger.error("Empty response text received from model")
                            raise ValueError("모델에서 빈 응답을 받았습니다.")
                            
                        # JSON 형식이 아닌 경우를 위한 처리
                        if not (text.strip().startswith('{') and text.strip().endswith('}')):
                            self.logger.error(f"Invalid JSON format in response: {text[:100]}")
                            raise ValueError("응답이 올바른 JSON 형식이 아닙니다.")
                            
                        # JSON 파싱 시도
                        try:
                            json_data = json.loads(text)
                        except json.JSONDecodeError:
                            # 마지막 중괄호를 찾아서 자르기 시도
                            last_brace = text.rfind('}')
                            if last_brace > text.rfind('{'):
                                text = text[:last_brace+1]
                                try:
                                    json_data = json.loads(text)
                                except json.JSONDecodeError as e:
                                    self.logger.error(f"JSON parsing error after truncation: {str(e)}")
                                    self.logger.error(f"Truncated text: {text}")
                                    raise
                            else:
                                raise
                                
                        content = response_model.model_validate(json_data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON parsing error: {str(e)}")
                        self.logger.error(f"Raw response text: {text}")
                        raise
                    except Exception as e:
                        self.logger.error(f"Validation error: {str(e)}")
                        raise
                else:
                    content = response.text if response.text else ""
                
                return ModelResponse(
                    content=content,
                    usage_metadata=response.usage_metadata,
                    model=model,
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count,
                    processing_time=processing_time
                )
            
        except Exception as e:
            self.logger.error(f"Error in generate_response: {str(e)}")
            self.logger.error(f"Full prompt: {prompt}")
            raise

    def _is_response_complete(self, text: str) -> bool:
        """응답이 완성되었는지 확인"""
        # JSON 응답의 경우
        if text.strip().endswith('}'):
            try:
                json.loads(text)
                return True
            except json.JSONDecodeError:
                return False
        
        # 일반 텍스트 응답의 경우
        # 문장이 완성된 것처럼 보이는 경우 (마침표, 물음표, 느낌표로 끝나는 경우)
        if text.strip().endswith(('.', '?', '!')):
            return True
            
        return False

    async def create_chat(self, chat_id: str, model: str = "gemini-2.0-flash") -> None:
        """새로운 채팅 세션 생성"""
        try:
            self.active_chats[chat_id] = self.regular_client.chats.create(model=model)
            self.logger.debug(f"Created new chat session with ID: {chat_id}")
        except Exception as e:
            self.logger.error(f"Error creating chat session: {str(e)}")
            raise

    async def send_chat_message(
        self,
        chat_id: str,
        message: str,
        stream: bool = False
    ) -> str:
        """채팅 메시지 전송"""
        try:
            if chat_id not in self.active_chats:
                raise ValueError(f"Chat session with ID {chat_id} not found")
            
            chat = self.active_chats[chat_id]
            
            if stream:
                response_text = ""
                for chunk in chat.send_message_stream(message):
                    if chunk.text:
                        response_text += chunk.text
                        print(chunk.text, end="", flush=True)
                return response_text
            else:
                response = chat.send_message(message)
                return response.text
                
        except Exception as e:
            self.logger.error(f"Error sending chat message: {str(e)}")
            raise

    def get_chat_history(self, chat_id: str) -> List[Dict[str, str]]:
        """채팅 히스토리 조회"""
        try:
            if chat_id not in self.active_chats:
                raise ValueError(f"Chat session with ID {chat_id} not found")
            
            chat = self.active_chats[chat_id]
            history = []
            
            for message in chat._curated_history:
                history.append({
                    "role": message.role,
                    "content": message.parts[0].text
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error retrieving chat history: {str(e)}")
            raise

    async def generate_response_with_images(
        self,
        prompt: str,
        images: List[str],
        system_instruction: str = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        stream: bool = False,
        response_model: Type[BaseModel] = None,
        model: str = "gemini-2.0-flash-lite",
    ) -> ModelResponse:
        """이미지와 텍스트를 함께 처리하여 응답 생성"""
        try:
            import time
            start_time = time.time()
            
            config = GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
                response_mime_type="application/json" if response_model else None,
                response_schema=response_model if response_model else None
            )
            
            self.logger.debug(f"Starting async generation with config: {config}")
            self.logger.debug(f"Model: {model}")
            self.logger.debug(f"Number of images: {len(images)}")
            self.logger.debug(f"Prompt length: {len(prompt)}")
            
            # 멀티모달 입력 구성
            contents = []
            contents.append({"text": prompt})
            
            for img_base64 in images:
                contents.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_base64
                    }
                })
            
            if stream:
                full_response = ""
                async for chunk in self.client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config
                ):
                    if chunk.text:
                        full_response += chunk.text
                        print(chunk.text, end="", flush=True)
                
                processing_time = time.time() - start_time
                
                return ModelResponse(
                    content=full_response,
                    usage_metadata=chunk.usage_metadata,
                    model=model,
                    input_tokens=chunk.usage_metadata.prompt_token_count,
                    output_tokens=chunk.usage_metadata.candidates_token_count,
                    total_tokens=chunk.usage_metadata.total_token_count,
                    processing_time=processing_time
                )
            else:
                self.logger.debug("Sending request to model...")
                response = await self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                
                processing_time = time.time() - start_time
                self.logger.debug(f"Received response from model. Processing time: {processing_time:.2f}s")
                self.logger.debug(f"Response type: {type(response)}")
                self.logger.debug(f"Response has text: {hasattr(response, 'text')}")
                self.logger.debug(f"Response text type: {type(response.text) if hasattr(response, 'text') else 'No text attribute'}")
                
                if hasattr(response, 'text'):
                    self.logger.debug(f"Response text length: {len(response.text) if response.text else 0}")
                    if response.text:
                        self.logger.debug(f"Response text preview: {response.text[:100]}")
                
                if response_model:
                    try:
                        text = response.text
                        if not text:
                            self.logger.error("Empty response text received from model")
                            self.logger.error(f"Response object: {response}")
                            self.logger.error(f"Response attributes: {dir(response)}")
                            raise ValueError("모델에서 빈 응답을 받았습니다.")
                            
                        # JSON 형식이 아닌 경우를 위한 처리
                        if not (text.strip().startswith('{') and text.strip().endswith('}')):
                            self.logger.error(f"Invalid JSON format in response: {text[:100]}")
                            raise ValueError("응답이 올바른 JSON 형식이 아닙니다.")
                            
                        # JSON 파싱 시도
                        try:
                            json_data = json.loads(text)
                        except json.JSONDecodeError:
                            # 마지막 중괄호를 찾아서 자르기 시도
                            last_brace = text.rfind('}')
                            if last_brace > text.rfind('{'):
                                text = text[:last_brace+1]
                                try:
                                    json_data = json.loads(text)
                                except json.JSONDecodeError as e:
                                    self.logger.error(f"JSON parsing error after truncation: {str(e)}")
                                    self.logger.error(f"Truncated text: {text}")
                                    raise
                            else:
                                raise
                                
                        content = response_model.model_validate(json_data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON parsing error: {str(e)}")
                        self.logger.error(f"Raw response text: {text}")
                        raise
                    except Exception as e:
                        self.logger.error(f"Validation error: {str(e)}")
                        raise
                else:
                    content = response.text if response.text else ""
                
                return ModelResponse(
                    content=content,
                    usage_metadata=response.usage_metadata,
                    model=model,
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count,
                    processing_time=processing_time
                )
            
        except Exception as e:
            self.logger.error(f"Error in async generate_response_with_images: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            self.logger.error(f"Error details: {str(e)}")
            raise

class AsyncModelHandler:
    """비동기 LLM API 호출을 처리하는 핸들러"""

    def __init__(self):
        load_dotenv()
        
        self.project_id = "postmath"
        self.location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        
        # 비동기 클라이언트 초기화
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        ).aio
        
        self.logger = setup_logger("async_model_handler")
        
        # 일반 클라이언트도 초기화
        self.regular_client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location
        )
        self.active_chats = {}

    async def count_tokens(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash-lite"
    ) -> Dict:
        """프롬프트의 토큰 수를 계산"""
        try:
            token_count = await self.client.models.count_tokens(
                model=model,
                contents=prompt
            )
            return {
                "total_tokens": token_count.total_tokens,
                "prompt_tokens": token_count.prompt_token_count,
                "candidates_tokens": token_count.candidates_token_count
            }
        except Exception as e:
            self.logger.error(f"토큰 카운팅 실패: {str(e)}")
            raise

    async def count_tokens_with_images(
        self,
        prompt: str,
        images: List[str],
        model: str = "gemini-2.0-flash-lite"
    ) -> Dict:
        """이미지와 텍스트를 포함한 프롬프트의 토큰 수를 계산"""
        try:
            # 멀티모달 입력 구성
            contents = []
            contents.append({"text": prompt})
            
            for img_base64 in images:
                contents.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_base64
                    }
                })
            
            # 토큰 수 계산
            token_count = await self.client.models.count_tokens(
                model=model,
                contents=contents
            )
            
            return {
                "total_tokens": token_count.total_tokens,
                "prompt_tokens": token_count.prompt_token_count,
                "candidates_tokens": token_count.candidates_token_count
            }
        except Exception as e:
            self.logger.error(f"이미지 토큰 카운팅 실패: {str(e)}")
            raise

    async def generate_response(
        self,
        prompt: str,
        system_instruction: str = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        stream: bool = False,
        response_model: Type[BaseModel] = None,
    ) -> ModelResponse:
        """응답 생성"""
        try:
            import time
            start_time = time.time()
            
            # 생성 설정
            config = GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
                response_mime_type="application/json" if response_model else None,
                response_schema=response_model if response_model else None
            )
            
            model = "gemini-2.0-flash-lite"
            
            if stream:
                # 스트리밍 응답 처리
                full_response = ""
                async for chunk in self.client.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config=config
                ):
                    if chunk.text:
                        full_response += chunk.text
                        print(chunk.text, end="", flush=True)
                processing_time = time.time() - start_time
                return ModelResponse(
                    content=full_response,
                    usage_metadata=chunk.usage_metadata,
                    model=model,
                    input_tokens=chunk.usage_metadata.prompt_token_count,
                    output_tokens=chunk.usage_metadata.candidates_token_count,
                    total_tokens=chunk.usage_metadata.total_token_count,
                    processing_time=processing_time
                )
            else:
                # 일반 응답 처리
                response = await self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )
                
                processing_time = time.time() - start_time
                
                if response_model:
                    try:
                        text = response.text
                        if not text:
                            self.logger.error("Empty response text received from model")
                            raise ValueError("모델에서 빈 응답을 받았습니다.")
                            
                        # JSON 형식이 아닌 경우를 위한 처리
                        if not (text.strip().startswith('{') and text.strip().endswith('}')):
                            self.logger.error(f"Invalid JSON format in response: {text[:100]}")
                            raise ValueError("응답이 올바른 JSON 형식이 아닙니다.")
                            
                        # JSON 파싱 시도
                        try:
                            json_data = json.loads(text)
                        except json.JSONDecodeError:
                            # 마지막 중괄호를 찾아서 자르기 시도
                            last_brace = text.rfind('}')
                            if last_brace > text.rfind('{'):
                                text = text[:last_brace+1]
                                try:
                                    json_data = json.loads(text)
                                except json.JSONDecodeError as e:
                                    self.logger.error(f"JSON parsing error after truncation: {str(e)}")
                                    self.logger.error(f"Truncated text: {text}")
                                    raise
                            else:
                                raise
                                
                        content = response_model.model_validate(json_data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON parsing error: {str(e)}")
                        self.logger.error(f"Raw response text: {text}")
                        raise
                    except Exception as e:
                        self.logger.error(f"Validation error: {str(e)}")
                        raise
                else:
                    content = response.text if response.text else ""
                
                return ModelResponse(
                    content=content,
                    usage_metadata=response.usage_metadata,
                    model=model,
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count,
                    processing_time=processing_time
                )
            
        except Exception as e:
            self.logger.error(f"Error in async generate_response: {str(e)}")
            raise

    async def create_chat(self, chat_id: str, model: str = "gemini-2.0-flash") -> None:
        """새로운 채팅 세션 생성"""
        try:
            self.active_chats[chat_id] = self.regular_client.chats.create(model=model)
            self.logger.debug(f"Created new chat session with ID: {chat_id}")
        except Exception as e:
            self.logger.error(f"Error creating chat session: {str(e)}")
            raise

    async def send_chat_message(
        self,
        chat_id: str,
        message: str,
        stream: bool = False
    ) -> str:
        """채팅 메시지 전송"""
        try:
            if chat_id not in self.active_chats:
                raise ValueError(f"Chat session with ID {chat_id} not found")
            
            chat = self.active_chats[chat_id]
            
            if stream:
                response_text = ""
                for chunk in chat.send_message_stream(message):
                    if chunk.text:
                        response_text += chunk.text
                        print(chunk.text, end="", flush=True)
                return response_text
            else:
                response = chat.send_message(message)
                return response.text
                
        except Exception as e:
            self.logger.error(f"Error sending chat message: {str(e)}")
            raise

    def get_chat_history(self, chat_id: str) -> List[Dict[str, str]]:
        """채팅 히스토리 조회"""
        try:
            if chat_id not in self.active_chats:
                raise ValueError(f"Chat session with ID {chat_id} not found")
            
            chat = self.active_chats[chat_id]
            history = []
            
            for message in chat._curated_history:
                history.append({
                    "role": message.role,
                    "content": message.parts[0].text
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error retrieving chat history: {str(e)}")
            raise

    async def generate_response_with_images(
        self,
        prompt: str,
        images: List[str],
        system_instruction: str = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        stream: bool = False,
        response_model: Type[BaseModel] = None,
        model: str = "gemini-2.0-flash-lite",
    ) -> ModelResponse:
        """이미지와 텍스트를 함께 처리하여 응답 생성"""
        try:
            import time
            start_time = time.time()
            config = GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
                response_mime_type="application/json" if response_model else None,
                response_schema=response_model if response_model else None,
                # thinking_config=genaitypes.ThinkingConfig(thinking_budget=8192)
            )
            
            self.logger.debug(f"Starting async generation with config: {config}")
            self.logger.debug(f"Model: {model}")
            self.logger.debug(f"Number of images: {len(images)}")
            self.logger.debug(f"Prompt length: {len(prompt)}")
            
            # 멀티모달 입력 구성
            contents = []
            contents.append({"text": prompt})
            
            for img_base64 in images:
                contents.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_base64
                    }
                })
            
            if stream:
                full_response = ""
                async for chunk in self.client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config
                ):
                    if chunk.text:
                        full_response += chunk.text
                        print(chunk.text, end="", flush=True)
                
                processing_time = time.time() - start_time
                
                return ModelResponse(
                    content=full_response,
                    usage_metadata=chunk.usage_metadata,
                    model=model,
                    input_tokens=chunk.usage_metadata.prompt_token_count,
                    output_tokens=chunk.usage_metadata.candidates_token_count,
                    total_tokens=chunk.usage_metadata.total_token_count,
                    processing_time=processing_time
                )
            else:
                self.logger.debug("Sending request to model...")
                response = await self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                print(f"response: {response}")
                processing_time = time.time() - start_time
                self.logger.debug(f"Received response from model. Processing time: {processing_time:.2f}s")
                self.logger.debug(f"Response type: {type(response)}")
                self.logger.debug(f"Response has text: {hasattr(response, 'text')}")
                self.logger.debug(f"Response text type: {type(response.text) if hasattr(response, 'text') else 'No text attribute'}")
                
                if hasattr(response, 'text'):
                    self.logger.debug(f"Response text length: {len(response.text) if response.text else 0}")
                    if response.text:
                        self.logger.debug(f"Response text preview: {response.text[:100]}")
                
                if response_model:
                    try:
                        text = response.text
                        if not text:
                            self.logger.error("Empty response text received from model")
                            self.logger.error(f"Response object: {response}")
                            self.logger.error(f"Response attributes: {dir(response)}")
                            raise ValueError("모델에서 빈 응답을 받았습니다.")
                            
                        # JSON 형식이 아닌 경우를 위한 처리
                        if not (text.strip().startswith('{') and text.strip().endswith('}')):
                            self.logger.error(f"Invalid JSON format in response: {text[:100]}")
                            raise ValueError("응답이 올바른 JSON 형식이 아닙니다.")
                            
                        # JSON 파싱 시도
                        try:
                            json_data = json.loads(text)
                        except json.JSONDecodeError:
                            # 마지막 중괄호를 찾아서 자르기 시도
                            last_brace = text.rfind('}')
                            if last_brace > text.rfind('{'):
                                text = text[:last_brace+1]
                                try:
                                    json_data = json.loads(text)
                                except json.JSONDecodeError as e:
                                    self.logger.error(f"JSON parsing error after truncation: {str(e)}")
                                    self.logger.error(f"Truncated text: {text}")
                                    raise
                            else:
                                raise
                                
                        content = response_model.model_validate(json_data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON parsing error: {str(e)}")
                        self.logger.error(f"Raw response text: {text}")
                        raise
                    except Exception as e:
                        self.logger.error(f"Validation error: {str(e)}")
                        raise
                else:
                    content = response.text if response.text else ""
                
                return ModelResponse(
                    content=content,
                    usage_metadata=response.usage_metadata,
                    model=model,
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count,
                    processing_time=processing_time
                )
            
        except Exception as e:
            self.logger.error(f"Error in async generate_response_with_images: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            self.logger.error(f"Error details: {str(e)}")
            raise