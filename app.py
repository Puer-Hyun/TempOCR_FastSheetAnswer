from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict
import uvicorn
from model_handler import AsyncModelHandler
import logging
import json

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
model_handler = AsyncModelHandler()

# 파일명 분석을 위한 모델
class FilenameRequest(BaseModel):
    filename: str

class FilenamesRequest(BaseModel):
    filenames: List[str]  # 최대 50개의 파일명 리스트

class FilenameAnalysis(BaseModel):
    year: Optional[int] = None
    grade: Optional[str] = None  # 고1, 고2, 고3, 중1, 중2, 중3 등
    track: Optional[str] = None  # 공통, 문과, 이과
    semester: Optional[str] = None  # 1학기, 2학기
    test_type: Optional[str] = None  # 1차(중간고사), 2차(기말고사)
    subject: Optional[str] = '수학'  # 수학
    detail_subject: Optional[str] = None  # 미적분, 기하, 확률, 통계, 공통수학 등
    school: Optional[str] = None  # 학교 이름
    location: Optional[str] = None  # 지역 (서울 서초구, 인천 연수구, 경기 오산시, 오산시 등)
    
class FilenameResponse(BaseModel):
    status: str
    analysis: FilenameAnalysis

class FilenamesResponse(BaseModel):
    status: str
    analyses: List[FilenameAnalysis]

# 이미지 분석을 위한 모델
class ImageRequest(BaseModel):
    images: List[str]  # base64 인코딩된 이미지 리스트

class ImageAnalysis(BaseModel):
    is_original: bool
    is_answer: bool
    
class CheckFilenameAnalysisRequest(BaseModel):
    """이미지와, FilenameResponse 비교를 위한 Input"""
    images: List[str]
    analysis: FilenameAnalysis
    
class CompareFilenameAnalysis(BaseModel):
    """
    이미지와, FilenameResponse 비교를 위한 Schema.
    맞음, 틀림, 알 수 없음 셋 중 하나.
    """
    year: Optional[str] = None
    grade: Optional[str] = None
    track: Optional[str] = None  # 공통, 문과, 이과
    semester: Optional[str] = None
    test_type: Optional[str] = None
    subject: Optional[str] = None
    detail_subject: Optional[str] = None
    school: Optional[str] = None
    location: Optional[str] = None
    
class CheckFilenameAnalysisResponse(BaseModel):
    """이미지와, FilenameResponse 비교를 위한 최종 Output"""
    status: str
    result: CompareFilenameAnalysis

class AnalysisResponse(BaseModel):
    status: str
    analysis: ImageAnalysis

class FieldResult(BaseModel):
    isContentMatchingFileName: bool
    value: Optional[Union[int, str]] = None

class NewCheckFilenameAnalysisResponse(BaseModel):
    status: str
    results: Dict[str, FieldResult]


class AnswerResponse(BaseModel):
    type: str
    number: int
    answer: str
    score: float
    
class ScoreResponse(BaseModel):
    status: str
    scores: List[float]  # 각 이미지별 점수 리스트
    answers: Optional[List[AnswerResponse]] = None


@app.post("/analyze/filename")
async def analyze_filename(request: FilenameRequest):
    try:
        logger.info(f"Received request to analyze filename: {request.filename}")
        
        filename_analysis_prompt = f"""
파일명 정보:
{request.filename}

파일명 분석 가이드:
- 파일명에서 학교, 학년과 학기, 시험 정보를 추출할 수 있습니다. 예를 들어:
  * "백석고 고1-2 중간" -> 백석고등학교 1학년 2학기 중간고사
  * "첨단고등학교 고2-1 기말" -> 첨단고등학교 2학년 1학기 기말고사

다음 정보를 파일명을 기반으로 추출해주세요:
1. year: 문서의 실시년도
2. grade: 문서의 학년 (고1, 고2, 고3, 중1, 중2, 중3 등)
3. track: 문서에서 나타나는 계열 정보 (공통, 문과, 이과)
4. semester: 문서의 학기 (1학기, 2학기)
5. test_type: 문서의 시험 유형 (중간고사, 기말고사)
6. subject: 문서의 과목 (수학)
7. detail_subject: 문서의 세부과목 (미적분, 기하, 확률과 통계, 공통수학, 수학(상), 수학(하), 기하와 벡터 등)
8. school: 문서의 학교 이름 (XX중학교, YY고등학교)
9. location: 지역 (서울 서초구, 인천 연수구, 경기 오산시, 오산시 등)
"""

        response = await model_handler.generate_response(
            prompt=filename_analysis_prompt,
            system_instruction="당신은 파일 분석의 전문가입니다. 파일명을 분석하여 시험지의 기본 정보를 추출해주세요. 반드시 JSON 형식으로 응답해주세요. 이때, grade는 축약어로 고1, 고2, 고3, 중1, 중2, 중3 중에서 골라야하고, test_type은 중간고사, 기말고사 중에 골라야합니다. 만약 '1차' 라는 정보가 있다면 그것은 중간고사를 의미하고, '2차'라는 정보가 있다면 그것은 기말고사를 의미합니다. school은 AA중학교 혹은 BB고등학교와 같이 축약어가 아닌 방식으로 적어주세요. 학기(semester)는 1학기, 2학기 중에 골라야합니다. 23-2-1-M 의 경우는 2023년 2학년 1학기 중간고사를 의미합니다. M은 중간고사, F는 기말고사를 의미할 수도 있습니다. track은 계열 정보를 의미합니다. 문과, 이과인지 공통인지를 의미합니다.",
            temperature=0.1,
            response_model=FilenameResponse
        )

        # 응답이 FilenameResponse 타입인 경우
        if isinstance(response.content, FilenameResponse):
            return {
                "status": "success with FilenameResponse",
                "analysis": response.content.model_dump()
            }
        # 응답이 문자열인 경우 JSON으로 파싱 시도
        elif isinstance(response.content, str):
            try:
                analysis_data = json.loads(response.content)
                return {
                    "status": "success with str...",
                    "analysis": analysis_data
                }
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"원본 응답: {response.content}")
                raise HTTPException(
                    status_code=500,
                    detail=f"JSON 파싱 오류: {str(e)}"
                )
        else:
            logger.error(f"예상치 못한 응답 타입: {type(response.content)}")
            logger.error(f"응답 내용: {response.content}")
            raise HTTPException(
                status_code=500,
                detail=f"예상치 못한 응답 타입: {type(response.content)}"
            )

    except Exception as e:
        logger.error(f"파일명 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analyze/comparefilename")
async def compare_filename_analysis(request: CheckFilenameAnalysisRequest):
    try:
        logger.info("Received request to compare filename analysis with images")
        logger.info(f"Request: {request.analysis.model_dump_json()}")
        
        # 1단계: OCR 수행
        ocr_prompt = """
시험지 이미지의 모든 텍스트를 OCR로 추출해주세요.
특히 다음 부분들을 주의 깊게 확인해주세요:

1. 시험지 상단, 하단, 사이드드 부분의 모든 텍스트
2. 시험 정보가 포함된 부분 (학교명, 학년, 시험 유형, 날짜 등)
3. 과목명이나 단원명이 표시된 부분

추출된 텍스트를 원래 위치와 함께 알려주세요. 예시:
{
    "header": "시험지 상단에서 발견된 모든 텍스트",
    "title": "시험지 제목 부분의 텍스트",
    "date": "발견된 날짜 정보",
    "school_info": "학교 관련 정보",
    "exam_info": "시험 관련 정보",
    "subject_info": "과목 관련 정보",
}"""

        # OCR 수행 (첫 번째 이미지만)
        ocr_response = await model_handler.generate_response_with_images(
            prompt=ocr_prompt,
            images=request.images,  # 모든 이미지 사용
            system_instruction="You're a Korean OCR expert. Extract all the meta information from the exam paper image accurately, especially don't miss the meta information at the top of the exam paper.",
            temperature=0.1
        )

        logger.info(f"OCR 결과: {ocr_response.content}")

        # 2단계: OCR 결과와 파일명 분석 결과 비교
        comparison_prompt = f"""
OCR로 추출된 텍스트:
{ocr_response.content}

파일명에서 추출한 정보:
- 연도: {request.analysis.year}년
- 학년: {request.analysis.grade}
- 계열: {request.analysis.track}
- 학기: {request.analysis.semester}
- 시험 유형: {request.analysis.test_type}
- 과목: {request.analysis.subject}
- 세부 과목: {request.analysis.detail_subject}
- 학교: {request.analysis.school}
- 지역: {request.analysis.location}

위의 OCR 결과와 파일명 정보를 비교하여 각 항목의 일치 여부를 판단해주세요.
특히 다음 사항들을 고려해주세요:

1. 연도 비교:
   - OCR에서 발견된 날짜/연도가 {request.analysis.year}와 일치하는지
   - 시험지 상단이나 날짜 표기에서 연도를 확인
   - 특히 페이지의 좌상단, 우상단에 '시행일'이라고 표기되어 있는 경우가 많음.

2. 학년 비교:
   - OCR에서 발견된 학년 표기가 {request.analysis.grade}와 일치하는지
   - '고1', '1학년', '1' 등 다양한 표기 방식 고려
   
3. 계열 비교:
   - OCR에서 발견된 계열 정보가 {request.analysis.track}와 일치하는지
   - 공통, 문과, 이과 중에 골라야합니다.

4. 학기 비교:
   - OCR에서 발견된 학기 정보가 {request.analysis.semester}와 일치하는지
   - '1학기', '2학기', '1', '2' 등 다양한 표기 고려

5. 시험 유형 비교:
   - OCR에서 발견된 시험 유형이 {request.analysis.test_type}와 일치하는지
   - '중간고사'는 '1차', '(1)차' 등으로 표기될 수 있음
   - '기말고사'는 '2차', '(2)차' 등으로 표기될 수 있음

6. 과목 비교:
   - OCR에서 발견된 과목명이 {request.analysis.subject}와 일치하는지
   - 제목이나 헤더에서 과목명 확인

7. 세부 과목 비교:
   - OCR에서 발견된 세부 과목명이 {request.analysis.detail_subject}와 일치하는지
   - 시험지 제목이나 단원 정보에서 확인

8. 학교명 비교:
   - OCR에서 발견된 학교명이 {request.analysis.school}와 일치하는지
   - 시험지 상단에서 학교명 확인

9. 지역 비교:
   - OCR에서 발견된 지역 정보가 {request.analysis.location}와 일치하는지
   - 학교 주소나 관련 정보에서 확인

각 항목에 대해 다음 기준으로 판단해주세요:
- "맞음": OCR 텍스트에서 해당 정보가 명확하게 확인되고 파일명의 정보와 일치
- "틀림": OCR 텍스트에서 다른 정보가 확인됨
- "알 수 없음": OCR 텍스트에서 해당 정보를 찾을 수 없음

응답은 반드시 다음 JSON 형식이어야 합니다:
{{
    "status": "success",
    "result": {{
        "year": "맞음/틀림/알 수 없음",
        "grade": "맞음/틀림/알 수 없음",
        "track": "맞음/틀림/알 수 없음",
        "semester": "맞음/틀림/알 수 없음",
        "test_type": "맞음/틀림/알 수 없음",
        "subject": "맞음/틀림/알 수 없음",
        "detail_subject": "맞음/틀림/알 수 없음",
        "school": "맞음/틀림/알 수 없음",
        "location": "맞음/틀림/알 수 없음"
    }}
}}"""

        comparison_response = await model_handler.generate_response_with_images(
            prompt=comparison_prompt,
            images=request.images,  # 모든 이미지 사용
            system_instruction="You're a text analytics expert. Compare the OCR-extracted text with the information extracted from the filename to accurately determine the match.",
            temperature=0.0,
            response_model=CheckFilenameAnalysisResponse
        )

        # 응답 처리 및 반환
        if isinstance(comparison_response.content, CheckFilenameAnalysisResponse):
            logger.info(f"비교 분석 결과: {comparison_response.content.model_dump()}")
            
            # 새로운 응답 형식으로 변환
            result = comparison_response.content.result
            new_response = {
                "status": "success",
                "results": {
                    "year": {
                        "isContentMatchingFileName": result.year == "맞음",
                        "value": request.analysis.year
                    },
                    "grade": {
                        "isContentMatchingFileName": result.grade == "맞음",
                        "value": request.analysis.grade
                    },
                    "track": {
                        "isContentMatchingFileName": result.track == "맞음",
                        "value": request.analysis.track
                    },
                    "semester": {
                        "isContentMatchingFileName": result.semester == "맞음",
                        "value": request.analysis.semester
                    },
                    "test_type": {
                        "isContentMatchingFileName": result.test_type == "맞음",
                        "value": request.analysis.test_type
                    },
                    "subject": {
                        "isContentMatchingFileName": result.subject == "맞음",
                        "value": request.analysis.subject
                    },
                    "detail_subject": {
                        "isContentMatchingFileName": result.detail_subject == "맞음",
                        "value": request.analysis.detail_subject
                    },
                    "school": {
                        "isContentMatchingFileName": result.school == "맞음",
                        "value": request.analysis.school
                    },
                    "location": {
                        "isContentMatchingFileName": result.location == "맞음",
                        "value": request.analysis.location
                    }
                }
            }
            return new_response
            
        elif isinstance(comparison_response.content, str):
            try:
                result_data = json.loads(comparison_response.content)
                logger.info(f"비교 분석 결과: {result_data}")
                
                # 문자열 응답도 새로운 형식으로 변환
                result = result_data["result"]
                new_response = {
                    "status": "success",
                    "results": {
                        "year": {
                            "isContentMatchingFileName": result["year"] == "맞음",
                            "value": request.analysis.year
                        },
                        "grade": {
                            "isContentMatchingFileName": result["grade"] == "맞음",
                            "value": request.analysis.grade
                        },
                        "track": {
                            "isContentMatchingFileName": result["track"] == "맞음",
                            "value": request.analysis.track
                        },
                        "semester": {
                            "isContentMatchingFileName": result["semester"] == "맞음",
                            "value": request.analysis.semester
                        },
                        "test_type": {
                            "isContentMatchingFileName": result["test_type"] == "맞음",
                            "value": request.analysis.test_type
                        },
                        "subject": {
                            "isContentMatchingFileName": result["subject"] == "맞음",
                            "value": request.analysis.subject
                        },
                        "detail_subject": {
                            "isContentMatchingFileName": result["detail_subject"] == "맞음",
                            "value": request.analysis.detail_subject
                        },
                        "school": {
                            "isContentMatchingFileName": result["school"] == "맞음",
                            "value": request.analysis.school
                        },
                        "location": {
                            "isContentMatchingFileName": result["location"] == "맞음",
                            "value": request.analysis.location
                        }
                    }
                }
                return new_response
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"원본 응답: {comparison_response.content}")
                raise HTTPException(
                    status_code=500,
                    detail=f"JSON 파싱 오류: {str(e)}"
                )
        else:
            logger.error(f"예상치 못한 응답 타입: {type(comparison_response.content)}")
            raise HTTPException(
                status_code=500,
                detail=f"예상치 못한 응답 타입: {type(comparison_response.content)}"
            )

    except Exception as e:
        logger.error(f"파일명 비교 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/originality")
async def analyze_originality(request: ImageRequest):
    """
    One-step originality / answer-sheet checker.
    Returns {"status": "...", "analysis": {"is_original": bool, "is_answer": bool}}
    """
    try:
        logger.info(f"Received request to analyze {len(request.images)} images")

        system_instruction = """
You are an expert in both mathematical content and image forensics.
For every image provided, carefully inspect:

1. Evidence that the problem was actually solved by a student
   – handwriting, markings, erased traces, etc.
   – OR the presence of an official school stamp / principal's seal.
     (A school seal counts as proof that the sheet is an original copy,
      even when no student scribbles are visible.)

2. Whether the image is an official answer sheet
   – ANY of the following makes it an answer sheet:
       • a table or list labelled "정답", "모범답안", "Answer Key", etc.
       • a grid of question numbers with corresponding answers
       • official explanations printed next to answers
   – Ignore the student's own writing when deciding if it is an answer sheet.

Return **one** JSON object with these two Boolean fields:

{
  "is_original": true | false,   // true if (1) handwriting OR (2) school stamp exists
  "is_answer":   true | false    // true if an official answer key / explanation exists
}

Respond **only** with valid JSON – no additional keys, comments, or text.
"""

        # single model call
        model_resp = await model_handler.generate_response_with_images(
            prompt="Analyze the images and output the JSON exactly as specified.",
            images=request.images,
            system_instruction=system_instruction,
            temperature=0.1,
            response_model=AnalysisResponse
        )

        # Accept either typed model response or raw JSON string
        if isinstance(model_resp.content, AnalysisResponse):
            analysis_data = model_resp.content.analysis
        else:
            analysis_data = json.loads(model_resp.content)["analysis"]

        # simple success envelope
        return {
            "status": "success",
            "analysis": analysis_data
        }

    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/filenames")
async def analyze_filenames(request: FilenamesRequest):
    try:
        if len(request.filenames) > 50:
            raise HTTPException(status_code=400, detail="최대 50개의 파일명만 처리할 수 있습니다.")
            
        logger.info(f"Received request to analyze {len(request.filenames)} filenames")
        
        filenames_analysis_prompt = f"""
다음 파일명들을 분석해주세요:

{chr(10).join([f"{i+1}. {filename}" for i, filename in enumerate(request.filenames)])}

파일명 분석 가이드:
- 파일명에서 학교, 학년과 학기, 시험 정보를 추출할 수 있습니다. 예를 들어:
  * "백석고 고1-2 중간" -> 백석고등학교 1학년 2학기 중간고사
  * "첨단고등학교 고2-1 기말" -> 첨단고등학교 2학년 1학기 기말고사

각 파일명에 대해 다음 정보를 추출해주세요:
1. year: 문서의 실시년도
2. grade: 문서의 학년 (고1, 고2, 고3, 중1, 중2, 중3 등)
3. track: 문서에서 나타나는 계열 정보 (공통, 문과, 이과)
4. semester: 문서의 학기 (1학기, 2학기)
5. test_type: 문서의 시험 유형 (중간고사, 기말고사)
6. subject: 문서의 과목 (수학)
7. detail_subject: 문서의 세부과목 (미적분, 기하, 확률과 통계, 공통수학, 수학(상), 수학(하), 기하와 벡터 등)
8. school: 문서의 학교 이름 (XX중학교, YY고등학교)
9. location: 지역 (서울 서초구, 인천 연수구, 경기 오산시, 오산시 등)

응답은 반드시 다음 JSON 형식이어야 합니다:
{{
    "status": "success",
    "analyses": [
        {{
            "year": 2024,
            "grade": "고1",
            "track": "공통",
            "semester": "1학기",
            "test_type": "중간고사",
            "subject": "수학",
            "detail_subject": "수학(상)",
            "school": "OO고등학교",
            "location": "서울 강남구"
        }},
        // ... 나머지 파일명들에 대한 분석 결과
    ]
}}
"""

        response = await model_handler.generate_response(
            prompt=filenames_analysis_prompt,
            system_instruction="당신은 파일 분석의 전문가입니다. 여러 파일명을 한 번에 분석하여 시험지의 기본 정보를 추출해주세요. 반드시 JSON 형식으로 응답해주세요. 이때, grade는 축약어로 고1, 고2, 고3, 중1, 중2, 중3 중에서 골라야하고, test_type은 중간고사, 기말고사 중에 골라야합니다. 만약 '1차' 라는 정보가 있다면 그것은 중간고사를 의미하고, '2차'라는 정보가 있다면 그것은 기말고사를 의미합니다. school은 AA중학교 혹은 BB고등학교와 같이 축약어가 아닌 방식으로 적어주세요. 학기(semester)는 1학기, 2학기 중에 골라야합니다. 23-2-1-M 의 경우는 2023년 2학년 1학기 중간고사를 의미합니다. M은 중간고사, F는 기말고사를 의미할 수도 있습니다. track은 계열 정보를 의미합니다. 문과, 이과인지 공통인지를 의미합니다.",
            temperature=0.1,
            response_model=FilenamesResponse
        )

        # 응답이 FilenamesResponse 타입인 경우
        if isinstance(response.content, FilenamesResponse):
            return {
                "status": "success with FilenamesResponse",
                "analyses": response.content.analyses
            }
        # 응답이 문자열인 경우 JSON으로 파싱 시도
        elif isinstance(response.content, str):
            try:
                analysis_data = json.loads(response.content)
                return {
                    "status": "success with str...",
                    "analyses": analysis_data["analyses"]
                }
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"원본 응답: {response.content}")
                raise HTTPException(
                    status_code=500,
                    detail=f"JSON 파싱 오류: {str(e)}"
                )
        else:
            logger.error(f"예상치 못한 응답 타입: {type(response.content)}")
            logger.error(f"응답 내용: {response.content}")
            raise HTTPException(
                status_code=500,
                detail=f"예상치 못한 응답 타입: {type(response.content)}"
            )

    except Exception as e:
        logger.error(f"파일명 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analyze/ocr_scores")
async def analyze_ocr_scores(request: ImageRequest):
    """
    이미지에서 수학 문제의 점수를 추출합니다.
    Returns {"status": "success", "scores": [3.7, 2.5, ...]}  # 각 이미지별 점수
    """
    try:
        logger.info(f"Received request to analyze scores from {len(request.images)} images")

        system_instruction = """
당신은 수학 시험지의 점수를 추출하는 전문가입니다.
주어진 이미지에서 다음 규칙에 따라 점수만 추출해주세요:

1. 점수 추출 규칙:
   - 0~15 사이의 숫자만 점수로 인식
   - 소수점이 있는 경우도 포함 (예: 3.7, 2.5)
   - 괄호, 중괄호 등은 무시하고 순수 숫자만 추출
   - 15보다 큰 숫자는 무시
   - 점수가 없는 경우 0으로 처리

2. 응답 형식:
   - 각 이미지별로 점수를 리스트로 반환
   - JSON 형식으로 응답
   - 예시: {"status": "success", "scores": [3.7, 2.5, 0, 1.0]}

3. 주의사항:
   - 문제 번호나 다른 숫자는 무시
   - 점수 표시가 있는 부분만 추출
   - 확실하지 않은 경우 0으로 처리
"""

        # 모델 호출
        model_resp = await model_handler.generate_response_with_images(
            prompt="이미지에서 수학 문제의 점수만 추출해주세요.",
            images=request.images,
            system_instruction=system_instruction,
            temperature=0.1,
            response_model=ScoreResponse
        )

        # 응답 처리
        if isinstance(model_resp.content, ScoreResponse):
            return {
                "status": "success",
                "scores": model_resp.content.scores
            }
        elif isinstance(model_resp.content, str):
            try:
                response_data = json.loads(model_resp.content)
                return {
                    "status": "success",
                    "scores": response_data["scores"]
                }
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"원본 응답: {model_resp.content}")
                raise HTTPException(
                    status_code=500,
                    detail=f"JSON 파싱 오류: {str(e)}"
                )
        else:
            logger.error(f"예상치 못한 응답 타입: {type(model_resp.content)}")
            raise HTTPException(
                status_code=500,
                detail=f"예상치 못한 응답 타입: {type(model_resp.content)}"
            )

    except Exception as e:
        logger.error(f"점수 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/analyze/fast_answer_sheet")
async def analyze_fast_answer_sheet(request: ImageRequest):
    """
    이미지에서 수학 문제의 점수를 추출합니다.
    Returns {"status": "success", "scores": [3.7, 2.5, ...]}  # 각 이미지별 점수
    """
    try:
        logger.info(f"Received request to analyze scores from {len(request.images)} images")
        logger.info(f"Request images count: {len(request.images)}")

        system_instruction = """
당신은 수학 시험지의 빠른 정답을 추출하는 전문가입니다.
주어진 이미지에서 다음 규칙에 따라 이 수학 문제의 정답과 배점을 추출해주세요:

1. 정답 추출 규칙:
   - 객관식, 주관식, 서술형을 구분할 것.
   - 대부분의 문제는 객관식임. "정답" 열 혹은 "정답" 행 에 있는 값들이 정답임.
   - 주관식의 경우 latex으로 정답을 추출할 것
   - 서술형의 경우 '해설참조'로 추출할 것.
   
   
2. 배점 추출 규칙
   - 0~15 사이의 숫자만 점수로 인식
   - 소수점이 있는 경우도 포함 (예: 3.7, 2.5)
   - 괄호, 중괄호 등은 무시하고 순수 숫자만 추출
   - 15보다 큰 숫자는 무시
   

3. 응답 형식:
   - 각 이미지들을 전부 참조하여 정답과 배점을 추출할 것.
   - JSON 형식으로 응답
   - 예시: {"status": "success", "answers": [{type: "객관식", number: 1, answer: "1", score: 1}, {type: "주관식", number: 2, answer: "2"}, {type: "서술형", number: 3, answer: "해설참조"}], 
   "scores": [{number: 1, score: 1.7}, {"number": 2, "score": 2.5}, {"number": 3, "score": 3.7}]}

4. 주의사항:
   - 문제 번호나 다른 숫자는 무시
   - 점수 표시가 있는 부분만 추출
   - 확실하지 않은 경우 0으로 처리
   - 여러 과목에 대한 정답표가 있는 경우 '수학'만 처리
   - 숫자만 나열된 경우 맨 위에서부터 차례대로 처리하되, 1행 = 1,2,3,4,5번, 2행=6,7,8,9,10번, 3행=11,12,13,14,15번 이러한 방식임.
"""

        logger.info("Calling model with system instruction and images")
        # 모델 호출
        model_resp = await model_handler.generate_response_with_images(
            model="gemini-2.5-pro-preview-05-06",
            prompt="이미지에서 수학 문제의 정답과 배점을 추출해주세요.",
            images=request.images,
            system_instruction=system_instruction,
            temperature=0.1,
            response_model=ScoreResponse
        )

        logger.info(f"Model response type: {type(model_resp.content)}")
        logger.info(f"Model response content: {model_resp.content}")

        # 응답 처리
        if isinstance(model_resp.content, ScoreResponse):
            logger.info("Processing ScoreResponse type response")
            return {
                "status": "success",
                "scores": model_resp.content.scores,
                "answers": model_resp.content.answers
            }
        elif isinstance(model_resp.content, str):
            logger.info("Processing string type response")
            try:
                response_data = json.loads(model_resp.content)
                logger.info(f"Parsed JSON response: {response_data}")
                return {
                    "status": "success",
                    "scores": response_data["scores"],
                    "answers": response_data["answers"]
                }
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"원본 응답: {model_resp.content}")
                logger.error(f"응답 길이: {len(model_resp.content) if model_resp.content else 0}")
                raise HTTPException(
                    status_code=500,
                    detail=f"JSON 파싱 오류: {str(e)}"
                )
        else:
            logger.error(f"예상치 못한 응답 타입: {type(model_resp.content)}")
            logger.error(f"응답 내용: {model_resp.content}")
            raise HTTPException(
                status_code=500,
                detail=f"예상치 못한 응답 타입: {type(model_resp.content)}"
            )

    except Exception as e:
        logger.error(f"점수 분석 중 오류 발생: {str(e)}")
        logger.error(f"오류 타입: {type(e)}")
        logger.error(f"오류 상세: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010) 