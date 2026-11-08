import os
import time
import base64

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler


GEMMA_MODEL_PATH = "../src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "../src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
IMAGE_PATH = "../src/output/vision_output.jpg"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 150


chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)

llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_WINDOW,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)


print("Vision JPG 입력 대기 중...")


# 마지막 이미지 수정 시간 저장
if os.path.exists(IMAGE_PATH):
    last_modified = os.path.getmtime(IMAGE_PATH)
else:
    last_modified = 0


while True:
    # 이미지 파일 존재 확인
    if os.path.exists(IMAGE_PATH):
        current_modified = os.path.getmtime(IMAGE_PATH)

        # 이미지 파일 변경 여부 확인
        if current_modified != last_modified:
            last_modified = current_modified

            # 이미지 파일 읽기
            with open(IMAGE_PATH, "rb") as file:
                image_bytes = file.read()

            # 이미지 Base64 인코딩
            image_base64 = (base64.b64encode(image_bytes).decode("utf-8"))
            image_data = ("data:image/jpeg;base64," + image_base64)

            # 이미지와 텍스트를 Gemma에 전달
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": """
                                    Instruction:
                                    주어진 이미지를 분석하여
                                    이미지에 나타난 객체가 무엇인지 설명하시오.

                                    Constraint:
                                    이미지에서 확인할 수 있는 정보만 사용하여 답변하시오.

                                    Output Format:
                                    객체에 대한 간결한 설명.
                                   """
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "이미지에 나타난 객체가 무엇인지 설명하시오."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data
                                },
                            },
                        ],
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )

            answer = response["choices"][0]["message"]["content"].strip()

            print("\n[Gemma]")
            print(answer)


    # CPU를 불필요하게 계속 사용하지 않도록 잠시 대기
    time.sleep(0.1)