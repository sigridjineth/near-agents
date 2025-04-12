import openai
import json
import nearai

# 1) NEAR AI base url
hub_url = "https://api.near.ai/v1"

# 2) nearai.config.load_config_file()를 통해 auth 정보 로드
auth = nearai.CONFIG.get_client_config().auth
# auth 정보를 JSON 형태로 직렬화
# signature = json.dumps(auth)

# 3) openai 패키지의 클라이언트 생성
client = openai.OpenAI(
    base_url=hub_url,
    api_key=auth
)

# 4) 모델 목록 불러오기
models = client.models.list()
print(models)

# providers(=각 모델 소유/제공자) 목록을 보려면:
providers = set([model.id.split("::")[0] for model in models])
print(providers)
