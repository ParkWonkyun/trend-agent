# CSV Trend Agent

CSV를 업로드하고 자연어로 원하는 추세(trend) 차트를 요청하는 Streamlit 앱입니다.

## 로컬 실행

```bash
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY 입력
streamlit run app.py
```

## Streamlit Community Cloud 배포

1. https://share.streamlit.io 접속 후 GitHub 계정으로 로그인
2. **New app** → 이 저장소(`ParkWonkyun/trend-agent`) 선택, 브랜치 `master`, 메인 파일 `app.py` 지정
3. **Advanced settings → Secrets**에 아래 내용 입력 (API 키는 저장소에 커밋하지 않습니다)

   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"
   ```

4. **Deploy** 클릭
