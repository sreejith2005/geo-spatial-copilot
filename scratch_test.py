import os
from dotenv import load_dotenv

load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY")
print(f"GEMINI: {gemini_key[:4] if gemini_key else None}")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    # Try initializing
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_key)
    res = llm.invoke("Reply with the word ACTIVE")
    print(res.content)
except Exception as e:
    print(f"Error: {e}")
