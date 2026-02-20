import os
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
google_api_key = os.getenv("GOOGLE_API_KEY")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

gemini_flash_model = OpenAIChatCompletionsModel(
    model="gemini-3-flash-preview", openai_client=gemini_client
)

gemini_pro_model = OpenAIChatCompletionsModel(
    model="gemini-3-pro-preview", openai_client=gemini_client
)
