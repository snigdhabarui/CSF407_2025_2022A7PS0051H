from google import genai
from google.genai import types
from IPython.display import HTML,Markdown,display
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API

client = genai.Client(api_key=GEMINI_API_KEY)
for m in client.models.list():
    print(m.name)