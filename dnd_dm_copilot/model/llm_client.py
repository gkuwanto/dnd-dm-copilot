import openai
import dotenv
import os

dotenv.load_dotenv()

client = openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
async_client = openai.AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

if __name__ == "__main__":
    print(client)  
    print(async_client)