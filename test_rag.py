import asyncio
import os
from dotenv import load_dotenv
import rag

load_dotenv()

async def main():
    query = "Healthcare startup with IoT patient monitoring platform telemedicine"
    cases = await rag.retrieve_cases(
        query=query,
        api_key=os.getenv("OPENAI_API_KEY"),
        top_k=2,
        threshold=0.45
    )
    print(f"Found {len(cases)} cases:")
    for c in cases:
        print(f"- {c['title']} [{c['industry']}]")

asyncio.run(main())