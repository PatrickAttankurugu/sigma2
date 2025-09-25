#!/usr/bin/env python3
"""Quick test to verify Google Gemini API connectivity"""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_minimal_api():
    """Test with minimal API call to check connectivity"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ No API key found")
            return False
            
        print(f"🔑 Using API key: {api_key[:20]}...")
        
        # Minimal call with fast Gemini model and lowest tokens
        llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model="gemini-1.5-flash",
            temperature=0.1,
            max_output_tokens=10
        )
        
        print("🔄 Making minimal API call...")
        response = await llm.ainvoke([HumanMessage(content="Hi")])
        print(f"✅ SUCCESS! Response: {response.content}")
        print("🎉 Gemini connectivity verified!")
        return True
        
    except Exception as e:
        print(f"❌ Still getting error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_minimal_api())
