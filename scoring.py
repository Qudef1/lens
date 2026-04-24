import os
import re
import json
import time
import glob
import asyncio
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Optional, List

import rag
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize client
client = OpenAI(api_key=OPENAI_API_KEY)

# File paths
PATH = 'patents-2'
INPUT_CSV = f'{PATH}.csv' # Source file with patents
OUTPUT_CSV = f'./classified/{PATH}-classified.csv'
CASES_DIR = './cases' # Directory containing .md case files
KNOWLEDGE_BASE_DIR = './knowledge_base' # RAG knowledge base

# Interexy Profile (Base)
INTEREXY_BASE_PROFILE = """
Interexy is a premium software development company (outstaffing/custom dev).
Target Clients: Product companies (Startups, Scaleups, SMBs) in Healthcare, Fintech, Energy, AI.
Ideal Client: Has a tech product, needs senior developers (top 2%), uses modern stack (Python, JS, AI, IoT).
"""

# ======================== STEP 1: DATA PREPARATION ========================

def prepare_data():
    """Loads CSV and groups patents by applicant."""
    print(f"📂 Loading file: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False)
    except FileNotFoundError:
        print(f"❌ File {INPUT_CSV} not found.")
        return None

    df.columns = df.columns.str.strip()
    
    if 'Applicants' not in df.columns:
        print("❌ Column 'Applicants' not found.")
        return None

    required_cols = ['Applicants', 'Title', 'Abstract']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Missing columns: {required_cols}")
        return None

    df['Title'] = df['Title'].fillna('')
    df['Abstract'] = df['Abstract'].fillna('')

    def join_texts(series):
        return ' '.join(series.unique())[:5000] 

    df_grouped = df.groupby('Applicants').agg({
        'Title': join_texts,
        'Abstract': join_texts
    }).reset_index()

    print(f"📊 Grouped into {len(df_grouped)} unique applicants.")
    return df_grouped

# ======================== STEP 2: LOAD CASES ========================

def load_cases(cases_dir: str) -> str:
    """
    Loads all .md files from the cases directory and combines them into a single string.
    """
    if not os.path.exists(cases_dir):
        print(f"⚠️ Cases directory '{cases_dir}' not found. Proceeding without cases.")
        return ""

    md_files = glob.glob(os.path.join(cases_dir, "*.md"))
    if not md_files:
        print(f"⚠️ No .md files found in '{cases_dir}'.")
        return ""

    cases_text = ""
    for filepath in md_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                filename = os.path.basename(filepath)
                cases_text += f"\n=== CASE FILE: {filename} ===\n{content}\n"
        except Exception as e:
            print(f"⚠️ Error reading {filepath}: {e}")
            
    print(f"✅ Loaded {len(md_files)} case files.")
    return cases_text

# ======================== STEP 3: FAST QUALIFICATION FILTER ========================

def qualify_company_fast(company_name: str, patent_snippet: str) -> Dict:
    """
    Quick check using GPT-4o-mini (no web search) to decide if we should proceed.
    """
    prompt = f"""
    You are a B2B Sales Qualifier for Interexy (software outstaffing/dev shop).
    
    Company Name: "{company_name}"
    Patent Context: "{patent_snippet[:1000]}..." (Truncated)

    Task: Determine if this entity is a POTENTIAL CLIENT for software development services.
    
    REJECT (is_target: false) if:
    1. It is an Individual/Person.
    2. It is a University or Academic Institution.
    3. It is a Tech Giant with massive in-house R&D (e.g., NVIDIA, Google, Apple, Microsoft, etc.).
    4. It is a Government Body or Non-Profit NGO.
    
    ACCEPT (is_target: true) if:
    1. It looks like a Product Company, Startup, SME, or Mid-sized Enterprise.
    2. It operates in Tech, Healthcare, Fintech, Energy, or Industrial IoT.

    Return STRICT JSON:
    {{
        "is_target": true/false,
        "reason": "Brief reason",
        "entity_type": "Individual/University/Giant/ProductCompany/Other"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"   ⚠️ Qualification error: {e}")
        return {"is_target": False, "reason": "Error", "entity_type": "Unknown"}

# ======================== STEP 4: WEB RESEARCH & DEEP SCORING ========================

def research_and_score_company(
    company_name: str,
    patent_context: str,
    cases_text: str,
    rag_cases: Optional[List[Dict]] = None,
) -> Dict:
    """
    Performs Web Search via GPT-4o and then Scores the company using Cases.
    If rag_cases are provided, also embeds their content into the scoring prompt
    so the LLM compares against actual retrieved cases rather than generic summaries.
    """
    print(f"   🔍 Deep Dive: {company_name[:50]}...")

    rag_context = ""
    if rag_cases:
        rag_blocks = []
        for c in rag_cases:
            title = c.get("title", "Unknown")
            body = c.get("body", "")[:600].strip()
            industries = c.get("industry", [])
            if isinstance(industries, list):
                industries = ", ".join(i for i in industries if i)
            rag_blocks.append(
                f"- Case: {title}\n  Industry: {industries}\n  Content: {body}"
            )
        rag_context = "\n\n".join(rag_blocks)
        print(f"   📚 RAG: {len(rag_cases)} case(s) available for scoring.")
    
    # 1. Web Research Prompt
    research_prompt = f"""
    Act as a Business Analyst. Research the company: "{company_name}".
    
    Find:
    1. What they do (Product/Service).
    2. Industry & Tech Stack (AI, IoT, Cloud, etc.).
    3. Company Stage (Startup, Scaleup, Enterprise).
    4. Region/HQ.
    5. Do they seem to build complex software products?

    If the company is unknown, small, or seems inactive, state that.
    Return a concise summary in English.
    """

    try:
        # Web Search
        completion = client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={
                "user_location": {
                    "type": "approximate",
                    "approximate": {"country": "US", "city": "New York", "region": "New York"}
                }
            },
            messages=[{"role": "user", "content": research_prompt}]
        )
        web_info = completion.choices[0].message.content
        
        if "NO_INFO_FOUND" in web_info or "unknown" in web_info.lower():
            context_for_scoring = f"Web search yielded little info. Patent Context: {patent_context[:2000]}"
        else:
            context_for_scoring = f"Web Info: {web_info}\n\nPatent Context: {patent_context[:1000]}"

    except Exception as e:
        print(f"   ⚠️ Web Search failed: {e}. Using Patents.")
        context_for_scoring = f"Web Search Failed. Patent Context: {patent_context[:2000]}"

    # 2. Scoring Prompt WITH CASES
    score_prompt = f"""
    You are a Senior Sales Director at Interexy.

    INTEREXY PROFILE:
    {INTEREXY_BASE_PROFILE}

    INTEREXY SUCCESSFUL CASES (Reference for Ideal Clients):
    {cases_text}

{f"""    RETRIEVED CASES FROM KNOWLEDGE BASE (prioritized by relevance):
    {rag_context}
    """ if rag_context else ""}    TARGET COMPANY: {company_name}
    CONTEXT:
    ---
    {context_for_scoring}
    ---

    TASK: Score the potential for selling software development services (outstaffing/custom dev) to this company.

    INSTRUCTIONS:
    1. Analyze the Target Company's context (Web/Patents).
    2. Compare it with the Interexy Successful Cases provided above.
    3. Also compare against the RETRIEVED CASES (if any) — these are the most semantically similar to the target company.
    4. Look for similarities in:
       - Industry (e.g., Healthcare, MedTech)
       - Technology (e.g., AI, IoT, Bluetooth, Mobile Apps)
       - Problem Domain (e.g., Remote monitoring, Diagnostics, Data integration)

    SCORING RULES (1-10):
    - 1-3: NO GO. (Individual, University, Giant, Non-Tech, No similarity to cases).
    - 4-6: MAYBE. (Relevant industry but weak tech match, or unclear need for outsourcing).
    - 7-8: GOOD FIT. (Strong industry match, similar tech stack to cases, likely needs dev help).
    - 9-10: PERFECT FIT. (Very similar to one of the provided cases. E.g., another MedTech startup using IoT/AI like MedKitDoc).

    Return STRICT JSON:
    {{
        "ai_score": <number 1-10>,
        "reasoning": "<Why this score? Explicitly mention which case it resembles (if any) and what technologies match.>",
        "industry": "<Industry>",
        "tech_stack": ["<tag1>", "<tag2>"],
        "recommendation": "<One sentence on how to approach them, referencing a specific case if possible>"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
                {"role": "user", "content": score_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"   ❌ Scoring error: {e}")
        return {"ai_score": 0, "reasoning": f"Error: {str(e)}", "industry": "Unknown", "tech_stack": [], "recommendation": ""}

# ======================== STEP 5: RAG-BASED MESSAGE GENERATION ========================

def build_rag_query(company_name: str, industry: str, tech_stack: str) -> str:
    """Construct a search query for RAG retrieval combining company context."""
    parts = [company_name]
    if industry and industry not in ("Unknown", "Other"):
        parts.append(industry)
    if tech_stack:
        parts.append(tech_stack)
    return " ".join(parts)


def format_case_for_message(case: Dict) -> str:
    """Extract a concise summary of a case for inclusion in a personalized message."""
    title = case.get("title", "Unknown case")
    body = case.get("body", "")

    # Take first 300 chars of body as a teaser
    snippet = body[:300].strip()
    if len(body) > 300:
        snippet += "..."

    industries = case.get("industry", [])
    if isinstance(industries, list):
        industries = ", ".join(i for i in industries if i)
    return f"- {title}: {snippet}"


async def generate_lead_message_async(
    company_name: str,
    industry: str,
    tech_stack: str,
    ai_score: int,
    rag_cases: Optional[List[Dict]] = None,
) -> str:
    """
    Generates a personalized outreach message using pre-retrieved RAG cases.
    If rag_cases are not provided, falls back to a new RAG lookup.
    """
    # Use pre-retrieved RAG cases if available, otherwise skip retrieval
    if rag_cases is None:
        query = build_rag_query(company_name, industry, tech_stack)
        try:
            rag_cases = await rag.retrieve_cases(
                query=query,
                api_key=OPENAI_API_KEY,
                top_k=3,
                threshold=0.25,
            )
        except Exception as e:
            print(f"   ⚠️ RAG retrieval failed: {e}.")
            rag_cases = []
        if rag_cases:
            print(f"   ✅ RAG retrieved {len(rag_cases)} case(s) for '{company_name[:30]}'")
        else:
            print(f"   ⚠️ No RAG cases retrieved — generating message without case reference.")

    if rag_cases:
        cases_context = "\n\n".join(format_case_for_message(c) for c in rag_cases)
        retrieval_note = f"Relevant case(s) from our portfolio:\n{cases_context}\n\n"
    else:
        retrieval_note = ""

    # Build the message generation prompt
    score_band = "premium" if ai_score >= 8 else ("promising" if ai_score >= 6 else "speculative")
    tone = "enthusiastic" if ai_score >= 8 else "warm and professional"

    prompt = f"""You are a B2B outreach writer at Interexy — a premium software development company (outstaffing/custom dev).

Write a personalized outreach message for a potential client.

=== TARGET COMPANY ===
Name: {company_name}
Industry: {industry}
Tech Stack: {tech_stack}
Score Band: {score_band}

=== OUR SUCCESSFUL CASES ===
{retrieval_note if retrieval_note else "No highly relevant cases found — use your general knowledge of Interexy's expertise in modern tech stacks."}

=== MESSAGE REQUIREMENTS ===
1. Friendly greeting, short and direct (under 200 words total).
2. Compliment the company's work — reference their specific industry and tech achievements.
3. If you have case references from above, mention them with concrete outcomes.
   If no strong match exists, reference a relevant Interexy specialty instead.
4. End with a clear, confident offer to help bring their idea to life.
5. Tone: {tone}.

=== OUTPUT ===
Return ONLY the message text, no labels, no JSON, just the message itself."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        message = response.choices[0].message.content.strip()
        return message
    except Exception as e:
        print(f"   ❌ Message generation failed: {e}")
        return ""


def generate_lead_message(
    company_name: str,
    industry: str,
    tech_stack: str,
    ai_score: int,
    rag_cases: Optional[List[Dict]] = None,
) -> str:
    """Synchronous wrapper around the async RAG message generator."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(
        generate_lead_message_async(company_name, industry, tech_stack, ai_score, rag_cases)
    )


# ======================== MAIN LOOP ========================

def main():
    # 1. Load Cases
    cases_text = load_cases(CASES_DIR)
    
    # 2. Prepare Data
    df = prepare_data()
    if df is None:
        return

    results = []
    total = len(df)
    
    print("🚀 Starting Pipeline: Qualification -> Web Research -> Scoring with Cases...")
    
    for index, row in df.iterrows():
        company_name = row['Applicants']
        patent_text = f"Titles: {row['Title']}\nAbstracts: {row['Abstract']}"
        
        print(f"[{index+1}/{total}] Processing: {company_name[:60]}...")

        # 3. Fast Qualification
        qual_result = qualify_company_fast(company_name, patent_text)
        
        if not qual_result.get("is_target", False):
            print(f"   🚫 Rejected by Qualifier: {qual_result.get('reason')}")
            reject_message = generate_lead_message(
                company_name,
                qual_result.get("entity_type", "Unknown"),
                "",
                ai_score=1,
            )
            results.append({
                "Applicants": company_name,
                "AI-score": 1,
                "Reasoning": f"Rejected: {qual_result.get('reason')}",
                "Industry": qual_result.get("entity_type", "Unknown"),
                "Technologies": "",
                "Data_Source": "Qualifier Reject",
                "Recommendation": "Do not contact",
                "Message": reject_message,
            })
            continue

        print(f"   ✅ Passed Qualifier. Proceeding to Deep Dive...")

        # 4. RAG retrieval (used for both scoring and message)
        query_for_rag = build_rag_query(company_name, "", patent_text[:500])
        try:
            rag_cases = asyncio.run(
                rag.retrieve_cases(query=query_for_rag, api_key=OPENAI_API_KEY, top_k=3, threshold=0.20)
            )
        except Exception as e:
            print(f"   ⚠️ RAG failed: {e}")
            rag_cases = []

        if rag_cases:
            print(f"   📚 RAG found {len(rag_cases)} relevant case(s) for '{company_name[:30]}'")
        else:
            print(f"   ⚠️ No RAG matches — using fallback for '{company_name[:30]}'")

        # 5. Deep Research & Scoring (with RAG cases injected)
        score_result = research_and_score_company(company_name, patent_text, cases_text, rag_cases=rag_cases)

        tech_stack = ", ".join(score_result.get("tech_stack", []))
        industry = score_result.get("industry", "")
        ai_score = score_result.get("ai_score", 0)

        # 6. Generate personalized outreach message using the same RAG cases
        lead_message = generate_lead_message(company_name, industry, tech_stack, ai_score, rag_cases=rag_cases)

        results.append({
            "Applicants": company_name,
            "AI-score": ai_score,
            "Reasoning": score_result.get("reasoning", ""),
            "Industry": industry,
            "Technologies": tech_stack,
            "Data_Source": "Web Search + Patents + Cases",
            "Recommendation": score_result.get("recommendation", ""),
            "Message": lead_message,
        })

        # Rate limit protection
        time.sleep(2)

    # Save Results
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(by="AI-score", ascending=False)
    
    df_result.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print("="*50)
    print(f"✅ Done! Saved to {OUTPUT_CSV}")
    print("🏆 Top 5 Potential Clients:")
    print(df_result.head(5)[['Applicants', 'AI-score', 'Industry', 'Recommendation']])

if __name__ == "__main__":
    main()