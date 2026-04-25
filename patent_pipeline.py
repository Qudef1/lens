"""
Patent Lead Generation Pipeline with AI Scoring
================================================

A complete pipeline that:
1. Loads and pre-scores patent data
2. Groups patents by company
3. Provides AI-powered deep scoring with RAG
4. Outputs results to CSV

Usage:
    python patent_pipeline.py --input patents-2.csv --output results.csv
"""

import os
import re
import json
import time
import asyncio
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

import rag

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

INTEREXY_BASE_PROFILE = """
Interexy is a premium software development company (outstaffing/custom dev).
Target Clients: Product companies (Startups, Scaleups, SMBs) in Healthcare, Fintech, Energy, AI.
Ideal Client: Has a tech product, needs senior developers (top 2%), uses modern stack (Python, JS, AI, IoT).
"""

COMMON_SUFFIXES = [
    'CO', 'CO.', 'CO,', 'COMPANY', 'INC', 'INC.', 'LLC', 'LLP', 'LTD', 'LTD.',
    'PLC', 'GMBH', 'AG', 'SA', 'S.A.', 'SARL', 'BV', 'PT', 'PTY', 'SP Z O O',
    'SP ZOO', 'SAS', 'S.R.L.', 'SRL', 'CORP', 'CORP.', 'CORPORATION', 'LIMITED',
    'health', 'healthcare'
]

EXCLUDE_KEYWORDS = [
    'university', 'institute', 'college', 'academy', 'foundation',
    'hospital', 'clinic', 'health system', 'laboratory', 'research center',
    'department', 'ministry', 'agency', 'government', 'nasa', 'darpa',
    'nih', 'nist', 'fraunhofer', 'mit', 'stanford', 'harvard', 'oxford',
    'eth', 'samsung', 'philips', 'siemens', 'medtronic', 'johnson', 'abbott',
    'roche', 'baxter', 'general electric', 'schneider', 'abb', 'honeywell',
    'eaton', 'emerson', 'rockwell', 'continental', 'google', 'microsoft',
    'apple', 'amazon', 'ibm', 'intel', 'meta', 'nvidia', 'oracle', 'cisco',
    'salesforce', 'adobe', 'sap', 'qualcomm', 'huawei', 'baidu', 'alibaba',
    'tencent', 'sony', 'panasonic', 'bosch', 'visa', 'mastercard',
    'american express', 'paypal', 'jpmorgan', 'goldman', 'citibank', 'hsbc',
    'fiserv', 'stripe', 'toyota', 'ford', 'bmw', 'volkswagen', 'daimler',
    'waymo', 'tesla', 'uber', 'trimble', 'pfizer', 'novartis', 'astrazeneca',
    'illumina', 'thermo', 'palo alto networks', 'crowdstrike', 'symantec',
    'mcafee', 'fortinet', 'shell', 'bp', 'exxon', 'john deere', 'basf',
    'bayer', 'syngenta', 'monsanto', 'cargill', 'openai', 'lg electronics',
    'ericsson', 'nokia', 'bank', 'insurance', 'financial services', 'telecom',
    'netflix', 'univ', 'lenovo', 'kaspersky'
]

SECTOR_KEYWORDS = {
    'Healthcare': [
        'telehealth', 'telemedicine', 'patient monitoring', 'remote diagnosis',
        'digital health', 'health platform', 'clinical decision', 'medical imaging',
        'drug delivery', 'diagnostic', 'biosensor', 'wearable', 'genomics',
        'drug discovery', 'clinical trial', 'bioinformatics', 'precision medicine',
        'digital pathology', 'laboratory automation', 'healthcare'
    ],
    'Energy': [
        'energy management', 'smart grid', 'energy storage', 'solar monitoring',
        'wind turbine', 'battery management', 'ev charging', 'demand response',
        'microgrid', 'power optimization', 'cleantech', 'renewable energy', 'energy'
    ],
    'AI/ML': [
        'machine learning', 'deep learning', 'neural network', 'natural language processing',
        'computer vision', 'predictive analytics', 'ai platform', 'mlops',
        'model deployment', 'inference engine', 'large language model', 'artificial intelligence'
    ],
    'Fintech': [
        'payment processing', 'fraud detection', 'risk scoring', 'open banking',
        'digital wallet', 'lending platform', 'credit scoring', 'kyc', 'regtech',
        'algorithmic trading', 'financial analytics', 'neobank', 'fintech'
    ],
    'Industrial IoT': [
        'predictive maintenance', 'condition monitoring', 'industrial iot',
        'smart factory', 'asset tracking', 'digital twin', 'supply chain visibility',
        'process automation', 'quality control', 'oee', 'iot'
    ],
    'Mobility': [
        'fleet management', 'route optimization', 'autonomous vehicle',
        'mobility platform', 'traffic management', 'logistics optimization',
        'vehicle telematics', 'last-mile delivery', 'cargo tracking', 'mobility'
    ],
    'PropTech': [
        'smart building', 'building automation', 'property management',
        'facility management', 'hvac optimization', 'occupancy monitoring',
        'real estate platform', 'space management', 'proptech'
    ],
    'Cybersecurity': [
        'threat detection', 'anomaly detection', 'network security',
        'endpoint protection', 'identity management', 'zero trust',
        'vulnerability management', 'siem', 'data privacy', 'compliance automation',
        'cybersecurity'
    ],
    'AgriTech': [
        'precision agriculture', 'crop monitoring', 'irrigation management',
        'soil analysis', 'yield prediction', 'farm management',
        'livestock monitoring', 'agricultural drone', 'agritech'
    ],
}

DATE_COLUMNS = ['Application Date', 'Publication Date', 'Earliest Priority Date']


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    column_mapping = {
        'Applicant': 'Applicants', 'Assignee': 'Applicants', 'Assignees': 'Applicants',
        'Owner': 'Applicants', 'Patent Assignee': 'Applicants',
        'Patent Title': 'Title', 'Document Title': 'Title',
        'Abstract (en)': 'Abstract', 'Summary': 'Abstract',
        'Inventor': 'Inventors', 'Author': 'Inventors',
        'Kind Code': 'Document Type', 'Publication Type': 'Document Type',
        'Country': 'Jurisdiction', 'Publication Country': 'Jurisdiction', 'Country Code': 'Jurisdiction',
        'Status': 'Legal Status',
        'Filing Date': 'Application Date',
        'Priority Date': 'Earliest Priority Date',
        'Cited Patent Count': 'Cites Patent Count', 'Forward Cites': 'Cites Patent Count',
        'Lens URL': 'URL', 'Link': 'URL',
    }
    existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    return df.rename(columns=existing_mapping)


def normalize_company_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ''
    name = name.strip().upper()
    name = re.sub(r"\(.*?\)", "", name)
    name = name.replace('&', ' AND ')
    name = re.sub(r"[^A-Z0-9\s]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    for suffix in COMMON_SUFFIXES:
        pattern = rf"\b{re.escape(suffix)}\b\.?$"
        name = re.sub(pattern, '', name).strip()
    name = re.sub(r'\s+', ' ', name).strip()
    if name.startswith('THE '):
        name = name[4:]
    return name


def detect_sector(title: str, abstract: str) -> str:
    text = f"{title or ''} {abstract or ''}".lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return sector
    return 'Other'


def is_excluded(company: str) -> bool:
    name = (company or '').lower()
    return any(keyword in name for keyword in EXCLUDE_KEYWORDS)


def compute_prescore(row: pd.Series) -> int:
    score = 0
    n = int(row.get('patent_count', 0))
    if n == 1:
        score += 30
    elif n <= 3:
        score += 25
    elif n <= 10:
        score += 18
    elif n <= 30:
        score += 8

    days = row.get('days_since_filing')
    if pd.notna(days):
        if days <= 180:
            score += 25
        elif days <= 365:
            score += 20
        elif days <= 730:
            score += 10
        else:
            score += 2

    doc_type = str(row.get('Document Type') or '').lower()
    if 'application' in doc_type:
        score += 15
    elif 'patent' in doc_type:
        score += 5

    inv = int(row.get('inventor_count', 0))
    if inv == 1:
        score += 15
    elif inv == 2:
        score += 10
    elif inv <= 4:
        score += 5

    jurisdiction = str(row.get('Jurisdiction')).upper()
    if jurisdiction == 'US':
        score += 10
    elif jurisdiction in ['GB', 'IL']:
        score += 8
    elif jurisdiction in ['AU', 'CA', 'EP']:
        score += 6

    if int(row.get('Cites Patent Count', 1)) == 0:
        score += 5

    return score


def load_and_prescore(input_path: str, max_patents: int = 30) -> pd.DataFrame:
    print(f"Loading file: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    df = normalize_column_names(df)

    required_cols = ['Applicants', 'Title', 'Abstract']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")

    for col in ['Applicants', 'Title', 'Abstract', 'Inventors', 'Document Type', 'Jurisdiction', 'Legal Status']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    df['normalized_company'] = df['Applicants'].apply(normalize_company_name)

    if 'Inventors' in df.columns:
        df['inventor_count'] = df['Inventors'].apply(
            lambda x: len([p for p in str(x).split(';;') if p.strip()])
        )
    else:
        df['inventor_count'] = 1

    if 'Application Date' in df.columns:
        df['application_date'] = pd.to_datetime(df['Application Date'], errors='coerce')
    else:
        df['application_date'] = pd.NaT

    today = pd.Timestamp(datetime.today().strftime('%Y-%m-%d'))
    df['days_since_filing'] = (today - df['application_date']).dt.days
    df['sector'] = df.apply(lambda row: detect_sector(row.get('Title', ''), row.get('Abstract', '')), axis=1)

    legal_status_col = 'Legal Status' if 'Legal Status' in df.columns else None
    if legal_status_col:
        df = df[
            (~df['normalized_company'].apply(is_excluded)) &
            (df[legal_status_col].isin(['PENDING', 'ACTIVE'])) &
            (df['Applicants'] != '')
        ].copy()
    else:
        df = df[
            (~df['normalized_company'].apply(is_excluded)) &
            (df['Applicants'] != '')
        ].copy()

    company_counts = df.groupby('normalized_company').size().reset_index(name='patent_count')
    df = df.merge(company_counts, on='normalized_company', how='left')
    df = df[df['patent_count'] <= max_patents]

    df['prescore'] = df.apply(compute_prescore, axis=1)

    agg_dict = {
        'Applicants': lambda x: '; '.join(sorted(set(str(v) for v in x if str(v)))[:3]),
        'Title': lambda x: ' || '.join(list(dict.fromkeys(str(v) for v in x if str(v)))[:3]),
        'Abstract': lambda x: ' '.join(list(dict.fromkeys(str(v) for v in x if str(v)))[:5])[:2000],
        'sector': lambda x: ', '.join(sorted(set(str(v) for v in x if str(v)))[:3]),
        'patent_count': 'first',
        'prescore': 'max',
    }

    if 'Application Date' in df.columns:
        agg_dict['Application Date'] = 'max'

    grouped = df.groupby('normalized_company').agg(agg_dict).reset_index()

    rename_dict = {
        'normalized_company': 'Company',
        'sector': 'Industry',
        'patent_count': 'Patents_number',
        'prescore': 'Prescore',
    }

    if 'Application Date' in grouped.columns:
        rename_dict['Application Date'] = 'Date_of_latest_publication'

    grouped = grouped.rename(columns=rename_dict)

    grouped['AI_score'] = None
    grouped['Message'] = None
    grouped['Website'] = None
    grouped['LinkedIn'] = None

    grouped = grouped.sort_values('Prescore', ascending=False)

    print(f"Pre-scoring complete: {len(grouped)} companies")
    return grouped


def build_rag_query(company_name: str, industry: str, patent_text: str) -> str:
    parts = [company_name]
    if industry and industry not in ("Unknown", "Other"):
        parts.append(industry)
    if patent_text:
        parts.append(patent_text[:500])
    return " ".join(parts)


async def ai_score_company_async(
    company_name: str,
    industry: str,
    patent_text: str,
) -> Dict:
    print(f"   AI Scoring: {company_name[:50]}...")

    # 🔹 1. RAG-поиск релевантных кейсов (единственный источник кейсов)
    query_for_rag = build_rag_query(company_name, industry, patent_text)
    try:
        rag_cases = await rag.retrieve_cases(
            query=query_for_rag,
            api_key=OPENAI_API_KEY,
            top_k=3,
            threshold=0.45  # 🔹 Более строгий порог для качества
        )
    except Exception as e:
        print(f"   RAG failed: {e}")
        rag_cases = []

    # 🔹 2. Формируем компактный RAG-контекст
    rag_context = ""
    if rag_cases:
        rag_blocks = []
        for c in rag_cases:
            title = c.get("title", "Unknown")
            body = c.get("body", "")[:400].strip()  # 🔹 Короткие сниппеты
            industries = c.get("industry", [])
            if isinstance(industries, list):
                industries = ", ".join(i for i in industries if i)
            sim = c.get("similarity", 0)
            rag_blocks.append(f"- [{sim}] {title} [{industries}]: {body}")
        rag_context = "\n".join(rag_blocks)
        print(f"   RAG: {len(rag_cases)} case(s) retrieved.")

    # 🔹 3. Web-поиск информации о компании
    research_prompt = f'Research company: "{company_name}". Return JSON: {{"summary": "concise summary", "website": "official website URL or null", "linkedin": "LinkedIn URL or null"}}'
    website = None
    linkedin = None
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={"user_location": {"type": "approximate", "approximate": {"country": "US"}}},
            messages=[
                {"role": "system", "content": "You must respond with valid JSON only, no additional text or explanations."},
                {"role": "user", "content": research_prompt}
            ]
        )
        web_info = completion.choices[0].message.content
        try:
            # Extract JSON from the response (GPT may add extra text)
            start = web_info.find('{')
            end = web_info.rfind('}') + 1
            if start != -1 and end > start:
                json_str = web_info[start:end]
                parsed = json.loads(json_str)
                summary = parsed.get("summary", "")
                website = parsed.get("website") if parsed.get("website") and parsed.get("website") != "null" else None
                linkedin = parsed.get("linkedin") if parsed.get("linkedin") and parsed.get("linkedin") != "null" else None
                context_for_scoring = f"Web: {summary[:800]}\nPatents: {patent_text[:800]}" if summary else f"Patents: {patent_text[:1200]}"
            else:
                raise json.JSONDecodeError("No JSON found", web_info, 0)
        except json.JSONDecodeError:
            print(f"   Web Search failed to parse JSON: {web_info[:200]}...")
            context_for_scoring = f"Patents: {patent_text[:1200]}"
    except Exception as e:
        print(f"   Web Search failed: {e}")
        context_for_scoring = f"Patents only: {patent_text[:1200]}"

    # 🔹 4. Финальный промпт — ТОЛЬКО с RAG-кейсами, без загрузки всех кейсов
    score_prompt = f"""You are a Senior Sales Director at Interexy.

INTEREXY PROFILE:
{INTEREXY_BASE_PROFILE}

{'RELEVANT CASES (from knowledge base, ranked by similarity):\n' + rag_context + '\n\n' if rag_context.strip() else ''}TARGET: {company_name} ({industry})
CONTEXT: {context_for_scoring}

TASK: Score potential for selling software dev services (1-10).

RULES:
- 1-3: No fit (university or giant or non-tech or no similarity to RAG cases)
- 4-6: Maybe (relevant industry, but weak tech match or unclear outsourcing need)
- 7-8: Good fit (product company in target sector, modern tech stack)
- 9-10: Perfect fit (very similar to one of the RAG cases above)

Return JSON ONLY:
{{"ai_score": N, "industry": "...", "tech_stack": ["..."], "recommendation": "..."}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Output valid JSON only."},
                {"role": "user", "content": score_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        
        tech_stack = ", ".join(result.get("tech_stack", [])[:5])
        ai_score = result.get("ai_score", 0)

        lead_message = ""

        return {"ai_score": ai_score, "industry": result.get("industry", ""), "tech_stack": tech_stack, "recommendation": result.get("recommendation", ""), "message": lead_message, "website": website, "linkedin": linkedin}

    except Exception as e:
        print(f"   Scoring error: {e}")
        return {"ai_score": 0, "industry": "Unknown", "tech_stack": "", "recommendation": f"Error: {str(e)[:100]}", "message": ""}


# async def generate_lead_message_async(
#     company_name: str,
#     industry: str,
#     tech_stack: str,
#     ai_score: int,
#     rag_cases: Optional[List[Dict]] = None,
# ) -> str:
#     if rag_cases:
#         cases_context = "\n\n".join([
#             f"- {c.get('title', 'Unknown')} [{c.get('industry', [])}]: {c.get('body', '')[:250].strip()}"
#             for c in rag_cases
#         ])
#         retrieval_note = f"Relevant case(s) from our portfolio:\n{cases_context}\n\n"
#     else:
#         retrieval_note = ""

#     score_band = "premium" if ai_score >= 8 else ("promising" if ai_score >= 6 else "speculative")
#     tone = "enthusiastic" if ai_score >= 8 else "warm and professional"

#     prompt = f"""You are a B2B outreach writer at Interexy — a premium software development company (outstaffing/custom dev).

# Write a personalized outreach message for a potential client.

# === TARGET COMPANY ===
# Name: {company_name}
# Industry: {industry}
# Tech Stack: {tech_stack}
# Score Band: {score_band}

# === OUR SUCCESSFUL CASES ===
# {retrieval_note if retrieval_note else "No highly relevant cases found — use your general knowledge of Interexy's expertise in modern tech stacks."}

# # === MESSAGE REQUIREMENTS ===
# 1. Friendly greeting, short and direct (under 200 words total).
# 2. Compliment the company's work — reference their specific industry and tech achievements.
# 3. If you have case references from above, mention them with concrete outcomes.
#     If no strong match exists, reference a relevant Interexy specialty instead.
# 4. End with a clear, confident offer to help bring their idea to life.
# 5. Tone: {tone}.

# === OUTPUT ===
# Return ONLY the message text, no labels, no JSON, just the message itself."""

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.7,
#             max_tokens=300,
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"   Message generation failed: {e}")
#         return ""


def ai_score_company(company_name: str, industry: str, patent_text: str) -> Dict:
    """Синхронная обёртка для ai_score_company_async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(
        ai_score_company_async(company_name, industry, patent_text)
    )


def main():
    parser = argparse.ArgumentParser(description='Patent Lead Generation Pipeline')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', default='patent_results.csv', help='Output CSV filename')
    parser.add_argument('--max-patents', type=int, default=30, help='Max patents per company')
    parser.add_argument('--top', type=int, default=None, help='Keep only top N results')
    args = parser.parse_args()

    df = load_and_prescore(args.input, max_patents=args.max_patents)

    if args.top:
        df = df.head(args.top)

    print(f"\nStarting AI scoring for {len(df)} companies...")

    for idx, row in df.iterrows():
        company_name = row['Company']
        industry = row['Industry']
        patent_text = f"Titles: {row['Title']}\nAbstracts: {row['Abstract']}"

        print(f"[{idx+1}/{len(df)}] Processing: {company_name[:60]}...")

        # 🔹 Вызов без cases_text — только RAG
        result = ai_score_company(company_name, industry, patent_text)

        df.at[idx, 'AI_score'] = result['ai_score']
        df.at[idx, 'Message'] = result['message']

        time.sleep(2)

    df = df.sort_values(by='AI_score', ascending=False)
    df.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\nDone! Saved to {args.output}")
    print(f"Top 5 companies by AI score:")
    print(df.head(5)[['Company', 'Prescore', 'AI_score', 'Industry']])


if __name__ == "__main__":
    main()