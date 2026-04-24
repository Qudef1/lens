"""
Patent Scoring Script for Interexy
====================================
Filters and scores patents from Lens.org CSV export
to identify startups and SMBs as potential clients.

Usage:
    python patent_scoring.py --input patents.csv
    python patent_scoring.py --input patents.csv --min-score 75 --sector Healthcare
    python patent_scoring.py --input patents.csv --top 200 --output my_results.csv

Requirements:
    pip install pandas numpy
"""

import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EXCLUDE_KEYWORDS = [
    'university', 'institute', 'college', 'academy', 'foundation',
    'hospital', 'clinic', 'health system', 'laboratory', 'research center',
    'department', 'ministry', 'agency', 'government', 'nasa', 'darpa',
    'nih', 'nist', 'fraunhofer', 'mit', 'stanford', 'harvard', 'oxford', 'eth',
    'samsung', 'philips', 'siemens', 'medtronic', 'johnson', 'abbott',
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
    'ericsson', 'nokia',
]

SECTOR_KEYWORDS = {
    'Healthcare': [
        'telehealth', 'telemedicine', 'patient monitoring', 'remote diagnosis',
        'digital health', 'health platform', 'clinical decision', 'medical imaging',
        'drug delivery', 'diagnostic', 'biosensor', 'wearable', 'genomics',
        'drug discovery', 'clinical trial', 'bioinformatics', 'precision medicine',
        'digital pathology', 'laboratory automation',
    ],
    'Energy': [
        'energy management', 'smart grid', 'energy storage', 'solar monitoring',
        'wind turbine', 'battery management', 'ev charging', 'demand response',
        'microgrid', 'power optimization', 'cleantech', 'renewable energy',
    ],
    'AI/ML': [
        'machine learning', 'deep learning', 'neural network', 'natural language processing',
        'computer vision', 'predictive analytics', 'ai platform', 'mlops',
        'model deployment', 'inference engine', 'large language model',
    ],
    'Fintech': [
        'payment processing', 'fraud detection', 'risk scoring', 'open banking',
        'digital wallet', 'lending platform', 'credit scoring', 'kyc', 'regtech',
        'algorithmic trading', 'financial analytics', 'neobank',
    ],
    'Industrial IoT': [
        'predictive maintenance', 'condition monitoring', 'industrial iot',
        'smart factory', 'asset tracking', 'digital twin', 'supply chain visibility',
        'process automation', 'quality control', 'oee',
    ],
    'Mobility': [
        'fleet management', 'route optimization', 'autonomous vehicle',
        'mobility platform', 'traffic management', 'logistics optimization',
        'vehicle telematics', 'last-mile delivery', 'cargo tracking',
    ],
    'PropTech': [
        'smart building', 'building automation', 'property management',
        'facility management', 'hvac optimization', 'occupancy monitoring',
        'real estate platform', 'space management',
    ],
    'Cybersecurity': [
        'threat detection', 'anomaly detection', 'network security',
        'endpoint protection', 'identity management', 'zero trust',
        'vulnerability management', 'siem', 'data privacy', 'compliance automation',
    ],
    'AgriTech': [
        'precision agriculture', 'crop monitoring', 'irrigation management',
        'soil analysis', 'yield prediction', 'farm management',
        'livestock monitoring', 'agricultural drone',
    ],
}

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"📂 Loading: {path}")
    df = pd.read_csv(path)
    print(f"   → {len(df)} rows, {len(df.columns)} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['Applicants', 'Title', 'Abstract', 'Inventors', 'Document Type',
                'Legal Status', 'Jurisdiction']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    # Patents per applicant
    counts = df.groupby('Applicants').size().reset_index(name='applicant_total_patents')
    df = df.merge(counts, on='Applicants', how='left')

    # Inventor count
    df['inventor_count'] = df['Inventors'].apply(
        lambda x: len([i.strip() for i in x.split(';;') if i.strip()]) if x else 0
    )

    # Days since filing
    df['Application Date'] = pd.to_datetime(df['Application Date'], errors='coerce')
    today = pd.Timestamp(datetime.today().strftime('%Y-%m-%d'))
    df['days_since_filing'] = (today - df['Application Date']).dt.days

    # Sector
    df['sector'] = df.apply(_detect_sector, axis=1)

    return df


def _detect_sector(row) -> str:
    text = (row.get('Title', '') + ' ' + row.get('Abstract', '')).lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return sector
    return 'Other'


def _is_excluded(applicant: str) -> bool:
    a = applicant.lower()
    return any(kw in a for kw in EXCLUDE_KEYWORDS)


def _score_row(row) -> tuple:
    score = 0
    signals = []

    # A) Company size by patent count (max 30)
    n = row['applicant_total_patents']
    if n == 1:
        score += 30; signals.append('solo_patent(+30)')
    elif n <= 3:
        score += 25; signals.append(f'{n}_patents(+25)')
    elif n <= 10:
        score += 18; signals.append(f'{n}_patents(+18)')
    elif n <= 30:
        score += 8;  signals.append(f'{n}_patents(+8)')
    # >30 → 0 points, likely a large company

    # B) Filing recency (max 25)
    days = row['days_since_filing']
    if pd.notna(days):
        if days <= 180:
            score += 25; signals.append('filed<6m(+25)')
        elif days <= 365:
            score += 20; signals.append('filed<1y(+20)')
        elif days <= 730:
            score += 10; signals.append('filed<2y(+10)')
        else:
            score += 2;  signals.append('filed>2y(+2)')

    # C) Document type (max 15)
    if row['Document Type'] == 'Patent Application':
        score += 15; signals.append('application(+15)')
    elif row['Document Type'] == 'Granted Patent':
        score += 5;  signals.append('granted(+5)')

    # D) Founder-led signal: 1–2 inventors (max 15)
    inv = row['inventor_count']
    if inv == 1:
        score += 15; signals.append('1_inventor=founder(+15)')
    elif inv == 2:
        score += 10; signals.append('2_inventors(+10)')
    elif inv <= 4:
        score += 5;  signals.append(f'{inv}_inventors(+5)')

    # E) Jurisdiction (max 10)
    j = row['Jurisdiction']
    if j == 'US':
        score += 10; signals.append('US(+10)')
    elif j in ['GB', 'IL']:
        score += 8;  signals.append(f'{j}(+8)')
    elif j in ['AU', 'CA', 'EP']:
        score += 6;  signals.append(f'{j}(+6)')

    # F) Original — no prior patent citations (max 5)
    if row.get('Cites Patent Count', 1) == 0:
        score += 5; signals.append('no_prior_cites(+5)')

    return score, ' | '.join(signals)


def score(df: pd.DataFrame) -> pd.DataFrame:
    df[['score', 'score_signals']] = df.apply(
        lambda r: pd.Series(_score_row(r)), axis=1
    )
    return df


def filter_data(df: pd.DataFrame, max_patents: int = 30) -> pd.DataFrame:
    filtered = df[
        (~df['Applicants'].apply(_is_excluded)) &
        (df['Legal Status'].isin(['PENDING', 'ACTIVE'])) &
        (df['applicant_total_patents'] <= max_patents) &
        (df['Applicants'] != '')
    ].copy()
    return filtered


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    cols_map = {
        'score':                   'Score',
        'Applicants':              'Company',
        'sector':                  'Sector',
        'Title':                   'Patent Title',
        'Jurisdiction':            'Jurisdiction',
        'Application Date':        'Filing Date',
        'Document Type':           'Doc Type',
        'Legal Status':            'Status',
        'applicant_total_patents': 'Company Total Patents',
        'inventor_count':          'Inventor Count',
        'days_since_filing':       'Days Since Filing',
        'Cited by Patent Count':   'Times Cited',
        'score_signals':           'Score Breakdown',
        'URL':                     'Lens URL',
    }
    available = {k: v for k, v in cols_map.items() if k in df.columns}
    result = df[list(available.keys())].copy()
    result.columns = list(available.values())
    return result.sort_values('Score', ascending=False)


def print_summary(df_raw: pd.DataFrame, df_filtered: pd.DataFrame, result: pd.DataFrame):
    print("\n" + "═" * 55)
    print("  PATENT SCORING SUMMARY")
    print("═" * 55)
    print(f"  Input patents:        {len(df_raw):>6}")
    print(f"  After filtering:      {len(df_filtered):>6}")
    print(f"  In output:            {len(result):>6}")
    print()
    print(f"  🔥 Hot   (75–100):    {len(result[result['Score'] >= 75]):>6}")
    print(f"  🟡 Warm  (55–74):     {len(result[(result['Score'] >= 55) & (result['Score'] < 75)]):>6}")
    print(f"  ❄️  Cold  (<55):       {len(result[result['Score'] < 55]):>6}")
    print()
    print("  Top sectors:")
    for sector, count in result['Sector'].value_counts().head(6).items():
        print(f"    {sector:<20} {count:>5}")
    print()
    print("  Top 5 companies:")
    top = result[['Score', 'Company', 'Sector', 'Filing Date']].head(5)
    for _, row in top.iterrows():
        print(f"    [{row['Score']:>3}] {str(row['Company'])[:35]:<35} {row['Sector']}")
    print("═" * 55)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Score Lens.org patent exports to find startup/SMB leads for Interexy'
    )
    parser.add_argument('--input',      required=True,          help='Path to input CSV from Lens.org')
    parser.add_argument('--output',     default='patents_scored.csv', help='Output CSV filename')
    parser.add_argument('--min-score',  type=int, default=0,    help='Minimum score to include (e.g. 75)')
    parser.add_argument('--max-patents',type=int, default=30,   help='Max patents per applicant (default: 30)')
    parser.add_argument('--sector',     default=None,           help='Filter by sector (e.g. Healthcare, AI/ML)')
    parser.add_argument('--top',        type=int, default=None, help='Keep only top N results')
    args = parser.parse_args()

    # Pipeline
    df = load_data(args.input)
    df = clean(df)
    df = enrich(df)
    df = score(df)
    df_filtered = filter_data(df, max_patents=args.max_patents)
    result = build_output(df_filtered)

    # Optional filters
    if args.min_score > 0:
        result = result[result['Score'] >= args.min_score]
        print(f"   → Applied min score filter: ≥{args.min_score} → {len(result)} rows")

    if args.sector:
        result = result[result['Sector'].str.lower() == args.sector.lower()]
        print(f"   → Applied sector filter: {args.sector} → {len(result)} rows")

    if args.top:
        result = result.head(args.top)
        print(f"   → Kept top {args.top} results")

    # Save
    result.to_csv(args.output, index=False)
    print(f"\n✅ Saved → {args.output}")

    print_summary(df, df_filtered, result)


if __name__ == '__main__':
    main()