"""
Patent Leadgen Pipeline
=======================

A user-friendly script for Lens.org patent exports.
It normalizes applicant names, groups patents by company,
applies a cheap relevance filter, and can expose a simple UI.

Usage:
    python patent_leadgen.py --input test.csv --output top_companies.csv --top 200
    python patent_leadgen.py --input test.csv --ui

Requirements:
    pandas
    optional: streamlit for the UI
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

COMMON_SUFFIXES = [
    'CO', 'CO.', 'CO,', 'COMPANY', 'INC', 'INC.', 'LLC', 'LLP', 'LTD', 'LTD.',
    'PLC', 'GMBH', 'AG', 'SA', 'S.A.', 'SARL', 'BV', 'PT', 'PTY', 'SP Z O O',
    'SP ZOO', 'SAS', 'S.R.L.', 'SRL', 'CORP', 'CORP.', 'CORPORATION', 'LIMITED','health','healthcare'
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
    'ericsson', 'nokia', 'bank', 'insurance', 'financial services', 'telecom','netflix','univ'
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
        'microgrid', 'power optimization', 'cleantech', 'renewable energy',
        'energy'
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

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит названия колонок Lens.org к ожидаемому формату."""
    column_mapping = {
        # Заявители
        'Applicant': 'Applicants',
        'Applicants': 'Applicants',
        'Assignee': 'Applicants',
        'Assignees': 'Applicants',
        'Owner': 'Applicants',
        'Patent Assignee': 'Applicants',
        
        # Заголовок
        'Title': 'Title',
        'Patent Title': 'Title',
        'Document Title': 'Title',
        
        # Аннотация
        'Abstract': 'Abstract',
        'Abstract (en)': 'Abstract',
        'Summary': 'Abstract',
        
        # Изобретатели
        'Inventor': 'Inventors',
        'Inventors': 'Inventors',
        'Author': 'Inventors',
        
        # Тип документа
        'Document Type': 'Document Type',
        'Kind Code': 'Document Type',
        'Publication Type': 'Document Type',
        
        # Юрисдикция
        'Jurisdiction': 'Jurisdiction',
        'Country': 'Jurisdiction',
        'Publication Country': 'Jurisdiction',
        'Country Code': 'Jurisdiction',
        
        # Правовой статус
        'Legal Status': 'Legal Status',
        'Status': 'Legal Status',
        
        # Даты
        'Application Date': 'Application Date',
        'Filing Date': 'Application Date',
        'Publication Date': 'Publication Date',
        'Priority Date': 'Earliest Priority Date',
        'Earliest Priority Date': 'Earliest Priority Date',
        
        # Цитирования
        'Cites Patent Count': 'Cites Patent Count',
        'Cited Patent Count': 'Cites Patent Count',
        'Forward Cites': 'Cites Patent Count',
        
        # URL
        'URL': 'URL',
        'Lens URL': 'URL',
        'Link': 'URL',
    }
    
    # Переименовываем только существующие колонки
    existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_mapping)
    
    return df

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


def _detect_sector(title: str, abstract: str) -> str:
    text = f"{title or ''} {abstract or ''}".lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return sector
    return 'Other'


def _is_excluded(company: str) -> bool:
    name = (company or '').lower()
    return any(keyword in name for keyword in EXCLUDE_KEYWORDS)


def compute_row_score(row: pd.Series) -> int:
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

    if str(row.get('Jurisdiction')).upper() == 'US':
        score += 10
    elif str(row.get('Jurisdiction')).upper() in ['GB', 'IL']:
        score += 8
    elif str(row.get('Jurisdiction')).upper() in ['AU', 'CA', 'EP']:
        score += 6

    if int(row.get('Cites Patent Count', 1)) == 0:
        score += 5

    return score


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = normalize_column_names(df)
    return df


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Applicants'] = df['Applicants'].fillna('').astype(str)
    for col in ['Title', 'Abstract', 'Inventors', 'Document Type', 'Jurisdiction']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    df = parse_dates(df)
    df['normalized_company'] = df['Applicants'].apply(normalize_company_name)
    df['inventor_count'] = df['Inventors'].apply(
        lambda x: len([p for p in str(x).split(';;') if p.strip()])
    )
    df['application_date'] = pd.to_datetime(df.get('Application Date'), errors='coerce')
    today = pd.Timestamp(datetime.today().strftime('%Y-%m-%d'))
    df['days_since_filing'] = (today - df['application_date']).dt.days
    df['sector'] = df.apply(lambda row: _detect_sector(row.get('Title', ''), row.get('Abstract', '')), axis=1)
    df['company_raw_count'] = df.groupby('Applicants')['Applicants'].transform('count')
    df['company_normalized_count'] = df.groupby('normalized_company')['normalized_company'].transform('count')
    df['row_score'] = df.apply(compute_row_score, axis=1)
    return df


def summarize_pipeline(df: pd.DataFrame, company_df: pd.DataFrame, filtered_df: pd.DataFrame, args: argparse.Namespace) -> None:
    print('\n' + '=' * 60)
    print('PATENT LEADGEN PIPELINE SUMMARY')
    print('=' * 60)
    print(f"  Input patents:              {len(df):>6}")
    print(f"  Unique raw applicants:      {df['Applicants'].nunique():>6}")
    print(f"  Normalized companies:       {df['normalized_company'].nunique():>6}")
    print(f"  After cheap filter:         {filtered_df['normalized_company'].nunique():>6}")
    print(f"  Output candidates:          {len(company_df):>6}")
    print()
    print(f"  Selected industry:          {args.industry or 'Any'}")
    print(f"  Selected jurisdiction:      {', '.join(args.jurisdiction) if args.jurisdiction else 'Any'}")
    print(f"  Date range:                 {args.start_date or 'Any'} to {args.end_date or 'Any'}")
    print('=' * 60)


def build_company_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    company_columns = [
        'normalized_company',
        'Applicants',
        'sector',
        'Jurisdiction',
        'company_normalized_count',
        'company_raw_count',
        'patent_count',
        'max_row_score',
        'min_days_since_filing',
        'application_date',
        'Title',
        'URL',
    ]

    agg = {
        'Applicants': lambda x: '; '.join(sorted(set(x))[:5]),
        'Title': lambda x: ' || '.join(list(dict.fromkeys(x))[:3]),
        'URL': lambda x: '; '.join(list(dict.fromkeys(x))[:3]),
        'sector': lambda x: ', '.join(sorted(set(x))[:3]),
        'Jurisdiction': lambda x: ', '.join(sorted(set(x))[:3]),
        'company_raw_count': 'max',
        'company_normalized_count': 'max',
        'patent_count': 'max',
        'row_score': 'max',
        'days_since_filing': 'min',
        'Application Date': 'min',
    }

    grouped = df.groupby('normalized_company').agg(agg).reset_index()
    grouped = grouped.rename(columns={
        'Applicants': 'Raw Applicants',
        'sector': 'Sector',
        'Jurisdiction': 'Jurisdiction',
        'company_raw_count': 'Raw Applicant Names',
        'company_normalized_count': 'Patent Rows',
        'patent_count': 'Patent Count',
        'row_score': 'Company Score',
        'days_since_filing': 'Most Recent Filing Days Ago',
        'Application Date': 'Earliest Filing Date',
        'Title': 'Top Patent Titles',
        'URL': 'Sample Links',
    })

    grouped['Company'] = grouped['normalized_company']
    grouped = grouped.drop(columns=['normalized_company'])
    grouped = grouped.sort_values(['Company Score', 'Patent Count', 'Most Recent Filing Days Ago'], ascending=[False, False, True])
    return grouped


def prepare_company_table(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    company_df = df.groupby('normalized_company').agg(
        patent_count=('Applicants', 'size'),
        company_raw_count=('Applicants', 'nunique'),
        row_score=('row_score', 'max'),
        min_days_since_filing=('days_since_filing', 'min'),
        Title=('Title', lambda x: ' || '.join(list(dict.fromkeys(x))[:3])),
        URL=('URL', lambda x: '; '.join(list(dict.fromkeys(x))[:3])),
        Applicants=('Applicants', lambda x: '; '.join(sorted(set(x))[:5])),
        sector=('sector', lambda x: ', '.join(sorted(set(x))[:3])),
        Jurisdiction=('Jurisdiction', lambda x: ', '.join(sorted(set(x))[:3])),
        ApplicationDate=('Application Date', 'min'),
    ).reset_index()

    company_df = company_df.rename(columns={
        'normalized_company': 'Company',
        'row_score': 'Company Score',
        'patent_count': 'Patent Count',
        'company_raw_count': 'Raw Applicant Names',
        'min_days_since_filing': 'Most Recent Filing Days Ago',
        'Title': 'Top Patent Titles',
        'URL': 'Sample Links',
        'Applicants': 'Raw Applicants',
        'sector': 'Sector',
        'Jurisdiction': 'Jurisdiction',
        'ApplicationDate': 'Earliest Filing Date',
    })

    company_df = company_df.sort_values(['Company Score', 'Patent Count', 'Most Recent Filing Days Ago'], ascending=[False, False, True])
    return company_df


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()
    if args.start_date:
        filtered = filtered[filtered['application_date'] >= pd.to_datetime(args.start_date)]
    if args.end_date:
        filtered = filtered[filtered['application_date'] <= pd.to_datetime(args.end_date)]
    if args.industry:
        filtered = filtered[filtered['sector'] == args.industry]
    if args.jurisdiction:
        filtered = filtered[filtered['Jurisdiction'].isin(args.jurisdiction)]
    if args.max_patents is not None:
        filtered = filtered[filtered['company_normalized_count'] <= args.max_patents]
    return filtered


def cheap_filter(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[
        (~df['normalized_company'].apply(_is_excluded)) &
        (df['Legal Status'].isin(['PENDING', 'ACTIVE'])) &
        (df['Applicants'] != '')
    ].copy()
    return filtered


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    df = load_data(Path(args.input))
    df = enrich_data(df)
    raw_companies = df['Applicants'].nunique()
    norm_companies = df['normalized_company'].nunique()

    filtered = cheap_filter(df)
    filtered = apply_filters(filtered, args)
    company_df = prepare_company_table(filtered, args)

    if args.top:
        company_df = company_df.head(args.top)

    summarize_pipeline(df, company_df, filtered, args)
    return company_df


def run_streamlit_ui(args: argparse.Namespace) -> None:
    try:
        import streamlit as st
    except ImportError:
        print('Streamlit is not installed. Install with: pip install streamlit')
        return

    st.set_page_config(page_title='Patent Leadgen Explorer', layout='wide')
    st.title('Patent Leadgen Explorer')

    st.sidebar.header('Controls')
    uploaded = st.sidebar.file_uploader('Upload Lens CSV', type=['csv'])
    input_path = uploaded if uploaded is not None else args.input

    sample_df = None
    if uploaded is not None:
        try:
            sample_df = pd.read_csv(uploaded, nrows=1000)
        except Exception:
            sample_df = None
    elif args.input:
        try:
            sample_df = load_data(Path(args.input))
        except Exception:
            sample_df = None

    start_date = st.sidebar.date_input('Start filing date', value=None)
    end_date = st.sidebar.date_input('End filing date', value=None)
    sector = st.sidebar.selectbox('Sector', ['Any'] + list(SECTOR_KEYWORDS.keys()))
    jurisdiction_options = sorted(sample_df['Jurisdiction'].dropna().unique().tolist()) if sample_df is not None and 'Jurisdiction' in sample_df.columns else []
    jurisdictions = st.sidebar.multiselect('Jurisdictions', jurisdiction_options)
    top_n = st.sidebar.number_input('Top companies', min_value=10, max_value=1000, value=200, step=10)

    if uploaded is None and not args.input:
        st.warning('Choose a CSV file or set --input on the command line.')
        return

    st.write('### Pipeline filters')
    st.write('- Cheap filter: excludes universities, giants, and inactive patents')
    st.write('- Normalized companies: groups variants of the same applicant')
    st.write('- Sector and jurisdiction filters')

    if input_path is not None:
        if uploaded is not None:
            df = pd.read_csv(uploaded, low_memory=False)
        else:
            df = load_data(Path(args.input))
        df = enrich_data(df)
        if start_date:
            df = df[df['application_date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['application_date'] <= pd.to_datetime(end_date)]
        if sector and sector != 'Any':
            df = df[df['sector'] == sector]
        if jurisdictions:
            df = df[df['Jurisdiction'].isin(jurisdictions)]

        filtered = cheap_filter(df)
        company_df = prepare_company_table(filtered, args)
        company_df = company_df.head(top_n)

        st.metric('Input patents', len(df))
        st.metric('Unique raw applicants', df['Applicants'].nunique())
        st.metric('Normalized companies', df['normalized_company'].nunique())
        st.metric('Filtered companies', company_df['Company'].nunique())

        st.dataframe(company_df)
        csv = company_df.to_csv(index=False)
        st.download_button('Download top companies', csv, 'leadgen_companies.csv', 'text/csv')


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Patent leadgen pipeline for Lens.org exports')
    parser.add_argument('--input', required=False, help='Path to input CSV file')
    parser.add_argument('--output', default='patent_leads.csv', help='Output CSV filename')
    parser.add_argument('--top', type=int, default=200, help='Keep only top N companies')
    parser.add_argument('--max-patents', type=int, default=30, help='Maximum patent rows per normalized company')
    parser.add_argument('--min-score', type=int, default=0, help='Minimum company score to keep')
    parser.add_argument('--industry', default=None, help='Filter by industry sector')
    parser.add_argument('--jurisdiction', nargs='*', default=[], help='Filter by jurisdiction code(s)')
    parser.add_argument('--start-date', default=None, help='Filter companies by earliest filing date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=None, help='Filter companies by latest filing date (YYYY-MM-DD)')
    parser.add_argument('--ui', action='store_true', help='Launch Streamlit UI')
    return parser.parse_args()


def main() -> None:
    args = build_args()
    if args.ui:
        run_streamlit_ui(args)
        return

    if not args.input:
        raise ValueError('Path to input CSV is required in CLI mode. Use --input <file.csv>')

    company_df = run_pipeline(args)
    if args.min_score > 0:
        company_df = company_df[company_df['Company Score'] >= args.min_score]
    company_df.to_csv(args.output, index=False)
    print(f'\nSaved top companies: {args.output}')


if __name__ == '__main__':
    main()
