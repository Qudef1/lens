"""
Patent Lead Generation UI
=========================

Streamlit UI for patent pre-scoring and AI-powered deep scoring.

Usage:
    streamlit run app.py
"""

import os
import time
import asyncio
import pickle
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

import rag
from patent_pipeline import (
    normalize_column_names,
    normalize_company_name,
    detect_sector,
    is_excluded,
    compute_prescore,
    ai_score_company_async,
    load_and_prescore as pipeline_load_and_prescore,
    generate_lead_message,
    SECTOR_KEYWORDS,
    DATE_COLUMNS,
    EXCLUDE_KEYWORDS,
)

load_dotenv()

STATE_FILE = 'prescored_state.pkl'
ALL_STATE_FILE = 'prescored_state_all.pkl'

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title='Patent Lead Generation', layout='wide')
st.title('Patent Lead Generation Pipeline')

if 'prescored_df' not in st.session_state:
    st.session_state.prescored_df = None
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                st.session_state.prescored_df = pickle.load(f)
            # Ensure dtypes for new columns
            for col in ['rag_cases', 'web_context']:
                if col in st.session_state.prescored_df.columns:
                    st.session_state.prescored_df[col] = st.session_state.prescored_df[col].astype(object)
        except Exception as e:
            st.warning(f"Could not load saved state: {e}")

if 'all_processed_df' not in st.session_state:
    st.session_state.all_processed_df = None
    if os.path.exists(ALL_STATE_FILE):
        try:
            with open(ALL_STATE_FILE, 'rb') as f:
                st.session_state.all_processed_df = pickle.load(f)
            # Ensure dtypes
            for col in ['rag_cases', 'web_context']:
                if col in st.session_state.all_processed_df.columns:
                    st.session_state.all_processed_df[col] = st.session_state.all_processed_df[col].astype(object)
        except Exception as e:
            st.warning(f"Could not load all processed state: {e}")

# 🔹 Убрано: if 'cases_text' not in st.session_state

def load_and_prescore_ui(uploaded_file, max_patents: int, start_date, end_date, sector, jurisdictions) -> pd.DataFrame:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        df = pipeline_load_and_prescore(tmp_path, max_patents=max_patents)
        if 'application_date' in df.columns and start_date:
            df = df[df['application_date'] >= pd.to_datetime(start_date)]
        if 'application_date' in df.columns and end_date:
            df = df[df['application_date'] <= pd.to_datetime(end_date)]
        if sector and sector != 'All' and 'Industry' in df.columns:
            df = df[df['Industry'] == sector]
        if jurisdictions and 'Jurisdiction' in df.columns:
            df = df[df['Jurisdiction'].isin(jurisdictions)]
        return df
    finally:
        os.unlink(tmp_path)

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)

def save_state():
    if st.session_state.prescored_df is not None:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(st.session_state.prescored_df, f)

def save_all_state():
    if st.session_state.prescored_df is not None:
        # Update all_processed_df with new data
        if st.session_state.all_processed_df is None:
            st.session_state.all_processed_df = st.session_state.prescored_df.copy()
        else:
            all_df = st.session_state.all_processed_df
            new_df = st.session_state.prescored_df
            cols_to_update = ['AI_score', 'Recommendation', 'Industry', 'Website', 'LinkedIn', 'rag_cases', 'web_context', 'Lead_Message']
            for col in cols_to_update:
                if col in new_df.columns and col in all_df.columns:
                    # Get processed companies from new_df
                    processed = new_df[new_df['AI_score'].notna()].copy()
                    for comp in processed['Company']:
                        new_val = processed.loc[processed['Company'] == comp, col].iloc[0]
                        mask = all_df['Company'] == comp
                        if mask.any() and pd.notna(new_val):
                            all_df.loc[mask, col] = new_val
            # Add new companies (those not in all_df)
            new_companies = new_df[~new_df['Company'].isin(all_df['Company'])]
            if not new_companies.empty:
                st.session_state.all_processed_df = pd.concat([all_df, new_companies], ignore_index=True)
            else:
                st.session_state.all_processed_df = all_df
        with open(ALL_STATE_FILE, 'wb') as f:
            pickle.dump(st.session_state.all_processed_df, f)

async def score_single_company_async_wrapper(idx: int, row: pd.Series) -> Dict:
    company_name = row['Company']
    industry = row.get('Industry', '')
    patent_text = f"Titles: {row.get('Title', '')}\nAbstracts: {row.get('Abstract', '')}"

    # Check if already processed successfully
    existing_score = row.get('AI_score')
    if pd.notna(existing_score) and existing_score > 0:
        print(f"   Company {company_name} already processed with score {existing_score}, skipping AI scoring.")
        return {
            'idx': idx,
            'ai_score': existing_score,
            'recommendation': row.get('Recommendation', ''),
            'industry': row.get('Industry', ''),
            'website': row.get('Website'),
            'linkedin': row.get('LinkedIn')
        }

    result = await ai_score_company_async(company_name, industry, patent_text)
    print(result)
    return {'idx': idx, 'ai_score': result.get('ai_score', 0), 'recommendation': result.get('recommendation', ''), 'industry': result.get('industry', ''), 'website': result.get('website'), 'linkedin': result.get('linkedin')}

def score_single_company(idx: int, row: pd.Series) -> Dict:
    coro = score_single_company_async_wrapper(idx, row)
    return run_async(coro)

# ==================== UI ====================

st.sidebar.header('1. Upload & Configure')
uploaded_file = st.sidebar.file_uploader('Upload Patent CSV', type=['csv'])

st.sidebar.subheader('Filters')
max_patents = st.sidebar.number_input('Max patents per company', min_value=1, max_value=100, value=30)
start_date = st.sidebar.date_input('Start filing date', value=None)
end_date = st.sidebar.date_input('End filing date', value=None)
sector_options = ['All'] + list(SECTOR_KEYWORDS.keys())
sector = st.sidebar.selectbox('Sector', sector_options)

jurisdiction_options = []
if uploaded_file is not None:
    try:
        sample_df = pd.read_csv(uploaded_file, nrows=50)
        sample_df = normalize_column_names(sample_df)
        if 'Jurisdiction' in sample_df.columns:
            jurisdiction_options = sorted(sample_df['Jurisdiction'].dropna().unique().tolist())
    except Exception as e:
        st.warning(f"Could not preview file: {e}")

jurisdictions = st.sidebar.multiselect('Jurisdictions', jurisdiction_options)

if st.sidebar.button('Run Pre-scoring', type='primary'):
    if uploaded_file is None:
        st.error('Please upload a CSV file first')
    else:
        with st.spinner('Running pre-scoring pipeline...'):
            try:
                # 🔹 Убрано: st.session_state.cases_text = load_cases(CASES_DIR)
                st.session_state.prescored_df = load_and_prescore_ui(
                    uploaded_file, max_patents, start_date, end_date, sector, jurisdictions
                )
                # Merge existing AI scores from all_processed_df
                if st.session_state.all_processed_df is not None:
                    all_df = st.session_state.all_processed_df
                    merge_cols = ['AI_score', 'Recommendation', 'Industry', 'Website', 'LinkedIn', 'rag_cases', 'web_context', 'Lead_Message']
                    for col in merge_cols:
                        if col in all_df.columns and col in st.session_state.prescored_df.columns:
                            # Build lookup dict for companies in all_processed_df
                            lookup = all_df.set_index('Company')[col].to_dict()
                            # Update prescored_df for companies that exist in lookup and have no value
                            for comp in st.session_state.prescored_df['Company']:
                                curr_val = st.session_state.prescored_df.loc[st.session_state.prescored_df['Company'] == comp, col].iloc[0]
                                if comp in lookup and (pd.isna(curr_val) or curr_val is None):
                                    st.session_state.prescored_df.loc[st.session_state.prescored_df['Company'] == comp, col] = lookup[comp]
                save_state()
                st.success(f'Pre-scoring complete! Found {len(st.session_state.prescored_df)} companies')
            except Exception as e:
                st.error(f'❌ Error during pre-scoring: {str(e)}')
                st.code(f"Available columns: {list(pd.read_csv(uploaded_file, nrows=1).columns)}")


st.sidebar.markdown('---')
st.sidebar.header('2. Display Options and analyze')
top_n = st.sidebar.number_input('Top companies to display (0 for all)', min_value=0, value=0)

st.sidebar.subheader('Reprocessing Options')
if st.session_state.prescored_df is not None:
    if st.sidebar.button('Reprocess Failed Companies (AI score 0 or errors)', type='secondary'):
        failed_df = st.session_state.prescored_df[
            (st.session_state.prescored_df['AI_score'] == 0) |
            (st.session_state.prescored_df['AI_score'].isna()) |
            (st.session_state.prescored_df['Recommendation'].str.contains('Error|429', na=False, case=False))
        ]
        if len(failed_df) == 0:
            st.sidebar.info('No failed companies to reprocess.')
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.expander('Reprocessing Log', expanded=True)
            for i, (idx, row) in enumerate(failed_df.iterrows()):
                company_name = row['Company']
                status_text.text(f'Reprocessing {i+1}/{len(failed_df)}: {company_name[:50]}...')
                with log_container:
                    st.write(f'**[{i+1}/{len(failed_df)}]** {company_name}')
            try:
                result = score_single_company(idx, row)
                existing_score = st.session_state.prescored_df.at[idx, 'AI_score']
                if pd.notna(existing_score) and existing_score > 0:
                    with log_container:
                        st.write(f"   ⏭️ Already processed with score {existing_score}/10, skipped")
                else:
                    st.session_state.prescored_df.at[idx, 'AI_score'] = result['ai_score']
                    st.session_state.prescored_df.at[idx, 'Recommendation'] = result['recommendation']
                    st.session_state.prescored_df.at[idx, 'rag_cases'] = json.dumps(result.get('rag_cases', []), ensure_ascii=False)
                    st.session_state.prescored_df.at[idx, 'web_context'] = result.get('web_context', '')
                    if result['industry']:
                        st.session_state.prescored_df.at[idx, 'Industry'] = result['industry']
                    if result.get('website') is not None:
                        st.session_state.prescored_df.at[idx, 'Website'] = result['website']
                    if result.get('linkedin') is not None:
                        st.session_state.prescored_df.at[idx, 'LinkedIn'] = result['linkedin']
                    with log_container:
                        st.write(f"   ✅ Score: {result['ai_score']}/10")
            except Exception as e:
                with log_container:
                    st.error(f"   ❌ Error: {str(e)[:200]}")
                save_all_state()
                progress_bar.progress((i + 1) / len(failed_df))
                time.sleep(0.5)
            status_text.text('✨ Reprocessing complete!')
            st.success(f'Reprocessed {len(failed_df)} companies!')
            st.rerun()

if st.session_state.prescored_df is not None:
    if st.sidebar.button('Analyze Top Companies with AI', type='primary'):
        df_display_for_ai = st.session_state.prescored_df.copy()
        if 'Prescore' in df_display_for_ai.columns:
            df_display_for_ai = df_display_for_ai.sort_values('Prescore', ascending=False)
        if top_n > 0:
            df_display_for_ai = df_display_for_ai.head(top_n)
        df = df_display_for_ai
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.expander('AI Scoring Log', expanded=False)

        for i, (idx, row) in enumerate(df.iterrows()):
                company_name = row['Company']
                status_text.text(f'Processing {i+1}/{len(df)}: {company_name[:50]}...')
                with log_container:
                    st.write(f'**[{i+1}/{len(df)}]** {company_name}')
                try:
                    # 🔹 Вызов без cases_text
                    result = score_single_company(idx, row)
                    existing_score = st.session_state.prescored_df.at[idx, 'AI_score']
                    if pd.notna(existing_score) and existing_score > 0:
                        with log_container:
                            st.write(f"   ⏭️ Already processed with score {existing_score}/10, skipped")
                    else:
                        if 'AI_score' not in st.session_state.prescored_df.columns:
                            st.session_state.prescored_df['AI_score'] = None
                        if 'Recommendation' not in st.session_state.prescored_df.columns:
                            st.session_state.prescored_df['Recommendation'] = None
                        if 'rag_cases' not in st.session_state.prescored_df.columns:
                            st.session_state.prescored_df['rag_cases'] = None
                        if 'web_context' not in st.session_state.prescored_df.columns:
                            st.session_state.prescored_df['web_context'] = None
                        st.session_state.prescored_df['rag_cases'] = st.session_state.prescored_df['rag_cases'].astype(object)
                        st.session_state.prescored_df['web_context'] = st.session_state.prescored_df['web_context'].astype(object)
                        print(f"Assigning to idx {idx}, rag_cases type {type(result.get('rag_cases', []))}")
                        st.session_state.prescored_df.at[idx, 'AI_score'] = result['ai_score']
                        st.session_state.prescored_df.at[idx, 'Recommendation'] = result['recommendation']
                        st.session_state.prescored_df.at[idx, 'rag_cases'] = json.dumps(result.get('rag_cases', []), ensure_ascii=False)
                        st.session_state.prescored_df.at[idx, 'web_context'] = result.get('web_context', '')
                        if result['industry']:
                            st.session_state.prescored_df.at[idx, 'Industry'] = result['industry']
                        if result.get('website') is not None:
                            st.session_state.prescored_df.at[idx, 'Website'] = result['website']
                        if result.get('linkedin') is not None:
                            st.session_state.prescored_df.at[idx, 'LinkedIn'] = result['linkedin']
                        with log_container:
                            st.write(f"   ✅ Score: {result['ai_score']}/10")
                except Exception as e:
                    with log_container:
                        st.error(f"   ❌ Error: {str(e)[:200]}")
                save_all_state()
                progress_bar.progress((i + 1) / len(df))
                time.sleep(0.5)
        status_text.text('✨ AI scoring complete!')
        st.success(f'{"Top" if top_n > 0 else "All"} companies scored!')
        st.rerun()

# ==================== Отображение результатов ====================

if st.session_state.prescored_df is not None:
    st.header('Pre-scoring Results')
    df_display = st.session_state.prescored_df.copy()
    if 'Prescore' in df_display.columns:
        df_display = df_display.sort_values('Prescore', ascending=False)
    if top_n > 0:
        df_display = df_display.head(top_n)
    base_columns = ['Company', 'Industry', 'Patents_number', 'Prescore', 'AI_score']
    if 'Date_of_latest_publication' in df_display.columns:
        base_columns.insert(3, 'Date_of_latest_publication')
    if 'Recommendation' in df_display.columns:
        base_columns.append('Recommendation')
    if 'Website' in df_display.columns:
        base_columns.append('Website')
    if 'LinkedIn' in df_display.columns:
        base_columns.append('LinkedIn')
    if 'Lead_Message' in df_display.columns:
        base_columns.append('Lead_Message')
    for col in base_columns:
        if col not in df_display.columns:
            df_display[col] = None
    st.dataframe(
        df_display[base_columns],
        use_container_width=True,
        height=400,
        column_config={
            "AI_score": st.column_config.NumberColumn("AI Score", format="%d/10", min_value=0, max_value=10),
            "Prescore": st.column_config.NumberColumn("Pre-Score", format="%d"),
            "Recommendation": st.column_config.TextColumn("Recommendation"),
            "Website": st.column_config.LinkColumn("Website"),
            "LinkedIn": st.column_config.LinkColumn("LinkedIn"),
            "Lead_Message": st.column_config.TextColumn("Lead Message"),
        }
    )

    st.subheader('Individual AI Scoring')

    st.markdown('### Batch Reprocessing')
    selected_companies = st.multiselect('Select companies to reprocess (optional)', options=df_display['Company'].tolist(), default=[])
    if st.button('Reprocess Selected Companies', type='secondary') and selected_companies:
        selected_indices = df_display[df_display['Company'].isin(selected_companies)].index.tolist()
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.expander('Batch Reprocessing Log', expanded=True)
        for i, idx in enumerate(selected_indices):
            row = st.session_state.prescored_df.loc[idx]
            company_name = row['Company']
            status_text.text(f'Reprocessing {i+1}/{len(selected_indices)}: {company_name[:50]}...')
            with log_container:
                st.write(f'**[{i+1}/{len(selected_indices)}]** {company_name}')
            try:
                result = score_single_company(idx, row)
                existing_score = st.session_state.prescored_df.at[idx, 'AI_score']
                if pd.notna(existing_score) and existing_score > 0:
                    with log_container:
                        st.write(f"   ⏭️ Already processed with score {existing_score}/10, skipped")
                else:
                    st.session_state.prescored_df.at[idx, 'AI_score'] = result['ai_score']
                    st.session_state.prescored_df.at[idx, 'Recommendation'] = result['recommendation']
                    st.session_state.prescored_df.at[idx, 'rag_cases'] = json.dumps(result.get('rag_cases', []), ensure_ascii=False)
                    st.session_state.prescored_df.at[idx, 'web_context'] = result.get('web_context', '')
                    if result['industry']:
                        st.session_state.prescored_df.at[idx, 'Industry'] = result['industry']
                    if result.get('website') is not None:
                        st.session_state.prescored_df.at[idx, 'Website'] = result['website']
                    if result.get('linkedin') is not None:
                        st.session_state.prescored_df.at[idx, 'LinkedIn'] = result['linkedin']
                    with log_container:
                        st.write(f"   ✅ Score: {result['ai_score']}/10")
            except Exception as e:
                with log_container:
                    st.error(f"   ❌ Error: {str(e)[:200]}")
            save_all_state()
            progress_bar.progress((i + 1) / len(selected_indices))
            time.sleep(0.5)
        status_text.text('✨ Batch reprocessing complete!')
        st.success(f'Reprocessed {len(selected_indices)} companies!')
        st.rerun()

    for idx, row in df_display.iterrows():
        with st.container(border=True):
            col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 2])
            with col1:
                st.markdown(f"**{row['Company'][:50]}**")
                st.caption(f"{row.get('Industry', 'Unknown')}")
            with col2:
                st.metric("Patents", row.get('Patents_number', '?'))
            with col3:
                st.metric("Pre", row.get('Prescore', '?'))
            with col4:
                ai_val = row.get('AI_score')
                if pd.notna(ai_val):
                    st.metric("AI", f"{int(ai_val)}/10")
                else:
                    st.write("—")
            with col5:
                if pd.isna(row.get('AI_score')):
                    if st.button('🤖 Analyze', key=f'btn_{idx}'):
                        try:
                            with st.spinner(f'Analyzing {row["Company"][:30]}...'):
                                result = score_single_company(idx, row)
                                st.session_state.prescored_df.at[idx, 'AI_score'] = result['ai_score']
                                st.session_state.prescored_df.at[idx, 'Recommendation'] = result['recommendation']
                                st.session_state.prescored_df.at[idx, 'rag_cases'] = json.dumps(result.get('rag_cases', []), ensure_ascii=False)
                                st.session_state.prescored_df.at[idx, 'web_context'] = result.get('web_context', '')
                                if result['industry']:
                                    st.session_state.prescored_df.at[idx, 'Industry'] = result['industry']
                                if result.get('website') is not None:
                                    st.session_state.prescored_df.at[idx, 'Website'] = result['website']
                                if result.get('linkedin') is not None:
                                    st.session_state.prescored_df.at[idx, 'LinkedIn'] = result['linkedin']
                                save_all_state()
                                st.success(f"Score: {result['ai_score']}/10")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)[:150]}")
                elif pd.notna(row.get('AI_score')):
                    if row.get('Recommendation'):
                        with st.popover("📝 View Recommendation"):
                            st.write(row['Recommendation'])
                    if pd.isna(row.get('Lead_Message')):
                        if st.button('Generate Lead Message', key=f'msg_{idx}'):
                            try:
                                message = generate_lead_message(row['Company'], row['Industry'], row.get('Tech_Stack_Web', ''), int(row['AI_score']), st.session_state.prescored_df.at[idx, 'web_context'] or '')
                                st.session_state.prescored_df.at[idx, 'Lead_Message'] = message
                                st.success("Lead message generated!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating message: {str(e)[:150]}")
                    else:
                        st.success("✓ Message ready")

    st.markdown('---')
    st.subheader('Export Results')
    col1, col2 = st.columns(2)
    with col1:
        csv = st.session_state.prescored_df.to_csv(index=False)
        st.download_button('📥 Download Full Results (CSV)', csv, 'patent_results.csv', 'text/csv', key='download_full')
    with col2:
        scored_df = st.session_state.prescored_df[st.session_state.prescored_df['AI_score'].notna()]
        if len(scored_df) > 0:
            csv_scored = scored_df.to_csv(index=False)
            st.download_button('📥 Download AI-Scored Only (CSV)', csv_scored, 'patent_results_ai_scored.csv', 'text/csv', key='download_scored')
        else:
            st.info('No AI-scored companies yet — run analysis first')
else:
    st.info('📤 Upload a CSV file and run pre-scoring to begin')

st.sidebar.markdown('---')
st.sidebar.markdown('### ℹ️ About')
st.sidebar.markdown("""
**Patent Lead Generation Pipeline**
Identifies potential software development clients from patent data.
**Workflow:**
1. Upload Lens.org CSV export
2. Apply filters (sector, jurisdiction, date)
3. Run pre-scoring (rule-based)
4. Run AI deep-scoring with RAG (only top-3 relevant cases)
5. Export results
**Tips:**
- Start with small test files (~100 rows)
- Use sector filters to reduce processing time
- AI scoring costs ~$0.02/company (GPT-4o-mini)
""")