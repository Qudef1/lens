"""
Test script for pre-scoring pipeline
"""
import sys
from patent_pipeline import load_and_prescore, load_cases, ai_score_company

def test_prescore():
    print("=" * 60)
    print("Testing Pre-scoring Pipeline")
    print("=" * 60)

    # Test pre-scoring
    try:
        print("\n1. Loading and pre-scoring patents...")
        df = load_and_prescore('patents-2.csv', max_patents=30)

        print(f"\n✓ Pre-scoring successful!")
        print(f"  Total companies: {len(df)}")
        print(f"\nTop 5 companies by Prescore:")
        print(df.head(5)[['Company', 'Industry', 'Patents_number', 'Prescore']])

        # Test AI scoring on top 2 companies
        print("\n" + "=" * 60)
        print("Testing AI Scoring (Top 2 Companies)")
        print("=" * 60)

        cases_text = load_cases('./knowledge_base/cases')

        for idx in range(min(2, len(df))):
            row = df.iloc[idx]
            company_name = row['Company']
            industry = row['Industry']
            patent_text = f"Titles: {row['Title']}\nAbstracts: {row['Abstract']}"

            print(f"\n[{idx+1}/2] AI Scoring: {company_name[:50]}...")

            result = ai_score_company(company_name, industry, patent_text, cases_text)

            print(f"  ✓ AI Score: {result['ai_score']}/10")
            print(f"  Industry: {result['industry']}")
            print(f"  Tech Stack: {result['tech_stack']}")
            print(f"  Message preview: {result['message'][:150]}...")

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_prescore()
