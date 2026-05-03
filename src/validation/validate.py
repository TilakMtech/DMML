"""
Data validation module for the RecoMart recommendation pipeline.

This module validates raw ingested datasets before downstream processing.
It checks schema completeness, missing values, duplicate records, and value
ranges such as rating validity.

Key responsibilities:
- Validate user interaction data
- Validate product metadata
- Detect missing, duplicate, and invalid records
- Generate data quality reports
- Prevent poor-quality data from entering later pipeline stages
"""
from __future__ import annotations
from pathlib import Path
import glob, pandas as pd, numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from src.utils.common import ROOT, load_config, setup_logger, save_json, utc_now
LOG = setup_logger('validation', 'logs/validation.log')

def latest(pattern):
    files = sorted(glob.glob(str(ROOT / pattern)))
    if not files: raise FileNotFoundError(pattern)
    return Path(files[-1])

def validate_interactions(path: Path, cfg: dict):
    """
    Validates raw user interaction data.

    Args:
        path: Path to the raw interaction dataset.
        cfg: Pipeline configuration containing validation rules.

    Returns:
        Tuple containing the validated DataFrame and validation issue summary.
    """
    df = pd.read_csv(path)
    issues=[]
    required = cfg['validation']['required_interaction_columns']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols: issues.append({'check':'schema', 'severity':'critical', 'detail':f'missing columns {missing_cols}', 'count':len(missing_cols)})
    dupes = int(df.duplicated().sum());
    if dupes: issues.append({'check':'duplicates','severity':'warning','detail':'duplicate interaction rows','count':dupes})
    miss = df[required].isna().sum().to_dict()
    for k,v in miss.items():
        if v and k not in ['rating']:
            issues.append({'check':'missing','severity':'error','detail':k,'count':int(v)})
    parsed_ts = pd.to_datetime(df.get('event_ts'), errors='coerce', utc=True)
    bad_ts = int(parsed_ts.isna().sum())
    if bad_ts: issues.append({'check':'format','severity':'error','detail':'invalid event_ts','count':bad_ts})
    bad_price = int((pd.to_numeric(df.get('price'), errors='coerce') < 0).sum())
    if bad_price: issues.append({'check':'range','severity':'error','detail':'negative price','count':bad_price})
    ratings = pd.to_numeric(df.get('rating'), errors='coerce').dropna()
    bad_rating = int(((ratings < cfg['validation']['rating_min']) | (ratings > cfg['validation']['rating_max'])).sum())
    if bad_rating: issues.append({'check':'range','severity':'error','detail':'rating outside 1-5','count':bad_rating})
    return df, issues

def validate_products(path: Path, cfg: dict):
    """
    Validates raw product metadata.

    Args:
        path: Path to the raw product metadata dataset.
        cfg: Pipeline configuration containing validation rules.

    Returns:
        Tuple containing the validated DataFrame and validation issue summary.
    """
    df = pd.read_json(path)
    issues=[]
    required = cfg['validation']['required_product_columns']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols: issues.append({'check':'schema','severity':'critical','detail':f'missing columns {missing_cols}','count':len(missing_cols)})
    dupes = int(df.duplicated('item_id').sum())
    if dupes: issues.append({'check':'duplicates','severity':'error','detail':'duplicate product item_id','count':dupes})
    for col in ['sentiment_score','popularity_score']:
        vals = pd.to_numeric(df.get(col), errors='coerce')
        bad = int(((vals < 0) | (vals > 1) | vals.isna()).sum())
        if bad: issues.append({'check':'range','severity':'error','detail':f'{col} outside 0-1/null','count':bad})
    return df, issues

def make_pdf(summary, issues, out='reports/data_quality_report.pdf'):
    path = ROOT / out; path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path)); styles=getSampleStyleSheet(); story=[]
    story.append(Paragraph('RecoMart Data Quality Report', styles['Title']))
    story.append(Paragraph(f"Generated: {summary['run_ts']}", styles['Normal'])); story.append(Spacer(1,12))
    rows=[['Dataset','Rows','Columns','Issues']]
    for ds, stats in summary['datasets'].items(): rows.append([ds, stats['rows'], stats['columns'], stats['issues']])
    tbl=Table(rows); tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),.5,colors.grey)])); story.append(tbl); story.append(Spacer(1,12))
    story.append(Paragraph('Validation Issues', styles['Heading2']))
    issue_rows=[['Dataset','Check','Severity','Detail','Count']] + [[i['dataset'],i['check'],i['severity'],i['detail'],i['count']] for i in issues]
    tbl2=Table(issue_rows, repeatRows=1); tbl2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),.4,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP')]))
    story.append(tbl2); doc.build(story); return out

def main():
    """
    Executes the validation stage of the pipeline.

    Reads the latest raw datasets, applies quality checks, writes validation
    reports, and logs validation outcomes.
    """
    cfg=load_config()
    ipath=latest('data/raw/source=csv/type=interactions/dt=*/hour=*/interactions.csv')
    ppath=latest('data/raw/source=rest_api/type=products/dt=*/hour=*/products.json')
    idf, ii=validate_interactions(ipath, cfg); pdf, pi=validate_products(ppath, cfg)
    issues=[dict(dataset='interactions', **x) for x in ii] + [dict(dataset='products', **x) for x in pi]
    summary={'run_ts':utc_now(),'datasets':{'interactions':{'rows':len(idf),'columns':len(idf.columns),'issues':len(ii)},'products':{'rows':len(pdf),'columns':len(pdf.columns),'issues':len(pi)}},'issues':issues}
    save_json(summary, 'reports/data_quality_report.json')
    make_pdf(summary, issues)
    LOG.info('Validation complete issues=%s', len(issues))
    return summary
if __name__ == '__main__': main()
