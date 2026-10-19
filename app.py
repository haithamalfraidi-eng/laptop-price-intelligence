"""Run with streamlit run app.py after the batch pipeline."""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from src.data_cleaning import ROOT, CLEAN
from src.train_model import FEATURES, NUMERIC, CATEGORICAL
from src.pricing_engine import compare_laptop
from src.prediction import validate_specs

st.set_page_config(page_title='Laptop Price Intelligence',page_icon=':computer:',layout='wide')
st.title('Laptop Price Intelligence')
st.caption('Explore the dataset | Compare specifications | Estimate prices')
st.info('Portfolio research prototype. Currency and collection date are unknown. All prices are dataset units, not current market quotes.')
try:
    d=pd.read_csv(CLEAN)
    required=set(FEATURES+['Price','Model Name','CPU','Storage'])
    if d.empty or not required.issubset(d.columns):
        raise ValueError('Cleaned dataset is empty or has missing columns.')
except (OSError,ValueError,pd.errors.ParserError) as exc:
    st.error(f'Dataset unavailable: {exc}. Run python run_pipeline.py.')
    st.stop()

market,comparison,prediction,performance=st.tabs(['Market analysis','Laptop comparison','Price prediction','ML performance'])
with market:
    st.subheader('Dataset overview')
    brands=st.multiselect('Manufacturers',sorted(d.Manufacturer.dropna().unique()))
    categories=st.multiselect('Categories',sorted(d.Category.dropna().unique()))
    filtered=d.copy()
    if brands: filtered=filtered[filtered.Manufacturer.isin(brands)]
    if categories: filtered=filtered[filtered.Category.isin(categories)]
    if filtered.empty:
        st.warning('No records match these filters. Choose another combination.')
    else:
        a,b,c=st.columns(3)
        a.metric('Devices',len(filtered));b.metric('Manufacturers',filtered.Manufacturer.nunique())
        c.metric('Median price | dataset units',f'{filtered.Price.median():,.0f}')
        st.plotly_chart(px.box(filtered,x='Category',y='Price',points='all',title='Recorded price distribution',labels={'Price':'Price (dataset units)'}),width='stretch')
        st.plotly_chart(px.scatter(filtered,x='RAM_GB',y='Price',color='Category',hover_data=['Manufacturer','Model Name'],labels={'Price':'Price (dataset units)','RAM_GB':'RAM (GB)'},title='RAM and recorded prices'),width='stretch')
        st.subheader('Business insights')
        counts=filtered.Category.value_counts()
        st.write(f'{counts.index[0]} accounts for {counts.iloc[0]} of {len(filtered)} filtered records ({counts.iloc[0]/len(filtered):.1%}). This describes dataset coverage, not market share.')
        summary=filtered.groupby('Category').Price.agg(['count','median','min','max']).sort_values('median',ascending=False)
        top=summary.index[0]
        st.write(f'{top} has the highest category median in this selection: {summary.loc[top,"median"]:,.0f} dataset units, based on {int(summary.loc[top,"count"])} records. Specification mix and small groups can explain differences.')
        st.caption('These associations do not establish causation, customer demand, profitability or future price movements.')
        st.dataframe(summary.rename(columns={'count':'Records','median':'Median price','min':'Minimum','max':'Maximum'}),width='stretch')
        with st.expander('Explore records'):
            st.dataframe(filtered,hide_index=True)
with comparison:
    st.subheader('Specification comparison')
    idx=st.selectbox('Choose laptop',list(d.index),format_func=lambda i:f'{d.loc[i,"Manufacturer"]} | {d.loc[i,"Model Name"]} | row {i}')
    r=compare_laptop(d,idx)
    st.write(r['group'])
    a,b,c=st.columns(3)
    a.metric('Recorded price',f'{d.loc[idx,"Price"]:,.0f}')
    b.metric('Exact specification peers',r['peer_count'])
    c.metric('Peer median',f'{r["peer_median"]:,.0f}' if r['peer_median'] is not None else 'Insufficient peers')
    if r['relative_pct'] is not None:
        st.write(f'Recorded price is {r["relative_pct"]:+.1f}% relative to the peer median (dataset units).')
    else:
        st.warning('At least three exact specification peers are required to show a median or percentage comparison.')
    st.caption('CPU matching uses exact source text. Storage matching uses SSD/HDD and total capacities; other storage media may remain ambiguous. GPU, display, condition, seller, date and exact configuration differences remain. No bargain or overpricing judgment is supported.')
    st.write('Closest same-category alternatives, ordered by CPU match, RAM difference, storage configuration and capacity difference. These may not be equivalent devices.')
    st.dataframe(r['alternatives'],hide_index=True)

try:
    metrics=json.loads((ROOT/'reports/model_metrics.json').read_text(encoding='utf8'))
    model=joblib.load(ROOT/'models/laptop_model.joblib')
    reference=pd.read_csv(ROOT/'reports/training_reference.csv')
    scored=metrics[metrics['selected_model']]
except (OSError,ValueError,KeyError,EOFError,ImportError,IndexError) as exc:
    with prediction: st.warning(f'Model artifacts unavailable or incompatible: {exc}. Run python run_pipeline.py.')
    with performance: st.warning('Run python run_pipeline.py to generate compatible evaluation artifacts.')
else:
    with prediction:
        st.subheader('Enter laptop specifications')
        st.caption('The model uses CPU brand, not CPU generation or exact SKU; GPU is not a model feature. Options and range checks come from training rows only.')
        specs={}
        with st.form('manual_specs'):
            cols=st.columns(2)
            for i,col in enumerate(CATEGORICAL):
                specs[col]=cols[i%2].selectbox(col.replace('_',' '),sorted(reference[col].dropna().unique()))
            for i,col in enumerate(NUMERIC):
                default=float(reference[col].median())
                specs[col]=cols[i%2].number_input(col.replace('_',' '),min_value=0.0,value=default,step=0.1 if col in ['Screen_Inches','Weight_KG'] else 1.0)
            submitted=st.form_submit_button('Predict dataset price',type='primary')
        if submitted:
            try:
                frame,warnings=validate_specs(specs,reference)
                for warning in warnings: st.warning(warning)
                estimate=float(model.predict(frame)[0])
                if not np.isfinite(estimate) or estimate <= 0:
                    raise ValueError('Model produced an invalid nonpositive or nonfinite estimate.')
                st.metric('Estimated price | dataset units',f'{estimate:,.0f}')
                st.caption(f'Grouped holdout MAE: {scored["test_mae"]:,.0f} dataset units. This is aggregate error, not a confidence interval. Unseen specification combinations may be unreliable even within individual ranges.')
            except (ValueError,TypeError) as exc:
                st.error(str(exc))
        with st.expander('Predict an existing dataset device'):
            existing=st.selectbox('Dataset device',list(d.index),format_func=lambda i:f'{d.loc[i,"Manufacturer"]} {d.loc[i,"Model Name"]} | {i}')
            value=float(model.predict(d.loc[[existing],FEATURES])[0])
            st.write(f'Model estimate: {value:,.0f} | Recorded: {d.loc[existing,"Price"]:,.0f} dataset units')
            st.caption('This row may belong to training data; this example is not unbiased evaluation.')
    with performance:
        st.subheader('Held-out model evaluation')
        st.write(f'Selected model: **{metrics["selected_model"]}**')
        st.caption(metrics['evaluation'])
        cards=st.columns(4)
        for card,label,value in zip(cards,['MAE | dataset units','RMSE | dataset units','R-squared','Holdout records'],[f'{scored["test_mae"]:,.0f}',f'{scored["test_rmse"]:,.0f}',f'{scored["test_r2"]:.3f}',metrics['test_rows']]):
            card.metric(label,value)
        st.caption(f'Training records: {metrics["train_rows"]}. Overlapping manufacturer/model groups: {metrics["overlapping_model_groups"]}. Lower MAE/RMSE is better. R-squared is a regression score, not percentage accuracy, and can be negative.')
        cv=pd.DataFrame([{'Model':name,'CV MAE':item['cv_mae_mean'],'Fold SD':item['cv_mae_std']} for name,item in metrics.items() if isinstance(item,dict) and 'cv_mae_mean' in item])
        st.plotly_chart(px.bar(cv,x='Model',y='CV MAE',error_y='Fold SD',title='Training-only grouped cross-validation (fold SD, not confidence intervals)'),width='stretch')
        try:
            holdout=pd.read_csv(ROOT/'reports/holdout_predictions.csv')
            fig=px.scatter(holdout,x='actual',y='predicted',title='Holdout: actual vs predicted',labels={'actual':'Actual (dataset units)','predicted':'Predicted (dataset units)'})
            low=float(holdout[['actual','predicted']].min().min());high=float(holdout[['actual','predicted']].max().max())
            fig.add_shape(type='line',x0=low,y0=low,x1=high,y1=high,line=dict(dash='dash'))
            st.plotly_chart(fig,width='stretch')
            holdout['Residual']=holdout.actual-holdout.predicted
            st.plotly_chart(px.scatter(holdout,x='predicted',y='Residual',title='Holdout residuals (actual minus predicted)'),width='stretch')
            comparison_data=json.loads((ROOT/'reports/performance_comparison.json').read_text(encoding='utf8'))
            rows=[]
            for key,label in [('original_recorded','Original recorded row split'),('updated_same_row_split','Updated identical row split')]:
                item=comparison_data[key]; score=item[item['selected_model']]
                rows.append({'Evaluation':label,'Model':item['selected_model'],'MAE':score['test_mae'],'RMSE':score['test_rmse'],'R-squared':score['test_r2']})
            st.subheader('Comparable original and updated evaluation')
            st.dataframe(pd.DataFrame(rows),hide_index=True)
            st.caption(comparison_data['note'])
        except (OSError,ValueError,KeyError) as exc:
            st.warning(f'Evaluation charts unavailable: {exc}. Rerun the pipeline.')
        st.warning('Small dataset; unknown currency/date. Grouping reduces related-model leakage but naming variations and unmeasured hardware differences remain. No current market valuation or guarantee for new devices.')
