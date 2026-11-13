from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / 'demo_data/predictions/prophet_forecast.csv'
CHURN = ROOT / 'demo_data/predictions/churn_predictions_demo.csv'
TEST = ROOT / 'demo_data/reports/temporal_final_test_v2.json'
VALIDATION = ROOT / 'demo_data/reports/temporal_validation_v2.json'

st.set_page_config(page_title='RetailPulse | Intelligence Hub', page_icon='📈', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp {font-family:Inter,sans-serif;}
.stApp {background: #0a1020; color:#e7edf7;}
[data-testid="stSidebar"] {background:#101b2e;border-right:1px solid #23334d;}
.block-container {padding-top:2rem;max-width:1450px;}
h1,h2,h3 {letter-spacing:-.035em;}
.hero {background:linear-gradient(115deg,#142d47,#10223b 55%,#182246);border:1px solid #294363;border-radius:20px;padding:28px 32px;margin-bottom:22px;}
.hero .eyebrow {font-size:12px;color:#61dfc2;letter-spacing:.16em;font-weight:700;}
.hero h1 {font-size:35px;margin:8px 0 6px;color:#f3f7ff;}
.hero p {color:#afc2d9;margin:0;}
.metric-card {background:#132139;border:1px solid #293a56;border-radius:15px;padding:20px 18px;min-height:115px;}
.metric-card .label {color:#a6b8d2;font-size:13px;}
.metric-card .value {font-size:30px;font-weight:800;color:#f5f9ff;margin-top:12px;letter-spacing:-.04em;}
.metric-card .detail {font-size:11px;color:#7b91ad;margin-top:6px;}
.section-label {font-size:13px;color:#6ee7cb;letter-spacing:.1em;text-transform:uppercase;font-weight:700;margin-top:18px;}
[data-testid="stDataFrame"] {border:1px solid #263955;border-radius:12px;}
.stTabs [data-baseweb="tab-list"] {gap:12px;}
.stTabs [data-baseweb="tab"] {background:#122039;border-radius:9px;padding:12px 18px;}
.stTabs [aria-selected="true"] {background:#1d3c50;color:#73f3d3;}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def csv(path):
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def json_file(path):
    return json.loads(path.read_text())

missing = [p for p in (FORECAST, CHURN, TEST) if not p.is_file()]
if missing:
    st.error('Public demo files are missing. Regenerate or restore the demo_data directory.')
    for p in missing:
        st.code(str(p))
    st.stop()

forecast, churn, metrics = csv(FORECAST), csv(CHURN), json_file(TEST)
validation = json_file(VALIDATION) if VALIDATION.is_file() else None
if not {'ds','yhat'}.issubset(forecast.columns):
    st.error('Forecast file must contain ds and yhat columns.')
    st.stop()
if not {'CustomerID','ChurnProbability','Churn','PredictedChurn'}.issubset(churn.columns):
    st.error('Churn file does not have the required columns.')
    st.stop()
forecast['ds'] = pd.to_datetime(forecast['ds'], errors='coerce')
forecast = forecast.dropna(subset=['ds','yhat']).sort_values('ds')
churn['ChurnProbability'] = pd.to_numeric(churn['ChurnProbability'], errors='coerce')
churn = churn.dropna(subset=['ChurnProbability']).copy()
churn['RiskSegment'] = pd.cut(churn['ChurnProbability'], bins=[-0.001,0.4,0.7,1.001], labels=['Low','Medium','High'])

COLORS = {'teal':'#58ddc4','blue':'#71a8ff','amber':'#f5bd68','red':'#ff7f8d','muted':'#8ea3c0'}
def card(label, value, detail=''):
    st.markdown(f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div><div class="detail">{detail}</div></div>', unsafe_allow_html=True)

def chart_style(fig, height=390):
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#bbcee4'),margin=dict(l=12,r=12,t=35,b=12),height=height,hovermode='x unified',legend=dict(orientation='h',y=1.14,x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='rgba(150,180,220,.12)',zeroline=False)
    return fig

st.warning(
    "PUBLIC DEMONSTRATION — Customer records, risk scores and "
    "feature contributions are synthetic. Forecasting and churn "
    "evaluation metrics are historical aggregate results."
)

with st.sidebar:
    st.markdown('## ◈ RetailPulse')
    st.caption('ML ANALYTICS WORKSPACE')
    st.divider()
    st.markdown('**Data snapshot**')
    st.write('September 10, 2011')
    st.caption('90-day customer churn horizon')
    st.divider()
    st.markdown('**Project modules**')
    st.write('📈 F03 · Demand forecasting')
    st.write('👥 F04 · Customer churn')
    st.divider()
    st.info('PUBLIC DEMO: Customer records and feature contributions are simulated. Historical aggregate evaluation results are separate.')

st.markdown('<div class="hero"><div class="eyebrow">RETAIL INTELLIGENCE / ANALYTICS SUITE</div><h1>RetailPulse Intelligence Hub</h1><p>Demand forecasting · Customer retention · Model insights</p></div>',unsafe_allow_html=True)

k1,k2,k3,k4 = st.columns(4)
with k1: card('Synthetic demo customers',f'{len(churn):,}','Simulated interface dataset')
with k2: card('Historical test churn rate',f"{metrics['Churn_Rate']:.1%}",'Historical 90-day outcome')
with k3: card('Out-of-time ROC-AUC',f"{metrics['ROC_AUC']:.3f}",'XGBoost v2')
with k4: card('Top 20% precision',f"{metrics['Precision_at_Top20']:.1%}",'Historical test ranking metric')

st.write('')
overview, demand, customers, performance, explainability = st.tabs(['◉ Overview','↗ Demand Forecast','♙ Customer Risk','▥ Model Performance','✦ Model Explainability'])

with overview:
    st.subheader('Synthetic customer analytics overview')
    a,b = st.columns([1.05,1])
    with a:
        counts = churn['RiskSegment'].value_counts().reindex(['Low','Medium','High'],fill_value=0)
        fig = go.Figure(go.Pie(labels=counts.index.tolist(),values=counts.values,hole=.69,marker=dict(colors=[COLORS['teal'],COLORS['amber'],COLORS['red']]),textinfo='label+percent'))
        fig.update_layout(title='Customer risk distribution',showlegend=False)
        st.plotly_chart(chart_style(fig,340),width='stretch')
    with b:
        fig=go.Figure()
        fig.add_trace(go.Histogram(x=churn['ChurnProbability'],nbinsx=25,marker_color=COLORS['blue'],name='Customers'))
        fig.update_layout(title='Predicted churn probability',xaxis_title='Probability',yaxis_title='Customers',bargap=.07)
        st.plotly_chart(chart_style(fig,340),width='stretch')
    st.caption('Risk groups are illustrative thresholds: Low < 40%, Medium 40–70%, High ≥ 70%. They are not calibrated business policies.')

with demand:
    st.subheader('F03 · Demand forecasting')
    st.caption('Historical Prophet forecasts beginning December 10, 2011. Amounts are shown in dataset currency units; verify the source currency before labeling them as revenue in a specific currency.')
    f1,f2,f3=st.columns(3)
    with f1: card('Forecast days',str(len(forecast)))
    with f2: card('Average daily forecast',f"{forecast['yhat'].mean():,.0f}",'Dataset monetary units')
    with f3: card('Total predicted demand',f"{forecast['yhat'].sum():,.0f}",'Sum across forecast dates')
    fig=go.Figure()
    if {'yhat_lower','yhat_upper'}.issubset(forecast.columns):
        fig.add_trace(go.Scatter(x=forecast['ds'],y=forecast['yhat_upper'],mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=forecast['ds'],y=forecast['yhat_lower'],mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(113,168,255,.16)',name='Model interval'))
    fig.add_trace(go.Scatter(x=forecast['ds'],y=forecast['yhat'],mode='lines+markers',name='Forecast',line=dict(color=COLORS['teal'],width=3),marker=dict(size=5)))
    fig.update_layout(title='Forecast trend and uncertainty',xaxis_title='Date',yaxis_title='Forecast (dataset units)')
    st.plotly_chart(chart_style(fig,440),width='stretch')
    st.dataframe(forecast,width='stretch',hide_index=True)
    st.download_button('⬇ Download forecast CSV',forecast.to_csv(index=False).encode(),file_name='retailpulse_prophet_forecast.csv',mime='text/csv')


    # F03_ROLLING_EVALUATION_V2
    st.divider()
    st.subheader("Forecast accuracy & backtesting")
    st.caption(
        "Five historical, fixed-origin 30-day forecast windows. "
        "Lower error is better. Results are retrospective."
    )

    rolling_path = (
        ROOT / "demo_data/reports/rolling_backtest_v2.csv"
    )
    detail_path = (
        ROOT / "demo_data/predictions/rolling_backtest_v2.csv"
    )

    if rolling_path.exists() and detail_path.exists():
        rolling = pd.read_csv(rolling_path)
        details = pd.read_csv(detail_path)

        summary = (
            rolling.groupby("Model")[["MAE", "RMSE", "WAPE"]]
            .mean()
            .reset_index()
        )

        prophet_row = summary[
            summary["Model"] == "Prophet"
        ].iloc[0]

        baseline_row = summary[
            summary["Model"] == "SeasonalNaive"
        ].iloc[0]

        improvement = (
            (baseline_row["MAE"] - prophet_row["MAE"])
            / baseline_row["MAE"] * 100
        )

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            card(
                "Prophet avg MAE",
                f"{prophet_row['MAE']:,.0f}",
                "Five backtest windows"
            )

        with m2:
            card(
                "Baseline avg MAE",
                f"{baseline_row['MAE']:,.0f}",
                "Seasonal naive"
            )

        with m3:
            card(
                "Prophet avg WAPE",
                f"{prophet_row['WAPE']:.2f}%",
                "Average window WAPE"
            )

        with m4:
            card(
                "MAE improvement",
                f"{improvement:+.1f}%",
                "Relative to baseline"
            )

        st.write("")

        fig_compare = go.Figure()

        for model_name, color in [
            ("Prophet", COLORS["teal"]),
            ("SeasonalNaive", "#7EA6FF")
        ]:
            part = rolling[
                rolling["Model"] == model_name
            ].sort_values("TestStart")

            fig_compare.add_trace(go.Bar(
                x=part["TestStart"],
                y=part["MAE"],
                name=model_name,
                marker_color=color
            ))

        fig_compare.update_layout(
            title="MAE by historical test window",
            barmode="group",
            xaxis_title="Test window start",
            yaxis_title="MAE (dataset monetary units)"
        )

        st.plotly_chart(
            chart_style(fig_compare, 420),
            width='stretch'
        )

        st.subheader("Actual vs predicted sales")

        windows = sorted(
            details["TestStart"].unique().tolist()
        )

        selected_window = st.selectbox(
            "Select historical test window",
            windows,
            index=len(windows) - 1
        )

        window_data = details[
            details["TestStart"] == selected_window
        ].copy()

        window_data["ds"] = pd.to_datetime(
            window_data["ds"]
        )

        fig_actual = go.Figure()

        actual_data = window_data[
            window_data["Model"] == "Prophet"
        ]

        fig_actual.add_trace(go.Scatter(
            x=actual_data["ds"],
            y=actual_data["Actual"],
            name="Actual sales",
            mode="lines+markers",
            line=dict(color="#F8FAFC", width=2)
        ))

        for model_name, color in [
            ("Prophet", COLORS["teal"]),
            ("SeasonalNaive", "#7EA6FF")
        ]:
            part = window_data[
                window_data["Model"] == model_name
            ]

            fig_actual.add_trace(go.Scatter(
                x=part["ds"],
                y=part["Predicted"],
                name=model_name,
                mode="lines",
                line=dict(color=color, width=2)
            ))

        fig_actual.update_layout(
            title=f"30-day backtest · {selected_window}",
            xaxis_title="Date",
            yaxis_title="Daily sales (dataset monetary units)"
        )

        st.plotly_chart(
            chart_style(fig_actual, 440),
            width='stretch'
        )

        with st.expander("View detailed backtest metrics"):
            st.dataframe(
                rolling,
                width='stretch',
                hide_index=True
            )

        st.caption(
            "Prophet won four of five selected windows. "
            "Yearly seasonality was fitted using less than "
            "two years of history, which can make estimates "
            "unstable. These results are not an untouched "
            "final evaluation."
        )

    else:
        st.warning(
            "Rolling backtest files were not found. "
            "Run rolling_backtest_v2.py first."
        )

with customers:
    st.subheader('F04 · Synthetic customer risk explorer')
    st.caption('Synthetic customer records and simulated risk scores for interface demonstration only. These are not actual XGBoost predictions.')
    c1,c2,c3=st.columns([1.2,1,1])
    with c1: search=st.text_input('Find customer ID',placeholder='e.g. 12918')
    with c2: risk=st.multiselect('Risk segments',['High','Medium','Low'],default=['High','Medium','Low'])
    with c3: threshold=st.slider('Minimum predicted probability',0.0,1.0,0.0,0.05)
    subset=churn[churn['RiskSegment'].astype(str).isin(risk)&(churn['ChurnProbability']>=threshold)].copy()
    if search.strip():
        subset=subset[subset['CustomerID'].astype(str).str.replace(r'\.0$','',regex=True).str.contains(search.strip(),regex=False)]
    subset=subset.sort_values('ChurnProbability',ascending=False)
    s1,s2,s3=st.columns(3)
    with s1: card('Matching customers',f'{len(subset):,}')
    with s2: card('Average predicted risk',f"{subset['ChurnProbability'].mean():.1%}" if len(subset) else '—')
    with s3: card('High-risk customers',f"{(subset['RiskSegment']=='High').sum():,}")
    st.write('')
    display=subset[['CustomerID','RiskSegment','ChurnProbability','PredictedChurn','Churn']].copy()
    display['ChurnProbability']=display['ChurnProbability']*100
    st.dataframe(display,width='stretch',hide_index=True,column_config={'ChurnProbability':st.column_config.ProgressColumn('Predicted churn risk (%)',min_value=0,max_value=100,format='%.1f%%')})
    st.download_button('⬇ Download filtered customer CSV',subset.to_csv(index=False).encode(),file_name='retailpulse_customer_risk.csv',mime='text/csv')

with performance:
    st.subheader('F04 · Historical model evaluation')
    st.caption('Out-of-time evaluation on September 10, 2011; threshold = 0.50. Historical validation is a different snapshot.')
    p1,p2,p3,p4=st.columns(4)
    with p1: card('Accuracy',f"{metrics['Accuracy']:.2%}")
    with p2: card('Precision',f"{metrics['Precision']:.2%}")
    with p3: card('Recall',f"{metrics['Recall']:.2%}")
    with p4: card('F1 score',f"{metrics['F1']:.3f}")
    a,b=st.columns(2)
    with a:
        cm=pd.crosstab(churn['Churn'].astype(int),churn['PredictedChurn'].astype(int)).reindex(index=[0,1],columns=[0,1],fill_value=0)
        fig=go.Figure(go.Heatmap(z=cm.values,x=['Predicted retained','Predicted churn'],y=['Actual retained','Actual churn'],text=cm.values,texttemplate='%{text:,}',textfont=dict(size=21),colorscale=[[0,'#18263d'],[1,'#36bba6']],showscale=False))
        fig.update_layout(title='Final test confusion matrix')
        st.plotly_chart(chart_style(fig,370),width='stretch')
    with b:
        names=['ROC_AUC','Accuracy','Precision','Recall','F1','Precision_at_Top20']
        labels=['ROC-AUC','Accuracy','Precision','Recall','F1','Top 20% precision']
        fig=go.Figure()
        if validation:
            fig.add_trace(go.Bar(name='Validation',x=labels,y=[validation[k]*100 for k in names],marker_color=COLORS['blue']))
        fig.add_trace(go.Bar(name='Final test',x=labels,y=[metrics[k]*100 for k in names],marker_color=COLORS['teal']))
        fig.update_layout(title='Validation vs final test',barmode='group',yaxis_title='Score (%)',yaxis_range=[0,100])
        st.plotly_chart(chart_style(fig,370),width='stretch')
    st.info('The model prioritizes recall, but false positives are substantial. Precision@Top20 is useful for ranking customers; it should be compared with the observed churn rate baseline.')
    with st.expander('View complete evaluation metrics'):
        st.json(metrics)

st.divider()
st.caption('RetailPulse ML Analytics · F03 demand forecasting + F04 90-day customer churn · Historical evaluation + synthetic public interface')


# ==========================================================
# F04 V2 — MODEL EXPLAINABILITY
# ==========================================================
with explainability:
    st.header("F04 · Simulated feature contributions")

    st.caption(
        "Simulated feature contributions for interface demonstration; not genuine model SHAP explanations. "
        "Global explanations use 1,000 sampled customers "
        "generated independently for the public interface."
    )

    shap_dir = ROOT / "demo_data/reports"
    figure_dir = ROOT / "demo_data/figures"

    importance_path = (
        shap_dir / "shap_feature_importance_demo.csv"
    )
    customer_path = (
        shap_dir / "customer_shap_values_demo.csv"
    )

    if not importance_path.exists() or not customer_path.exists():
        st.warning("Simulated contribution files are missing.")
    else:
        importance = pd.read_csv(importance_path)
        customer_shap = pd.read_csv(customer_path)

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Simulated features",
                len(importance)
            )

        with b:
            st.metric(
                "Synthetic customers",
                len(customer_shap)
            )

        with c:
            st.metric(
                "Top feature",
                importance.iloc[0]["Feature"]
            )

        st.subheader("Illustrative feature importance")

        top_features = importance.head(15).sort_values(
            "ImportancePercent"
        )

        fig = px.bar(
            top_features,
            x="ImportancePercent",
            y="Feature",
            orientation="h",
            color="ImportancePercent",
            color_continuous_scale=[
                "#17304C",
                "#2DD4BF"
            ],
            labels={
                "ImportancePercent": "Simulated contribution share (%)",
                "Feature": "Feature"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=520,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, width='stretch')

        st.info(
            "Importance is calculated from random simulated contributions. "
            "These values do not measure actual model influence."
        )

        st.subheader("Simulated customer contributions")

        customer_ids = sorted(
            customer_shap["CustomerID"].unique().tolist()
        )

        selected_id = st.selectbox(
            "Select customer ID",
            customer_ids,
            format_func=lambda x: str(int(x))
        )

        selected = customer_shap[
            customer_shap["CustomerID"] == selected_id
        ].iloc[0]

        contributions = selected.drop("CustomerID")

        explanation = pd.DataFrame({
            "Feature": contributions.index,
            "ContributionValue": contributions.values.astype(float)
        })

        explanation["Magnitude"] = (
            explanation["ContributionValue"].abs()
        )

        explanation = explanation.nlargest(
            12, "Magnitude"
        ).sort_values("ContributionValue")

        explanation["Direction"] = explanation[
            "ContributionValue"
        ].apply(
            lambda x: "Positive simulated contribution"
            if x > 0 else "Negative simulated contribution"
        )

        fig_customer = px.bar(
            explanation,
            x="ContributionValue",
            y="Feature",
            color="Direction",
            orientation="h",
            color_discrete_map={
                "Positive simulated contribution": "#FB7185",
                "Negative simulated contribution": "#2DD4BF"
            }
        )

        fig_customer.update_layout(
            template="plotly_dark",
            height=470,
            margin=dict(l=10, r=10, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_customer,
            width='stretch'
        )

        st.caption(
            "These are randomly generated values, not actual SHAP explanations. "
            "They are not linked to model scores or percentage-point "
            "changes in churn probability."
        )

        st.download_button(
            "Download simulated feature importance",
            data=importance.to_csv(index=False),
            file_name="shap_feature_importance_demo.csv",
            mime="text/csv"
        )
