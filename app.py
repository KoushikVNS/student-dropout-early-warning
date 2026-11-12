import streamlit as st
import pandas as pd
import joblib
import shap

FEATURES = ['total_clicks', 'active_days', 'days_inactive', 'avg_score',
            'num_submitted', 'num_late', 'num_of_prev_attempts', 'studied_credits']

NICE = {
    'total_clicks': 'Total clicks (first 60 days)',
    'active_days': 'Active days',
    'days_inactive': 'Days since last activity',
    'avg_score': 'Average assignment score',
    'num_submitted': 'Assignments submitted',
    'num_late': 'Late submissions',
    'num_of_prev_attempts': 'Previous attempts',
    'studied_credits': 'Studied credits',
}

# Slider limits for the What-if page: (min, max, step), taken from describe() in the notebook
LIMITS = {
    'total_clicks': (0, 5000, 10),
    'active_days': (0, 90, 1),
    'days_inactive': (0, 85, 1),
    'avg_score': (0, 100, 1),
    'num_submitted': (0, 4, 1),
    'num_late': (0, 4, 1),
    'num_of_prev_attempts': (0, 6, 1),
    'studied_credits': (30, 630, 5),
}

st.set_page_config(page_title='Student Early-Warning Dashboard', page_icon='🎓', layout='wide')


@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    explainer = shap.TreeExplainer(model)
    return model, explainer


model, explainer = load_model()


def risk_level(p):
    if p >= 0.6:
        return '🔴 High'
    if p >= 0.3:
        return '🟡 Medium'
    return '🟢 Low'


# ---------- Sidebar ----------
st.sidebar.title('🎓 Early-Warning')
page = st.sidebar.radio('Go to', ['Class overview', 'Student detail', 'What-if simulator'])
uploaded = st.sidebar.file_uploader('Upload class CSV', type='csv')
st.sidebar.caption('No file uploaded? A sample class of 150 students is used.')

data = pd.read_csv(uploaded) if uploaded else pd.read_csv('sample_class.csv')

missing = [c for c in FEATURES if c not in data.columns]
if missing:
    st.error(f'Your CSV is missing these columns: {missing}')
    st.stop()
if 'student_id' not in data.columns:
    data.insert(0, 'student_id', range(1, len(data) + 1))

data['risk'] = model.predict_proba(data[FEATURES])[:, 1]
data['level'] = data['risk'].apply(risk_level)
data['risk_pct'] = (data['risk'] * 100).round(1)

# ---------- Page 1: Class overview ----------
if page == 'Class overview':
    st.title('Class overview')
    c1, c2, c3 = st.columns(3)
    c1.metric('Students', len(data))
    c2.metric('High-risk students', int((data['risk'] >= 0.6).sum()))
    c3.metric('Average risk', f"{data['risk'].mean():.0%}")

    table = data.sort_values('risk', ascending=False)[['student_id', 'level', 'risk_pct'] + FEATURES]
    st.dataframe(
        table.rename(columns=NICE),
        hide_index=True,
        column_config={
            'risk_pct': st.column_config.ProgressColumn('Risk', format='%.0f%%', min_value=0, max_value=100),
            'level': 'Level',
            'student_id': 'Student ID',
        },
    )

# ---------- Page 2: Student detail ----------
elif page == 'Student detail':
    st.title('Student detail')
    order = data.sort_values('risk', ascending=False).index
    idx = st.selectbox(
        'Choose a student (highest risk first)', order,
        format_func=lambda i: f"Student {data.loc[i, 'student_id']} ({data.loc[i, 'risk']:.0%} risk)")

    row = data.loc[[idx], FEATURES]
    p = data.loc[idx, 'risk']
    st.subheader(f'{risk_level(p)} risk: {p:.0%}')

    sv = explainer(row)
    values = sv.values[0]
    contrib = pd.Series(values, index=[NICE[f] for f in FEATURES]).sort_values(ascending=False)

    st.markdown('#### Why? (SHAP explanation)')
    st.caption('Positive bars push the risk up, negative bars push it down.')
    st.bar_chart(contrib, horizontal=True, sort=False)   # sort=False keeps the biggest push-up on top

    st.markdown('#### Main reasons')
    top = pd.Series(values, index=FEATURES).sort_values(ascending=False).head(3)
    reasons = [(f, val) for f, val in top.items() if val > 0]
    for f, val in reasons:
        st.write(f'⬆️ **{NICE[f]}** is {row.iloc[0][f]:.0f}, which increases risk')
    if not reasons:
        st.write('✅ Nothing in this student\'s first 60 days pushes their risk up.')

    st.markdown('#### Student data')
    st.dataframe(row.rename(columns=NICE), hide_index=True)

# ---------- Page 3: What-if simulator ----------
else:
    st.title('What-if simulator')
    st.caption("Pick a student, then move the sliders to see how a change in behaviour changes their risk.")

    order = data.sort_values('risk', ascending=False).index
    idx = st.selectbox(
        'Start from student (highest risk first)', order,
        format_func=lambda i: f"Student {data.loc[i, 'student_id']} ({data.loc[i, 'risk']:.0%} risk)")
    # the student's own values, rounded and kept inside the slider limits
    start = {}
    for f in FEATURES:
        lo, hi, step = LIMITS[f]
        start[f] = int(min(max(round(data.loc[idx, f]), lo), hi))
    p0 = float(model.predict_proba(pd.DataFrame([start], columns=FEATURES))[0, 1])

    values = {}
    c1, c2 = st.columns(2)
    for i, f in enumerate(FEATURES):
        lo, hi, step = LIMITS[f]
        with (c1 if i < 4 else c2):
            # the key includes the student, so the sliders reset when you pick another student
            values[f] = st.slider(NICE[f], lo, hi, start[f], step=step, key=f'{f}_{idx}')

    x = pd.DataFrame([values], columns=FEATURES)
    p = float(model.predict_proba(x)[0, 1])

    st.subheader(f'{risk_level(p)} risk: {p:.0%}')
    change = round((p - p0) * 100)
    if change == 0:
        st.metric('Risk now', f'{p:.0%}', delta=f'No change from starting {p0:.0%}',
                  delta_color='off', delta_arrow='off')
    else:
        st.metric('Risk now', f'{p:.0%}', delta=f'{change:+d} points vs. starting {p0:.0%}',
                  delta_color='inverse')
    st.progress(p)
