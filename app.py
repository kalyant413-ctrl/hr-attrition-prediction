import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="centered")

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 28px 32px; border-radius: 16px; color: white; margin-bottom: 20px;
}
.hero h1 { color: white; margin: 0; font-size: 2rem; }
.hero p { margin: 6px 0 0 0; opacity: 0.9; }
.risk-card { padding: 24px; border-radius: 16px; text-align: center; margin: 10px 0 16px 0; }
.risk-card h2 { margin: 0; font-size: 2.6rem; }
.risk-card p { margin: 4px 0 0 0; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

model = joblib.load("model.pkl")
opt = joblib.load("options.pkl")

st.markdown("""
<div class="hero">
  <h1>👥 Employee Attrition Predictor</h1>
  <p>Estimate how likely an employee is to leave, using a machine learning model trained on 311 employees.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar inputs ----------
st.sidebar.header("Employee details")
salary = st.sidebar.number_input("Salary", min_value=opt["salary_min"],
                                 max_value=opt["salary_max"] + 50000, value=65000, step=1000)
dept = st.sidebar.selectbox("Department", opt["departments"])
manager = st.sidebar.selectbox("Manager", opt["managers"])
source = st.sidebar.selectbox("Recruitment source", opt["sources"])
perf = st.sidebar.selectbox("Performance score", opt["perf_scores"])
engagement = st.sidebar.slider("Engagement survey score", 1.0, 5.0, 4.0, 0.1)
satisfaction = st.sidebar.slider("Employee satisfaction (1-5)", 1, 5, 4)
projects = st.sidebar.number_input("Special projects count", 0, 10, 0)
days_late = st.sidebar.number_input("Days late in last 30 days", 0, 30, 0)
absences = st.sidebar.number_input("Absences", 0, 30, 5)
predict = st.sidebar.button("🔮 Predict", use_container_width=True, type="primary")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Key Insights", "🧠 About the Model"])

# ---------- Tab 1: Prediction ----------
with tab1:
    if not predict:
        st.info("👈 Enter the employee's details in the sidebar and click **Predict**.")
    else:
        row = pd.DataFrame([{
            "Salary": salary, "EngagementSurvey": engagement, "EmpSatisfaction": satisfaction,
            "SpecialProjectsCount": projects, "DaysLateLast30": days_late, "Absences": absences,
            "Department": dept, "RecruitmentSource": source,
            "ManagerName": manager, "PerformanceScore": perf,
        }])
        prob = float(model.predict_proba(row)[0][1])

        if prob < 0.33:
            label, color, bg, tip = "LOW RISK", "#15803d", "#dcfce7", \
                "Below the company average attrition of 33%."
        elif prob < 0.50:
            label, color, bg, tip = "MODERATE RISK", "#b45309", "#fef3c7", \
                "Around or above the company average of 33%. Worth keeping an eye on."
        else:
            label, color, bg, tip = "HIGH RISK", "#b91c1c", "#fee2e2", \
                "Well above the company average of 33%. Consider a check-in conversation."

        st.markdown(
            f'<div class="risk-card" style="background:{bg};color:{color};">'
            f'<p>{label}</p><h2>{prob:.0%}</h2><p>estimated probability of leaving</p></div>',
            unsafe_allow_html=True)
        st.progress(prob)
        st.write(tip)

        c1, c2, c3 = st.columns(3)
        c1.metric("Engagement", f"{engagement:.1f} / 5")
        c2.metric("Absences", absences)
        c3.metric("Salary", f"${salary:,}")

        st.caption("This is a screening estimate from a small dataset (about 70% accuracy). "
                   "Use it to start a conversation, not to make decisions about people.")

# ---------- Tab 2: Insights ----------
with tab2:
    st.subheader("What the data showed")
    m1, m2, m3 = st.columns(3)
    m1.metric("Employees", "311")
    m2.metric("Left", "104")
    m3.metric("Attrition rate", "33%")

    st.markdown("**Attrition rate by recruitment source (%)**")
    src = pd.DataFrame({
        "Source": ["Google Search", "Diversity Job Fair", "CareerBuilder", "Indeed",
                   "LinkedIn", "Employee Referral", "Website"],
        "Attrition %": [61.2, 55.2, 47.8, 24.1, 23.7, 16.1, 7.7],
    }).set_index("Source")
    st.bar_chart(src)

    st.markdown("""
- **Production** has the most leavers (about 40% attrition).
- **Employee referrals and the company website** keep their hires best.
- **Three managers** account for about 37% of all leavers.
- Engagement, satisfaction and absences alone are only weak predictors of leaving.
    """)

# ---------- Tab 3: About the model ----------
with tab3:
    st.subheader("Model comparison")
    st.write("Four models were trained on 248 employees and tested on 63. "
             "**SVM** was selected because it had the best cross-validated F1 score.")
    res = pd.DataFrame({
        "Model": ["SVM", "Logistic Regression", "Random Forest", "Decision Tree"],
        "Accuracy": [0.698, 0.635, 0.698, 0.635],
        "Recall": [0.667, 0.667, 0.619, 0.381],
        "F1": [0.596, 0.549, 0.578, 0.410],
        "CV F1 (5-fold)": [0.572, 0.568, 0.528, 0.399],
    }).set_index("Model")
    st.dataframe(res, use_container_width=True)

    st.markdown("""
**Limitations**
- Small dataset (311 employees), so results are indicative, not exact.
- Gender, race and marital status were deliberately excluded from the model.
- The model finds patterns, not causes.
    """)