import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Explain My Learning Risk", page_icon="📚", layout="wide")


# ---------- Load the saved model ----------
@st.cache_resource
def load_artifacts():
    art = joblib.load("models/risk_model.joblib")
    explainer = shap.Explainer(art["pipe"].named_steps["clf"], art["X_bg"])
    return art, explainer


art, explainer = load_artifacts()
pipe, columns = art["pipe"], art["columns"]
prep = pipe.named_steps["prep"]
feature_names = art["feature_names"]
median = art["median"]
LOW_CUT, HIGH_CUT = art["thresholds"]

# ---------- Wording helpers ----------
LABELS = {
    "failures": "Past class failures", "higher": "Plans for higher education",
    "goout": "Going out with friends", "famrel": "Family relationships",
    "internet": "Internet at home", "freetime": "Free time after school",
    "famsup": "Family support", "schoolsup": "Extra school support",
    "nursery": "Nursery attendance", "school": "School attended",
    "absences": "School absences", "studytime": "Weekly study time",
    "activities": "Extracurricular activities", "paid": "Paid extra classes",
    "traveltime": "Travel time to school", "health": "Health status",
}
YESNO = {
    "higher": ("Planning to pursue higher education", "Not planning to pursue higher education"),
    "internet": ("Having internet access at home", "Having no internet access at home"),
    "famsup": ("Receiving family educational support", "Not receiving family educational support"),
    "schoolsup": ("Receiving extra school support", "Not receiving extra school support"),
    "nursery": ("Having attended nursery school", "Not having attended nursery school"),
    "activities": ("Taking part in extracurricular activities", "Not taking part in extracurricular activities"),
    "paid": ("Taking paid extra classes", "Not taking paid extra classes"),
}
SCALE = {
    "goout": "going out with friends", "famrel": "family relationship quality",
    "freetime": "free time after school", "health": "health status",
    "studytime": "weekly study time", "traveltime": "travel time to school",
}
SUPPORT = {"schoolsup", "famsup"}
BAND_TEXT = {
    "Low": "In our test data, roughly 1 in 10 students in this band were truly at risk.",
    "Medium": "In our test data, roughly 1 in 3 students in this band were truly at risk.",
    "High": "In our test data, roughly 7 in 10 students in this band were truly at risk.",
}


def risk_level(p):
    return "High" if p >= HIGH_CUT else "Medium" if p >= LOW_CUT else "Low"


def to_raw_col(name):
    for c in columns:
        if name == c or name.startswith(c + "_"):
            return c


def describe(col, val):
    if col in YESNO:
        return YESNO[col][0] if val == "yes" else YESNO[col][1]
    if col == "failures":
        return f"Having {val} past class failure{'s' if val != 1 else ''}"
    if col == "absences":
        return f"Having {val} school absence{'s' if val != 1 else ''}"
    if col == "school":
        return "The school attended"
    if val == median[col]:
        return f"A typical level of {SCALE[col]}"
    direction = "higher" if val > median[col] else "lower"
    return f"A {direction}-than-typical level of {SCALE[col]} ({val} on a 1-5 scale)"


def explain_in_words(sv, raw, top_k=3):
    lines = []
    for name in sv[sv > 0].sort_values(ascending=False).head(top_k).index:
        col = to_raw_col(name)
        d = describe(col, raw[col])
        if col in SUPPORT:
            lines.append(("up", f"{d} appears alongside higher risk in this data, "
                                "probably because support goes to students who already need it."))
        else:
            lines.append(("up", f"{d} is increasing the model's risk estimate."))
    for name in sv[sv < 0].sort_values().head(2).index:
        col = to_raw_col(name)
        lines.append(("down", f"{describe(col, raw[col])} is lowering the model's risk estimate."))
    return lines


def contribution_chart(sv):
    by_col = {}
    for name, v in sv.items():
        col = to_raw_col(name)
        by_col[col] = by_col.get(col, 0) + v
    s = pd.Series(by_col).sort_values(key=abs, ascending=False).head(8).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh([LABELS[c] for c in s.index], s.values,
            color=["#C8432E" if v > 0 else "#2F6DB5" for v in s.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Lowers estimated risk  <-   ->  Raises estimated risk")
    fig.tight_layout()
    return fig


# ---------- Header ----------
st.title("📚 Explain My Learning Risk")
st.caption("An early-warning decision-support prototype. It estimates risk and explains why. "
           "It does not make judgments about any student.")

# ---------- Sidebar: student information ----------
st.sidebar.header("Student information")
st.sidebar.caption("Enter student details by changing values to estimate risk.")
st.sidebar.caption("The displayed result is from default values.")
nursery = st.sidebar.selectbox("Attended nursery school", ["yes", "no"])
activities = st.sidebar.selectbox("Extracurricular activities", ["yes", "no"])
higher = st.sidebar.selectbox("Plans higher education", ["yes", "no"])
internet = st.sidebar.selectbox("Internet at home", ["yes", "no"])
schoolsup = st.sidebar.selectbox("Receives extra school support", ["no", "yes"])
famsup = st.sidebar.selectbox("Receives family educational support", ["yes", "no"])
paid = st.sidebar.selectbox("Takes paid extra classes", ["no", "yes"])
school = st.sidebar.selectbox("School", ["GP", "MS"])
absences = st.sidebar.slider("School absences", 0, 75, 4)
studytime = st.sidebar.select_slider(
    "Weekly study time", options=[1, 2, 3, 4], value=2,
    format_func=lambda v: {1: "under 2 h", 2: "2-5 h", 3: "5-10 h", 4: "over 10 h"}[v])
failures = st.sidebar.slider("Past class failures", 0, 3, 0)
goout = st.sidebar.slider("Going out with friends (1 low - 5 high)", 1, 5, 3)
freetime = st.sidebar.slider("Free time after school (1 low - 5 high)", 1, 5, 3)
famrel = st.sidebar.slider("Family relationship quality (1 bad -5 excellent)", 1, 5, 4)
health = st.sidebar.slider("Health status (1 bad - 5 good)", 1, 5, 4)
traveltime = st.sidebar.slider("Travel time to school (1-4)", 1, 4, 1)

student = pd.DataFrame([{
    "school": school, "traveltime": traveltime, "studytime": studytime,
    "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
    "activities": activities, "nursery": nursery, "higher": higher,
    "internet": internet, "famrel": famrel, "freetime": freetime,
    "goout": goout, "health": health, "absences": absences,
}])[columns]

# ---------- Prediction and explanation ----------
p = pipe.predict_proba(student)[0, 1]
level = risk_level(p)
Xs = pd.DataFrame(prep.transform(student), columns=feature_names)
sv_obj = explainer(Xs)
sv = pd.Series(sv_obj.values[0], index=feature_names)

tab1, tab2, tab3 = st.tabs(["Prediction", "Why?", "What-if"])

with tab1:
    c1, c2 = st.columns(2)
    c1.metric("Estimated risk level", level)
    c2.metric("Model probability", f"{p:.0%}")
    st.progress(float(p))
    # st.write(BAND_TEXT[level])
with tab2:
    st.subheader("Why did the model give this estimate?")
    for kind, text in explain_in_words(sv, student.iloc[0]):
        st.markdown(("⚠️ " if kind == "up" else "✅ ") + text)
    st.pyplot(contribution_chart(sv))
    st.caption("These explanations describe how the model behaves. They are associations "
               "in the data, not proven causes.")
    with st.expander("Technical view (SHAP waterfall)"):
        shap.plots.waterfall(sv_obj[0], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.close("all")

with tab3:
    st.subheader("What-if view")
    st.caption("Change a factor to see how the model's estimate moves. These are model "
               "scenarios, not guarantees of what would happen in real life.")
    w1, w2, w3 = st.columns(3)
    n_abs = w1.slider("Absences", 0, 75, absences)
    n_go = w2.slider("Going out (1-5)", 1, 5, goout)
    n_st = w3.slider("Study time (1-4)", 1, 4, studytime)
    s2 = student.copy()
    s2["absences"], s2["goout"], s2["studytime"] = n_abs, n_go, n_st
    p2 = pipe.predict_proba(s2)[0, 1]
    a, b = st.columns(2)
    a.metric("Current estimate", f"{p:.0%}")
    b.metric("Scenario estimate", f"{p2:.0%}", f"{(p2 - p) * 100:+.0f} points", delta_color="inverse")
    st.caption("Study time has very little effect in this model, so moving it may change nothing.")

