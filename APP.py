import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from datetime import datetime

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

X_test = pd.read_excel('X_test.xlsx')

feature_names = ['Age', 'BMI', 'SBP', 'DBP', 'FPG', 'Chol', 'Tri', 'HDL', 'LDL', 'ALT', 'BUN', 'CCR', 'FFPG', 'smoking', 'drinking']
# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "Age": {"type": "numerical", "min": 18, "max": 100, "default": 30},
    "BMI": {"type": "numerical", "min": 10.0, "max": 50.0, "default": 24.0},
    "SBP": {"type": "numerical", "min": 50, "max": 200, "default": 120},
    "DBP": {"type": "numerical", "min": 30, "max": 120, "default": 80},
    "FPG": {"type": "numerical", "min": 0.0, "max": 20.0, "default": 5.0},
    "Chol": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 4.5},
    "Tri": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.0},
    "HDL": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.5},
    "LDL": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 3.0},
    "ALT": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 30.0},
    "BUN": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 20.0},
    "CCR": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 50.0},
    "FFPG": {"type": "numerical", "min": 0.0, "max": 20.0, "default": 5.0},
    "smoking": {"type": "categorical", "options": [1, 2, 3]},
    "drinking": {"type": "categorical", "options": [1, 2, 3]},
}

# Streamlit 界面
st.title("Diabetes Insight")

# 页面选择
page = st.selectbox("Select a page", ["Model Training", "Online Prediction", "SHAP Force Plot", "LIME Explanation", "History"])

# 页面1：模型训练
if page == "Model Training":
    st.header("Model Training")
    # 四个按钮分别显示对应的图片
    button_1 = st.button("Show Feature Selection Image")
    button_2 = st.button("Show Model Evaluation Image")
    button_3 = st.button("Show ROC Curve Image")
    button_4 = st.button("Show Confusion Matrix Image")

    # 根据按钮点击情况显示对应图片
    if button_1:
        st.image("Feature Selection.png", caption="Feature Selection")
    elif button_2:
        st.image("Model Evaluation.png", caption="Model Evaluation")
    elif button_3:
        st.image("ROC Curve.png", caption="ROC Curve")
    elif button_4:
        st.image("Confusion Matrix.png", caption="Confusion Matrix")

# 页面2：在线预测
elif page == "Online Prediction":
    st.header("Online Prediction")
    st.markdown("""
    This application uses a Random Forest model to predict diabetes based on several health features such as age, BMI, blood pressure, cholesterol levels, etc. The features and their corresponding values are dynamically entered by the user, and the model's prediction is displayed along with SHAP values to explain the model's decision.

    The following functionalities are implemented:
    1. **Explanation of Variables**: An "Explain Variables" button provides an explanation for each feature used in the model.
    2. **Feature Inputs**: Users can input health data through number inputs and select boxes.
    3. **Prediction**: Upon pressing "Predict", the model computes the diabetes risk percentage.
    """)

    # 解释变量按钮
    if "show_explanation" not in st.session_state:
        st.session_state.show_explanation = False

    if st.button("Explain Variables"):
        st.session_state.show_explanation = not st.session_state.show_explanation

    # 根据按钮状态显示或隐藏解释
    if st.session_state.show_explanation:
        st.markdown("""
        **Variable Explanations**:
        - **Age**: The age of the individual (in years).
        - **BMI**: Body Mass Index, a measure of body fat based on height and weight.
        - **SBP**: Systolic blood pressure, the upper number in a blood pressure reading.
        - **DBP**: Diastolic blood pressure, the lower number in a blood pressure reading.
        - **FPG**: Fasting plasma glucose, a test to measure the glucose levels after an overnight fast.
        - **Chol**: Total cholesterol level in the blood.
        - **Tri**: Triglyceride level, a type of fat found in the blood.
        - **HDL**: High-density lipoprotein cholesterol, "good" cholesterol.
        - **LDL**: Low-density lipoprotein cholesterol, "bad" cholesterol.
        - **ALT**: Alanine aminotransferase, an enzyme found in the liver, used to monitor liver health.
        - **BUN**: Blood urea nitrogen, a test for kidney function.
        - **CCR**: Creatinine clearance rate, an estimate of kidney function.
        - **FFPG**: Fasting free fatty acid levels, a measure of fatty acid metabolism.
        - **Smoking**: Smoking status (1 = currently smoking, 2 = previously smoked, 3 = never smoked).
        - **Drinking**: Drinking status (1 = currently drinking, 2 = previously drank, 3 = never drank).
        """)

    # 动态生成输入项
    st.header("Enter the following feature values:")
    feature_values = []
    for feature, properties in feature_ranges.items():
        if properties["type"] == "numerical":
            value = st.number_input(
                label=f"{feature} ({properties['min']} - {properties['max']})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
            )
        elif properties["type"] == "categorical":
            if feature == "smoking":
                value = st.selectbox(
                    label="Smoking Status (1 = currently smoking, 2 = previously smoked, 3 = never smoked)",
                    options=properties["options"],
                )
            elif feature == "drinking":
                value = st.selectbox(
                    label="Drinking Status (1 = currently drinking, 2 = previously drank, 3 = never drank)",
                    options=properties["options"],
                )
            else:
                value = st.selectbox(
                    label=f"{feature} (Select a value)",
                    options=properties["options"],
                )
        feature_values.append(value)

    # 转换为模型输入格式
    features = np.array([feature_values])

    if st.button("Predict"):
        # 模型预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # 提取患病的概率
        probability = predicted_proba[1] * 100  # 选择患病的概率

        # 保存预测结果到 session_state
        st.session_state.prediction = f"Based on feature values, predicted possibility of having diabetes is {probability:.2f}%"

        # 保存历史记录并附加时间戳
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "features": feature_values,
            "prediction": st.session_state.prediction,
            "timestamp": datetime.now()  # 当前时间戳
        })

    # 显示预测结果为文字
    if 'prediction' in st.session_state:
        st.write(st.session_state.prediction)

# 页面3：SHAP力图解释
elif page == "SHAP Force Plot":
    st.header("SHAP Force Plot")
    feature_values = []
    for feature, properties in feature_ranges.items():
        if properties["type"] == "numerical":
            value = st.number_input(
                label=f"{feature} ({properties['min']} - {properties['max']})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
            )
        elif properties["type"] == "categorical":
            if feature == "smoking":
                value = st.selectbox(
                    label="Smoking Status (1 = currently smoking, 2 = previously smoked, 3 = never smoked)",
                    options=properties["options"],
                )
            elif feature == "drinking":
                value = st.selectbox(
                    label="Drinking Status (1 = currently drinking, 2 = previously drank, 3 = never drank)",
                    options=properties["options"],
                )
            else:
                value = st.selectbox(
                    label=f"{feature} (Select a value)",
                    options=properties["options"],
                )
        feature_values.append(value)

    features = np.array([feature_values])

    # 计算 SHAP 值
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 获取当前预测类别的SHAP值
    class_index = model.predict(features)[0]
    if class_index == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values], columns=feature_ranges.keys()), matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_ranges.keys()), matplotlib=True)

    # 保存并显示 SHAP 力图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

# 页面4：LIME解释
elif page == "LIME Explanation":
    st.header("LIME Explanation")

    # 获取用户输入的特征值
    feature_values = []
    for feature, properties in feature_ranges.items():
        if properties["type"] == "numerical":
            value = st.number_input(
                label=f"{feature} ({properties['min']} - {properties['max']})",
                min_value=float(properties["min"]),
                max_value=float(properties["max"]),
                value=float(properties["default"]),
            )
        elif properties["type"] == "categorical":
            if feature == "smoking":
                value = st.selectbox(
                    label="Smoking Status (1 = currently smoking, 2 = previously smoked, 3 = never smoked)",
                    options=properties["options"],
                )
            elif feature == "drinking":
                value = st.selectbox(
                    label="Drinking Status (1 = currently drinking, 2 = previously drank, 3 = never drank)",
                    options=properties["options"],
                )
            else:
                value = st.selectbox(
                    label=f"{feature} (Select a value)",
                    options=properties["options"],
                )
        feature_values.append(value)

    features = np.array([feature_values])

    # LIME 解释
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,  # 使用训练数据集
        feature_names=X_test.columns.tolist(),
        class_names=['Not sick', 'Sick'],
        mode='classification'
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )

    # 显示 LIME 解释
    lime_html = lime_exp.as_html(show_table=False)  # 不显示特征值表
    st.components.v1.html(lime_html, height=800, scrolling=True)


# 页面5：历史记录
elif page == "History":
    st.header("Prediction History")

    if "history" in st.session_state and st.session_state.history:
        for record in st.session_state.history:
            st.subheader("Prediction")
            st.write(record["prediction"])
            st.write("Features:")
            for i, feature in enumerate(feature_ranges.keys()):
                st.write(f"{feature}: {record['features'][i]}")

            # 格式化并显示时间戳（精确到时分秒）
            timestamp = record["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"Prediction Time: {timestamp}")

            st.markdown("---")
    else:
        st.write("No predictions yet.")

