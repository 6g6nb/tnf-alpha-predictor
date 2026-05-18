import streamlit as st
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import time

st.set_page_config(page_title="TNF-α 亲和力预测", page_icon="💊")
st.title("🧬 亲和力预测虚拟平台")
st.markdown("### 基于随机森林的亲和力预测模型")

@st.cache_resource
def load_model():
    with st.spinner("加载模型中..."):
        time.sleep(1)
        with open("2random_forest_model.pkl", "rb") as f:
            model = pickle.load(f)
    return model

model = load_model()

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    return np.array(fp)

# 使用说明（不用 expander，直接 markdown）
st.markdown("### 📖 使用说明")
st.markdown("1. 输入 SMILES 字符串\n2. 点击「预测」按钮")

# 单分子预测
st.header("🔬 单分子预测")
col1, col2 = st.columns([3, 1])
with col1:
    smiles_input = st.text_input("SMILES 字符串", placeholder="例如: CC1=CC=C(C=C1)C(=O)O", label_visibility="collapsed")
with col2:
    predict_button = st.button("🚀 预测", use_container_width=True)

if predict_button:
    if smiles_input:
        fp = get_fingerprint(smiles_input)
        if fp is not None:
            pred = model.predict([fp])[0]
            st.success(f"**预测 pChEMBL: {pred:.4f}**")
        else:
            st.error("无效的 SMILES")
    else:
        st.warning("请输入 SMILES")

# 批量预测
st.header("📁 批量预测")
uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    smiles_col = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_col is None:
        st.error("CSV 中必须有 SMILES 列")
    else:
        with st.spinner("计算中..."):
            preds = []
            for smi in df[smiles_col]:
                if pd.isna(smi):
                    preds.append(np.nan)
                    continue
                fp = get_fingerprint(smi)
                preds.append(model.predict([fp])[0] if fp is not None else np.nan)
        df["Predicted_pChEMBL"] = preds
        st.dataframe(df.head(20))
        st.download_button("下载结果", df.to_csv(index=False), "predictions.csv", "text/csv")