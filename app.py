import streamlit as st
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import time

# 页面设置
st.set_page_config(page_title="TNF-α 亲和力预测", page_icon="💊")
st.title("🧬 亲和力预测虚拟平台")
st.markdown("### 基于随机森林的亲和力预测模型")

# 加载模型（使用缓存加速）
@st.cache_resource
def load_model():
    with st.spinner("🔄 模型加载中，首次启动可能需要几十秒，请稍候..."):
        time.sleep(2)  # 模拟依赖加载，确保稳定
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

# ========== 使用说明（展开块）==========
with st.expander("📖 使用说明", expanded=True):
    st.markdown("""
    1. 在下方输入一个分子的SMILES字符串。
    2. 点击「预测」按钮，模型将立即给出预测的pChEMBL值。
    """)

# ========== 单分子预测（主区域，输入框放在说明下方）==========
st.header("🔬 单分子预测")
smiles_input = st.text_input("SMILES 字符串", placeholder="例如: CC1=CC=C(C=C1)C(=O)O")
if st.button("🚀 预测", type="primary", use_container_width=True):
    if smiles_input:
        fp = get_fingerprint(smiles_input)
        if fp is not None:
            pred = model.predict([fp])[0]
            st.success(f"**预测 pChEMBL: {pred:.4f}**")
            st.caption("pChEMBL = -log₁₀(IC₅₀), 数值越高活性越强。")
        else:
            st.error("❌ 无效的 SMILES，请检查输入！")
    else:
        st.warning("请先输入 SMILES。")

# ========== 批量预测 ==========
st.header("📁 批量预测")
uploaded_file = st.file_uploader("上传一个包含 SMILES 列的 CSV 文件", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    smiles_col = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_col is None:
        st.error("文件中未找到 SMILES 列，请确保列名为 SMILES。")
    else:
        with st.spinner("🔥 计算中，请稍候... (一次批量预测可能需要几分钟)"):
            preds = []
            for _, smi in df[smiles_col].items():
                if pd.isna(smi):
                    preds.append(np.nan)
                    continue
                fp = get_fingerprint(smi)
                preds.append(model.predict([fp])[0] if fp is not None else np.nan)

        df["Predicted_pChEMBL"] = preds
        st.dataframe(df.head(20))
        st.download_button("📥 下载预测结果 (CSV)", df.to_csv(index=False), "predictions.csv", "text/csv")