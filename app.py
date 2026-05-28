import streamlit as st
import os
import json
import warnings
import tempfile
from datetime import datetime

# --- 1. 基础环境配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入核心组件
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# 定义数据库存储路径
DB_PATH = "./drug_db"
os.makedirs(DB_PATH, exist_ok=True)

# --- 2. 逻辑类定义 ---

class KnowledgeManager:
    """知识库管理：负责 PDF 解析、向量化存储与检索"""
    def __init__(self, model_name, db_path):
        # HuggingFace 模式：直接传入模型名称，库会自动从云端下载并缓存
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, file_path):
        """解析单份 PDF 并存入向量库"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        splits = splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, query):
        """根据查询内容检索最相关的知识片段"""
        results = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([res.page_content for res in results])

class PharmacyAgent:
    """AI 药师：负责逻辑推理与报告生成"""
    def __init__(self, api_key, base_url="https://api.deepseek.com"):
        clean_key = str(api_key).strip()
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=clean_key,
            base_url=base_url,
            temperature=0
        )

    def audit(self, prescription_json, context):
        system_instruction = """你是一位极其严谨的资深临床药师。请根据提供的【参考资料】对【处方数据】进行审核。
        
        ### 核心逻辑：
        1. **儿科专项复核**：针对 <18岁患者，必须识别 weight(体重) 并核算剂量。
           - 必须列出计算公式。示例：[推荐标准(mg/kg)] × [患者体重(kg)] = [剂量]。
        2. **社保合规审核**：核对 diagnosis(诊断) 是否符合资料中的“医保支付限定范围”。
           - 若不符（如阿奇霉素限肺炎，处方诊断为感冒），必须提示“医保拒付风险”。
        3. **用法用量及安全性**：检查频次、途径及禁忌。

        请输出：【剂量复核】(含公式)、【社保合规核对】、【药学结论】、【修改建议】。"""

        prompt = ChatPromptTemplate.from_template(system_instruction + "\n\n资料: {context}\n\n处方: {prescription}")
        chain = prompt | self.llm
        
        return chain.invoke({
            "context": context,
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

# --- 3. 资源缓存初始化 ---

@st.cache_resource
def get_knowledge_manager():
    """初始化知识库 (HuggingFace 自动下载模式)"""
    # 直接使用模型 ID，Streamlit Cloud 会自动下载
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return KnowledgeManager(model_name, DB_PATH)

# --- 4. Streamlit UI 界面 ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="⚖️")
    
    km = get_knowledge_manager()

    # --- 侧边栏：登录与资料管理 ---
    st.sidebar.title("🔐 系统管理")
    
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ""

    input_key = st.sidebar.text_input(
        "请输入 DeepSeek API Key:", 
        type="password", 
        placeholder="sk-...",
        value=st.session_state['api_key']
    )

    if not input_key:
        st.sidebar.warning("请输入 API Key 登录系统。")
        st.stop()
    else:
        st.session_state['api_key'] = input_key
        agent = PharmacyAgent(st.session_state['api_key'])

    st.sidebar.markdown("---")
    st.sidebar.header("📂 核心资料库")
    
    uploaded_files = st.sidebar.file_uploader(
        "上传 PDF 资料 (多选)", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.sidebar.button("✨ 批量学习资料"):
            success_count = 0
            progress_bar = st.sidebar.progress(0)
            for i, file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                try:
                    km.upload_docs(tmp_path)
                    success_count += 1
                finally:
                    if os.path.exists(tmp_path): os.unlink(tmp_path)
                progress_bar.progress((i + 1) / len(uploaded_files))
            st.sidebar.success(f"✅ 已导入 {success_count} 份资料！")
            st.rerun()

    if st.sidebar.button("🚪 退出系统"):
        st.session_state['api_key'] = ""
        st.rerun()

    # --- 主界面 ---
    st.title("🏥 药剂科 AI 处方审核平台")
    st.caption("增强版：支持儿科剂量复核与医保合规核对")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1.3])

    with col_in:
        st.subheader("📋 录入处方信息")
        with st.form("audit_form"):
            r1 = st.columns(2)
            age = r1[0].number_input("年龄", value=5, min_value=0)
            weight = r1[1].number_input("体重 (kg)", value=18.0, step=0.1)
            
            r2 = st.columns(2)
            diagnosis = r2[0].text_input("临床诊断", value="急性支气管炎")
            insurance = r2[1].selectbox("医保类型", ["统筹医保", "自费", "门诊大病"])
            
            st.markdown("---")
            med_name = st.text_input("药品名称", value="阿奇霉素")
            dosage = st.text_input("单次剂量", value="250mg")
            freq = st.text_input("给药频次", value="一日一次")
            btn = st.form_submit_button("🧪 提交审核")

    with col_out:
        st.subheader("📝 审核报告")
        if btn:
            prescription_data = {
                "patient": {"age": age, "weight": weight, "diagnosis": diagnosis, "insurance_type": insurance},
                "medications": [{"name": med_name, "dosage": dosage, "frequency": freq}]
            }
            with st.spinner("AI 分析中..."):
                context_info = km.retrieve_context(f"{med_name} 说明书 医保政策 报销规定")
                try:
                    report_content = agent.audit(prescription_data, context_info)
                    
                    # 修复 SyntaxError: 避免在 f-string 中直接使用反斜杠
                    html_report = report_content.replace('\n', '<br>')
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff;">
                        {html_report}
                    </div>
                    """, unsafe_allow_html=True)
                    st.download_button("📥 导出报告", report_content, file_name=f"报告_{med_name}.txt")
                except Exception as e:
                    st.error(f"审核出错: {e}")
