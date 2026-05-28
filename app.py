import streamlit as st
import os
import json
import warnings
import tempfile
from datetime import datetime

# --- 1. 基础环境配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

DB_PATH = "./drug_db"
os.makedirs(DB_PATH, exist_ok=True)

# --- 2. 逻辑类定义 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        print(f"[{datetime.now()}] 📦 初始化检索模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, query):
        results = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([res.page_content for res in results])

class PharmacyAgent:
    def __init__(self, api_key, base_url):
        # 清理 Key 中的空格或特殊换行
        clean_key = str(api_key).strip()
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=clean_key,
            base_url=base_url,
            temperature=0
        )

    def audit(self, prescription_json, context):
        system_prompt = """你是一位资深临床药师。请根据【参考资料】审核【处方数据】。
        必须核算儿科剂量(weight)并检查社保合规性(diagnosis)。"""
        prompt = ChatPromptTemplate.from_template(system_prompt + "\n\n资料: {context}\n处方: {prescription}")
        chain = prompt | self.llm
        return chain.invoke({
            "context": context,
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

# --- 3. 缓存初始化 (将模型与 Agent 分开初始化) ---

@st.cache_resource
def get_knowledge_manager():
    # 检索模型是公共的，可以预加载
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return KnowledgeManager(model_name, DB_PATH)

# --- 4. Streamlit UI 界面 ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="💊")
    
    # 初始化知识库
    km = get_knowledge_manager()

    # --- 登录逻辑 ---
    st.sidebar.title("🔐 系统登录")
    
    # 检查 session_state 中是否有 key
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ""

    # 在侧边栏输入 Key
    input_key = st.sidebar.text_input(
        "请输入 DeepSeek API Key:", 
        type="password", 
        placeholder="sk-...",
        value=st.session_state['api_key']
    )

    if not input_key:
        st.info("👋 欢迎！请在侧边栏输入您的 DeepSeek API Key 以激活 AI 药师。")
        st.stop()  # 停止执行后续代码，直到输入 Key
    else:
        st.session_state['api_key'] = input_key
        # 实例化 Agent（每次 Key 变化会重新创建）
        agent = PharmacyAgent(st.session_state['api_key'], "https://api.deepseek.com")

    # --- 登录后的主界面 ---
    st.title("🏥 药剂科 AI 处方审核平台")
    st.markdown("---")

    # 侧边栏：知识库管理
    with st.sidebar:
        st.header("📂 知识库管理")
        uploaded_file = st.file_uploader("上传药品说明书 PDF", type="pdf")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                if st.button("✨ 立即同步知识"):
                    with st.spinner("学习中..."):
                        count = km.upload_docs(tmp.name)
                        st.success(f"新增 {count} 条知识。")
            os.unlink(tmp.name)
        
        if st.button("🚪 退出登录"):
            st.session_state['api_key'] = ""
            st.rerun()

    # 主界面：处方录入
    col_in, col_out = st.columns([1, 1.2])

    with col_in:
        st.subheader("📋 录入处方")
        with st.form("audit_form"):
            r1 = st.columns(2)
            age = r1[0].number_input("年龄", value=5, min_value=0)
            weight = r1[1].number_input("体重 (kg)", value=18.0)
            
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
            prescription = {
                "patient": {"age": age, "weight": weight, "diagnosis": diagnosis, "insurance_type": insurance},
                "medications": [{"name": med_name, "dosage": dosage, "frequency": freq}]
            }
            
            with st.spinner("AI 药师分析中..."):
                context = km.retrieve_context(med_name)
                try:
                    report = agent.audit(prescription, context)
                    st.markdown(report)
                    st.download_button("📥 导出报告", report, file_name=f"报告_{med_name}.txt")
                except Exception as e:
                    st.error(f"审核失败。可能是 API Key 无效或余额不足。错误详情: {e}")
        else:
            st.info("请在左侧录入数据。")

if __name__ == "__main__":
    main()
