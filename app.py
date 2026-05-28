import streamlit as st
import os
import json
import warnings
import tempfile
from datetime import datetime

# --- 1. 基础环境配置 (必须在最前面) ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 导入核心库
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# 自动创建本地目录
DB_PATH = "./drug_db"
os.makedirs(DB_PATH, exist_ok=True)

# --- 2. 逻辑类定义 ---

class KnowledgeManager:
    def __init__(self, model_name, db_path):
        print(f"[{datetime.now()}] 📦 正在初始化 Embedding 模型: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        print(f"[{datetime.now()}] 🗄️ 正在连接向量数据库...")
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, file_path):
        """解析 PDF 并存入数据库"""
        if not os.path.exists(file_path):
            return 0
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, query):
        """检索知识"""
        results = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([res.page_content for res in results])

class PharmacyAgent:
    def __init__(self, api_key, base_url):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url=base_url,
            temperature=0
        )

    def audit(self, prescription_json, context):
        system_prompt = """你是一位资深临床药师。请根据【参考资料】审核【处方数据】。
        
        ### 核心审核逻辑：
        1. **儿科剂量复核**：若患者 <18岁，必须识别 weight(体重) 并计算。
           - 必须列出公式：建议标准(mg/kg) × 体重(kg) = 建议单次/每日剂量。
           - 对比处方中的 dosage，判断是否超标。
        2. **社保合规审核**：识别 diagnosis(诊断) 和 insurance_type(医保类型)。
           - 核对资料中的“医保支付范围”。若诊断不符（如阿奇霉素限肺炎，诊断为感冒），提示“医保拒付风险”。
        3. **安全性**：核对禁忌症、频次、重复用药。

        ### 输出结构：
        - 【儿科剂量计算】：(含公式分析)
        - 【医保合规核对】：(含诊断匹配分析)
        - 【药学审核结论】：(合理/不合理/医保风险)
        - 【专家建议】："""
        
        prompt = ChatPromptTemplate.from_template(system_prompt + "\n\n资料: {context}\n处方: {prescription}")
        chain = prompt | self.llm
        return chain.invoke({
            "context": context,
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

# --- 3. 系统初始化 (带缓存和日志) ---

@st.cache_resource
def init_system():
    print(f"[{datetime.now()}] 🚀 开始全系统初始化...")
    
    # 在线上环境，直接指定模型名，HuggingFace 库会自动处理下载（比魔搭在海外快）
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # 从 Secrets 获取 API Key
    api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
    base_url = "https://api.deepseek.com"

    km = KnowledgeManager(model_name, DB_PATH)
    agent = PharmacyAgent(api_key, base_url)
    
    print(f"[{datetime.now()}] ✅ 系统初始化完成！")
    return km, agent

# --- 4. Streamlit UI 界面 ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="⚖️")
    
    # 尝试加载系统
    try:
        km, agent = init_system()
    except Exception as e:
        st.error(f"系统启动失败，请检查 Logs。错误: {e}")
        return

    st.title("🏥 药剂科 AI 处方审核平台")
    st.markdown("---")

    # 侧边栏：知识库管理
    with st.sidebar:
        st.header("📂 知识库增强")
        uploaded_file = st.file_uploader("上传药品说明书/医保政策 PDF", type="pdf")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                if st.button("✨ 立即学习"):
                    with st.spinner("解析并存入向量库..."):
                        count = km.upload_docs(tmp.name)
                        st.success(f"学习完成！新增 {count} 条知识。")
            os.unlink(tmp.name)
        
        st.info("提示：在线上环境，上传的知识仅在本次运行有效。如需持久化，请将 drug_db 上传至 GitHub。")

    # 主界面分列
    col_in, col_out = st.columns([1, 1.2])

    with col_in:
        st.subheader("📋 录入处方")
        with st.form("audit_form"):
            row1 = st.columns(2)
            age = row1[0].number_input("患者年龄", value=5, min_value=0)
            weight = row1[1].number_input("患者体重 (kg)", value=18.0, step=0.1)
            
            row2 = st.columns(2)
            diagnosis = row2[0].text_input("临床诊断", value="急性支气管炎")
            insurance = row2[1].selectbox("医保类型", ["统筹医保", "自费", "门诊大病"])
            
            st.markdown("---")
            med_name = st.text_input("药品名称", value="阿奇霉素")
            dosage = st.text_input("单次剂量 (mg/支/袋)", value="250mg")
            freq = st.text_input("给药频次", value="一日一次")
            
            btn = st.form_submit_button("🧪 开始审核")

    with col_out:
        st.subheader("📝 审核报告")
        if btn:
            if not st.secrets.get("DEEPSEEK_API_KEY"):
                st.error("未发现 API Key，请在 Streamlit Secrets 中配置。")
                return

            prescription = {
                "patient": {"age": age, "weight": weight, "diagnosis": diagnosis, "insurance_type": insurance},
                "medications": [{"name": med_name, "dosage": dosage, "frequency": freq}]
            }
            
            with st.spinner("AI 药师深度分析中..."):
                # 检索
                context = km.retrieve_context(med_name)
                if not context.strip():
                    st.warning("⚠️ 库中未找到说明书，将基于通用医学常识。")
                
                # 审核
                report = agent.audit(prescription, context)
                st.markdown(report)
                
                # 下载
                st.download_button("📥 导出报告", report, file_name=f"审核报告_{med_name}.txt")
        else:
            st.info("请在左侧录入数据并点击提交。")

if __name__ == "__main__":
    main()
