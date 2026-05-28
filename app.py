# --- 1. 环境补丁 (必须放在最顶端) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # 本地 Windows 环境可能没有 pysqlite3，忽略即可
    pass

import streamlit as st
import os
import json
import warnings
import tempfile
import shutil
from datetime import datetime

# 导入核心组件
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 2. 逻辑类定义 ---

class KnowledgeManager:
    """知识库管理：负责 PDF 解析、向量化存储与检索"""
    def __init__(self, model_name, db_path):
        self.db_path = db_path
        # 使用多语言轻量化模型，适合 Streamlit Cloud 内存限制
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
        """检索相关知识"""
        try:
            # 检查数据库是否为空
            count = self.vectorstore._collection.count()
            if count == 0:
                return "（提醒：当前知识库为空，审核将仅依赖 AI 通识）"
            
            results = self.vectorstore.similarity_search(query, k=3)
            return "\n".join([res.page_content for res in results])
        except Exception:
            return "（检索失败或知识库未初始化）"

class PharmacyAgent:
    """AI 药师核心逻辑"""
    def __init__(self, api_key, base_url="https://api.deepseek.com"):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key.strip(),
            base_url=base_url,
            temperature=0,
            max_retries=2
        )

    def audit(self, prescription_json, context):
        system_instruction = """你是一位极其严谨的资深临床药师。请根据【参考资料】审核【处方数据】。
        
        ### 审核要求：
        1. **儿科剂量复核**：针对 <18岁患者，必须根据 weight(体重) 核算剂量。
           - 必须列出计算公式：[推荐标准(mg/kg)] × [患者体重(kg)] = [每日/每次应给剂量]。
        2. **医保合规性**：核对 diagnosis(诊断) 是否符合资料中的“医保支付范围”。
        3. **安全性检查**：检查频次、给药途径及是否有重复用药或禁忌。

        ### 请按以下结构输出报告：
        #### 1️⃣ 剂量复核 (含详细计算公式)
        #### 2️⃣ 医保支付合规性核对
        #### 3️⃣ 药学审核结论
        #### 4️⃣ 修改建议"""

        prompt = ChatPromptTemplate.from_template(
            "系统指引：{system}\n\n参考资料：{context}\n\n待审处方数据：{prescription}"
        )
        chain = prompt | self.llm
        
        response = chain.invoke({
            "system": system_instruction,
            "context": context,
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        })
        return response.content

# --- 3. 资源缓存与初始化 ---

@st.cache_resource
def get_km():
    """缓存知识库管理器，避免重复加载嵌入模型"""
    DB_PATH = "./drug_db"
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    # 使用该模型因为它在处理中文和多语言时体积适中且准确度高
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return KnowledgeManager(model_name, DB_PATH)

# --- 4. 主界面展示 ---

def main():
    st.set_page_config(page_title="AI 智能审方系统", layout="wide", page_icon="💊")
    
    # 初始化知识库
    km = get_km()

    # --- 侧边栏 ---
    st.sidebar.title("⚙️ 系统管理")
    
    api_key = st.sidebar.text_input("请输入 DeepSeek API Key:", type="password")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📚 知识库更新")
    uploaded_files = st.sidebar.file_uploader("上传 PDF 资料 (医保目录/说明书)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.sidebar.button("开始学习新资料"):
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                with st.sidebar:
                    with st.spinner(f"正在解析: {file.name}..."):
                        km.upload_docs(tmp_path)
                os.unlink(tmp_path)
            st.sidebar.success("资料入库成功！")

    if st.sidebar.button("清空本地数据库"):
        if os.path.exists("./drug_db"):
            shutil.rmtree("./drug_db")
            st.rerun()

    # --- 主页面内容 ---
    st.title("🏥 临床药师 AI 处方审核平台")
    st.info("本系统支持：儿科剂量自动换算、医保限定范围核对、用法用量安全性评价。")

    if not api_key:
        st.warning("⚠️ 请先在侧边栏输入 API Key 以激活 AI 审核功能。")
        st.stop()

    agent = PharmacyAgent(api_key)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("📋 处方信息录入")
        with st.form("rx_input"):
            c1, c2 = st.columns(2)
            age = c1.number_input("年龄 (岁)", 0, 120, 5)
            weight = c2.number_input("体重 (kg)", 0.0, 200.0, 18.0)
            
            diagnosis = st.text_input("临床诊断", "急性支气管炎")
            
            st.write("**药品明细**")
            med_name = st.text_input("药品名称", "阿奇霉素干混悬剂")
            dosage = st.text_input("单次剂量 (例如: 0.25g)", "0.25g")
            freq = st.selectbox("给药频次", ["QD (一日一次)", "BID (一日两次)", "TID (一日三次)", "ST! (立即使用)"])
            
            submitted = st.form_submit_button("🧪 提交 AI 智能审核")

    with col2:
        st.subheader("📑 审核报告")
        if submitted:
            rx_data = {
                "patient": {"age": age, "weight": weight, "diagnosis": diagnosis},
                "medications": [{"name": med_name, "dosage": dosage, "frequency": freq}]
            }
            
            with st.spinner("🔍 正在检索相关规范并分析..."):
                # 1. 检索知识库
                search_query = f"{med_name} {diagnosis} 儿童剂量 医保报销限定"
                context = km.retrieve_context(search_query)
                
                # 2. 调用 AI 审核
                try:
                    report = agent.audit(rx_data, context)
                    
                    # 3. 展示结果
                    st.success("分析完成")
                    st.markdown(report)
                    
                    # 下载功能
                    st.download_button(
                        label="📥 导出审核报告",
                        data=report,
                        file_name=f"审方报告_{med_name}_{datetime.now().strftime('%H%M')}.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"审核过程发生错误: {str(e)}")

if __name__ == "__main__":
    main()
