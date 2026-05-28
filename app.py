import streamlit as st
import os
import json
import warnings
import tempfile
import shutil
from datetime import datetime

# --- 基础环境配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        # 优化：显式指定模型目录，避免 Streamlit 重复下载
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, file_path):
        """解析单份 PDF 并存入向量库"""
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
            splits = splitter.split_documents(docs)
            self.vectorstore.add_documents(splits)
            return len(splits)
        except Exception as e:
            st.error(f"解析文件失败: {e}")
            return 0

    def retrieve_context(self, query):
        """根据查询内容检索最相关的知识片段"""
        # 增加判断：如果库中没有数据，返回空
        try:
            results = self.vectorstore.similarity_search(query, k=3)
            return "\n".join([res.page_content for res in results])
        except:
            return "（当前知识库为空，请先上传参考资料）"

class PharmacyAgent:
    """AI 药师：负责逻辑推理与报告生成"""
    def __init__(self, api_key, base_url="https://api.deepseek.com"):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key.strip(),
            base_url=base_url,
            temperature=0
        )

    def audit(self, prescription_json, context):
        system_instruction = """你是一位极其严谨的资深临床药师。请根据提供的【参考资料】对【处方数据】进行审核。
        
        ### 核心逻辑：
        1. **儿科专项复核**：针对 <18岁患者，必须识别 weight(体重) 并核算剂量。
           - 必须列出计算公式。示例：[推荐标准(mg/kg)] × [患者体重(kg)] = [应给剂量]。
        2. **社保合规审核**：核对 diagnosis(诊断) 是否符合资料中的“医保支付限定范围”。
           - 若不符，必须提示“医保拒付风险”。
        3. **用法用量及安全性**：检查频次、途径及禁忌。

        请严格按以下格式输出：
        #### 1. 剂量复核 (含公式)
        #### 2. 社保合规核对
        #### 3. 药学结论
        #### 4. 修改建议"""

        prompt = ChatPromptTemplate.from_template("{system}\n\n参考资料: {context}\n\n待审处方: {prescription}")
        chain = prompt | self.llm
        
        return chain.invoke({
            "system": system_instruction,
            "context": context,
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

# --- 3. 资源管理与初始化 ---

@st.cache_resource
def init_km():
    DB_PATH = "./drug_db"
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    # 推荐使用性能较好的多语言模型
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return KnowledgeManager(model_name, DB_PATH)

# --- 4. Streamlit UI ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="⚖️")
    
    # 初始化知识库
    km = init_km()

    # --- 侧边栏 ---
    st.sidebar.title("🔐 系统设置")
    
    api_key = st.sidebar.text_input("DeepSeek API Key:", type="password", placeholder="sk-...")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📂 药品知识库导入")
    uploaded_files = st.sidebar.file_uploader("上传药品说明书/医保目录 (PDF)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.sidebar.button("🚀 开始学习资料"):
        with st.sidebar:
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                with st.spinner(f"正在学习: {file.name}..."):
                    km.upload_docs(tmp_path)
                os.unlink(tmp_path)
            st.success("资料入库完成！")

    if st.sidebar.button("🗑️ 清空知识库"):
        if os.path.exists("./drug_db"):
            shutil.rmtree("./drug_db")
            st.rerun()

    # --- 主界面 ---
    st.title("🏥 临床药师 AI 审方决策支持系统")
    
    if not api_key:
        st.info("💡 请在左侧侧边栏输入 API Key 以启动 AI 引擎。")
        st.stop()

    agent = PharmacyAgent(api_key)

    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.subheader("📋 处方录入")
        with st.form("rx_form"):
            r1 = st.columns(2)
            age = r1[0].number_input("患者年龄", 0, 120, 5)
            weight = r1[1].number_input("体重 (kg)", 0.0, 200.0, 18.0)
            
            diagnosis = st.text_input("临床诊断", "急性支气管炎")
            
            st.markdown("---")
            med_name = st.text_input("药品名称", "阿奇霉素干混悬剂")
            dosage = st.text_input("单次给药剂量 (如: 0.2g)", "0.25g")
            freq = st.selectbox("给药频次", ["一日一次 (QD)", "一日两次 (BID)", "一日三次 (TID)", "必要时 (PRN)"])
            
            submit = st.form_submit_button("⚖️ 提交智能审核")

    with c2:
        st.subheader("📝 审核报告")
        if submit:
            rx_data = {
                "patient": {"age": age, "weight": weight, "diagnosis": diagnosis},
                "meds": [{"name": med_name, "dosage": dosage, "frequency": freq}]
            }
            
            with st.spinner("正在检索知识库并分析中..."):
                # 检索知识
                context = km.retrieve_context(f"{med_name} 用法用量 医保限制 儿童剂量")
                
                try:
                    result = agent.audit(rx_data, context)
                    st.markdown(result)
                    
                    st.download_button(
                        label="📥 下载审核报告",
                        data=result,
                        file_name=f"审方报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"AI 服务调用失败: {e}")

if __name__ == "__main__":
    main()
