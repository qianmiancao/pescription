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
from modelscope.hub.snapshot_download import snapshot_download

# 定义数据库存储路径
DB_PATH = "./drug_db"
os.makedirs(DB_PATH, exist_ok=True)

# --- 2. 逻辑类定义 ---

class KnowledgeManager:
    """知识库管理：负责 PDF 解析、向量化存储与检索"""
    def __init__(self, model_path, db_path):
        # 初始化嵌入模型 (Embedding)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}
        )
        # 初始化或加载本地向量库
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, file_path):
        """解析单份 PDF 并存入向量库"""
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # 医药文档建议切分稍微细一点，保证检索精度
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
        splits = splitter.split_documents(docs)
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, query):
        """根据查询内容检索最相关的 3 个知识片段"""
        results = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([res.page_content for res in results])

class PharmacyAgent:
    """AI 药师：负责逻辑推理与报告生成"""
    def __init__(self, api_key, base_url="https://api.deepseek.com"):
        # 药学审核必须严谨，temperature 设为 0
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=str(api_key).strip(),
            base_url=base_url,
            temperature=0
        )

    def audit(self, prescription_json, context):
        """处方审核核心提示词"""
        system_instruction = """你是一位极其严谨的资深临床药师。请根据提供的【参考资料】对【处方数据】进行审核。
        
        ### 核心逻辑：
        1. **儿科专项复核**：若患者 <18岁，必须识别 weight(体重) 并进行剂量核算。
           - 必须列出计算公式。示例：建议标准(mg/kg) × 体重(kg) = 建议剂量。
        2. **社保合规审核**：识别 diagnosis(诊断) 和 insurance_type(医保类型)。
           - 核对参考资料中关于该药的“医保支付限定范围”。
           - 若诊断不符（如阿奇霉素限肺炎，处方诊断为感冒），必须提示“医保拒付风险”。
        3. **用法用量及安全性**：检查频次、途径、禁忌症及重复用药。

        请输出结构化报告：【剂量核算】(含公式)、【社保合规核对】、【药学结论】、【修改建议】。"""

        prompt = ChatPromptTemplate.from_template(system_instruction + "\n\n资料库背景: {context}\n\n当前待审核处方: {prescription}")
        chain = prompt | self.llm
        
        return chain.invoke({
            "context": context,
            "prescription": json.dumps(prescription_json, ensure_ascii=False, indent=2)
        }).content

# --- 3. 资源缓存初始化 (防止重复加载) ---

@st.cache_resource
def get_knowledge_manager():
    """同步下载模型并初始化 KM"""
    try:
        # 优先通过 ModelScope 下载，适合国内环境
        model_dir = snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    except:
        # 海外环境备用
        model_dir = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return KnowledgeManager(model_dir, DB_PATH)

# --- 4. Streamlit UI 界面 ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="⚖️")
    
    # 初始化知识库管理对象
    km = get_knowledge_manager()

    # --- 侧边栏：登录与资料管理 ---
    st.sidebar.title("🔐 系统管理")
    
    # API Key 登录逻辑
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ""

    input_key = st.sidebar.text_input(
        "请输入 DeepSeek API Key:", 
        type="password", 
        placeholder="sk-...",
        value=st.session_state['api_key']
    )

    if not input_key:
        st.sidebar.warning("请输入有效 Key 以激活 AI 药师。")
        st.info("👋 欢迎！请在侧边栏输入 API Key 登录系统。")
        st.stop()
    else:
        st.session_state['api_key'] = input_key
        agent = PharmacyAgent(st.session_state['api_key'])

    st.sidebar.markdown("---")
    st.sidebar.header("📂 核心资料库")
    st.sidebar.caption("支持批量上传：药品说明书、医保政策、审方制度")

    # 批量上传组件
    uploaded_files = st.sidebar.file_uploader(
        "上传 PDF 资料", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.sidebar.button("✨ 批量同步资料"):
            success_count = 0
            progress_bar = st.sidebar.progress(0)
            
            for i, file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                try:
                    with st.sidebar.status(f"正在学习: {file.name}", expanded=False):
                        km.upload_docs(tmp_path)
                        success_count += 1
                finally:
                    if os.path.exists(tmp_path): os.unlink(tmp_path)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.sidebar.success(f"✅ 成功导入 {success_count} 份资料！")
            st.rerun()

    if st.sidebar.button("🚪 退出系统"):
        st.session_state['api_key'] = ""
        st.rerun()

    # --- 主界面：业务处理 ---
    st.title("🏥 药剂科 AI 处方审核平台")
    st.caption("集成儿科剂量精准核算与医保合规性专项审核")
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
            dosage = st.text_input("单次剂量 (mg/支/袋)", value="250mg")
            freq = st.text_input("给药频次", value="一日一次")
            
            btn = st.form_submit_button("🧪 提交 AI 药师审核")

    with col_out:
        st.subheader("📝 药学审核报告")
        if btn:
            prescription_data = {
                "patient": {
                    "age": age, "weight": weight, 
                    "diagnosis": diagnosis, "insurance_type": insurance
                },
                "medications": [{"name": med_name, "dosage": dosage, "frequency": freq}]
            }
            
            with st.spinner("AI 药师正在分析资料并核算中..."):
                # 1. 检索：同时搜索药品名和医保关键词
                context_info = km.retrieve_context(f"{med_name} 说明书 医保限制 报销规定")
                
                if not context_info.strip():
                    st.warning("⚠️ 资料库中未找到该药记录，将基于通用医学常识审核。")
                
                # 2. 调用 AI 审核
                try:
                    report = agent.audit(prescription_data, context_info)
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff;">
                        {report.replace('\n', '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button("📥 导出审核报告", report, file_name=f"审核报告_{med_name}.txt")
                except Exception as e:
                    st.error(f"审核过程发生错误: {e}")
        else:
            st.info("💡 请在左侧录入处方，并在侧边栏确保已上传对应资料。")

if __name__ == "__main__":
    main()
