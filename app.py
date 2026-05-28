import streamlit as st
import os
import json
import warnings
import tempfile
from modelscope.hub.snapshot_download import snapshot_download

# 导入 LangChain 相关组件
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# --- 1. 基础环境配置 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 自动创建必要的本地文件夹，防止 GitHub 路径缺失报错
DB_PATH = "./drug_db"
LOCAL_MANUALS_PATH = "./manuals"
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(LOCAL_MANUALS_PATH, exist_ok=True)

# --- 2. 逻辑类定义 ---

class KnowledgeManager:
    """知识库管理类：处理 PDF 加载与语义检索"""
    def __init__(self, model_path, db_path):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

    def upload_docs(self, path):
        """导入 PDF 文档并建立索引"""
        if not os.path.exists(path) or (os.path.isdir(path) and not os.listdir(path)):
            return 0
        
        # 加载与切分
        if os.path.isdir(path):
            loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        else:
            loader = PyPDFLoader(path)
            
        docs = loader.load()
        if not docs:
            return 0

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = splitter.split_documents(docs)
        
        self.vectorstore.add_documents(splits)
        return len(splits)

    def retrieve_context(self, medication_name):
        """检索相关知识片段"""
        results = self.vectorstore.similarity_search(medication_name, k=3)
        return "\n".join([res.page_content for res in results])

class PharmacyAgent:
    """AI 药师类：处理处方审核逻辑"""
    def __init__(self, api_key, base_url):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url=base_url,
            temperature=0  # 严谨审核必须设为 0
        )

    def audit_prescription(self, prescription_json, context):
        """核心审核提示词逻辑：包含儿科剂量核算与社保合规"""
        system_instruction = """你是一位极其严谨的资深临床药师。请根据提供的【参考资料】对【处方数据】进行深度审核。
        
        ### 审核要求：
        1. **儿科专项复核**：针对 <18 岁患者，必须利用处方中的 weight(体重) 进行核算。
           - 必须列出计算公式。示例：[标准剂量] × [体重] = [建议剂量]。
           - 严判处方剂量是否超标或不足。
        2. **社保合规审核**：识别诊断(diagnosis)和医保类型(insurance_type)。
           - 核对参考资料中的“医保支付限定范围”。
           - 若诊断与支付范围不符（如：阿奇霉素限细菌感染，诊断为感冒），必须提示“医保拒付风险”。
        3. **用法用量及安全性**：检查频次、给药途径、重复用药及禁忌。

        请输出结构化报告：【剂量复核】(含公式)、【医保合规分析】、【结论】、【修改建议】。"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "参考资料: {context}\n\n处方数据: {prescription}")
        ])

        chain = prompt | self.llm
        prescription_str = json.dumps(prescription_json, ensure_ascii=False, indent=2)
        
        return chain.invoke({
            "context": context,
            "prescription": prescription_str
        }).content

# --- 3. Streamlit 资源缓存 ---

@st.cache_resource
def load_system():
    """初始化模型和类"""
    # 同步嵌入模型
    try:
        model_dir = snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    except:
        model_dir = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # 获取 API Key
    api_key = st.secrets.get("DEEPSEEK_API_KEY", "YOUR_KEY_HERE")

    km = KnowledgeManager(model_path=model_dir, db_path=DB_PATH)
    agent = PharmacyAgent(api_key=api_key, base_url="https://api.deepseek.com")
    
    return km, agent

# --- 4. Streamlit UI 界面 ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="💊")
    
    km, agent = load_system()

    st.title("🏥 药剂科 AI 处方审核测试平台")
    st.caption("增强功能：儿科剂量精准核算 + 社保支付范围合规性核对")

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统管理")
        # 允许手动覆盖 API Key
        user_key = st.text_input("DeepSeek API Key", type="password", help="若云端 Secrets 已设置则可留空")
        if user_key:
            agent.llm.openai_api_key = user_key
        
        st.markdown("---")
        st.subheader("📚 知识库更新")
        
        # 网页直接上传 PDF 功能
        uploaded_file = st.file_uploader("上传单份 PDF 说明书/医保政策", type="pdf")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                if st.button("✨ 学习此文件"):
                    with st.spinner("正在解析并存入向量库..."):
                        km.upload_docs(tmp.name)
                        st.success("学习完成！")
            os.unlink(tmp.name)

        if st.button("🔄 扫描全库 (manuals 文件夹)"):
            with st.spinner("正在读取服务器 manuals 目录..."):
                num = km.upload_docs(LOCAL_MANUALS_PATH)
                st.success(f"同步完成！共处理 {num} 个片段。")

    # 主界面分列
    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        st.subheader("📝 录入处方")
        with st.form("audit_form"):
            c1, c2, c3 = st.columns(3)
            with c1: age = st.number_input("患者年龄", value=5, min_value=0)
            with c2: weight = st.number_input("患者体重 (kg)", value=18.0, step=0.1)
            with c3: insurance = st.selectbox("医保类型", ["统筹医保", "自费", "门诊大病"])
            
            diagnosis = st.text_input("临床诊断", value="急性支气管炎")
            
            st.markdown("---")
            med_name = st.text_input("药品名称", value="阿奇霉素干混悬剂")
            dosage = st.text_input("单次剂量 (如 250mg)", value="250mg")
            freq = st.text_input("频次 (如 一日一次)", value="一日一次")
            
            submitted = st.form_submit_button("🔍 提交 AI 药师审核")

    with right_col:
        st.subheader("📋 审核报告")
        if submitted:
            prescription_data = {
                "patient": {
                    "age": age, "weight": weight, 
                    "diagnosis": diagnosis, "insurance_type": insurance
                },
                "medications": [{"name": med_name, "dosage": dosage, "frequency": freq}]
            }
            
            with st.spinner("AI 药师深度分析中..."):
                # 1. 检索
                context = km.retrieve_context(med_name)
                if not context.strip():
                    st.warning("⚠️ 库中未找到说明书，将基于通用医学常识。建议在左侧上传 PDF 完善知识库。")
                
                # 2. 审核
                try:
                    report = agent.audit_prescription(prescription_data, context)
                    st.markdown(report)
                    st.download_button("📥 导出报告", report, file_name=f"审核报告_{med_name}.txt")
                except Exception as e:
                    st.error(f"审核服务暂时不可用: {e}")
        else:
            st.info("提示：请在左侧录入处方，AI 会自动为您核算儿科剂量及医保权限。")

if __name__ == "__main__":
    main()
