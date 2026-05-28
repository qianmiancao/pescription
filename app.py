import streamlit as st
import os
import json
import warnings
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
        if not os.path.exists(path):
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
        
        # 存入向量数据库
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
            temperature=0  # 严谨审核建议设为 0
        )

    def audit_prescription(self, prescription_json, context):
        """核心审核提示词逻辑"""
        system_instruction = """你是一位资深临床药师。请根据提供的【参考资料】对【处方数据】进行审核。
        
        ### 必须包含的审核维度：
        1. **儿科剂量复核**：针对 <18 岁患者，必须利用 weight(kg) 计算。公式：推荐标准 × 体重。
        2. **社保合规审核**：核对 diagnosis(诊断) 是否符合医保报销限制。
        3. **用法用量及安全性**：核对频次、禁忌症。
        
        请输出结构清晰的报告，包含：剂量分析（列出计算过程）、医保风险提示、审核结论、修改建议。"""

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

# --- 3. Streamlit 资源缓存 (避免重复下载和加载) ---

@st.cache_resource
def load_system():
    """初始化模型和类"""
    # 从 ModelScope 下载模型 (适配国内网络)
    try:
        model_dir = snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    except:
        model_dir = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # 优先从 Streamlit Secrets 获取 API Key
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except:
        api_key = "请在侧边栏填入 Key" # 兜底逻辑

    km = KnowledgeManager(model_path=model_dir, db_path="./drug_db")
    agent = PharmacyAgent(api_key=api_key, base_url="https://api.deepseek.com")
    
    return km, agent

# --- 4. Streamlit UI 界面 ---

def main():
    st.set_page_config(page_title="AI 药师审方系统", layout="wide", page_icon="💊")
    
    # 初始化
    km, agent = load_system()

    st.title("🏥 药剂科 AI 处方审核测试平台")
    st.caption("集成儿科剂量核算与社保合规性检查功能")

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统管理")
        # 允许用户在页面手动更改 Key（可选）
        user_key = st.text_input("API Key (若 Secrets 已设置则留空)", type="password")
        if user_key:
            agent.llm.openai_api_key = user_key
        
        st.markdown("---")
        st.subheader("📚 知识库更新")
        pdf_path = st.text_input("PDF 文件夹路径", value=r"C:\Users\HUAWEI\prescription\manuals")
        if st.button("🚀 导入说明书"):
            with st.spinner("正在解析 PDF 并建立索引..."):
                num = km.upload_docs(pdf_path)
                st.success(f"导入成功！新增 {num} 个知识片段。")
        st.info("注：线上部署后需确保服务器能访问该路径或上传 drug_db 文件夹。")

    # 主界面分列
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📝 录入处方")
        with st.form("audit_form"):
            # 患者信息
            c1, c2, c3 = st.columns(3)
            with c1: age = st.number_input("年龄", value=5, min_value=0)
            with c2: weight = st.number_input("体重 (kg)", value=18.0, step=0.1)
            with c3: insurance = st.selectbox("医保类型", ["统筹医保", "自费", "门诊大病"])
            
            diagnosis = st.text_input("临床诊断", value="急性支气管炎")
            
            st.markdown("---")
            # 药品信息
            med_name = st.text_input("药品名称", value="阿奇霉素干混悬剂")
            dosage = st.text_input("单次剂量", value="250mg")
            freq = st.text_input("频次", value="一日一次")
            
            submitted = st.form_submit_button("🧪 提交 AI 审核")

    with right_col:
        st.subheader("📋 药学审核报告")
        if submitted:
            prescription_data = {
                "patient": {
                    "age": age, "weight": weight, 
                    "diagnosis": diagnosis, "insurance_type": insurance
                },
                "medications": [{"name": med_name, "dosage": dosage, "frequency": freq}]
            }
            
            with st.spinner("AI 药师分析中..."):
                # 1. 检索知识库
                context = km.retrieve_context(med_name)
                if not context.strip():
                    st.warning("⚠️ 本地库未找到相关说明书，将使用 AI 通用药学知识审核。")
                
                # 2. 调用 AI 审核
                try:
                    report = agent.audit_prescription(prescription_data, context)
                    st.markdown(f"**审核结果：**\n\n{report}")
                    st.download_button("📥 下载报告", report, file_name=f"审核报告_{med_name}.txt")
                except Exception as e:
                    st.error(f"审核出错：{e}")
        else:
            st.info("请在左侧填写处方信息并点击提交。")

if __name__ == "__main__":
    main()