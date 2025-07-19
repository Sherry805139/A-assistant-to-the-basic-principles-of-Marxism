import os
import re
import dashscope
from typing import Dict, List, TypedDict, Optional, Any
from langchain_community.vectorstores import FAISS
from langchain_dashscope.embeddings import DashScopeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langgraph.graph import StateGraph, END

#  API Key Setup (与之前相同) 
# 重要：运行前必须设置环境变量
# 方法1：在终端中运行
#    export DASHSCOPE_API_KEY='your_api_key_here'      # Linux/Mac
#    $env:DASHSCOPE_API_KEY='your_api_key_here'         # Windows PowerShell
# 方法2：取消下面这行注释并填入您的API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-67eb31fc296f46728913a60ad6c03e32" # 请替换为您的实际API Key并取消注释

# 让 dashscope SDK 立刻获取到正确的 key（import dashscope 已经在上面执行过）
import dashscope as _ds_internal
if os.environ.get("DASHSCOPE_API_KEY"):
    _ds_internal.api_key = os.environ["DASHSCOPE_API_KEY"]

#  CustomChatDashScope (与之前相同，确保稳定调用) 
class CustomChatDashScope(BaseChatModel):
    model: str = "qwen-turbo"
    temperature: float = 0.7

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        prompt_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_messages.append({'role': 'system', 'content': msg.content})
            elif isinstance(msg, HumanMessage):
                prompt_messages.append({'role': 'user', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                prompt_messages.append({'role': 'assistant', 'content': msg.content})

        response = dashscope.Generation.call(
            model=self.model,
            messages=prompt_messages,
            result_format='message',
            temperature=self.temperature,
            stream=False,
            **kwargs
        )

        if hasattr(response, "status_code"):
            if response.status_code == 200:
                ai_content = response.output.choices[0]["message"]["content"]
                return AIMessage(content=ai_content)
            raise Exception(
                f"DashScope API Error: Code {getattr(response, 'code', 'unknown')}, Message: {getattr(response, 'message', 'unknown')}"
            )

        if hasattr(response, "__iter__"):
            content_chunks: List[str] = []
            for chunk in response:
                try:
                    content_chunks.append(chunk.choices[0].delta.content)
                except Exception:
                    pass
            return AIMessage(content="".join(content_chunks))

        raise Exception("DashScope 返回了未知的响应类型，无法解析。")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        ai_msg = self._call(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    @property
    def _llm_type(self) -> str:
        return "custom_chat_dashscope_wrapper"



### **LangGraph 状态定义（GraphState）**

# 这里会扩展 `GraphState` 来存储苏格拉底对话所需的额外信息：


class GraphState(TypedDict):
    """定义图状态，管理整个工作流中的数据状态"""
    user_input: str  # 用户输入的原始指令
    current_topic: str  # 当前对话的主题
    simulated_character: str  # 模拟的历史人物
    conversation_history: List[Dict[str, str]]  # 存储完整对话历史，用于维持苏格拉底对话的上下文
    retrieved_docs: List[str]  # 检索到的相关文档内容
    socratic_response: str  # 苏格拉底式回应
    turn_count: int # 对话轮次计数
    error_message: Optional[str]  # 错误信息
    # 新增字段用于表示对话是否结束，或者是否需要用户继续输入
    dialogue_status: str # "continue", "end", "error"

#####################以下是SocratesAgent部分###########################################################


class SocratesAgent:
    """特定人物语气的苏格拉底对话 Agent"""
    
    def __init__(self):
        """初始化 Agent"""
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise Exception("请设置 DASHSCOPE_API_KEY 环境变量")
        
        try:
            self.embeddings = DashScopeEmbeddings(model="text-embedding-v2")
            print("Embedding 模型初始化成功")
        except Exception as e:
            raise Exception(f"Embedding 模型初始化失败: {e}")
        
        try:
            self.llm = CustomChatDashScope(
                model="qwen-max", # 使用更强大的qwen-max来处理复杂对话和语气模仿
                temperature=0.8 # 稍高温度，让语气更自然，苏格拉底式提问更具启发性
            )
            print("LLM 模型初始化成功")
        except Exception as e:
            raise Exception(f"LLM 模型初始化失败: {e}")
        
        self.vectorstore = None
        self._load_knowledge_base() # 加载知识库

        self.graph = None
        self._build_graph() # 构建 LangGraph 工作流
    
    def _load_knowledge_base(self):
        """加载本地向量知识库"""
        try:
            print("正在加载向量知识库...")
            # 确保 'database_agent_mayuan' 目录存在且包含有效的FAISS索引文件
            self.vectorstore = FAISS.load_local(
                "database_agent_mayuan", 
                self.embeddings,
                allow_dangerous_deserialization=True # 允许从不安全来源反序列化，注意风险
            )
            print("向量知识库加载成功！")
        except Exception as e:
            print(f"加载向量知识库失败: {e}")
            print("Agent将在没有知识库的情况下运行（功能受限）")
            self.vectorstore = None

    def parse_user_intent_node(self, state: GraphState) -> Dict:
        """
        解析用户输入节点 - 提取主题和期望模拟的历史人物。
        这个节点只在对话的开始（turn_count == 0）执行。
        """
        print("正在解析用户意图...")
        user_input = state["user_input"]
        current_topic = state["current_topic"]
        simulated_character = state["simulated_character"]

        # 如果是首次输入，尝试解析主题和人物
        if state["turn_count"] == 0:
            # 尝试通过LLM解析，更智能和灵活
            prompt = PromptTemplate.from_template("""
            用户希望进行一场关于马克思主义基本原理的苏格拉底式对话，并希望我模仿特定人物的语气。
            请从用户的输入中识别出“对话主题”和“希望模仿的历史人物”。
            如果未明确指定人物，请默认“马克思”。如果未明确指定主题，请默认“马克思主义哲学”。

            请以 JSON 格式输出，例如：
            {{
                "topic": "实践与认识的关系",
                "character": "马克思"
            }}

            用户输入: {user_input}
            """)
            
            messages = [
                SystemMessage(content="你是一个意图识别专家。"),
                HumanMessage(content=prompt.format(user_input=user_input))
            ]

            try:
                llm_response = self.llm.invoke(messages)
                # 尝试从LLM响应中解析JSON
                match = re.search(r"\{.*\}", llm_response.content, re.DOTALL)
                if match:
                    parsed_data = eval(match.group(0)) # 使用eval解析JSON字符串，注意安全性，此处假设LLM返回格式可控
                    current_topic = parsed_data.get("topic", "马克思主义哲学")
                    simulated_character = parsed_data.get("character", "马克思")
                else:
                    raise ValueError("LLM未能返回有效的JSON格式。")
            except Exception as e:
                print(f"LLM解析用户意图失败: {e}，将使用默认值。")
                current_topic = "马克思主义哲学"
                simulated_character = "马克思"
            
            # 初始化对话历史，加入用户的第一句话
            new_conversation_history = [{"role": "user", "content": user_input}]
        else:
            # 非首次输入，保持原有主题和人物，更新对话历史
            new_conversation_history = state["conversation_history"]
            new_conversation_history.append({"role": "user", "content": user_input})


        return {
            "current_topic": current_topic,
            "simulated_character": simulated_character,
            "conversation_history": new_conversation_history,
            "error_message": None,
            "dialogue_status": "continue" # 默认继续
        }

    def retrieve_knowledge_node(self, state: GraphState) -> Dict:
        """知识库检索节点 - 根据主题检索相关文档"""
        print(f"正在检索主题 '{state['current_topic']}' 的相关资料...")
        
        try:
            if self.vectorstore is None:
                error_msg = "向量知识库未正确加载，请检查database_agent_mayuan目录是否存在"
                print(error_msg)
                return {
                    "retrieved_docs": [],
                    "error_message": error_msg,
                    "dialogue_status": "error"
                }
            
            # 使用当前主题进行检索
            query = f"{state['current_topic']} 马克思主义基本原理 {state['simulated_character']}"
            docs = self.vectorstore.similarity_search(query, k=5) # 增加k值获取更多上下文
            retrieved_docs = list(dict.fromkeys([doc.page_content for doc in docs]))[:5]

            print(f"成功检索到 {len(retrieved_docs)} 个相关文档片段。")
            
            return {
                "retrieved_docs": retrieved_docs,
                "error_message": None,
                "dialogue_status": "continue"
            }
            
        except Exception as e:
            error_msg = f"检索过程中出现错误: {e}"
            print(error_msg)
            return {
                "retrieved_docs": [],
                "error_message": error_msg,
                "dialogue_status": "error"
            }

    def generate_socratic_response_node(self, state: GraphState) -> Dict:
        """
        生成苏格拉底式回应节点 - 根据对话历史、知识和人物语气生成启发性问题。
        """
        print("正在生成苏格拉底式回应...")
        
        current_topic = state["current_topic"]
        simulated_character = state["simulated_character"]
        conversation_history = state["conversation_history"]
        retrieved_docs = state["retrieved_docs"]

        # 构建系统消息，设定角色和语气
        system_message_content = f"""你是一个资深的马克思主义基本原理课程教师，现在你正在扮演 {simulated_character}，与学生进行一场关于 {current_topic} 的苏格拉底式对话。
你的目标是：
1. 模仿 {simulated_character} 的说话语气、风格和常用词汇。例如，如果扮演马克思，可以表现出严谨、辩证、批判的口吻。
2. 保持苏格拉底式对话的核心：不直接给出答案，而是通过一系列启发性的问题，一步一步引导学生深入思考，帮助他们发现问题、理解概念、形成自己的结论。
3. 你的问题应该基于当前的对话内容和提供的参考资料，聚焦于一个具体点，并促进思考。
4. 如果学生回答偏离主题，尝试巧妙引导回主题。
5. 当你认为学生对某个概念或问题已经有了足够深入的思考时，可以适当进行总结或提出下一个层次的问题。
6. 不要分点回答，回答不超过300字

参考资料：
{'   '.join(retrieved_docs)}

当前对话历史：
"""
        # 将对话历史转换为LLM可以理解的格式
        llm_messages = [SystemMessage(content=system_message_content)]
        for msg in conversation_history:
            if msg["role"] == "user":
                llm_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                llm_messages.append(AIMessage(content=msg["content"])) # 这里的assistant是历史对话中Agent自己的回答


        try:
            response = self.llm.invoke(llm_messages)
            socratic_response_text = response.content
            
            # 更新对话历史
            new_conversation_history = list(conversation_history) # 复制一份，避免直接修改原始对象
            new_conversation_history.append({"role": "assistant", "content": socratic_response_text})
            
            print("苏格拉底式回应生成完成！")
            
            return {
                "socratic_response": socratic_response_text,
                "conversation_history": new_conversation_history,
                "error_message": None,
                "turn_count": state["turn_count"] + 1,
                "dialogue_status": "continue"
            }
            
        except Exception as e:
            error_msg = f"生成苏格拉底式回应过程中出现错误: {e}"
            print(error_msg)
            return {
                "socratic_response": "抱歉，我在生成回应时遇到了问题。请稍后再试。",
                "error_message": error_msg,
                "dialogue_status": "error"
            }
    
    def _build_graph(self):
        """构建 LangGraph 工作流"""
        print("正在构建工作流图...")
        
        try:
            workflow = StateGraph(GraphState)
            
            # 添加节点
            workflow.add_node("parse_user_intent", self.parse_user_intent_node)
            workflow.add_node("retrieve_knowledge", self.retrieve_knowledge_node)
            workflow.add_node("generate_socratic_response", self.generate_socratic_response_node)
            
            # 定义流转关系
            # 入口：首先解析用户意图
            workflow.set_entry_point("parse_user_intent")
            
            # 意图解析后，进入知识检索
            workflow.add_edge("parse_user_intent", "retrieve_knowledge")
            
            # 知识检索后，进入生成苏格拉底式回应
            workflow.add_edge("retrieve_knowledge", "generate_socratic_response")
            
            # 苏格拉底式回应生成后，我们希望回到等待用户输入的状态
            # 这里使用一个条件判断，模拟对话的循环
            # 我们不直接跳到END，而是让外部程序决定何时结束对话
            # 所以这里让它返回，等待下次invoke
            # 实际的对话循环在 process_dialogue 方法中管理
            workflow.add_edge("generate_socratic_response", END)
            
            # 编译图
            self.graph = workflow.compile()
            print("工作流图构建完成！")
            
        except Exception as e:
            print(f"构建工作流图失败: {e}")
            print("Agent将无法正常工作")
            self.graph = None

    def process_dialogue(self, user_input: str, current_state: Optional[GraphState] = None) -> Dict:
        """
        处理用户请求，进行一轮对话。
        current_state 用于从上一轮对话继承状态。
        """
        print(f"\n收到用户输入: {user_input}")
        print("=" * 50)
        
        if self.graph is None:
            return {"response": "工作流图未正确构建，请检查系统初始化", "status": "error"}
        
        # 初始化状态或继承上一轮状态
        if current_state:
            # 确保传递给GraphState的字典符合其TypedDict定义
            initial_state: GraphState = {
                "user_input": user_input,
                "current_topic": current_state["current_topic"],
                "simulated_character": current_state["simulated_character"],
                "conversation_history": current_state["conversation_history"],
                "retrieved_docs": current_state["retrieved_docs"], # 持续使用之前检索的文档，避免重复检索
                "socratic_response": "",
                "turn_count": current_state["turn_count"],
                "error_message": None,
                "dialogue_status": "continue"
            }
        else:
            # 首次调用
            initial_state: GraphState = {
                "user_input": user_input,
                "current_topic": "", # 待解析
                "simulated_character": "", # 待解析
                "conversation_history": [],
                "retrieved_docs": [],
                "socratic_response": "",
                "turn_count": 0, # 初始化轮次计数
                "error_message": None,
                "dialogue_status": "continue"
            }
        
        try:
            # 运行工作流
            # LangGraph 的 invoke 每次执行一次从入口到 END 的路径
            # 对于持续对话，我们需要在外部循环调用
            final_state = self.graph.invoke(initial_state)
            
            # 检查是否有错误
            if final_state["error_message"]:
                return {"response": f"处理过程中出现错误: {final_state['error_message']}", "status": "error", "state": final_state}
            
            return {"response": final_state["socratic_response"], "status": "continue", "state": final_state}
            
        except Exception as e:
            return {"response": f"系统错误: {e}", "status": "error", "state": initial_state}


#####################以下是主函数部分#############################################

def main():
    """主程序入口 - 提供命令行交互界面"""
    print("=" * 60)
    print("      特定人物语气的苏格拉底对话 Agent")
    print("        基于 LangGraph 和 LangChain 构建")
    print("=" * 60)
    
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("\n❌ 错误：未设置 DASHSCOPE_API_KEY 环境变量")
        print("\n🔧 请按以下步骤设置 API Key：")
        print("1. 在终端中运行：")
        print("   export DASHSCOPE_API_KEY='your_api_key_here'  # Linux/Mac")
        print("   或")
        print("   $env:DASHSCOPE_API_KEY='your_api_key_here'   # Windows PowerShell")
        print("\n2. 或者在代码开头添加：")
        print("   os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here'")
        print("\n💡 您可以在阿里云控制台获取 API Key")
        return
    
    try:
        agent = SocratesAgent()
        print("\n🎉 Agent 初始化完成！")
        
        print("\n💡 使用说明:")
        print("   这是一个特定人物语气的苏格拉底对话 Agent。")
        print("   您可以指定主题和模拟人物，例如：")
        print("   - '我想和马克思探讨一下唯物辩证法。'")
        print("   - '我们来谈谈历史唯物主义，你就像恩格斯一样提问吧。'")
        print("   - '我想深入思考一下实践的本质。'")
        print("   - 输入 'quit' 或 'exit' 退出对话。")
        print("=" * 60)
        
        current_dialogue_state: Optional[GraphState] = None
        
        while True:
            try:
                user_input = input("\n您想探讨什么 (输入 'quit' 退出): ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("\n👋 谢谢使用，再见！")
                    break
                
                if not user_input:
                    print("⚠️  请输入您的问题或想探讨的主题。")
                    continue
                
                # 每次调用都传递上一次的对话状态
                response_data = agent.process_dialogue(user_input, current_dialogue_state)
                
                print("\n" + "=" * 50)
                print(f"📖 {response_data['state']['simulated_character']}（{response_data['state']['current_topic']}）的回应:")
                print("=" * 50)
                print(response_data["response"])
                print("=" * 50)
                
                # 更新当前对话状态，用于下一轮
                current_dialogue_state = response_data["state"]

                if response_data["status"] == "error":
                    print("对话遇到错误，可能需要重新开始。")
                    current_dialogue_state = None # 清空状态，重新开始
            
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 处理请求时出现错误: {e}")
                print("请重试或输入新的请求")
                current_dialogue_state = None # 清空状态，重新开始
                
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        print("请检查:")
        print("1. DASHSCOPE_API_KEY 环境变量是否设置正确")
        print("2. 向量数据库文件是否存在于 'database_agent_mayuan' 目录")
        print("3. 网络连接是否正常")


#####################以下是Flask Web应用部分#############################################

from flask import Flask, request, jsonify, render_template, session
import uuid

# 创建Flask应用
app = Flask(__name__)
app.secret_key = 'socrates_dialogue_secret_key_2024'  # 用于session管理

# 全局变量存储对话状态
dialogue_sessions = {}

# 初始化Agent
socrates_agent = None

def init_agent():
    """初始化SocratesAgent"""
    global socrates_agent
    try:
        socrates_agent = SocratesAgent()
        print("✅ SocratesAgent 初始化成功！")
        return True
    except Exception as e:
        print(f"❌ SocratesAgent 初始化失败: {e}")
        return False

@app.route('/')
def index():
    """主页路由"""
    return render_template('role_chat.html')

@app.route('/start_dialogue', methods=['POST'])
def start_dialogue():
    """开始新对话"""
    if not socrates_agent:
        return jsonify({"error": "AI助手未正确初始化，请检查系统配置"}), 500
    
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    
    if not user_message:
        return jsonify({"error": "请输入您想探讨的话题"}), 400
    
    try:
        # 生成新的会话ID
        session_id = str(uuid.uuid4())
        
        # 处理第一轮对话
        response_data = socrates_agent.process_dialogue(user_message, None)
        
        if response_data["status"] == "error":
            return jsonify({"error": f"对话启动失败: {response_data['response']}"}), 500
        
        # 保存对话状态
        dialogue_sessions[session_id] = response_data["state"]
        
        return jsonify({
            "session_id": session_id,
            "response": response_data["response"],
            "character": response_data["state"]["simulated_character"],
            "topic": response_data["state"]["current_topic"],
            "turn_count": response_data["state"]["turn_count"]
        })
        
    except Exception as e:
        print(f"启动对话时出错: {e}")
        return jsonify({"error": f"启动对话时发生内部错误: {str(e)}"}), 500

@app.route('/continue_dialogue', methods=['POST'])
def continue_dialogue():
    """继续对话"""
    if not socrates_agent:
        return jsonify({"error": "AI助手未正确初始化"}), 500
    
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    user_message = data.get("message", "").strip()
    
    if not session_id or session_id not in dialogue_sessions:
        return jsonify({"error": "会话已过期，请重新开始对话"}), 400
    
    if not user_message:
        return jsonify({"error": "请输入您的回应"}), 400
    
    try:
        # 获取当前对话状态
        current_state = dialogue_sessions[session_id]
        
        # 继续对话
        response_data = socrates_agent.process_dialogue(user_message, current_state)
        
        if response_data["status"] == "error":
            return jsonify({"error": f"对话处理失败: {response_data['response']}"}), 500
        
        # 更新对话状态
        dialogue_sessions[session_id] = response_data["state"]
        
        return jsonify({
            "response": response_data["response"],
            "character": response_data["state"]["simulated_character"],
            "topic": response_data["state"]["current_topic"],
            "turn_count": response_data["state"]["turn_count"]
        })
        
    except Exception as e:
        print(f"继续对话时出错: {e}")
        return jsonify({"error": f"处理对话时发生内部错误: {str(e)}"}), 500

@app.route('/end_dialogue', methods=['POST'])
def end_dialogue():
    """结束对话"""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    
    if session_id and session_id in dialogue_sessions:
        del dialogue_sessions[session_id]
        return jsonify({"message": "对话已结束"})
    
    return jsonify({"message": "会话未找到或已结束"})

@app.route('/health')
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "agent_loaded": socrates_agent is not None,
        "active_sessions": len(dialogue_sessions)
    })

def run_role_app():
    """运行Flask应用"""
    print("=" * 60)
    print("    🎭 历史人物苏格拉底对话 Web 应用")
    print("=" * 60)
    
    # 检查API Key
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("\n❌ 警告: 未找到 DASHSCOPE_API_KEY 环境变量")
        print("请设置API Key后重启应用")
        print("Windows PowerShell: $env:DASHSCOPE_API_KEY='your_api_key'")
        print("Linux/Mac: export DASHSCOPE_API_KEY='your_api_key'")
        print("=" * 60)
    
    # 初始化Agent
    if not init_agent():
        print("❌ 应用启动失败：AI助手初始化失败")
        return
    
    print("\n🚀 应用启动成功！")
    print("📱 访问地址: http://localhost:5002")
    print("💡 您可以与马克思、恩格斯等历史人物进行深度对话")
    print("=" * 60)
    
    try:
        app.run(debug=True, port=5002, host='0.0.0.0')
    except Exception as e:
        print(f"❌ 应用运行失败: {e}")

if __name__ == "__main__":
    # 如果直接运行此文件，可以选择命令行模式或Web模式
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        run_role_app()
    else:
        main()  # 原有的命令行模式
