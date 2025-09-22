import ast
import json
import inspect
from dashscope import Assistants, Threads, Messages, Runs

from ..rag_line.questions_processing import QuestionsProcessor


def function_to_schema(func) -> dict:
    # 将 Python 类型映射为 JSON schema 类型
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    # 尝试获取函数的签名
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        # 如果签名获取失败，则抛出错误并附带错误信息
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    # 初始化一个字典来存储参数类型
    parameters = {}
    # 遍历函数的参数，并映射它们的类型
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            # 如果参数的类型注解未知，则抛出错误
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    # 创建必需参数的列表（那些没有默认值的参数）
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    # 返回函数的 schema 作为字典
    function_schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),  # 获取函数描述（docstring）
            "parameters": {
                "type": "object",
                "properties": parameters,  # 参数类型
                "required": required,  # 必需参数列表
            },
        },
    }
    return function_schema


PlannerAssistant = Assistants.create(
    model="qwen-plus",
    name='流程编排机器人',
    # 定义Agent的功能描述
    description='你是团队的leader，你的手下有很多assistant，你需要根据用户的输入，决定使用哪些assistant以及采用怎样的顺序去使用这些assistant',
    # 定义对Agent的指示语句，Agent会按照指示语句进行工具的调用并返回结果。
    instructions=
    """
    你的团队中有以下assistant。
    FinancialAssistant：一个财务数据分析工具，提供基础的数据查询功能，同时包含ARIMA预测分析、布林带检测、Prophet周期性分析功能进行高级的时间序列分析和预测
    RagAssistant：一个知识库查询工具，提供金融领域的专业词汇、概念、问题的解答，同时包含文本总结、知识库更新等功能；
    ChatAssistant：对于日常问题的回答，则调用该assistant。
    你需要根据用户的问题，判断要以什么顺序使用这些assistant，你的返回形式是一个列表，不能返回其它信息。
    比如：["FinancialAssistant", "FinancialAssistant","RagAssistant"]或者["ChatAssistant"]，列表中的元素只能为上述的assistant
    """
)

FinancialAssistant = Assistants.create(
    model="qwen-plus",
    name='财务数据分析机器人',
    description='一个财务数据分析工具，提供基础的数据查询功能，同时包含ARIMA预测分析、布林带检测、Prophet周期性分析功能进行高级的时间序列分析和预测',
    instructions='你是一个财务数据分析工具，提供基础的数据查询功能，同时包含ARIMA预测分析、布林带检测、Prophet周期性分析功能进行高级的时间序列分析和预测',
    tools=[]
)

RagAssistant = Assistants.create(
    model="qwen-plus",
    name='Rag知识库助手',
    description='一个知识库查询助手，通过对话的方式，提供金融领域的专业词汇、概念、问题的解答，同时包含文本总结、知识库更新等功能，依据提供的内容的进行总结，不使用网络数据',
    instructions='一个知识库查询助手，通过对话的方式，提供金融领域的专业词汇、概念、问题的解答，同时包含文本总结、知识库更新等功能，依据提供的内容的进行总结，不使用网络数据',
    tools=[function_to_schema(QuestionsProcessor().process_question)]
)

ChatAssistant = Assistants.create(
    model="qwen-turbo",
    name='回答日常问题的机器人',
    description='一个智能助手，用于解答用户非知识库领域的日常问题，提供网络搜索等工具',
    instructions='请礼貌地回答用户的问题',
    tools=[{
        "mcpServers": {
            "tavily-mcp": {
                "type": "sse",
                "url": "https://mcp.api-inference.modelscope.net/efa7d02a481748/sse"
            },
            "amap-maps": {
                "type": "sse",
                "url": "https://mcp.api-inference.modelscope.cn/sse/2990a5ee6dd542"
            }
        }
    }]
)

SummaryAssistant = Assistants.create(
    model="qwen-plus",
    name='总结机器人',
    description='一个智能助手，根据用户的问题与参考信息，全面、完整地回答用户问题',
    instructions='你是一个智能助手，根据用户的问题与参考信息，全面、完整地回答用户问题'
)

function_mapper = {
    "process_question": {
        "function": QuestionsProcessor().process_question,
        "submit_flag": False
    }
}

assistant_mapper = {
    "RagAssistant": RagAssistant,
    "FinancialAssistant": FinancialAssistant,
    "ChatAssistant": ChatAssistant,
}


def get_agent_response(assistant, message=''):
    # 打印出输入Agent的信息
    thread = Threads.create()
    message = Messages.create(thread.id, content=message)
    run = Runs.create(thread.id, assistant_id=assistant.id)
    run_status = Runs.wait(run.id, thread_id=thread.id)
    # 如果响应失败，会打印出run failed
    if run_status.status == 'failed':
        print('run failed:')
    # 如果需要工具来辅助大模型输出，则进行以下流程
    if run_status.required_action:
        f = run_status.required_action.submit_tool_outputs.tool_calls[0].function
        # 获得function name
        func_name = f['name']
        # 获得function 的入参
        param = json.loads(f['arguments'])
        # 根据function name，通过function_mapper映射到函数，并将参数输入工具函数得到output输出，输出为str
        output = function_mapper[func_name]["function"](**param) if func_name in function_mapper else ""
        submit_flag = function_mapper[func_name]['submit_flag'] if func_name in function_mapper else True
        if submit_flag:
            tool_outputs = [{
                'output':
                    output
            }]

            run = Runs.submit_tool_outputs(run.id,
                                           thread_id=thread.id,
                                           tool_outputs=tool_outputs)
            Runs.wait(run.id, thread_id=thread.id)
        else:
            # 创建本地消息记录
            Messages.update(
                message_id=message.id,
                thread_id=thread.id,
                metadata={
                    'local_processing': True,
                    'tool_name': func_name,
                    'content': output
                }
            )

            # Runs.update(run.id, thread_id=thread.id, metadata={'local_processed': True})

    # 获取消息时合并本地结果
    msgs = Messages.list(thread.id)
    if any(getattr(msg, 'metadata', {}).get('local_processing') for msg in msgs['data']):
        return {
            'type': 'local_processed',
            'content': [msg['metadata']['content'] for msg in msgs['data'] if msg['metadata'].get('local_processing')]
        }
    else:
        return {
            'type': 'remote_processed',
            'content': msgs['data'][0]['content'][0]['text']['value']
        }


# 获得Multi Agent的回复，输入与输出需要与Gradio前端展示界面中的参数对齐
def get_multi_agent_response(query):
    # 获取Agent的运行顺序
    assistant_order = get_agent_response(PlannerAssistant, query)['content']
    order_stk = ast.literal_eval(assistant_order)
    cur_query = query
    Agent_Message = ""
    multi_agent_response = ""
    # 依次运行Agent
    for i in range(len(order_stk)):
        cur_assistant = assistant_mapper[order_stk[i]]
        response = get_agent_response(cur_assistant, cur_query)
        # 如果当前Agent为最后一个Agent，则将其输出作为Multi Agent的输出
        if i == len(order_stk) - 1:
            if response['type'] == 'local_processed':
                multi_agent_response = response['content']
            else:
                Agent_Message += f"*{order_stk[i]}*的回复为：{response}\n\n"
                prompt = f"请参考已知的信息：{Agent_Message}，回答用户的问题：{query}。"
                multi_agent_response = get_agent_response(SummaryAssistant, prompt)['content']
        # 如果当前Agent不是最后一个Agent，则将上一个Agent的输出response添加到下一轮的query中，作为参考信息
        else:
            # 在参考信息前后加上特殊标识符，可以防止大模型混淆参考信息与提问
            cur_query = f"你可以参考已知的信息：{response}你要完整地回答用户的问题。问题是：{query}。"
    return multi_agent_response
