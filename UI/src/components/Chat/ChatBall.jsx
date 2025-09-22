import React, {useState} from 'react';
import {FloatButton, Modal, List, Input, Spin} from 'antd';
import {MessageOutlined, CloseOutlined} from '@ant-design/icons';


export default function ChatBall() {
    const [open, setOpen] = useState(false);
    const [message, setMessage] = useState('');
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSend = async () => {
        if (!message.trim()) return;

        setLoading(true);
        setMessages(prev => [...prev, {
            content: message,
            isUser: true,
            time: new Date().toLocaleTimeString()
        }]);

        // 调用后端API
        try {
            const response = await fetch('/api/rag/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: message})
            });
            if (response.ok) {
                const data = await response.json();
                const msg = data.message;
                setMessages(prev => [
                    ...prev,
                    {
                        isUser: false,
                        time: new Date().toLocaleTimeString(),
                        content: msg // 直接存 message 对象，渲染时判断
                    }
                ]);
                setLoading(false); // 回复后关闭 loading
            }
        } catch (error) {
            console.error('API请求失败:', error);
        }

        setMessage('');
    };

    return (
        <>
            <FloatButton
                icon={open ? <CloseOutlined/> : <MessageOutlined/>}
                type="primary"
                style={{right: 24, bottom: 24}}
                tooltip={{
                    title: '智能助手',
                    color: 'blue',
                    placement: 'top',
                }}
                onClick={() => setOpen(!open)}
            />

            <Modal
                title="智能助手"
                open={open}
                footer={null}
                onCancel={() => setOpen(false)}
                bodyStyle={{padding: 0}}
                width={400}
            >
                <div style={{height: '60vh', display: 'flex', flexDirection: 'column'}}>
                    <div style={{flex: 1, overflowY: 'auto', padding: 16}}>
                        <Spin spinning={loading}>
                            <List
                                dataSource={messages}
                                renderItem={item => (
                                    <List.Item style={{
                                        flexDirection: item.isUser ? 'row-reverse' : 'row'
                                    }}>
                                        <div style={{
                                            background: item.isUser ? '#1890ff' : '#f5f5f5',
                                            color: item.isUser ? 'white' : 'black',
                                            padding: 8,
                                            borderRadius: 8,
                                            maxWidth: '80%'
                                        }}>
                                            {item.isUser ? (
                                                <>
                                                    {item.content}
                                                    <div style={{fontSize: 12, marginTop: 4}}>
                                                        {item.time}
                                                    </div>
                                                </>
                                            ) : (
                                                <>
                                                    {item.content.step_by_step_analysis && (
                                                        <div style={{marginBottom: 8}}>
                                                            <b>分析过程：</b><br/>
                                                            <span style={{whiteSpace: 'pre-line'}}>{item.content.step_by_step_analysis}</span>
                                                        </div>
                                                    )}
                                                    {item.content.final_answer && (
                                                        <div style={{marginBottom: 8}}>
                                                            <b>答复：</b>{item.content.final_answer}
                                                        </div>
                                                    )}
                                                    {item.content.relevant_report && item.content.relevant_report.length > 0 && (
                                                        <div style={{marginBottom: 8}}>
                                                            <b>相关报告：</b>
                                                            <ul style={{paddingLeft: 20}}>
                                                                {item.content.relevant_report.map((r, idx) => (
                                                                    <li key={idx}>{r}</li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    )}
                                                    <div style={{fontSize: 12, marginTop: 4}}>
                                                        {item.time}
                                                    </div>
                                                </>
                                            )}
                                        </div>
                                    </List.Item>
                                )}
                            />
                        </Spin>
                    </div>

                    <Input.Search
                        placeholder="输入您的问题"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onSearch={handleSend}
                        enterButton="发送"
                        style={{borderTop: '1px solid #f0f0f0'}}
                    />
                </div>
            </Modal>
        </>
    );
}