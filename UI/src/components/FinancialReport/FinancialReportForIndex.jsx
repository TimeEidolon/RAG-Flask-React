import React, {useState, useEffect} from 'react';
import {Table, Button, Space} from 'antd';
import moment from 'moment';

export default function FinancialReportForIndex() {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(false);

    // 表格列配置
    const columns = [
        {
            title: '股票代码',
            dataIndex: 'stock_code',
            key: 'stock_code',
        },
        {
            title: '公司名称',
            dataIndex: 'company_name',
            key: 'company_name',
        },
        {
            title: '财报名称',
            dataIndex: 'report_name',
            key: 'report_name',
        },
        {
            title: '发布时间',
            dataIndex: 'publish_time',
            key: 'publish_time',
            render: (text) => moment(text).format('YYYY-MM-DD')
        },
        {
            title: '操作',
            key: 'action',
            render: (_, record) => (
                <Space size="middle">
                    <Button type="link">查看详情</Button>
                    <Button danger type="link">删除</Button>
                </Space>
            ),
        },
    ];

    // 模拟数据获取
    useEffect(() => {
        setLoading(true);
        // TODO: 替换为真实API调用
        setTimeout(() => {
            setData([{
                stock_code: '600519',
                company_name: '贵州茅台',
                report_name: '2023年度财务报告',
                publish_time: '2024-03-15T14:30:00'
            }]);
            setLoading(false);
        }, 500);
    }, []);

    return (
        <Table
            columns={columns}
            dataSource={data}
            loading={loading}
            rowKey="id"
            pagination={{pageSize: 10}}
        />
    );
}