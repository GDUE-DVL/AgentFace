# AgentFace — 智能美颜 Agent

MIMO 多模态分析 + 4090 美颜推理 + 双向偏好学习。

## 架构

```
                         ┌──────────────────────┐
                         │     🌐 MIMO API       │
                         │  xiaomimimo.com/v1    │
                         │  多模态视觉分析        │
                         └──────────┬───────────┘
                                    │ HTTPS
                                    ▼
┌─────────┐    HTTP     ┌──────────────────────┐    SSH Tunnel    ┌─────────────────┐
│  用户    │ ────────→  │    🧠 AgentFace      │ ───────────────→ │  🎨 4090 服务器  │
│  浏览器   │ ←──────── │    :8000 FastAPI     │ ←─────────────── │  :8899 deepfrr  │
│  Web UI  │           │                      │                  │  美颜推理模型     │
└─────────┘            │  ┌────────────────┐  │                  └─────────────────┘
                       │  │ LangGraph 状态机 │  │
                       │  │ 9节点工作流      │  │
                       │  │ HITL 确认/反馈   │  │
                       │  ├────────────────┤  │
                       │  │ SQLite 永久记忆  │  │
                       │  │ 偏好·历史·反馈   │  │
                       │  └────────────────┘  │
                       └──────────────────────┘
```

## 快速开始

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 配置
cp .env.example .env
# 编辑 .env 填入 MIMO_API_KEY 和美颜模型地址

# 3. 启动（只需一个进程）
PYTHONPATH=src python -m uvicorn agent_face.main:app --host 0.0.0.0 --port 8000
```

打开 http://localhost:8000/app

## API

```
POST   /api/v1/sessions                 创建美颜会话
GET    /api/v1/sessions/{id}            查询会话状态
POST   /api/v1/sessions/{id}/confirm    确认/调整方案
POST   /api/v1/sessions/{id}/feedback   提交反馈
GET    /api/v1/users/{id}/preferences    获取偏好
GET    /api/v1/users/{id}/history        历史会话
DELETE /api/v1/users/{id}/preferences    重置偏好
GET    /api/v1/health                    健康检查
```

## 工作流

```
上传照片 → MIMO 分析(脸型/肤质/光线) → 确认/调整参数 → 4090 美颜 → 打分 → 记忆更新
```

## 项目结构

```
src/agent_face/
├── main.py              # FastAPI 入口
├── config.py            # 配置
├── api/                 # REST API 层
├── langgraph_brain/     # 状态机 + 记忆
│   ├── nodes/           # 9 个工作流节点
│   └── memory/          # 偏好/历史/反馈
├── maf_body/            # 模型调度 + 安全
├── bridge/              # 桥接层
├── models/              # MIMO + 美颜客户端
└── static/              # Web UI
```
