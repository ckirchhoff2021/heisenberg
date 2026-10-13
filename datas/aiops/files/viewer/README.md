# 数据预览网站

用于浏览时序曲线图像及其标签（`curve`、`recover`）的网页服务，支持数据集切换、多条件筛选、分页与大图预览。

## 依赖

服务端仅使用 Python 标准库（`http.server`、`json`、`pathlib` 等），**无第三方依赖**，无需安装任何包即可运行。

运行环境统一使用项目根目录下的 `.venv`（由 `uv` 根据根目录 `pyproject.toml` 管理）。

## 启动

在**项目根目录**下执行：

```bash
# 首次使用：创建/同步 .venv（viewer 本身不会安装额外依赖）
uv sync

# 方式一：直接使用 .venv 解释器
.venv/bin/python datas/aiops/files/viewer/server.py

# 方式二：通过 uv 运行（自动使用 .venv）
uv run python datas/aiops/files/viewer/server.py
```

默认监听 `http://0.0.0.0:8000`，可自定义地址与端口：

```bash
uv run python datas/aiops/files/viewer/server.py --host 127.0.0.1 --port 8123
```

停止服务：在启动终端按 `Ctrl+C`。

## 使用

- **切换数据集**：页面右上角下拉框，自动列出 `files/jsons/` 下的所有 JSON 文件；
- **筛选**：支持按 `curve`、`recover`、图像来源过滤，并可按图像文件名搜索；
- **预览**：点击任意卡片查看大图。

## 数据约定

- 输入为 `files/jsons/` 下的 JSON，每条记录包含 `curve`、`recover`、`image`、`from` 字段；
- `image` 仅为文件名，服务按 `from` 字段从 `files/gen` 或 `files/labeled` 目录读取图像，`from` 缺失时自动探测；
- 新增图像来源目录时，在 [server.py](server.py) 的 `IMAGE_DIRS` 中登记即可。

## 文件结构

```
viewer/
├── README.md
├── server.py            # HTTP 服务与 JSON API（标准库实现）
└── static/
    ├── index.html       # 页面结构
    ├── style.css        # 深色渐变样式
    └── app.js           # 前端交互
```
