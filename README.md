# EduEval Pipeline

本仓库提供 Edu-Eval 评测集及交互式教学 HTML 的生成、浏览器测试、
L1–L4 自动评估和反馈修复工具。代码采用 MIT License；公开仓库前请确认
`data/README.md` 所述数据来源与再发布授权。

## 安装与运行

建议在独立 Python 3.10+ 环境中运行：

```sh
python -m pip install -r requirements.txt
python -m playwright install chromium
python -m streamlit run app.py --server.address 127.0.0.1
```

在界面输入待测模型和裁判模型的 OpenAI-compatible 接口配置，上传
`examples/tasks.sample.jsonl` 做小批量测试，或上传 `data/edu_eval.jsonl` 运行完整
Edu-Eval。模型名必须与服务端实际部署一致。

API Key 可通过界面临时输入，或通过 JUDGE_API_KEY、VLM_API_KEY、TARGET_API_KEY
环境变量设置。`.env.example` 是变量说明，程序不会自动加载 `.env`。
保存配置不会保存这些 API Key；配置文件及运行结果均已加入 `.gitignore`。
真实模型评测会发送页面代码/截图到所配置服务，可能产生费用。

## 数据和产物

`data/edu_eval.jsonl` 收录 226 条中文 K–12 交互式教学 HTML 生成任务，覆盖数学、
语文、英语和物理。字段、分布、来源与使用限制见 [数据集说明](data/README.md)。
校验数据与仓库完整性：

```sh
python scripts/validate_release.py
```

每行 JSON 必须包含非空字符串 `generated_prompt`，可额外包含 `id`。
当前批量程序用输入行序号标识任务；断点续评只适用于输入顺序、模型和评分配置
均未改变的同一轮实验。变更配置后应选择新的输出目录。

流程：任务 → HTMLGenerator → PlaywrightHTMLTester → HTMLErrorDetector → JSON/CSV。
产物按模型分目录保存为 html、screenshots 和 reports，并提供汇总图和 CSV 导出。
`refine_loop.py --help` 提供迭代修复参数。

单独测试已有页面的浏览器行为：

```sh
python playwright_html_tester.py examples/demo.html --output-dir evaluation_outputs/demo
```

该命令不调用大模型，因此不会生成完整 L1–L4 裁判评分。
完整已有页面评估使用 `main.py --html_input ... --output-dir ...`，通过
OPENAI_API_KEY/OPENAI_BASE_URL 配置默认裁判；建议正式实验使用界面指定模型。

## 评分与复现

见 [评分说明](docs/SCORING.md)。L3/L4 是评分，不能当作执行成功率或交互成功率。
整理版默认采用已抽查历史报告记录的 0.20/0.25/0.25/0.30 权重及原有 L4 门控。
尚未逐份审核全部报告权重；论文公式与历史实现的差异需要在正式发布说明中处理。
本次整理没有改变评分提示词或 L4 判分逻辑。

截图抽样、模型服务版本及采样参数会影响复测结果。当前整理版不是精确复现保证。
历史训练数据、模型权重、内部配置和历史评测输出未随代码打包；Edu-Eval 的
226 条评测任务已包含在 `data/edu_eval.jsonl`。

## 浏览器运行环境

生成的 HTML 会执行 JavaScript 并可能访问外网。请在无个人账号、无敏感文件、
网络受控的独立环境中运行。Playwright 本身不等于完整的安全隔离环境。
界面仅绑定本机，不应直接作为公网多用户服务部署。

## 本地验证

```sh
python -m unittest discover -s tests -v
```

运行前阅读 [发布检查项](docs/OPEN_SOURCE_CHECKLIST.md) 和 [整理记录](docs/CHANGES.md)。

## 引用

若本仓库对研究有帮助，请使用 `CITATION.cff` 中的论文信息引用。论文 DOI：
<https://doi.org/10.16383/j.aas.c260145>。
