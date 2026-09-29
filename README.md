# Index-Translate

[Index-Translate](https://huggingface.co/collections/IndexTeam/index-translate) 是专为翻译任务训练的模型系列，覆盖百余种语言方向，包含 2B / 9B 两个规格，可在消费级显卡上本地运行。

## 模型下载

- **Hugging Face**：[IndexTeam · index-translate collection](https://huggingface.co/collections/IndexTeam/index-translate)
  - [`IndexTeam/Index-Translate-2B`](https://huggingface.co/IndexTeam/Index-Translate-2B)
  - [`IndexTeam/Index-Translate-9B`](https://huggingface.co/IndexTeam/Index-Translate-9B)
- **ModelScope**：[IndexTeam 组织主页](https://modelscope.cn/organization/IndexTeam)（同名仓库）

## 网页翻译插件（extension/）

[`extension/`](./extension) 是一个配合本地部署 Index-Translate 模型使用的浏览器双语翻译插件（Chrome / Edge / Firefox，Manifest V3，纯原生 JS 零构建）：

- 整页双语对照 / 替换原文 / 划词翻译，一键还原
- 通过 vLLM / SGLang 的 OpenAI 兼容接口调用本地模型，网页内容不出本机
- 内含完整的本地部署教程（vLLM / SGLang 启动命令、显存参考、配置对照表）

详见 [extension/README.md](./extension/README.md)。

## 开源协议

[Apache License 2.0](LICENSE)
