# inference · Index-Translate 模型推理快速上手

[English](README.md)

面向 [Index-Translate 开源模型家族](https://huggingface.co/collections/IndexTeam/index-translate)
的最小化推理脚本与实测用例（发布前全部对线上同款权重跑通）。

## 模型一览

| 模型 | 任务 | 入口 |
|---|---|---|
| [Index-Translate-2B](https://huggingface.co/IndexTeam/Index-Translate-2B) / [9B](https://huggingface.co/IndexTeam/Index-Translate-9B) | 通用文本翻译（中/英/日等 20+ 语种） | [`llm/`](llm/) |
| [Index-Nailong-2B](https://huggingface.co/IndexTeam/Index-Nailong-2B) / [9B](https://huggingface.co/IndexTeam/Index-Nailong-9B) | 长文档翻译（262K 上下文，中↔英 / 中↔日） | [`llm/`](llm/) |
| [Index-Homura-2B](https://huggingface.co/IndexTeam/Index-Homura-2B) / [9B](https://huggingface.co/IndexTeam/Index-Homura-9B) | 音节控制翻译（对齐配音时长） | [`llm/`](llm/) |
| [Index-Echo-S2ST-2B](https://huggingface.co/IndexTeam/Index-Echo-S2ST-2B) / [9B](https://huggingface.co/IndexTeam/Index-Echo-S2ST-9B) | 语音到语音配音（中/英源 → 英/西/日/中，保留音色） | [`echo-s2st/`](echo-s2st/) |
| [Index-Echo-S2TT-2B](https://huggingface.co/IndexTeam/Index-Echo-S2TT-2B) / [9B](https://huggingface.co/IndexTeam/Index-Echo-S2TT-9B) | 语音转文字翻译（中文音视频 → 英/日/西字幕） | [`echo-s2tt/`](echo-s2tt/) |

## 一分钟跑通

```bash
# --- 文本类 LLM（Translate / Nailong / Homura 通用）---
pip install -U vllm            # 需要带 Qwen3.5 支持的版本
./llm/serve_vllm.sh translate-9b
python llm/translate.py "你好，世界" --target en

# --- 语音配音 ---
huggingface-cli download IndexTeam/Index-Echo-S2ST-2B --local-dir ./Index-Echo-S2ST-2B
python echo-s2st/dub.py input.wav --lang en -o dub_en.wav

# --- 语音转字幕 ---
python echo-s2tt/s2tt.py input.mp4 --lang en   # 自动下载 2B 包
```

各目录 README 有详细说明；`llm/cases/` 收录了真实运行的输入输出样例。

相关项目：[`../video-dub/`](../video-dub/) 把这些模型串成完整视频配音管线
（人声分离、VAD 切分、时间轴回贴）。
