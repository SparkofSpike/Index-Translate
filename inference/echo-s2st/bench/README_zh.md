# S2ST 后端提速基准

[English](README.md) | 中文

对应 2026-09-29 单卡基准报告（`REPORT.md` §5）列的两个提速方向的验证实验：

1. **CosyVoice3 LM 解码换 vLLM 后端**——包里自带开关
   （`CosyVoice3(load_vllm=...)`，当前在 `code/tts/pipeline.py` 里写死关闭）。
   管线的 `synth()` 走 `llm.inference_wrapper(...)` 生成语音 token，该方法在
   挂载 vLLM 引擎后会自动切换路径——所以整个实验是**纯运行时补丁**，
   不改动模型包任何文件。
2. **ST LM `torch.compile`**——ST LM 解码走 HF `generate()`；
   编译 `llm.forward` 可加速 decode 循环（采用 `mode="default"`——
   CUDA graph 模式与该架构不兼容，见结果部分的排除路径说明）。

`bench_dub.py` 按与原始报告相同的方法论分别测量各模式（zh→en，
从包内 `input_zh.wav` 裁剪 1/3/6/10s 四档，预热 1 次 + 计时 3 次取均值，
指标含 wall / stlm / tts / 峰值显存）。

## 用法

```bash
# 基线（对照，复现原始数据）
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode baseline

# CosyVoice3 LM 走 vLLM（一次性导出到旁路目录，模型包保持只读）
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode vllm \
    --vllm-export /path/to/s2st_2b_vllm_export

# ST LM torch.compile
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode compile

# 两者叠加
python bench_dub.py --model-dir /path/to/dubbing_2b_fulldir_cv3 --mode vllm+compile
```

每个进程只跑一种模式——vLLM 补丁会释放 torch 侧 LM 层
（`del ...layers`），除提供的 `vllm+compile` 组合外不可在单进程内切换。

说明：

- **不要在服务线上推理的 GPU 上跑基准**（例如挂着在线 S2ST/S2TT worker
  的机器）。用空闲卡；显存不够就换机器或排队，**严禁停线上进程腾显存**。
- TensorRT（`load_trt`）**暂不覆盖**：包内未附带 flow decoder 的
  `.plan`（GPU 相关产物），需要先在本机做 ONNX→TRT 导出。vLLM 是该方向
  性价比最高的 80%。
- vLLM 引擎按上游默认 `gpu_memory_utilization=0.2` 创建——A100 80G 上约
  额外占 16 GB。
- JSON 输出保留了每个 clip 第 0 次的转写/译文文本，可与 baseline 逐字
  对比做质量抽检。

## 结果（2026-09-29，A100 80G，`dubbing_2b_fulldir_cv3`）

隔离 venv：py3.12 + torch 2.13+cu129 + vllm 0.29.0+cu129；baseline/vllm
两组用 transformers 5.6.0（与生产一致），compile 两组用 transformers
5.17.0（原因见下方排除路径）。RTF = 耗时 / 片段时长，越小越好。
四组模式在全部片段上**转写/译文逐条一致**（质量无损）。

| 片段 | baseline | vllm | compile | vllm+compile |
|---|---|---|---|---|
| 1s | 4.52 | 2.94 | 2.99 | **1.05** |
| 3s | 1.72 | 1.00 | 0.98 | **0.44** |
| 6s | 1.18 | 0.69 | 0.81 | **0.35** |
| 10s | 0.68 | 0.42 | 0.50 | **0.19** |

10s 档阶段拆分（wall 秒数均值）：ST LM 2.47 → 2.44（vllm）→ 0.39
（compile）→ 0.36（叠加）；TTS 4.28 → 1.72（vllm）→ 4.57（compile）→
1.54（叠加）。vllm 组峰值显存反降 8.2G → 6.8G。

**叠加后端到端提速 3.6×（10s 档）**——远超原报告的 30% 预期。

### 已排除路径（留档备查）

- `mode="reduce-overhead"` + 静态 KV cache 在该混合线性注意力架构上
  **不可行**：transformers 5.6 报 `KeyError: linear_attention`
  （masking_utils 映射表无混合架构条目），5.17 下 inductor 编译阶段
  段错误（`conv_state.copy_` 与 cudagraphs 不兼容）。上表 compile
  数据采用 **`mode="default"`**（无 CUDA graph），稳定且已拿到大部分
  收益（ST LM 阶段 ~7×）。
- vLLM 组需一处环境补丁：vllm 0.29 sampler 常量
  `MAX_NUM_STOP_TOKEN_IDS=128` 小于 CosyVoice3 的 200 个 stop token，
  在 bench venv 中改为 256（tensor 宽度由该常量决定、kernel 按 shape
  自适应，安全）。属环境注记，不改上游代码。

## 输出

`bench_<mode>.json` + stdout 表格：

```
clip    wall    stlm     tts   peak   RTF
   1s   4.36    ...      ...   8.3G  4.36
```
