# INT8 卷积累加器位宽检查器（单层）

端侧部署场景：确认现有 INT8 卷积层能否换用**更窄的有符号累加器**。
本服务对单个量化卷积层做**精确**的最坏情况分析，回答两个问题：

1. 覆盖所有输出位置、所有累加步骤的**最小安全有符号位宽**是多少？
2. 若目标位宽不足，**最早可能溢出**发生在第几步、哪个输出坐标？并给出
   能复算该溢出的局部输入（激活取值）。

## 语义约定（与分析器严格一致）

- 二维卷积，**步长 1**，**奇数核**（`kh`、`kw` 均为奇数），输出尺寸 = 输入尺寸。
- 边界用**对应通道的激活零点**填充，因此填充抽头的项恒为
  `(zp - zp) * w = 0`。
- 累加顺序：**偏置为第 0 步**，之后按 **输入通道 → 核行 → 核列** 的顺序累加，
  第一个卷积抽头是第 1 步。
- 每一项 = `(激活值 - 该通道零点) * 权重`。
- 每个位置的激活可在其通道的 `[min, max]` 内**独立**取任意整数值。
- 由于同一输出位置的各抽头读取互不相同的输入单元且各单元独立，逐步区间
  算术是**精确**的（非保守）：第 k 步部分和的极值 = 前 k 项逐项极值之和，
  且该极值可达（溢出响应中给出的局部输入即为证人）。

## 运行

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务监听 http://127.0.0.1:8000 ，交互式文档见 /docs
```

### 本地开发

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r app/requirements.txt
cd app && uvicorn main:app --port 8000
```

### 演示

```bash
./examples/demo.sh                 # 对 http://127.0.0.1:8000 发起两个样例请求
```

## API

### `POST /v1/analyze`

请求体：

| 字段 | 说明 |
| --- | --- |
| `feature_map` | `{height, width}`，输入特征图尺寸（1..1024） |
| `activations` | 逐输入通道 `[{min, max, zero_point}]`，uint8 域 `0..255`，`min <= max` |
| `weights` | INT8 权重，嵌套数组 `[out_ch][in_ch][ky][kx]`，值域 `[-128, 127]`，核两维均为奇数 |
| `bias` | 逐输出通道整数偏置（int64），即累加第 0 步 |
| `target_bits` | 待评估的有符号累加器位宽（1..64） |

响应要点：

| 字段 | 说明 |
| --- | --- |
| `min_safe_bits` | 所有位置、所有步骤都安全的最小有符号位宽 |
| `target_ok` | `target_bits` 是否足够 |
| `signed_range_for_target` | 目标位宽对应的有符号区间 `[lo, hi]` |
| `stats.accumulation_steps` | 累加步数 = `in_ch * kh * kw`（不含第 0 步偏置） |
| `extremes` | 全程最小/最大部分和及其首次出现位置（步号 + 输出坐标） |
| `overflow` | 目标位宽不足时非空：最早可能溢出的 `step`、`output` 坐标、<br>`direction`、`partial_sum`、`exceeds_by`，以及 `terms` ——<br>第 1..step 步的局部输入（输入坐标、激活取值、零点、权重、<br>项值、逐步部分和），重放 `bias + terms` 即可复算该溢出 |

`GET /health` 健康检查；`GET /` 服务信息；`GET /docs` Swagger UI。

### 样例

`examples/request_ok.json`：8×8 特征图、3 输入通道、2 输出通道、3×3 核，
目标 16 位 →

```json
{ "min_safe_bits": 14, "target_ok": true, "overflow": null, ... }
```

`examples/request_overflow.json`：同一层、目标 12 位 →

```json
{
  "min_safe_bits": 14, "target_ok": false,
  "overflow": {
    "step": 5, "output": {"y": 1, "x": 1, "channel": 0},
    "direction": "positive", "accumulator_range": [-2048, 2047],
    "partial_sum": 2472, "exceeds_by": 425, "bias": 1200,
    "terms": [
      {"step": 1, "padded": false, "input": {"y": 0, "x": 0, "channel": 0},
       "activation": 255, "zero_point": 128, "weight": 3, "term": 381,
       "partial_sum": 1581},
      ...
    ]
  }
}
```

复算：`1200 + 381 + 256 + 127 + 0 + 508 = 2472 > 2047`，第 5 步溢出成立。

## 正确性验证

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

- **暴力枚举交叉验证**：对 60 组随机小规格（含零点落在 `[min,max]` 之外、
  负权重、多通道、边界位置），穷举所有激活组合逐位置逐步模拟，与分析器的
  `min_safe_bits`、极值、最早溢出步号逐一比对一致。
- **溢出证人重放**：响应中的 `terms` 逐步重放，部分和与报告值一致且越界；
  每个激活取值都在其通道范围内；填充项恒为 0。
- 手算用例：1×1 核、纯偏置溢出（第 0 步）、2×2 地图 3×3 核的填充行为等。
- API 层：422 校验（偶数核、越界 INT8、形状不匹配、`min>max` 等）。

## 复杂度与限制

- 扫描量 ≈ `输出通道数 × 位置类型数 × 累加步数`；位置类型数 ≤
  `min(kh, H) × min(kw, W)`（3×3 核至多 9 类）。纯 Python 实现，工作量上限
  2×10⁷（超出返回 413）。典型单层（如 3×3、64→64）为亚秒级。
- 仅处理步长 1、奇数核、零点填充、输出尺寸不变的单层；不做网络级联分析。
- 权重经 JSON 嵌套数组传输，适合单层检查；超大层会受上述工作量上限约束。

## 目录

```
app/            FastAPI 服务（main.py / analyzer.py / schemas.py）
app/Dockerfile  镜像定义（python:3.12-slim）
docker-compose.yml
tests/          分析器单测（含暴力枚举交叉验证）+ API 测试
examples/       样例请求与 demo.sh
```
