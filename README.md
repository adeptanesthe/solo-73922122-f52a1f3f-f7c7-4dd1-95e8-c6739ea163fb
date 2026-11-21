# INT8 卷积检查器（单层）

端侧部署场景下的单层 INT8 卷积检查服务，提供两个接口：

1. **`POST /v1/analyze`** — 累加器位宽检查：确认现有 INT8 卷积层能否换用
   **更窄的有符号累加器**。对该层做**精确**的最坏情况分析，回答：
   - 覆盖所有输出位置、所有累加步骤的**最小安全有符号位宽**是多少？
   - 若目标位宽不足，**最早可能溢出**发生在第几步、哪个输出坐标？并给出
     能复算该溢出的局部输入（激活取值）。

2. **`POST /v1/layout-check`** — 权重布局等价性校验：设备上传 base64 打包
   权重，按声明的分块与轴顺序解码回规范布局，先**逐元素核对有效权重**，
   再用一张具体 uint8 特征图**逐步重放卷积**，定位首个差异的完整索引、
   输出坐标与两路部分和。布局不合法、块尾非零填充或字节数不符返回 422。

## 语义约定（两个接口一致）

- 二维卷积，**步长 1**，**奇数核**（`kh`、`kw` 均为奇数），输出尺寸 = 输入尺寸。
- 边界用**对应通道的激活零点**填充，因此填充抽头的项恒为
  `(zp - zp) * w = 0`。
- 累加顺序：**偏置为第 0 步**，之后按 **输入通道 → 核行 → 核列** 的顺序累加，
  第一个卷积抽头是第 1 步。
- 每一项 = `(激活值 - 该通道零点) * 权重`。
- `/v1/analyze` 中每个位置的激活可在其通道的 `[min, max]` 内**独立**取任意
  整数值；由于同一输出位置的各抽头读取互不相同的输入单元且各单元独立，
  逐步区间算术是**精确**的（非保守）。
- `/v1/layout-check` 使用**一张给定**的 uint8 特征图，因此重放得到的是该
  输入下的具体两路累加序列。

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
./examples/demo.sh                 # 对 http://127.0.0.1:8000 发起四个样例请求
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

### `POST /v1/layout-check`

请求体：

| 字段 | 说明 |
| --- | --- |
| `weights` | 参考 INT8 权重张量，**按其 `layout.decoded_axis_order` 嵌套** |
| `bias` | 逐输出通道整数偏置（int64），累加第 0 步 |
| `feature_map` | `{height, width, data}`，`data` 为 `[in_ch][y][x]` 的 uint8 值 |
| `input_zero_points` | 逐输入通道 uint8 零点，长度为 `dimensions.input_channels` |
| `dimensions` | 各轴尺寸 `output_channels / input_channels / kernel_height / kernel_width` 与分块 `output_channel_block / input_channel_block / kernel_block`（kernel 两轴共用一个块大小），核两维必须为奇数 |
| `layout.decoded_axis_order` | 参考权重的轴顺序，`oc/cin/ky/kx` 的一个排列 |
| `layout.packed_axis_order` | 设备打包缓冲的轴顺序，`oc/cin/ky/kx` 的一个排列 |
| `packed_weights_base64` | 设备打包权重，base64（标准字母表，严格校验）；按**有符号 int8** 解码 |

打包缓冲格式：对每个轴，按 `ceil(extent / block)` 个块、每块 `block` 项存放，
**块号在外、块内下标在内**；未分块即 `block == 1`。因此填充只出现在某轴最后
一个块的尾部，**所有填充字节解码后必须为 0**。四个轴按
`packed_axis_order` 嵌套（第一个轴最外层）。字节数必须精确等于
`∏ ceil(extent/block)`；不足或多余均返回 422。

响应要点：

| 字段 | 说明 |
| --- | --- |
| `matches` | `weights_match && replay_match` |
| `weights_match` | 有效权重逐元素完全一致 |
| `replay_match` | 该特征图下两路累加逐步一致 |
| `layout` | 回显轴顺序、分块尺寸、`signed_int8: true` |
| `packed` | `bytes`、`valid_elements`、`padding_elements` |
| `weight_check.mismatch_count` | 有效权重差异个数 |
| `weight_check.first_mismatch` | 按 `(oc, cin, ky, kx)` 规范序的**首个**差异：`index`、`reference`、`packed` |
| `replay.first_difference` | 首个两路部分和不等的抽头：`index`、`output`、`step`、激活/零点/`delta`、两项贡献、`reference_partial_sum`、`packed_partial_sum`；未分叉为 `null` |
| `replay.target_occurrence` | 首个差异权重（若有权重差异）首次被实际读取的位置及其两路部分和；该特征图未读到则为 `null` |
| `difference` | 汇总定位：`kind`（当前恒为 `weight_mismatch`）、`index`、`reference_weight`、`packed_weight`、`output`、`step`、`reference_partial_sum`、`packed_partial_sum`。以元素核对的首个差异为准，`output`/`step` 取自重放中该抽头首次被实际读取的位置；若给定特征图从未读到该抽头（如 `delta == 0`），则这几项为 `null`，但 `weight_check` 仍给出差异索引。 |

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

`examples/layout_match.json`：2 输出通道 × 2 输入通道 × 3×3 核，
`packed_axis_order = ["kx","ky","cin","oc"]`、三轴均按 2 分块（64 字节，
其中 28 个零填充槽），4×4 特征图 →

```json
{ "matches": true, "weights_match": true, "replay_match": true,
  "packed": {"bytes": 64, "valid_elements": 36, "padding_elements": 28},
  "difference": null, ... }
```

`examples/layout_mismatch.json`：同一缓冲，把 `(oc=0, cin=1, ky=1, kx=2)`
的打包字节由 `1` 改为 `2` →

```json
{
  "matches": false,
  "weight_check": {"mismatch_count": 1,
    "first_mismatch": {"index": {"output_channel": 0, "input_channel": 1,
      "kernel_y": 1, "kernel_x": 2}, "reference": 1, "packed": 2}},
  "difference": {
    "kind": "weight_mismatch",
    "index": {"output_channel": 0, "input_channel": 1, "kernel_y": 1, "kernel_x": 2},
    "output": {"y": 0, "x": 0, "channel": 0}, "step": 15,
    "reference_partial_sum": -103, "packed_partial_sum": -101
  }
}
```

复算：该位置为 padding 边界，输出 `(0,0)` 的第 15 步读到输入 `(1,2)`，
激活 102、零点 100，`delta = 2`；参考项 `2×1 = 2`、打包项 `2×2 = 4`，
两路部分和相差 2。

样例由 `examples/make_layout_samples.py` 用标准库确定性生成，可重复运行。


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
- **布局等价性**：`tests/test_layout.py` 用一份**独立实现**的打包器
  （不复用生产代码的坐标生成）交叉验证——全部 24 种轴排列、多组分块
  （含 28 个尾填充槽）、`[-128, 127]` 全域取值往返一致；篡改单个权重后
  断言首个差异索引、首次读取的输出坐标与两路部分和；非零尾填充、字节不足/
  多余、非法 base64、非排列轴顺序、尺寸/零点/特征图不匹配均断言 422。

## 复杂度与限制

- `/v1/analyze` 扫描量 ≈ `输出通道数 × 位置类型数 × 累加步数`；位置类型数 ≤
  `min(kh, H) × min(kw, W)`（3×3 核至多 9 类）。
- `/v1/layout-check` 扫描量 ≈ `输出通道数 × H × W × 累加步数`（重放整张
  特征图，不提前退出），打包解码 ≈ `∏ ceil(extent/block)`。
- 两者共用纯 Python 工作量上限 2×10⁷，超出返回 413；打包元素上限 4×10⁶。
  典型单层（如 3×3、64→64）为亚秒级。
- 仅处理步长 1、奇数核、零点填充、输出尺寸不变的单层；不做网络级联分析。
- 权重经 JSON 嵌套数组传输，适合单层检查；超大层会受上述工作量上限约束。
- 打包缓冲约定：每轴 `ceil(extent/block)` 块、块内连续，块尾仅允许零填充；
  不支持块内其他填充值或非零 padding。

## 目录

```
app/            FastAPI 服务（main.py / analyzer.py / layout.py / schemas.py）
app/Dockerfile  镜像定义（python:3.12-slim）
docker-compose.yml
tests/          分析器单测（含暴力枚举）+ 布局等价性交叉验证 + API 测试
examples/       样例请求（旧接口 2 个、新接口 2 个）、打包器与 demo.sh
```
