# 动态水面自适应单目相机视觉测距系统

> Adaptive Monocular Visual Ranging System for Dynamic Water Surfaces

![Web Demo](https://github.com/user-attachments/assets/753802cf-d440-4026-9266-9125e3f51aae)

## 项目简介

本项目为"面向动态水面场景的自适应单目相机视觉测距系统"。用户在视频画面中点选任意两点，系统输出两点在水面参考坐标系下的实际距离。系统适应水位变化、相机姿态漂移与户外视觉干扰（反光、遮挡、噪声、低纹理等），并提供可解释的诊断信息。

典型使用场景：河道、渠道、水库等水域的边坡/水工监测，用于替代或补充人工丈量，提升巡检与工程监测效率。

## 合同验收指标

| # | 指标 | 要求 | 测试口径 |
|---|------|------|----------|
| 1 | 测距误差 | 相对误差 ≤ 5%（2–50 m 工况） | 多高度 × 多距离逐点断言 |
| 2 | 外参自动修正成功率 | ≥ 95%（平移 >5 cm 或倾斜 >2°） | Monte Carlo N=120（合同测试）/ N=100（扩展测试），旋转/平移误差阈值 |
| 3 | 修正后精度恢复 | ≥ 90%（Recovery = 1 − \|D_post−D_true\|/(|D_pre−D_true\|+ε)） | 断言 recovery ≥ 0.9 且后验误差 ≤ 5% |
| 4 | 响应时间 | ≤ 1 s | P95 口径，60 次采样 |
| 5 | 标定通用性 | ≥ 3 种相机型号 | 3 套不同 camera_model 走通初始化 + 测距 |
| 6 | 基准点兼容性 | ≥ 2 类（ArUco + 圆形标志） | ArUco 失效时 circle fallback 实际更新外参 |
| 7 | 输出成果 | 技术报告 + 全部源代码 | 存在性检查 + 可导入检查 |

## 快速开始

### 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 运行测试

```bash
python -m pytest -v
```

测试包含：
- **tests/test_basic.py** — 基础单元测试（去畸变、射线求交、距离计算、PnP 回归）
- **tests/test_contract_metrics.py** — 合同 7 条验收指标对应的自动化测试
- **tests/test_complex_scenarios.py** — 扩展复杂场景测试（像素抖动、水位变化、姿态漂移、标记退化、退化几何、畸变噪声、极值边界）

所有测试通过断言控制，不满足阈值则测试失败。

### 运行 CLI Demo

```bash
python main.py --config config.yaml --input 0
```

`--input` 支持：相机 ID（如 `0`）、视频文件路径、图片序列目录。

交互：
- **左键**：连续点两点触发测距
- **右键或 C**：清除点位
- **Q**：退出

终端输出 JSON 结果，包含距离、置信度、耗时、外参质量、水位信息。

### 启动 Web 演示

1. 启动后端：

```bash
uvicorn web.app:app --host 0.0.0.0 --port 8765
```

2. 浏览器访问 http://localhost:8765

3. 在画布上点选两点，右侧显示测距结果与诊断信息（置信度、耗时、入射角、状态）。可通过水位输入框调整水面高度。

**API 端点：**

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/config` | 当前配置摘要 |
| POST | `/measure` | 测距（传入两像素坐标，返回距离 + 诊断） |
| GET | `/` | 前端页面 |

## 目录结构

```
├── vision_ranging.py        # 核心视觉算法（相机参数、基准点检测、PnP外参、测距）
├── water_level.py           # 水位输入适配（静态值/文件回放/融合）
├── main.py                  # CLI 入口（配置读取、视频流、鼠标点选、结果输出）
├── config.yaml              # 默认配置文件
├── requirements.txt         # Python 依赖
├── pytest.ini               # pytest 配置
├── TECH_REPORT.md           # 技术报告（合同验收版）
├── README.md                # 本文件
├── tests/
│   ├── test_basic.py                # 基础单元测试
│   ├── test_contract_metrics.py     # 合同 7 条指标测试
│   └── test_complex_scenarios.py    # 扩展复杂场景测试
└── web/
    ├── app.py               # FastAPI 后端
    └── static/
        └── index.html       # 前端页面（点选测距 + 诊断面板）
```

## 配置说明（config.yaml）

| 配置项 | 说明 |
|--------|------|
| `camera` | 内参 K、畸变 D、分辨率、型号名称 |
| `marker` | 主检测类型（aruco/circle）、fallback 类型、各类型参数 |
| `water_plane` | 水面平面定义（高度或一般平面方程） |
| `water_level` | 水位源（static / file） |
| `extrinsics` | PnP/RANSAC 参数与质量门控阈值 |
| `runtime` | 并行阈值、显示刷新、点选后清除 |
| `output` | 标注帧保存目录 |
| `logging` | 日志级别与文件路径 |

## 已知限制

- 当前版本使用合成/mock 数据验证，未接入真实 RTSP 流；生产部署需对接实际相机。
- 水位源仅支持静态值和文件回放；生产环境可扩展 API/数据库源。
- 近水平视角（入射角 < 5°）或极远距离（>50 m）情况下测距误差可能增大，系统通过置信度分数和入射角诊断提示用户。
- Web 演示使用固定默认外参，不包含实时标记检测流程。

## 诊断输出示例

```json
{
  "success": true,
  "distance_m": 12.345,
  "confidence": 0.92,
  "elapsed_ms": 0.38,
  "message": "ok",
  "diagnostics": {
    "plane": {"normal": [0.0, 0.0, 1.0], "d": -0.5},
    "grazing": [0.45, 0.47],
    "extrinsics_score": 0.88
  }
}
```

- `confidence`：综合几何评分（入射角）与外参质量。
- `grazing`：射线与水面的入射角余弦值，越小表示越接近平行（不稳定）。
- `extrinsics_score`：基于重投影误差和内点比例的外参质量评分。
