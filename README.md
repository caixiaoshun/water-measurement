# 动态水面自适应单目测距系统

本仓库提供一套可运行的单目视觉测距实现，并补齐了与合同 7 条验收指标一一对应的自动化测试。

## 一、项目结构

- `main.py`：系统入口与 demo（读取配置、打开视频/相机、鼠标点选、输出距离与诊断）
- `vision_ranging.py`：核心视觉算法
  - 相机参数/去畸变
  - ArUco + Circle 基准点检测
  - 检测器 fallback（Aruco 失败时切到 Circle）
  - PnP 外参更新（RANSAC + 质量门控 + 回退）
  - 射线-平面求交与两点距离计算
- `water_level.py`：水位输入适配（静态值/文件回放）
- `config.yaml`：配置文件
- `tests/test_contract_metrics.py`：合同验收指标测试
- `TECH_REPORT.md`：技术报告

## 二、环境安装

```bash
python -m pip install -r requirements.txt
```

## 三、运行 Demo

```bash
python main.py --config config.yaml --input 0
```

`--input` 支持：
- 相机 ID（如 `0`）
- 本地视频文件路径
- 图片序列目录路径

交互说明：
- 左键：连续点两点触发测距
- 右键或 `C`：清除点位
- `Q`：退出

终端会输出 JSON 结果，包含：
- `distance_m`
- `confidence`
- `elapsed_ms`
- 外参质量（重投影误差、内点比例、状态）
- 水位时间戳与来源

## 四、合同验收测试（严格阈值断言）

执行：

```bash
python -m pytest -q
```

测试文件：`tests/test_contract_metrics.py`

覆盖的合同条款：
1. 测距误差 ≤ 5%（2~50m 工况）
2. 外参自动修正成功率 ≥ 95%（Monte Carlo）
3. 修正后精度恢复 ≥ 90%（并要求后验误差 ≤ 5%）
4. 响应时间 ≤ 1s（采用 P95 口径）
5. 标定通用性 ≥ 3 套相机模型
6. 基准点兼容性 ≥ 2 类，且支持 ArUco 失败后的 Circle fallback
7. 技术报告与核心源码存在性检查

> 注意：测试不做“打印通过”，而是直接按合同阈值做断言，不满足即失败。

## 五、配置说明（`config.yaml`）

关键配置项：
- `camera`：内参 `K`、畸变 `D`、分辨率
- `marker`：主检测类型、fallback 类型、各类型参数
- `water_plane`：水面平面定义
- `water_level`：静态或文件回放源
- `extrinsics`：PnP/RANSAC 与质量阈值
- `runtime`：并行阈值、显示刷新等

## 六、备注

- 统一单位为米（m）。
- 坐标定义采用 `Xc = R * Xw + t`。
- 本仓库侧重离线可复现与验收对齐，便于后续对接真实现场数据。
