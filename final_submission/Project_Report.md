# EOG + IMU 智能人机交互系统 — 项目交付报告

**项目名称**: EOG + IMU Smart Human-Computer Interaction System
**团队成员**: Chenxu Zhang
**所属机构**: Imperial College London — AML Lab
**报告日期**: 2026 年 3 月 7 日

---

## 1. Motivation 与背景

### 1.1 问题背景

传统的人机交互依赖键盘、鼠标和触摸屏，对于肢体功能受限的用户（如渐冻症患者、高位截瘫患者、老年退行性疾病患者）而言，这些输入方式几乎不可用。全球约有 7500 万人需要使用轮椅，其中大量用户同时存在上肢运动障碍，亟需 **免手操作（hands-free）** 的交互方案。

### 1.2 技术动机

- **EOG（Electrooculography，眼电图）** 是一种通过皮肤表面电极检测眼球运动产生的电位变化的技术。相比 EEG 脑电接口，EOG 信号幅度大（数百微伏级）、信噪比高、硬件成本低，且用户几乎无需训练。
- **IMU（Inertial Measurement Unit，惯性测量单元）** 中的陀螺仪可以精确追踪头部运动，提供连续、低延迟的光标定位能力。
- **边缘 AI 部署**：将机器学习模型直接部署到 ESP32 微控制器上，实现端侧实时推理，无需依赖 PC 或云端计算，降低延迟并提高系统独立性。

### 1.3 项目目标

构建一个基于 EOG + IMU 的多模态人机交互系统，实现：

1. **头部控制光标移动** — IMU 陀螺仪追踪头部姿态，映射为屏幕光标位移
2. **眼球动作触发离散操作** — EOG 信号经 AI 模型实时分类，触发鼠标点击、滚轮等操作
3. **双应用场景** — 既能直接控制 PC 鼠标（USB HID），也能通过 ADB 桥接控制安卓虚拟机

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────┐
│        传感器层 (Sensor Layer)        │
│  EOG 电极 (A0 水平, A1 垂直)          │
│  IMU 陀螺仪 (ICM20600, I2C 0x69)     │
└──────────────┬──────────────────────┘
               ↓ 模拟信号 / I2C
┌─────────────────────────────────────┐
│      处理层 (ESP32-S3 微控制器)       │
│                                     │
│  IMU 通路:                           │
│   陀螺仪读取 → 死区滤波 → 灵敏度映射    │
│   → Mouse.move() 连续光标控制         │
│                                     │
│  EOG 通路:                           │
│   ADC 采集 → EMA 低通滤波 → 滑动窗口   │
│   → 14 维特征提取 → RF 模型推理        │
│   → 极性校验 + 动作锁定               │
│   → Mouse.click() / Scroll 离散触发   │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│         输出层 (两种模式)             │
│                                     │
│  模式 A: USB HID Mouse → 直接控制 PC  │
│  模式 B: 串口 → mumu_bridge.py       │
│          → ADB → 安卓虚拟机控制       │
└─────────────────────────────────────┘
```

### 2.2 硬件配置

| 组件 | 型号/规格 | 说明 |
|------|----------|------|
| 微控制器 | ESP32-S3 | 支持 USB HID、Wi-Fi、BLE |
| IMU 传感器 | ICM20600 | 6 轴（3 轴加速度 + 3 轴陀螺仪），I2C 地址 0x69 |
| EOG 电极 | 模拟输入 A0/A1 | 双通道：水平（左右眼动）+ 垂直（上下眼动） |
| ADC 分辨率 | 10-bit | 0–1023，基线约 500 |

---

## 3. 实现过程

### 3.1 数据采集

使用 ESP32 + EOG 电极以 50Hz 采样率采集双通道眼电信号，数据通过串口传输至 PC 端保存为 CSV 文件。

**采集规模**：

| 数据集 | 来源 | 样本量 | 类别 |
|--------|------|--------|------|
| 受试者 1 真实数据 | 实验室采集 | ~79 个 CSV | 6 类（Rest/Up/Down/Left/Right/Blink），含标准/快速/伴眨眼多种变体 |
| 受试者 2 真实数据 | 实验室采集 | ~50 个 CSV | 5 类（标准速度） |
| 受试者 3 真实数据 | 实验室采集 | ~40 个 CSV | 4 类（Up/Down/Left/Right） |
| 合成训练数据 | 算法生成 | 720 个 CSV | 6 类 × 120 个/类 |
| 切片训练数据 | 真实数据切片 | 169 个样本 | 6 类 |

**数据格式**: 每个 CSV 包含 `Program Time [s]`, `data 0` (H 通道), `data 1` (V 通道) 三列。

### 3.2 合成数据生成

由于真实采集数据量有限，我们设计了基于物理模型的合成数据生成方案：

- **基线模拟**：随机游走漂移 + 中频肌电干扰噪声 + 偶发尖峰毛刺（5% 概率）
- **动作波形**：基于正弦函数，加入时间扭曲（模拟扫视非对称性）、非对称包络（峰值位置偏移）、正负半周期幅度不对称（模拟过冲/回弹）
- **眨眼波形**：快速正向尖峰 + 负向回弹，H/V 双通道独立随机
- 每类生成 120 个样本，总计 720 个合成样本，与真实数据混合训练

### 3.3 信号预处理

```
原始信号 → DC 去偏 (减去均值) → EMA 低通滤波 (α=0.2) → 滑动窗口切分 (50 帧)
```

- **DC 去偏**：消除基线漂移，使波形围绕零点振荡
- **EMA 低通滤波**：平滑高频噪声，保留眼动波形特征
- **滑动窗口**：50 帧窗口（1 秒），步进 10 帧，实现准实时检测

### 3.4 特征工程

从每个窗口的 H/V 双通道分别提取 7 个统计特征，共 14 维：

| 编号 | 特征名称 | 计算方式 | 物理含义 |
|------|----------|----------|----------|
| 0 | Std (标准差) | √(Σ(x-μ)²/N) | 信号波动幅度 |
| 1 | Peak-to-Peak | max - min | 峰峰值，衡量动作强度 |
| 2 | Mean | Σx/N (去直流后) | 信号偏移方向 |
| 3 | Max | max(x-μ) | 正向峰值 |
| 4 | Min | min(x-μ) | 负向谷值 |
| 5 | Mean\|Diff\| | Σ\|x[i]-x[i-1]\|/(N-1) | 平均变化速率 |
| 6 | Max\|Diff\| | max(\|x[i]-x[i-1]\|) | 最大瞬时变化 |

### 3.5 模型训练

采用 **Random Forest（随机森林）** 分类器，选择理由：

1. **嵌入式部署友好**：可通过 `micromlgen` 库直接转换为 C++ 代码，适合在 ESP32 上运行
2. **低推理延迟**：决策树的 if-else 结构在微控制器上执行极快
3. **对小样本鲁棒**：集成学习天然抗过拟合

**训练配置**：
- 算法：Random Forest + Gradient Boosting（对比实验）
- 数据：合成数据 + 真实数据混合训练
- 特征标准化：StandardScaler（均值和缩放参数同步导出至 ESP32）
- 交叉验证：Stratified K-Fold

**模型迭代版本**：

| 版本 | 文件 | 训练数据 | 说明 |
|------|------|----------|------|
| v2 | eog_model_v2.joblib | 受试者 1 | 初始版本 |
| v3 | eog_model_v3.joblib | 受试者 1 | 参数调优 |
| v4 | eog_model_v4.joblib | 受试者 1 | 特征优化 |
| esp32 | eog_model_esp32.joblib | 受试者 1 | ESP32 部署版 |
| **esp32_multi** | **eog_model_esp32_multi.joblib** | **受试者 1+2+3** | **最终版，多受试者泛化** |

### 3.6 模型导出与嵌入式部署

**部署流水线**：

```
Python scikit-learn 模型
       ↓ micromlgen 库
C++ 头文件 (EOG_AI_Engine_esp32_multi.h)
  ├─ RandomForest::predict() 函数（决策树 if-else 展开）
  ├─ scale_features() 函数（StandardScaler 参数）
  └─ 编译烧录至 ESP32
```

ESP32 端侧实时推理流程（50Hz 循环）：

1. 读取 EOG 模拟信号 (A0, A1)
2. EMA 低通滤波 (α=0.2)
3. 维护 50 帧滑动窗口，每 10 帧提取 14 维特征
4. 信号门槛检查：P2P > 60 时才触发推理
5. 特征标准化 → Random Forest 分类
6. 后处理：极性校验 + Blink/Up 消歧 + 动作锁定

### 3.7 后处理与稳定性机制

| 机制 | 说明 |
|------|------|
| 信号门槛 | Peak-to-Peak > 60 才触发推理，跳过静止段 |
| 极性校验 | 正/负峰值比 > 1.2 确认 Left vs Right 方向 |
| Blink/Up 消歧 | V 通道 Max\|Diff\| > 42 → Up；< 35 → Blink |
| 动作锁定 | 每次触发后 40 帧（0.8 秒）冷却，防止重复触发 |

---

## 4. 具体完成的内容

### 4.1 PC 鼠标控制（USB HID 模式）

**固件文件**: `final_submission/IMU_EOG_Mouse.ino`

ESP32 通过 USB HID 协议直接模拟为标准鼠标设备，即插即用：

| 输入 | 动作映射 | 实现方式 |
|------|----------|----------|
| 头部转动 (IMU) | 光标移动 | `Mouse.move(gx * 0.02, gy * 0.02)` |
| 向左看 (EOG) | 鼠标左键单击 | `Mouse.click(MOUSE_LEFT)` |
| 向右看 (EOG) | 鼠标右键单击 | `Mouse.click(MOUSE_RIGHT)` |
| 向上看 (EOG) | 滚轮向上 | `Mouse.move(0, 0, 3)` |
| 向下看 (EOG) | 滚轮向下 | `Mouse.move(0, 0, -3)` |
| 眨眼 (EOG) | 保留（不触发） | — |

**关键参数**：
- IMU 死区：|gyro| < 250 视为静止
- IMU 灵敏度：0.02（角速度 → 像素映射系数）
- 采样/推理频率：50Hz（20ms 循环）

### 4.2 安卓虚拟机控制（ADB 桥接模式）

**固件文件**: `final_submission/IMU_EOG_MuMu.ino`
**桥接程序**: `final_submission/mumu_bridge.py`

MuMu 版采用串口输出 + PC 端桥接的架构：

- ESP32 不使用 USB HID，改为通过串口输出结构化数据：`H:{ema_h}\tV:{ema_v}\tCMD:{action}\tGX:{gx}\tGY:{gy}`
- PC 端 `mumu_bridge.py` 解析串口数据，通过 ADB 命令控制 MuMu 安卓模拟器

| 输入 | 虚拟机操作 | ADB 实现 |
|------|-----------|----------|
| 头部转动 (IMU) | 虚拟光标移动 | `adb shell input tap x y` |
| 向上看 (EOG) | 页面上滑 | `adb shell input swipe ... (向上)` |
| 向下看 (EOG) | 页面下滑 | `adb shell input swipe ... (向下)` |
| 向左看 (EOG) | 点击/翻页 | `adb shell input tap` 或边缘翻页 |
| 向右看 (EOG) | 返回 | `adb shell input keyevent BACK` |

**mumu_bridge.py 功能亮点**：
- 支持 `--sim` 键盘模拟模式（无 ESP32 时可用键盘测试）
- 虚拟光标边缘检测：光标在屏幕左/右 5% 区域时，左看触发翻页
- 线程安全的光标位置管理

### 4.3 模型训练与评估体系

| 脚本 | 功能 |
|------|------|
| `main_project/generate_and_train_synthetic.py` | 合成数据生成 + 模型训练完整流程 |
| `main_project/test_model.py` | 模型评估（单文件/批量目录测试） |
| `main_project/predict_continuous.py` | 连续信号滑窗预测与可视化 |
| `main_project/realtime_detect.py` | 实时串口读取 + 模型推理 + 波形可视化 |
| `main_project/transform_params.py` | Scaler/滤波器参数导出为 Arduino C++ 代码 |

### 4.4 数据采集工具

| 工具 | 功能 |
|------|------|
| `EOG_data_collection.py` | PC 端串口数据采集，保存为 CSV |
| `EOG_data_collection_only/` | 纯数据采集用 Arduino 固件（Arduino Nano + ESP32 两版） |
| `main_project/show_dataset.py` | 数据集可视化浏览 |

### 4.5 实时可视化工具

| 工具 | 功能 |
|------|------|
| `main_project/realtime_detect.py` | 实时 EOG 波形 + 模型推理结果显示 |
| `realtime_visualization.py` | EOG 原始信号实时波形显示 |
| `eog_esp32_visulise.py` | ESP32 数据实时可视化 |

---

## 5. 功能效果

### 5.1 PC 鼠标控制演示效果

- 用户佩戴 EOG 电极 + IMU 传感器，通过 USB 连接 ESP32 至电脑
- **头部转动** → 光标实时跟随移动，延迟约 20ms
- **向左看** → 触发鼠标左键单击（可用于打开文件、点击链接等）
- **向右看** → 触发鼠标右键单击（打开右键菜单）
- **向上/下看** → 页面滚动（浏览网页、文档）
- 整体实现免手操作电脑的基本交互

### 5.2 安卓虚拟机控制演示效果

- 通过 MuMu 模拟器运行安卓应用（如电子书阅读器、浏览器）
- 头部控制虚拟光标在屏幕上移动
- 眼球动作触发页面滑动、点击、返回等操作
- 支持键盘模拟测试模式，方便开发调试

### 5.3 模型性能

- 6 类动作（Rest/Up/Down/Left/Right/Blink）分类
- 多受试者泛化训练（3 名受试者数据 + 合成数据）
- ESP32 端侧推理，无需 PC 参与分类计算

---

## 6. 后续展望

### 6.1 短期改进

- **增加训练数据**：采集更多受试者、更多场景的数据，提升模型泛化能力
- **模型精度优化**：尝试 LightGBM、XGBoost 等更强的集成学习算法，或轻量级 CNN
- **参数自适应**：根据不同用户自动校准 IMU 灵敏度和 EOG 信号门槛

### 6.2 中期发展

- **无线化**：将 USB 连接替换为 BLE（蓝牙低功耗），实现无线鼠标
- **Android 原生支持**：通过 BLE HID 直接控制手机/平板，无需 ADB 桥接
- **多模态融合**：将 IMU 和 EOG 数据在特征层融合，提升复杂场景下的识别准确率

### 6.3 长期愿景

- **可穿戴产品化**：将传感器集成到眼镜框架或头带中，实现轻便佩戴
- **更多交互手势**：支持连续凝视、双击眨眼、眼球追踪等更丰富的交互模式
- **辅助功能生态**：与轮椅控制、智能家居等系统联动，构建完整的无障碍辅助生态

---

## 7. 项目文件附件清单

### 7.1 最终交付代码（final_submission/）

| 文件 | 说明 |
|------|------|
| `IMU_EOG_Mouse.ino` | ESP32 固件 — PC 鼠标控制版（USB HID） |
| `IMU_EOG_MuMu.ino` | ESP32 固件 — 安卓虚拟机控制版（串口输出） |
| `EOG_AI_Engine_esp32_multi.h` | AI 推理引擎 — Random Forest 模型 C++ 头文件 |
| `mumu_bridge.py` | PC 端桥接程序 — 串口读取 → ADB 控制 MuMu 模拟器 |

### 7.2 模型文件（main_project/models/）

| 文件 | 说明 |
|------|------|
| `rf_model.pkl` | Random Forest 模型（Python sklearn） |
| `gb_model.pkl` | Gradient Boosting 模型（对比实验） |
| `scaler.pkl` | StandardScaler 预处理器 |
| `models_xkp/eog_model_esp32_multi.joblib` | 最终多受试者 RF 模型 |
| `models_xkp/eog_scaler_esp32_multi.joblib` | 最终多受试者 Scaler |

### 7.3 训练代码（main_project/）

| 文件 | 说明 |
|------|------|
| `generate_and_train_synthetic.py` | 合成数据生成 + 模型训练 |
| `test_model.py` | 模型测试与评估 |
| `predict_continuous.py` | 连续信号预测 |
| `realtime_detect.py` | 实时检测与可视化 |
| `transform_params.py` | 参数导出工具 |
| `show_dataset.py` | 数据集可视化 |
| `head_mouse_sim.py` | 头控鼠标模拟器 |
| `EOG_data_collection.py` | 数据采集脚本 |

### 7.4 训练数据

| 目录 | 说明 |
|------|------|
| `DATA/EOG_data/1/` | 受试者 1 真实采集数据（6 类，~79 个 CSV） |
| `DATA/EOG_data/2/` | 受试者 2 真实采集数据（5 类，~50 个 CSV） |
| `DATA/EOG_data/3/` | 受试者 3 真实采集数据（4 类，~40 个 CSV） |
| `main_project/synthetic_EOG/` | 合成训练数据（6 类 × 120 个，共 720 个 CSV） |
| `main_project/training_data_real/` | 切片后的训练数据（169 个样本） |
| `EOG_data/Real_Test/` | 真实连续测试数据（5 个 CSV） |
| `IMU_EOG/demo case/` | 演示用典型波形数据 |

### 7.5 Arduino 固件历史版本

| 目录/文件 | 说明 |
|-----------|------|
| `EOG_Smart_Mouse/` | 纯 EOG 智能鼠标（无 IMU） |
| `IMU_mouse/` | 纯 IMU 鼠标控制（无 EOG AI） |
| `IMU_EOG/IMU_EOG_Mouse/` | IMU + EOG 集成鼠标（开发迭代版本） |
| `EOG_data_collection_only/` | 纯数据采集固件 |
| `cw_20251120/` | 早期课堂作业版本 |
| `main_project/realtime_eog_imu/` | 实时 EOG+IMU 数据采集固件 |

---

*报告完*
