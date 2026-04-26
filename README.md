# 三角洲行动 GameGPT 决赛技术报告

## 📋 代码文件说明

|文件名 |功能说明 |
|:---|:---|
|`compress.py` |数据二次压缩处理 |
|`config.py` |配置文件 |
|`improve_labels.py` |使用 LLM 进行后 5s 行为描述 |
|`main.py` |数据初步处理主程序 |
|`processor.py` |数据初步处理主要代码 |
|`train_infer.ipynb` |微调训练和预测推理 notebook |


---

## 一、任务理解与整体思路

本赛题决赛的核心任务是：**给定前 20 秒对局状态序列（Input），生成主玩家在未来 5 秒内的完整行为续写（Output）**。

### 1.1 主要难点

🔹 **稳定解析**：将非结构化 `txt` 日志转成可操作的结构化中间表示
🔹 **语义压缩**：将 20 秒内数千行高频帧数据压缩成对 LLM 最有价值的战术文本
🔹 **续写质量**：生成既符合游戏逻辑、又符合评审期望的三段式行为叙述
🔹 **资源约束下微调**：在单卡 24 GB 显存上高效完成 9B 参数模型的精调

### 1.2 整体技术路线

整体技术路线遵循"**数据驱动质量、两阶段训练分治、有限资源高效利用**"的设计原则。

> 📌 **完整流水线**

```text
原始 TXT 日志（管道符格式）
  │
  ├─→ parse_file()              → 结构化中间表示（5类数据结构）
  │
  ├─→ generate_input_text()     → 四层战术文本压缩（~936 tokens 中位）
  │
  ├─→ generate_label_text()     → 三段式标签续写（行为归因 + 过程 + 动作）
  │
  ├─→ LLM 高质量润色            → improved_*.jsonl（2,540 条精标数据）
  │
  ├─→ Stage 1 全量 SFT          → 学习六类行为基础映射（88,105 条，lr=2e-4）
  │
  ├─→ Stage 2 高质量精调         → 强化表达质量（2,540 条×2.5权重，lr=5e-5）
  │
  └─→ 推理：ChatML Prompt        → 三段式行为续写输出
```


---

## 二、数据解析与结构化

### 2.1 `parse_file()` 全覆盖解析架构

原始数据以 `|` 为分隔符，每行包含时间戳、事件类型和若干字段。`parse_file()` 函数对全部事件类型做了枚举覆盖，解析结果统一存入五类核心数据结构：

|数据结构 |类型 |内容说明 |
|:---|:---|:---|
|`players_info` |`{pid: {team, name}}` |全场玩家身份映射 |
|`frames` |`{ts: {pid: {x,y,z,yaw,pitch,speed,scope}}}` |全部帧位置/状态快照 |
|`events` |`[{ts, type, ...}]` 按时间排序 |技能/伤害/击倒/死亡/标点事件 |
|`labels` |`[{ts, label_type, pid, is_test, ...}]` |决策行及其元数据 |
|`decision_actions` |`[{ts, pid, action}]` |带`（决策）`标记的动作序列 |

#### 速度计算

玩家速度不直接读取字段，而是通过速度向量合成计算：

$$
v = \sqrt{v_x^2 + v_y^2 + v_z^2}
$$

这保证了在不同坐标系、不同帧率下速度计算的一致性。

#### 容错机制

当 `pid` 字段在决策行中缺失时，系统会统计全局帧中出现频率最高的玩家 ID 作为主玩家兜底，增强了数据格式不规整时的鲁棒性。

### 2.2 测试集的正确识别

测试集的决策行格式为 `ts|（决策）|玩家xxxx`，没有具体决策类型，且决策行之后没有数据，因此不生成标签文本。


---

## 三、上文的四层语义压缩

- 初步处理直接将结构化数据（`数据/output_jsonl` 目录）进行训练微调。
- 发现原始提取的数据序列极长，包含了大量静止、视野微小晃动等冗余信息。
- 导致极易超出模型处理的 Token 上限导致上下文截断，严重稀释了真正对决策有关键影响的战术事件
- 若增大max_tokens参数容易导致训练极度缓慢甚至显存溢出。

因此，compress.py对初步处理的数据进行了压缩。按**战术意义优先级**，将上文组织为四个独立层次，大幅度降低了中位 token 分布：

### 3.1 层次一：全局态势快照

提供对局上下文基底，作为 LLM 推理的全局锚点，内容固定不随时间变化：

```text
【全局态势】
主玩家: 红狼(ID=6558, 队伍4) | 共17名干员 | 我方3人 / 敌方14人
初始位置: (234,891) | 初始朝向: 北 | 开局最近敌方: 暗影(6712) 距87m
……
```

主玩家的初始帧通过遍历 `sorted_ts` 找到第一个有该玩家数据的时间戳，避免了坐标全零的空帧问题。

### 3.2 层次二：全窗口战斗事件

对 20 秒内发生的**所有伤害、击倒、死亡事件**做完整记录，不做时间压缩。这类离散事件对决策意图具有极强的指示作用——"队友刚被击倒"会直接触发救援行为，"主玩家造成伤害"意味着正在交战中。

### 3.3 层次三：前段摘要（0~15s）

对较早期的帧数据做智能聚合，区分两种模式：

- **潜伏模式**（静止率 > 85%）：直接输出"长时间潜伏观察"，附带累计位移和警戒方向，节省大量 token
- **机动模式**：按 `EARLY_SUMMARY_INTERVAL` 时间窗口滑动，**过滤掉低信息量的静止片段**，只保留有移动或速度变化的时间段

### 3.4 层次四：关键窗口（15~20s，事件驱动触发）

在决策前最后几秒，逐帧扫描并按**五个独立触发条件**决定是否输出当前帧，这是整个压缩方案最关键的策略：

```py
# 五个触发条件（满足任一即输出）
1. 位移 > DISP_THRESHOLD_LATE           # 空间移动
2. 朝向变化 > YAW_THRESHOLD_LATE        # 视角转动
3. scope 状态切换（开镜/关镜）           # 进入/退出战斗姿态
4. 速度变化 > SPEED_JUMP_THRESHOLD      # 移动状态突变
5. 与最近敌方距离跨越 ENEMY_DIST_THRESHOLDS  # 敌距档位变化

# 兜底：每 KEY_WINDOW_INTERVAL 秒强制至少输出一帧
```

这种设计保证了"**只输出有信息量的帧**"，从而把中位 token 数控制在 936，远低于 2048 上限，但不丢失关键决策证据。


---

## 四、三段式标签续写生成

`generate_label_text()` 为六类决策分别生成具备**行为归因 + 战术过程 + 最终动作**结构的高质量续写文本：

```text
主玩家{name}{前置状态}；随后{战术过程}；最后{具体决策动作}。
```

### 4.1 动态前置状态生成

**前置状态**由关键窗口 5 秒内的平均速度动态生成：

|速度范围 |生成状态 |
|:---|:---|
|> 7 m/s |"高速冲刺机动" |
|> 3 m/s |"持续跑步推进" |
|≤ 3 m/s 且静止率高 |"长时间潜伏" |

这使得即使是同一决策类型，不同速度状态的样本描述也会有自然差异，避免了标签文本的机械化重复。

### 4.2 特殊决策类型处理

#### `SkillStart` 类型

代码会反向查询事件流，在决策时刻前后 2 秒内匹配对应的 `技能生效` 事件，**用真实技能名称替代抽象 ID**：

```py
for ev in events:
    if (ev['type'] == '技能生效' and ev.get('caster') == main_pid
            and abs(ev['ts'] - label_ts) <= 2.0):
        buf = ev.get('buff', '')
        if not is_noise_buff(buf):
            skill_name = buf   # ← 用真实技能名，非抽象ID
            break
```

#### `Grenade` 类型

利用标签行中的落点坐标 `(land_x, land_z)` 与主玩家当前坐标，计算实际投掷距离和方向，使生成文本包含可验证的战术信息。

#### `BeingRescue` 类型

检查被救者 ID 是否等于主玩家自身，避免生成"救援自己"这类语义错误。


---

## 五、LLM 高质量数据增强

训练数据被划分为**两个质量层次**，以文件名前缀区分：

```text
compressed_jsonl/
├── Fire1.jsonl / SkillStart1.jsonl / ...   ← 规则标注（88,105 条）
└── improved_Fire.jsonl / improved_Action.jsonl / ...  ← LLM 润色（2,977 条）
```

- **规则标注样本**：数量大但表达机械
- **人工标注**：太过费时
- `improved_` 样本：经调用 LLM 对后 5s 的结构化数据进行总结描述，生成因果链更清晰、语言更自然的续写，是训练的**质量锚点**

对六类样本各约 500 条数据经过 LLM 加工，过滤超长（2048 tokens）后分布均衡，无明显类别偏倚。

### 5.1 Stage 1 样本分布

```text
规则标注样本:  88,105
LLM高质量样本: 2,977
过滤超长样本:  91,082 → 68,241  (丢弃 22,841 条，25.1%)
```

|决策类型 |样本数量 |占比 |
|:---|:---|:---|
|Fire |22,435 |32.9% |
|Looting |18,490 |27.1% |
|Action |15,851 |23.2% |
|SkillStart |8,123 |11.9% |
|BeingResuce |2,233 |3.3% |
|Grenade |1,109 |1.6% |
|**总计** |**68,241** |**100%** |

> 📊 估算步数：17,061(epoch=1)

### 5.2 Stage 2 样本分布

```text
LLM高质量样本: 2,977
过滤超长样本:  2,977 → 2,540  (丢弃 437 条，14.7%)
```

|决策类型 |样本数量 |占比 |
|:---|:---|:---|
|BeingResuce |482 |19.0% |
|Looting |467 |18.4% |
|Action |405 |15.9% |
|SkillStart |399 |15.7% |
|Grenade |394 |15.5% |
|Fire |393 |15.5% |
|**总计** |**2,540** |**100%** |

> 📊 估算步数：1,270(epoch=2)

### 5.3 两阶段训练参数对比

|指标 |Stage 1 |Stage 2 |
|:---|:---|:---|
|原始样本 |91,082 |2,977 |
|过滤后 |68,241（⭐75%） |2,540（⭐85%） |
|丢弃比例 |25.1% |14.7% |
|估算步数 |17,061 |1,270 |
|学习率 |2e-4 |5e-5 |
|样本权重 |1.0 / 2.5 |2.5 |

### 5.4 测试集分布

|决策类型 |样本数 |占比 |
|:---|:---|:---|
|BeingRescue |15 |13.6% |
|Looting |20 |18.2% |
|Action |20 |18.2% |
|SkillStart |20 |18.2% |
|Grenade |15 |13.6% |
|Fire |20 |18.2% |
|**总计** |**110** |**100%** |


---

## 六、模型架构与训练微调

### 6.1 模型选型：Qwen3.5-9B Vision + INT8

底座模型选用 **Qwen3.5-9B**，通过 Unsloth 的 `FastVisionModel` 加载，有两个关键选型决策：

|决策项 |选择 |理由 |
|:---|:---|:---|
|**Vision 版本** |✅ 采用 |为后续**图文联合推理增强**预留接口（详见第八章） |
|**量化方式** |INT8 |bf16 容易爆显存，INT8 保留足够的梯度与激活缓存空间 |

> 📌 `FastVisionModel` 与 Unsloth 的加速补丁完全兼容。

### 6.2 LoRA 参数配置

```py
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,   # 冻结视觉层，纯文本任务
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r          = 32,
    lora_alpha = 64,         # scaling factor = alpha/r = 2.0
    lora_dropout = 0.0,      # Unsloth 推荐无 Dropout
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention 全覆盖
        "gate_proj", "up_proj", "down_proj",        # MLP 全覆盖
    ],
    use_gradient_checkpointing = "unsloth",         # 显存节省
)
```

LoRA 同时覆盖 Attention 的 4 个投影和 MLP 的 3 个投影，确保模型在**注意力分配和前馈推理**两个层面都能被战术语言微调影响。

> 📊 最终可训练参数仅占总参数的 **0.61%**，以极低的参数量实现领域适配。

> 📌 **黄金配比**：`lora_alpha = 2 × r` 是 LoRA 的经典配比，缩放因子恰好为 2.0，确保新知识注入速度适中。

### 6.3 自定义 DataCollator（重要策略）

标准 `SFTTrainer` 默认对**全序列**计算损失，包括系统提示和游戏上文部分，导致模型把大量梯度浪费在"记住输入"上，而不是"学习如何回答"。

我们完全自定义了 `DataCollatorForCompletionOnlyLM`，实现了**仅对 assistant 回复部分计算 loss**的精确掩码：

```py
class DataCollatorForCompletionOnlyLM:
    def __call__(self, instances):
        # 步骤1：提取并暂存 __weight__，不传入模型
        weights = torch.tensor([inst["__weight__"] for inst in instances])
      
        # 步骤2：标准 padding
        batch = self.tokenizer.pad(instances, padding=True, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
      
        # 步骤3：Mask pad token → -100
        batch["labels"][batch["labels"] == pad_id] = -100
      
        # 步骤4：滑动窗口定位 <|im_start|>assistant\n，mask 之前所有 token
        for i in range(batch_size):
            found_idx = sliding_window_search(input_ids[i], response_token_ids)
            if found_idx != -1:
                batch["labels"][i, :found_idx + response_len] = -100
            else:
                batch["labels"][i, :] = -100   # 未找到则整条 mask（安全降级）
      
        batch["__weight__"] = weights
        return batch
```

> 🔑 **技术细节**：
> Qwen3.5 的 `<|im_start|>assistant\n` 对应 token IDs 为 `[248045, 74455, 198]`。
> collator 通过**滑动窗口精确匹配**这三个连续 token，而非字符串分割，避免了 tokenizer 行为不一致导致的 mask 位置偏移。

> 📊 **实测效果**：样本0总 token = 767，有效 label token = 88（11.5%）；样本1总 token = 767，有效 label token = 97（12.6%）。约 **88% 的 token 被屏蔽**，只有行为续写部分真正驱动梯度更新。

### 6.4 加权损失机制（WeightedSFTTrainer）

`DataCollatorForCompletionOnlyLM` 完成 label masking 后，`WeightedSFTTrainer` 接管损失缩放，将高质量样本的影响力显式注入训练过程：

```py
class WeightedSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("__weight__", None)   # 从 batch 中取出权重张量
        result = super().compute_loss(model, inputs, return_outputs, **kwargs)
        loss = result[0] if return_outputs else result
        if weights is not None:
            # 用 batch 内权重均值缩放损失
            loss = loss * weights.to(loss.device).float().mean()
        return (loss, result[1]) if return_outputs else loss
```

最终决定 loss 权重为 `IMPROVED_WEIGHT = 2.5`，基于如下考量：

- 高质量样本约占总样本 3%，若不加权重其梯度贡献会被大量规则样本淹没
- 2.5 倍权重将有效样本量提升到约 7.5%，在不引入过拟合风险的前提下显著放大了高质量标签的引导力

> 📌 这种设计对标准 SFT 框架**几乎零侵入**：权重信息从 dataset 构建阶段直接透传到损失层，无需修改 Trainer 其他逻辑。


---

## 七、两阶段训练策略

两阶段训练的核心逻辑是**分治**：先学会"识别行为类型"，再学会"表达行为过程"。

### 7.1 Stage 1：全量混合训练

- 使用 88,105 条规则标注样本 + 2,977 条高质量样本
- 学习率：`lr = 2e-4`
- 目标：让模型建立稳定的"上文结构 → 六类决策行为"的基础映射
- 高质量样本此阶段仍参与训练，作为"锚点"保持输出格式正确

> 📌 出出于对训练时间和成本的考量，只训练了 1000 步，初步学会如何对后 5s 的行为进行预测，作为 Stage 2 的基础。而在高质量训练数据的 Stage 2 上进行重点训练才更能教会模型如何进行行为描述。

### 7.2 Stage 2：高质量精调

- 从 Stage 1 的 `checkpoint-1000` 出发
- 仅使用 `improved_` 精标数据（2,540 条）
- 学习率降至：`lr = 5e-5`
- 样本权重：`2.5×`
- 目标：在不破坏已建立的行为分类能力的前提下，**精细调整措辞风格、因果逻辑和叙述连贯性**

> 📌 **检查点加载方式**（手动注入，保证可追溯性）：
>
> ```py
> state_dict = load_file("gamegpt_stage1/checkpoint-1000/adapter_model.safetensors")
> model.load_state_dict(state_dict, strict=False)
> # strict=False 允许 missing/unexpected keys 兼容多版本检查点
> ```

### 7.3 Stage 1 Checkpoint 权重分析

Stage 1 checkpoint_1000 的 LoRA 权重占比：**mean = 0.007807**

### 7.4 Stage 2 Loss 曲线

Stage 2 的 Loss 如下，无明显震荡或过拟合：

|Step |Training Loss |Step |Training Loss |
|:---|:---|:---|:---|
|50 |5.029249 |650 |2.526019 |
|100 |3.623250 |700 |2.394319 |
|150 |3.294871 |750 |2.390584 |
|200 |3.123952 |800 |2.303506 |
|250 |2.905822 |850 |2.246557 |
|300 |2.779895 |900 |2.241733 |
|350 |2.911753 |950 |2.263947 |
|400 |2.914152 |1000 |2.342830 |
|450 |2.771355 |1050 |2.271257 |
|500 |2.690706 |1100 |2.177100 |
|550 |2.730449 |1150 |2.288743 |
|600 |2.701624 |1200 |2.230583 |
|650 |2.526019 |1250 |2.191063 |

> ⚠️ **注**：Stage 2 全是高质量数据且带 2.5× 权重，因此显示 loss 较高。实际损失位于 **0.8-1.0** 区间（除以 2.5 后），对生成式模型是适宜区间。


---

## 八、图文融合推理构想

### 8.1 概述

Qwen3.5-9B 原生具备多模态能力。即便经过了结构化处理和压缩，但小模型的参数决定了其对于大量数据的理解能力，因此加入方位图更能辅助模型理解信息。

> 📌 示例文件位于：`数据/location_map`。由generate_map.py处理测试集生成，由于方位图对于目标信息的显示效果不算太好，时间紧促，因此最终没有进一步优化方位图的显示效果以及实现图文融合推理的代码。

### 8.2 动机

游戏日志中隐含了大量**空间关系信息**：玩家坐标轨迹、移动路径、敌我距离变化。纯文本表达这些信息时，LLM 理解"先绕点、后压近、再开火"这类路径与时序复合决策的效率并不高。

### 8.3 方案设计

提出**战术轨迹图 + 结构化文本**联合推理方案：

1. **轨迹图生成**：以决策时刻为原点，提取关键窗口内主玩家的坐标序列，渲染包含移动折线、最终朝向扇形、最近敌方相对位置、关键动作发生点（开镜/速度突变）的战术俯视图
2. **图文联合推理**：将轨迹图与压缩后的文本上文一起送入 Vision 底座，利用视觉感知能力理解空间路径，由语言层完成行为归因与续写生成

### 8.4 权衡与取舍

我们最终将图文方案定位为**推理阶段创新增强**而非训练主线，原因如下：

|约束因素 |说明 |
|:---|:---|
|**显存约束** |同时训练视觉层会额外消耗 3~4 GB 显存，可能超出当前设备显存可用上限 |
|**任务定义约束** |竞赛官方数据为纯文本，图像需额外生成，引入对齐风险 |
|**接口预留** |选用 Vision 版底座并冻结视觉层，是对后续图文方案的前瞻性准备，不增加训练成本 |

> 📌 对于 `Grenade`（落点预测）、`Fire`（交战距离判断）等空间感知要求高的类型，推理时可将轨迹图拼接到 prompt 中，在不重新训练的前提下提升生成质量。


---

## 九、推理准备与 Prompt 工程

### 9.1 ChatML Prompt 设计

推理 prompt 遵循与训练完全一致的 ChatML 三轮结构，消除训练-推理分布偏移：

```text
<|im_start|>system
你是三角洲行动游戏的战术 AI 助手，负责根据前20秒的对局上文，
预测并描述主玩家在接下来5秒内的具体行为过程。
用【主玩家...随后...最后...】的格式输出，不超过150字。
<|im_end|>
<|im_start|>user
【全局态势】
主玩家: 红狼(ID=6558, 队伍4) | 共17名干员 | 我方3人/敌方14人
【战斗事件】
  17.32s: 主玩家 → [队伍2]暗影(6712) 造成134伤害(剩余66)
【前段摘要 (0.0s~15.0s)】
  0.0~5.0s: 朝北移动41.2m 均速8.3m/s | 最近敌[暗影]312m
【关键窗口 (15.0s~19.95s)】
  19.10s: (234.1,891.3) 朝北 开镜 跑步 [→北位移2.1m, ★开镜] | 最近敌43m
<|im_end|>
<|im_start|>assistant
```

系统提示词做了三项精确约束：

|约束类型 |说明 |
|:---|:---|
|**角色定位** |绑定游戏场景，避免通用泛化 |
|**格式约束** |三段式，与训练标签格式完全一致 |
|**长度约束** |不超过 150 字，防止冗余重复 |

### 9.2 模型加载与 Tokenizer 适配

Qwen3.5 Vision 版返回的是 `Qwen3VLProcessor` 而非标准 tokenizer，文本编码功能需通过子属性访问。若直接使用外层 processor，会因接口差异导致维度不匹配：

```py
# ✅ 关键适配：提取子 tokenizer
text_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
print(f"Processor 类型: {type(tokenizer).__name__}")    # Qwen3VLProcessor
print(f"文本 Tokenizer: {type(text_tokenizer).__name__}") # TokenizersBackend
```

推理前还需手动修正 `max_position_embeddings`，避免 Vision 版默认配置对长序列的静默截断：

```py
model.config.max_position_embeddings = 262144
if hasattr(model.config, 'text_config'):
    model.config.text_config.max_position_embeddings = 262144
```

### 9.3 推理 System Message

```py
SYSTEM_MSG = (
    "你是三角洲行动游戏的战术分析专家，擅长根据玩家的历史行为数据预测其接下来的行为。\n"
    "你会收到一段结构化的对局记录，包含：\n"
    "  - 【全局态势】：地图上的兵力分布与初始状态\n"
    "  - 【前段摘要】：0~15秒的移动方向、速度与关键动作\n"
    "  - 【关键窗口】：15~20秒的坐标、朝向、移动状态与事件\n\n"
    "请根据以上信息，预测主玩家在第20~25秒的行为过程，严格遵守要求：\n"
    "  1. 以【主玩家...随后...最后...】的格式描述行为链\n"
    "  2. 重点描述：移动方向与状态、开镜/关镜、关键动作（跳/趴/站/滑铲等）\n"
    "  3. 不超过150字，不要出现推测性语气（如'可能'、'也许'）\n"
    "  4. 禁止在行文中出现具体时间戳（如20.05s、21.3s等），只描述动作顺序和因果关系。如必须使用时间描述，则必须描述到24.50s到25.00s才能完成描述。\n"
    "  5. 结尾完成总结动作描述：\n"
    "     必须是人类可读的战术动作词汇，"
    "     例如：完成开火并击倒对手、丢雷后冲锋、释放技能、救援队友、搜刮物资、战术规避、协同作战等，"
    "     严禁直接输出英文和代码标签（Action、BeingResuce、Fire、Move等）。\n"
    "  6. 只输出描述内容，不要解释推理过程"
)
```

### 9.4 Hint 追加策略

```py
# 将 hint 追加在 input 的下一行
user_content = sample_input + "\n" + hint
messages = [
    {"role": "system",    "content": SYSTEM_MSG},
    {"role": "user",      "content": user_content},
    {"role": "assistant", "content": ""},
]
```

> 📌 如果测试集没有决策类型，可以采用初赛的预测模型进行决策预测，然后将预测结果和置信度作为 hint 追加到 input 信息的下一列中。决赛已给出决策类型，直接采用并将置信度默认为 1。

### 9.5 SFT 训练关键参数

```py
trainer = WeightedSFTTrainer(
    args = SFTConfig(
        per_device_train_batch_size = 2,        # 每个 batch 训练两个样本数据
        gradient_accumulation_steps = 2,        # 缓解显存压力
        warmup_steps                = 20,       # 预热步数，stage1 设为 100，stage2 设为 20
        lr_scheduler_type           = "cosine",
        logging_steps               = 50,
        save_strategy               = "steps", 
        save_steps                  = 250,       # 每 250 步保存一次 checkpoint
        save_total_limit            = 2,       
        optim                       = "adamw_8bit", # adamw 优化器
        weight_decay                = 0.01,
        max_grad_norm               = 1.0,    
        max_seq_length              = MAX_SEQ_LENGTH,  # 2048，过大训练速度减半，显存翻倍
        packing                     = False,   #input数据与max_tokens差距并不大，采用padding提升的训练速度不明显
    ),
)
```


---

## 十、创新点总结

本方案在标准 SFT 微调基础上提出了以下六项技术创新：

|序号 |创新点 |核心价值 |
|:---|:---|:---|
|1 |**事件驱动的时序压缩算法** |五个战术触发条件替代均匀采样，信息密度最大化，中位 token 936，全量样本不超过 2048 上限 |
|2 |**针对 Qwen3VLProcessor 的自定义 DataCollator** |通过 token 序列滑动窗口精确定位 response 边界，未找到 template 时整条 mask 的安全降级机制保证训练稳定性 |
|3 |**全链路样本权重透传** |`__weight__` 字段从 dataset 构建阶段透传至 Trainer 损失层，对标准框架零侵入 |
|4 |**两阶段分治微调** |Stage1 学行为识别（大数据宽泛）→ Stage2 学表达质量（精标低学习率） |
|5 |**图文融合预留接口** |选择 Vision 底座冻结视觉层，以零训练成本预留多模态推理接口 |
|6 |**初赛决策分类模型融合** |将预测结果和置信度作为 hint 追加到 input 信息中，专用的分类模型在决策六分类上更可靠 |


