# =============================================================================
# config.py  —  GameGPT 数据处理全局配置
# =============================================================================

# ── 路径 ──────────────────────────────────────────────────────────────────────
INPUT_DIR    = r".\决赛测试100题"
OUTPUT_DIR   = "./数据/output_jsonl"
OUTPUT_FILE  = "test.jsonl"

# ── 决策类型：文件名关键词 → 决策标签 ────────────────────────────────────────
# 用于测试集（无决策行）时从文件名推断 label
FILENAME_LABEL_MAP = {
    "fire":       "Fire",
    "skillstart": "SkillStart",
    "grenade":    "Grenade",
    "looting":    "Looting",
    "beingresuce":"BeingResuce",
    "action":     "Action",
}

# ── 决策类型元信息 ─────────────────────────────────────────────────────────────
LABEL_META = {
    "Fire":        {"camp": "交战", "action": "开火"},
    "SkillStart":  {"camp": "交战", "action": "放技能"},
    "Grenade":     {"camp": "交战", "action": "丢雷"},
    "Looting":     {"camp": "避战", "action": "搜索物资"},
    "BeingResuce": {"camp": "避战", "action": "救援队友"},
    "Action":      {"camp": "交战", "action": "执行动作"},
}

# Action 类决策中的关键动作（在续写里重点描述）
ACTION_COMBAT_KEYWORDS = ["换弹", "开镜", "左探头", "右探头", "回正探头"]
ACTION_MOVE_KEYWORDS   = ["蹲", "趴", "站", "跳", "滑铲"]
ACTION_SCOPE_KEYWORDS  = ["关镜"]

# ── 压缩阈值 ───────────────────────────────────────────────────────────────────
DISP_THRESHOLD_EARLY   = 1.5    # 前段位移触发阈值(m)
DISP_THRESHOLD_LATE    = 0.4    # 关键窗口位移触发阈值(m)
YAW_THRESHOLD_EARLY    = 20.0   # 前段朝向变化阈值(°)
YAW_THRESHOLD_LATE     = 8.0    # 关键窗口朝向变化阈值(°)
SPEED_JUMP_THRESHOLD   = 3.0    # 速度突变阈值(m/s)
ENEMY_DIST_THRESHOLDS  = [60, 40, 25, 15, 8]
LATE_WINDOW_SEC        = 5.0    # 关键窗口长度(s)
KEY_WINDOW_INTERVAL    = 0.25   # 关键窗口最大采样间隔(s)
EARLY_SUMMARY_INTERVAL = 2.0    # 前段每N秒一条摘要

# ── 噪声过滤 ───────────────────────────────────────────────────────────────────
NOISE_BUFF_KEYWORDS     = ["耳机", "头盔声音", "纯表现", "展示用", "【装备】移速",
                            "【装备】瞄准速度"]
IMPORTANT_BUFF_KEYWORDS = ["医疗", "治疗", "侦察", "屏蔽", "护盾", "入水",
                            "消耗品", "止痛"]

# ── 方向 ───────────────────────────────────────────────────────────────────────
DIR_8 = ['北', '东北', '东', '东南', '南', '西南', '西', '西北']
