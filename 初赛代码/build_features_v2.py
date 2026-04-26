# -*- coding: utf-8 -*-
"""
增强版 build_features_v2.py

目标：
1. 面向初赛表格模型（XGBoost / LightGBM）
2. 尽量利用完整20秒上下文，但把总特征数控制在200以内
3. 每个txt样本提取为一行
4. 支持目录结构：
   训练数据/little/Action/*.txt
   训练数据/little/BeingResuce/*.txt
   训练数据/little/Fire/*.txt
   训练数据/little/Grenade/*.txt
   训练数据/little/Looting/*.txt
   训练数据/little/SkillStart/*.txt
5. 兼顾：末态 + 多窗口统计 + 趋势 + 事件时间差

说明：
- 标签优先使用父文件夹名
- 主玩家优先从决策行读取；若缺失，则用动作/伤害/技能日志兜底推断
- 只使用决策时刻及之前的数据，避免时序泄漏
"""

import os
import math
import glob
from collections import defaultdict
import numpy as np
import pandas as pd


# =========================================================
# 1. 基础工具函数
# =========================================================

def safe_float(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except:
        return default


def norm_player_id(x):
    if x is None:
        return None
    x = str(x).strip().replace("玩家", "")
    return x


def euclidean_distance_3d(x1, y1, z1, x2, y2, z2):
    vals = [x1, y1, z1, x2, y2, z2]
    if any(pd.isna(v) for v in vals):
        return np.nan
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


def circular_diff(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def ensure_columns(df, columns):
    if df is None or len(df.columns) == 0:
        df = pd.DataFrame(columns=columns)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df


def safe_mode(series, default="NONE"):
    s = series.dropna().astype(str)
    s = s[s != ""]
    if len(s) == 0:
        return default
    return s.value_counts().idxmax()


def get_time_window(df, time_col, end_time, left_sec):
    if len(df) == 0 or pd.isna(end_time):
        return df.iloc[0:0].copy()
    return df[(df[time_col] >= end_time - left_sec) & (df[time_col] <= end_time)].copy()


def add_basic_stats(feat, series, prefix):
    s = pd.to_numeric(series, errors="coerce").dropna()
    feat[f"{prefix}_count"] = int(len(s))
    if len(s) == 0:
        feat[f"{prefix}_mean"] = np.nan
        feat[f"{prefix}_std"] = np.nan
        feat[f"{prefix}_min"] = np.nan
        feat[f"{prefix}_max"] = np.nan
        feat[f"{prefix}_sum"] = 0.0
    else:
        feat[f"{prefix}_mean"] = float(s.mean())
        feat[f"{prefix}_std"] = float(s.std()) if len(s) > 1 else 0.0
        feat[f"{prefix}_min"] = float(s.min())
        feat[f"{prefix}_max"] = float(s.max())
        feat[f"{prefix}_sum"] = float(s.sum())


def last_time_gap(df, time_col, decision_time):
    if len(df) == 0 or pd.isna(decision_time):
        return np.nan
    t = pd.to_numeric(df[time_col], errors="coerce").dropna()
    if len(t) == 0:
        return np.nan
    return float(decision_time - t.max())


# =========================================================
# 2. 决策标签解析
# =========================================================

def parse_decision_line(parts):
    decision_type = parts[1].strip()
    info = {
        "decision_time": safe_float(parts[0]),
        "label_raw": decision_type,
        "label": None,
        "main_player_id": None,
        "decision_extra": {}
    }

    if "开火" in decision_type:
        info["label"] = "Fire"
        info["main_player_id"] = norm_player_id(parts[2]) if len(parts) > 2 else None
    elif "丢雷" in decision_type:
        info["label"] = "Grenade"
        info["main_player_id"] = norm_player_id(parts[2]) if len(parts) > 2 else None
        if len(parts) > 8:
            info["decision_extra"]["target_x"] = safe_float(parts[4])
            info["decision_extra"]["target_y"] = safe_float(parts[5])
            info["decision_extra"]["target_z"] = safe_float(parts[6])
            info["decision_extra"]["effect_radius"] = safe_float(parts[7])
            info["decision_extra"]["effect_time"] = safe_float(parts[8])
    elif "放技能" in decision_type:
        info["label"] = "SkillStart"
        info["main_player_id"] = norm_player_id(parts[2]) if len(parts) > 2 else None
        if len(parts) > 3:
            info["decision_extra"]["skill_name"] = parts[3]
    elif "搜" in decision_type:
        info["label"] = "Looting"
        info["main_player_id"] = norm_player_id(parts[2]) if len(parts) > 2 else None
    elif "救援" in decision_type:
        info["label"] = "BeingResuce"
        info["main_player_id"] = norm_player_id(parts[3]) if len(parts) > 3 else None
    return info


def find_decision_info(lines):
    for line in reversed(lines):
        parts = line.strip().split("|")
        if len(parts) >= 2 and "决策" in parts[1]:
            return parse_decision_line(parts)
    return None


# =========================================================
# 3. 各日志类型解析
# =========================================================

def parse_player_basic(parts):
    return {
        "time": safe_float(parts[0]),
        "player_id": norm_player_id(parts[2]),
        "x": safe_float(parts[3]),
        "y": safe_float(parts[4]),
        "z": safe_float(parts[5]),
        "weapon_yaw": safe_float(parts[6]),
        "weapon_pitch": safe_float(parts[7]),
        "vx": safe_float(parts[8]),
        "vy": safe_float(parts[9]),
        "vz": safe_float(parts[10]),
        "cam_x": safe_float(parts[11]),
        "cam_y": safe_float(parts[12]),
        "cam_z": safe_float(parts[13]),
        "fov": safe_float(parts[14]),
        "ray_visibility": parts[18] if len(parts) > 18 else "",
        "scope_state": parts[19] if len(parts) > 19 else ""
    }


def parse_action(parts):
    return {
        "time": safe_float(parts[0]),
        "player_id": norm_player_id(parts[2]),
        "action_name": parts[3] if len(parts) > 3 else ""
    }


def parse_skill(parts):
    return {
        "time": safe_float(parts[0]),
        "target_player_id": norm_player_id(parts[2]) if len(parts) > 2 else None,
        "caster_player_id": norm_player_id(parts[3]) if len(parts) > 3 else None,
        "buff_type": parts[4] if len(parts) > 4 else ""
    }


def parse_damage(parts):
    return {
        "time": safe_float(parts[0]),
        "attacker_id": norm_player_id(parts[2]) if len(parts) > 2 else None,
        "victim_id": norm_player_id(parts[3]) if len(parts) > 3 else None,
        "attacker_x": safe_float(parts[4]) if len(parts) > 4 else np.nan,
        "attacker_y": safe_float(parts[5]) if len(parts) > 5 else np.nan,
        "attacker_z": safe_float(parts[6]) if len(parts) > 6 else np.nan,
        "victim_x": safe_float(parts[7]) if len(parts) > 7 else np.nan,
        "victim_y": safe_float(parts[8]) if len(parts) > 8 else np.nan,
        "victim_z": safe_float(parts[9]) if len(parts) > 9 else np.nan,
        "visible": parts[10] if len(parts) > 10 else "",
        "hp_damage": safe_float(parts[12]) if len(parts) > 12 else np.nan,
        "victim_hp_left": safe_float(parts[13]) if len(parts) > 13 else np.nan,
        "armor_damage": safe_float(parts[14]) if len(parts) > 14 else np.nan
    }


def parse_knockdown(parts):
    return {
        "time": safe_float(parts[0]),
        "attacker_id": norm_player_id(parts[2]) if len(parts) > 2 else None,
        "victim_id": norm_player_id(parts[3]) if len(parts) > 3 else None
    }


def parse_death(parts):
    return {
        "time": safe_float(parts[0]),
        "dead_player_id": norm_player_id(parts[2]) if len(parts) > 2 else None,
        "death_reason": parts[3] if len(parts) > 3 else "",
        "killer_id": norm_player_id(parts[7]) if len(parts) > 7 else None
    }


# =========================================================
# 4. 单文件解析
# =========================================================

def parse_single_file(file_path, label_from_folder=True):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    decision_info = find_decision_info(lines)
    if decision_info is None:
        decision_info = {
            "decision_time": np.nan,
            "label_raw": None,
            "label": None,
            "main_player_id": None,
            "decision_extra": {}
        }

    if label_from_folder:
        decision_info["label"] = os.path.basename(os.path.dirname(file_path))

    player_basic_rows, action_rows, skill_rows = [], [], []
    damage_rows, knockdown_rows, death_rows = [], [], []

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        log_type = parts[1].strip()
        try:
            if log_type == "玩家基础信息":
                player_basic_rows.append(parse_player_basic(parts))
            elif log_type == "动作":
                action_rows.append(parse_action(parts))
            elif log_type == "技能生效":
                skill_rows.append(parse_skill(parts))
            elif log_type == "玩家造成伤害":
                damage_rows.append(parse_damage(parts))
            elif log_type == "玩家击倒":
                knockdown_rows.append(parse_knockdown(parts))
            elif log_type == "玩家死亡":
                death_rows.append(parse_death(parts))
        except Exception as e:
            print(f"[WARN] 解析失败，文件={file_path}, 行={line[:100]}, 错误={e}")

    player_basic_df = ensure_columns(pd.DataFrame(player_basic_rows), [
        "time", "player_id", "x", "y", "z", "weapon_yaw", "weapon_pitch",
        "vx", "vy", "vz", "cam_x", "cam_y", "cam_z", "fov", "ray_visibility", "scope_state"
    ])
    action_df = ensure_columns(pd.DataFrame(action_rows), ["time", "player_id", "action_name"])
    skill_df = ensure_columns(pd.DataFrame(skill_rows), ["time", "target_player_id", "caster_player_id", "buff_type"])
    damage_df = ensure_columns(pd.DataFrame(damage_rows), [
        "time", "attacker_id", "victim_id", "attacker_x", "attacker_y", "attacker_z",
        "victim_x", "victim_y", "victim_z", "visible", "hp_damage", "victim_hp_left", "armor_damage"
    ])
    knockdown_df = ensure_columns(pd.DataFrame(knockdown_rows), ["time", "attacker_id", "victim_id"])
    death_df = ensure_columns(pd.DataFrame(death_rows), ["time", "dead_player_id", "death_reason", "killer_id"])

    return {
        "sample_id": os.path.splitext(os.path.basename(file_path))[0],
        "decision_info": decision_info,
        "player_basic_df": player_basic_df,
        "action_df": action_df,
        "skill_df": skill_df,
        "damage_df": damage_df,
        "knockdown_df": knockdown_df,
        "death_df": death_df,
    }


# =========================================================
# 5. 主玩家兜底识别
# =========================================================

def infer_main_player_id(parsed):
    info = parsed["decision_info"]
    if info["main_player_id"] is not None:
        return str(info["main_player_id"])

    for df, col in [
        (parsed["action_df"], "player_id"),
        (parsed["damage_df"], "attacker_id"),
        (parsed["skill_df"], "caster_player_id")
    ]:
        s = df[col].dropna().astype(str)
        s = s[s != "None"]
        if len(s) > 0:
            return s.value_counts().idxmax()
    return None


# =========================================================
# 6. 精简但增强的特征提取（<120列）
# =========================================================

def extract_features_from_sample(parsed):
    sample_id = parsed["sample_id"]
    info = parsed["decision_info"]
    label = info["label"]
    decision_time = info["decision_time"]

    basic_df = parsed["player_basic_df"].copy()
    action_df = parsed["action_df"].copy()
    skill_df = parsed["skill_df"].copy()
    damage_df = parsed["damage_df"].copy()
    knockdown_df = parsed["knockdown_df"].copy()
    death_df = parsed["death_df"].copy()

    main_player_id = infer_main_player_id(parsed)

    feat = {
        "sample_id": sample_id,
        "label": label,
        "main_player_id": main_player_id,
    }

    if main_player_id is None:
        return feat

    main_basic_all = basic_df[basic_df["player_id"].astype(str) == str(main_player_id)].copy().sort_values("time")
    if pd.isna(decision_time) and len(main_basic_all) > 0:
        decision_time = pd.to_numeric(main_basic_all["time"], errors="coerce").max()
    feat["decision_time"] = decision_time

    if len(main_basic_all) == 0 or pd.isna(decision_time):
        return feat

    main_basic_before = main_basic_all[main_basic_all["time"] <= decision_time].copy().sort_values("time")
    if len(main_basic_before) == 0:
        main_basic_before = main_basic_all.copy().sort_values("time")
    last_row = main_basic_before.iloc[-1]

    # A. 末态特征（12）
    feat["last_x"] = last_row["x"]
    feat["last_y"] = last_row["y"]
    feat["last_z"] = last_row["z"]
    feat["last_weapon_yaw"] = last_row["weapon_yaw"]
    feat["last_weapon_pitch"] = last_row["weapon_pitch"]
    feat["last_vx"] = last_row["vx"]
    feat["last_vy"] = last_row["vy"]
    feat["last_vz"] = last_row["vz"]
    feat["last_speed"] = math.sqrt(
        (0 if pd.isna(last_row["vx"]) else last_row["vx"])**2 +
        (0 if pd.isna(last_row["vy"]) else last_row["vy"])**2 +
        (0 if pd.isna(last_row["vz"]) else last_row["vz"])**2
    )
    feat["last_fov"] = last_row["fov"]
    feat["last_is_scoped"] = 1 if str(last_row["scope_state"]).strip() == "开镜" else 0
    feat["last_visible_state"] = str(last_row["ray_visibility"]).strip() if not pd.isna(last_row["ray_visibility"]) else "NONE"

    # 准备基础序列
    main_basic_before["speed"] = np.sqrt(
        main_basic_before["vx"].fillna(0)**2 +
        main_basic_before["vy"].fillna(0)**2 +
        main_basic_before["vz"].fillna(0)**2
    )
    main_basic_before["is_scoped"] = main_basic_before["scope_state"].apply(lambda x: 1 if str(x).strip() == "开镜" else 0)

    yaw_vals = pd.to_numeric(main_basic_before["weapon_yaw"], errors="coerce").tolist()
    pitch_vals = pd.to_numeric(main_basic_before["weapon_pitch"], errors="coerce").tolist()
    yaw_changes, pitch_changes = [], []
    for i in range(1, len(yaw_vals)):
        yaw_changes.append(circular_diff(yaw_vals[i], yaw_vals[i-1]))
        if not pd.isna(pitch_vals[i]) and not pd.isna(pitch_vals[i-1]):
            pitch_changes.append(abs(pitch_vals[i] - pitch_vals[i-1]))
    main_basic_before["yaw_change"] = [np.nan] + yaw_changes if len(main_basic_before) > 0 else []
    main_basic_before["pitch_change"] = [np.nan] + pitch_changes if len(main_basic_before) > 0 else []

    # B. 全20秒统计（24）
    recent20 = get_time_window(main_basic_before, "time", decision_time, 20)
    feat["recent20_frames"] = len(recent20)
    add_basic_stats(feat, recent20["speed"], "recent20_speed")
    add_basic_stats(feat, recent20["fov"], "recent20_fov")
    add_basic_stats(feat, recent20["yaw_change"], "recent20_yawchg")
    add_basic_stats(feat, recent20["pitch_change"], "recent20_pitchchg")
    feat["recent20_scope_ratio"] = recent20["is_scoped"].mean() if len(recent20) > 0 else np.nan
    feat["recent20_x_disp"] = recent20["x"].iloc[-1] - recent20["x"].iloc[0] if len(recent20) > 1 else 0.0
    feat["recent20_y_disp"] = recent20["y"].iloc[-1] - recent20["y"].iloc[0] if len(recent20) > 1 else 0.0
    feat["recent20_z_disp"] = recent20["z"].iloc[-1] - recent20["z"].iloc[0] if len(recent20) > 1 else 0.0

    # C. 10秒/5秒/3秒窗口（每窗6个，共18）
    for sec in [10, 5, 3]:
        win = get_time_window(main_basic_before, "time", decision_time, sec)
        feat[f"recent{sec}_frames"] = len(win)
        feat[f"recent{sec}_speed_mean"] = win["speed"].mean() if len(win) > 0 else np.nan
        feat[f"recent{sec}_speed_std"] = win["speed"].std() if len(win) > 1 else 0.0
        feat[f"recent{sec}_scope_ratio"] = win["is_scoped"].mean() if len(win) > 0 else np.nan
        feat[f"recent{sec}_fov_mean"] = win["fov"].mean() if len(win) > 0 else np.nan
        feat[f"recent{sec}_yawchg_sum"] = win["yaw_change"].fillna(0).sum() if len(win) > 0 else 0.0

    # D. 分段趋势（12）
    seg1 = recent20[(recent20["time"] >= decision_time-20) & (recent20["time"] < decision_time-10)]
    seg2 = recent20[(recent20["time"] >= decision_time-10) & (recent20["time"] <= decision_time)]
    feat["seg1_speed_mean"] = seg1["speed"].mean() if len(seg1) > 0 else np.nan
    feat["seg2_speed_mean"] = seg2["speed"].mean() if len(seg2) > 0 else np.nan
    feat["seg_speed_mean_diff"] = feat["seg2_speed_mean"] - feat["seg1_speed_mean"] if pd.notna(feat["seg1_speed_mean"]) and pd.notna(feat["seg2_speed_mean"]) else np.nan
    feat["seg1_scope_ratio"] = seg1["is_scoped"].mean() if len(seg1) > 0 else np.nan
    feat["seg2_scope_ratio"] = seg2["is_scoped"].mean() if len(seg2) > 0 else np.nan
    feat["seg_scope_ratio_diff"] = feat["seg2_scope_ratio"] - feat["seg1_scope_ratio"] if pd.notna(feat["seg1_scope_ratio"]) and pd.notna(feat["seg2_scope_ratio"]) else np.nan
    feat["seg1_fov_mean"] = seg1["fov"].mean() if len(seg1) > 0 else np.nan
    feat["seg2_fov_mean"] = seg2["fov"].mean() if len(seg2) > 0 else np.nan
    feat["seg_fov_mean_diff"] = feat["seg2_fov_mean"] - feat["seg1_fov_mean"] if pd.notna(feat["seg1_fov_mean"]) and pd.notna(feat["seg2_fov_mean"]) else np.nan
    feat["seg1_yawchg_sum"] = seg1["yaw_change"].fillna(0).sum() if len(seg1) > 0 else 0.0
    feat["seg2_yawchg_sum"] = seg2["yaw_change"].fillna(0).sum() if len(seg2) > 0 else 0.0
    feat["seg_yawchg_sum_diff"] = feat["seg2_yawchg_sum"] - feat["seg1_yawchg_sum"]

    # E. 动作特征（14）
    main_actions = action_df[action_df["player_id"].astype(str) == str(main_player_id)].copy()
    main_actions = main_actions[main_actions["time"] <= decision_time].sort_values("time")
    main_actions = ensure_columns(main_actions, ["time", "player_id", "action_name"])
    main_actions["action_name"] = main_actions["action_name"].fillna("").astype(str)
    feat["action_total_count"] = len(main_actions)
    feat["action_nunique"] = main_actions["action_name"].nunique() if len(main_actions) > 0 else 0
    feat["action_mode_20"] = safe_mode(main_actions["action_name"], default="NONE")
    feat["recent10_action_count"] = len(get_time_window(main_actions, "time", decision_time, 10))
    feat["recent5_action_count"] = len(get_time_window(main_actions, "time", decision_time, 5))
    feat["recent3_action_count"] = len(get_time_window(main_actions, "time", decision_time, 3))

    if len(main_actions) > 0:
        last_action_row = main_actions.iloc[-1]
        feat["current_action_name"] = last_action_row.get("action_name", "")
        feat["current_action_time"] = last_action_row["time"]
        feat["current_action_gap_to_decision"] = decision_time - last_action_row["time"]
        feat["prev_action_name"] = main_actions.iloc[-2].get("action_name", "") if len(main_actions) > 1 else "NONE"
        feat["action_switch_count"] = int((main_actions["action_name"].astype(str) != main_actions["action_name"].astype(str).shift(1)).sum() - 1) if len(main_actions) > 1 else 0
        same_last = main_actions[main_actions["action_name"].astype(str) == str(last_action_row.get("action_name", ""))].copy()
        feat["last_action_occurs"] = len(same_last)
    else:
        feat["current_action_name"] = "NONE"
        feat["current_action_time"] = np.nan
        feat["current_action_gap_to_decision"] = np.nan
        feat["prev_action_name"] = "NONE"
        feat["action_switch_count"] = 0
        feat["last_action_occurs"] = 0

    # F. 伤害特征（18）
    damage_out = damage_df[(damage_df["attacker_id"].astype(str) == str(main_player_id)) & (damage_df["time"] <= decision_time)].copy()
    damage_in = damage_df[(damage_df["victim_id"].astype(str) == str(main_player_id)) & (damage_df["time"] <= decision_time)].copy()
    for sec in [20, 5, 3]:
        dout = get_time_window(damage_out, "time", decision_time, sec)
        din = get_time_window(damage_in, "time", decision_time, sec)
        feat[f"recent{sec}_damage_out_count"] = len(dout)
        feat[f"recent{sec}_damage_in_count"] = len(din)
        feat[f"recent{sec}_damage_out_hp_sum"] = dout["hp_damage"].fillna(0).sum()
        feat[f"recent{sec}_damage_in_hp_sum"] = din["hp_damage"].fillna(0).sum()
        feat[f"recent{sec}_damage_out_armor_sum"] = dout["armor_damage"].fillna(0).sum()
        feat[f"recent{sec}_damage_in_armor_sum"] = din["armor_damage"].fillna(0).sum()
    feat["last_damage_out_gap"] = last_time_gap(damage_out, "time", decision_time)
    feat["last_damage_in_gap"] = last_time_gap(damage_in, "time", decision_time)
    vis_out = damage_out["visible"].astype(str).str.contains("可见|True|true", na=False)
    vis_in = damage_in["visible"].astype(str).str.contains("可见|True|true", na=False)
    feat["damage_out_visible_ratio"] = vis_out.mean() if len(vis_out) > 0 else np.nan
    feat["damage_in_visible_ratio"] = vis_in.mean() if len(vis_in) > 0 else np.nan
    feat["damage_hp_net_20"] = feat["recent20_damage_out_hp_sum"] - feat["recent20_damage_in_hp_sum"]

    # G. 技能特征（10）
    skill_cast = skill_df[(skill_df["caster_player_id"].astype(str) == str(main_player_id)) & (skill_df["time"] <= decision_time)].copy()
    skill_hit = skill_df[(skill_df["target_player_id"].astype(str) == str(main_player_id)) & (skill_df["time"] <= decision_time)].copy()
    for sec in [20, 5]:
        sc = get_time_window(skill_cast, "time", decision_time, sec)
        sh = get_time_window(skill_hit, "time", decision_time, sec)
        feat[f"recent{sec}_skill_cast_count"] = len(sc)
        feat[f"recent{sec}_skill_hit_count"] = len(sh)
        feat[f"recent{sec}_skill_cast_buff_nunique"] = sc["buff_type"].nunique() if len(sc) > 0 else 0
        feat[f"recent{sec}_skill_hit_buff_nunique"] = sh["buff_type"].nunique() if len(sh) > 0 else 0
    feat["last_skill_cast_gap"] = last_time_gap(skill_cast, "time", decision_time)
    feat["last_skill_hit_gap"] = last_time_gap(skill_hit, "time", decision_time)
    feat["decision_skill_name"] = info.get("decision_extra", {}).get("skill_name", "NONE")
    feat["skill_cast_mode_20"] = safe_mode(skill_cast["buff_type"], default="NONE")

    # H. 击倒/死亡特征（8）
    knock_out = knockdown_df[(knockdown_df["attacker_id"].astype(str) == str(main_player_id)) & (knockdown_df["time"] <= decision_time)].copy()
    knock_in = knockdown_df[(knockdown_df["victim_id"].astype(str) == str(main_player_id)) & (knockdown_df["time"] <= decision_time)].copy()
    death_out = death_df[(death_df["killer_id"].astype(str) == str(main_player_id)) & (death_df["time"] <= decision_time)].copy()
    death_in = death_df[(death_df["dead_player_id"].astype(str) == str(main_player_id)) & (death_df["time"] <= decision_time)].copy()
    feat["recent20_knock_out_count"] = len(get_time_window(knock_out, "time", decision_time, 20))
    feat["recent20_knock_in_count"] = len(get_time_window(knock_in, "time", decision_time, 20))
    feat["recent20_death_out_count"] = len(get_time_window(death_out, "time", decision_time, 20))
    feat["recent20_death_in_count"] = len(get_time_window(death_in, "time", decision_time, 20))
    feat["last_knock_out_gap"] = last_time_gap(knock_out, "time", decision_time)
    feat["last_knock_in_gap"] = last_time_gap(knock_in, "time", decision_time)
    feat["last_death_out_gap"] = last_time_gap(death_out, "time", decision_time)
    feat["last_death_in_gap"] = last_time_gap(death_in, "time", decision_time)

    # I. 空间关系特征（9）
    near_df = basic_df[basic_df["time"] <= decision_time].copy()
    latest_state = near_df.sort_values("time").groupby("player_id").tail(1).copy() if len(near_df) > 0 else basic_df.copy()
    main_row_df = latest_state[latest_state["player_id"].astype(str) == str(main_player_id)]
    if len(main_row_df) > 0:
        main_row = main_row_df.iloc[0]
        dists = []
        for _, row in latest_state.iterrows():
            if str(row["player_id"]) == str(main_player_id):
                continue
            d = euclidean_distance_3d(main_row["x"], main_row["y"], main_row["z"], row["x"], row["y"], row["z"])
            if not pd.isna(d):
                dists.append(d)
        feat["nearest_player_dist"] = np.min(dists) if len(dists) > 0 else np.nan
        feat["mean_player_dist"] = np.mean(dists) if len(dists) > 0 else np.nan
        feat["std_player_dist"] = np.std(dists) if len(dists) > 1 else 0.0
    else:
        feat["nearest_player_dist"] = np.nan
        feat["mean_player_dist"] = np.nan
        feat["std_player_dist"] = np.nan

    recent5 = get_time_window(main_basic_before, "time", decision_time, 5)
    recent20_for_dist = get_time_window(main_basic_before, "time", decision_time, 20)
    feat["recent5_path_len"] = float(recent5["speed"].fillna(0).sum()) if len(recent5) > 0 else 0.0
    feat["recent20_path_len"] = float(recent20_for_dist["speed"].fillna(0).sum()) if len(recent20_for_dist) > 0 else 0.0
    feat["recent5_disp_norm"] = math.sqrt((feat.get("recent5_speed_mean", 0) if pd.notna(feat.get("recent5_speed_mean", np.nan)) else 0)**2)
    feat["recent20_disp_norm"] = math.sqrt(
        (feat.get("recent20_x_disp", 0) if pd.notna(feat.get("recent20_x_disp", np.nan)) else 0)**2 +
        (feat.get("recent20_y_disp", 0) if pd.notna(feat.get("recent20_y_disp", np.nan)) else 0)**2 +
        (feat.get("recent20_z_disp", 0) if pd.notna(feat.get("recent20_z_disp", np.nan)) else 0)**2
    )
    feat["path_straightness_20"] = feat["recent20_disp_norm"] / (feat["recent20_path_len"] + 1e-6) if pd.notna(feat["recent20_disp_norm"]) else np.nan
    feat["decision_effect_radius"] = info.get("decision_extra", {}).get("effect_radius", np.nan)

    return feat


# =========================================================
# 7. 批量遍历目录
# =========================================================

def _infer_label_from_path(root_dir, file_path):
    rel = os.path.relpath(file_path, root_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 2:
        return parts[0]
    return os.path.basename(os.path.dirname(file_path))


def collect_txt_files(root_dir, per_class_max=30000):
    pattern = os.path.join(root_dir, "**", "*.txt")
    all_files = sorted(glob.glob(pattern, recursive=True))

    if per_class_max is None or per_class_max <= 0:
        return all_files

    class_counter = defaultdict(int)
    selected = []
    for file_path in all_files:
        label = _infer_label_from_path(root_dir, file_path)
        if class_counter[label] < per_class_max:
            selected.append(file_path)
            class_counter[label] += 1
    return selected


def build_dataset(input_root, output_csv, label_from_folder=True, per_class_max=30000):
    txt_files = collect_txt_files(input_root, per_class_max=per_class_max)
    print(f"发现 {len(txt_files)} 个txt文件（每类最多 {per_class_max}）")

    selected_dist = defaultdict(int)
    for p in txt_files:
        selected_dist[_infer_label_from_path(input_root, p)] += 1
    print("每类采样数量：", dict(sorted(selected_dist.items(), key=lambda x: x[0])))

    all_features = []

    for i, file_path in enumerate(txt_files, 1):
        try:
            parsed = parse_single_file(file_path, label_from_folder=label_from_folder)
            feat = extract_features_from_sample(parsed)
            all_features.append(feat)
            if i % 50 == 0 or i == len(txt_files):
                print(f"[{i}/{len(txt_files)}] 已处理: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"[ERROR] 处理失败: {file_path}, 错误: {e}")

    df = pd.DataFrame(all_features)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"特征表已保存到: {output_csv}")
    print(f"数据形状: {df.shape}")
    print(f"总列数: {len(df.columns)}")
    if "label" in df.columns:
        print("\n标签分布：")
        print(df["label"].value_counts(dropna=False))
    return df


# =========================================================
# 8. 主程序入口
# =========================================================

if __name__ == "__main__":
    # input_root = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\训练数据\classified_samples_1"
    # output_csv = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\output\features_compact_v2.csv"

    # input_root = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\测试1000题"
    # output_csv = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\output\features_test1000_v2.csv"

    # input_root = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\训练数据\classified_samples_2"
    # output_csv = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\output\features_sample_2.csv"

    # input_root = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\训练数据\little"
    # output_csv = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\output\features_little.csv"

    # input_root = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\训练数据\classified_samples_3"
    # output_csv = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\output\features_sample_3.csv"
    input_root = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\训练数据\classified_samples_0"
    output_csv = r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\output\features_sample_0.csv"


    df = build_dataset(input_root, output_csv, label_from_folder=True, per_class_max=30000)
    #df=pd.read_csv(r"E:\EOne\2026游戏安全技术竞赛-游戏安全AI方向-初赛\output\features_compact_v2.csv", encoding="utf-8-sig")
    print(df.head())
    print(df.columns.tolist())
