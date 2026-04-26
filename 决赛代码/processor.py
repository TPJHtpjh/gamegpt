# =============================================================================
# processor.py  —  解析 + 特征计算 + 压缩 + 格式化（合并核心逻辑）
# =============================================================================

import math
import re
from collections import defaultdict
from config import (
    DISP_THRESHOLD_EARLY, DISP_THRESHOLD_LATE,
    YAW_THRESHOLD_EARLY,  YAW_THRESHOLD_LATE,
    SPEED_JUMP_THRESHOLD, ENEMY_DIST_THRESHOLDS,
    LATE_WINDOW_SEC, KEY_WINDOW_INTERVAL, EARLY_SUMMARY_INTERVAL,
    NOISE_BUFF_KEYWORDS, IMPORTANT_BUFF_KEYWORDS,
    FILENAME_LABEL_MAP, LABEL_META,
    ACTION_COMBAT_KEYWORDS, ACTION_MOVE_KEYWORDS, ACTION_SCOPE_KEYWORDS,
    DIR_8,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def yaw_to_dir(yaw: float) -> str:
    return DIR_8[int((yaw % 360 + 22.5) / 45) % 8]

def dist_2d(ax, az, bx, bz) -> float:
    return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)

def yaw_delta(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d

def classify_speed(speed: float) -> str:
    if speed < 0.3:   return "静止"
    if speed < 3.0:   return "慢走"
    if speed < 8.0:   return "跑步"
    return "冲刺"

def is_noise_buff(buff: str) -> bool:
    return any(k in buff for k in NOISE_BUFF_KEYWORDS)

def is_important_buff(buff: str) -> bool:
    return any(k in buff for k in IMPORTANT_BUFF_KEYWORDS)

def crossed_threshold(prev: float, curr: float, thresholds: list) -> bool:
    for t in thresholds:
        if (prev > t >= curr) or (prev < t <= curr):
            return True
    return False

def player_tag(pid: str, players_info: dict, main_pid: str) -> str:
    info = players_info.get(pid, {})
    name = info.get('name', '?')
    team = info.get('team', '?')
    return '主玩家' if pid == main_pid else f"[队伍{team}]{name}({pid})"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1  解析原始 TXT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_file(filepath: str) -> dict:
    """
    返回:
      players_info : {pid: {team, name}}
      frames       : {ts: {pid: {x,y,z,yaw,pitch,speed,scope}}}
      events       : [{ts, type, ...}]  按时间排序
      labels       : [{ts, label_type, pid, ...}]
      sorted_ts    : [float]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    players_info = {}
    frames       = defaultdict(dict)
    events       = []
    labels       = []
    decision_actions = []

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.split('|')
        if len(parts) < 2:
            continue
        try:
            ts = float(parts[0])
        except ValueError:
            continue
        etype = parts[1]

        # ── 游戏开始 ──────────────────────────────────────────────────────────
        if etype == '游戏开始':
            pid  = parts[2] if len(parts) > 2 else ''
            team = parts[3] if len(parts) > 3 else ''
            name = parts[4] if len(parts) > 4 else ''
            players_info[pid] = {'team': team, 'name': name}

        # ── 玩家基础信息 ───────────────────────────────────────────────────────
        elif etype == '玩家基础信息':
            if len(parts) < 12:
                continue
            pid = parts[2].replace('玩家', '')
            try:
                px, py, pz = float(parts[3]), float(parts[4]), float(parts[5])
                yaw   = float(parts[6])  if parts[6]  else 0.0
                pitch = float(parts[7])  if parts[7]  else 0.0
                vx    = float(parts[8])  if parts[8]  else 0.0
                vy    = float(parts[9])  if parts[9]  else 0.0
                vz    = float(parts[10]) if parts[10] else 0.0
            except (ValueError, IndexError):
                continue
            scope = parts[19] if len(parts) > 19 else '关镜'
            speed = math.sqrt(vx**2 + vy**2 + vz**2)
            frames[ts][pid] = {
                'x': px, 'y': py, 'z': pz,
                'yaw': yaw, 'pitch': pitch,
                'speed': speed, 'scope': scope,
            }

        # ── 技能生效 ───────────────────────────────────────────────────────────
        elif etype == '技能生效':
            target = parts[2].replace('玩家', '') if len(parts) > 2 else ''
            caster = parts[3].replace('玩家', '') if len(parts) > 3 else ''
            buff   = parts[4] if len(parts) > 4 else ''
            events.append({'ts': ts, 'type': '技能生效',
                           'target': target, 'caster': caster, 'buff': buff})

        # ── 动作 ───────────────────────────────────────────────────────────────
        elif etype == '动作':
            pid    = parts[2].replace('玩家', '') if len(parts) > 2 else ''
            action = parts[3] if len(parts) > 3 else ''
            if action.startswith('（决策）') or action.startswith('(决策)'):
                clean_action = action.replace('（决策）', '').replace('(决策)', '').strip()
                decision_actions.append({'ts': ts, 'pid': pid, 'action': clean_action})
            else:
                events.append({'ts': ts, 'type': '动作', 'pid': pid, 'action': action})

        # ── 玩家造成伤害 ────────────────────────────────────────────────────────
        elif etype == '玩家造成伤害':
            atk = parts[2].replace('玩家', '') if len(parts) > 2 else ''
            vic = parts[3].replace('玩家', '') if len(parts) > 3 else ''
            try:
                hp_dmg  = float(parts[12]) if len(parts) > 12 else 0.0
                hp_left = float(parts[13]) if len(parts) > 13 else 0.0
                is_down = parts[15]        if len(parts) > 15 else '0'
            except (ValueError, IndexError):
                hp_dmg, hp_left, is_down = 0, 0, '0'
            events.append({'ts': ts, 'type': '伤害',
                           'attacker': atk, 'victim': vic,
                           'hp_dmg': hp_dmg, 'hp_left': hp_left,
                           'is_down': is_down})

        # ── 玩家击倒 ───────────────────────────────────────────────────────────
        elif etype == '玩家击倒':
            atk = parts[2].replace('玩家', '') if len(parts) > 2 else ''
            vic = parts[3].replace('玩家', '') if len(parts) > 3 else ''
            events.append({'ts': ts, 'type': '击倒', 'attacker': atk, 'victim': vic})

        # ── 玩家死亡 ───────────────────────────────────────────────────────────
        elif etype == '玩家死亡':
            dead   = parts[2].replace('玩家', '') if len(parts) > 2 else ''
            cause  = parts[3] if len(parts) > 3 else ''
            killer = parts[7].replace('玩家', '') if len(parts) > 7 else ''
            events.append({'ts': ts, 'type': '死亡',
                           'dead': dead, 'cause': cause, 'killer': killer})

        # ── 标点 ───────────────────────────────────────────────────────────────
        elif etype == '标点':
            pid  = parts[2].replace('玩家', '') if len(parts) > 2 else ''
            mark = parts[3] if len(parts) > 3 else ''
            events.append({'ts': ts, 'type': '标点', 'pid': pid, 'mark': mark})

        # ── 决策行 —— 多种格式统一处理 ────────────────────────────────────────
        # 格式1: ts|（决策）开火|玩家xxx|yaw|pitch
        # 格式2: ts|（决策）放技能|玩家xxx|skill_id
        # 格式3: ts|（决策）|玩家xxx          ← 测试集（只有pid，无具体类型）
        elif '决策' in etype:
            pid = parts[2].replace('玩家', '') if len(parts) > 2 else ''

            # 从 etype 中提取具体决策子类型
            sub = etype.replace('（决策）', '').replace('(决策)', '').strip()

            if '开火' in sub or 'Fire' in sub:
                yaw_l   = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
                pitch_l = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
                labels.append({'ts': ts, 'label_type': 'Fire', 'pid': pid,
                                'yaw': yaw_l, 'pitch': pitch_l})

            elif '放技能' in sub or 'SkillStart' in sub:
                skill_id = parts[3] if len(parts) > 3 else ''
                labels.append({'ts': ts, 'label_type': 'SkillStart', 'pid': pid,
                                'skill_id': skill_id})

            elif '丢雷' in sub or 'Grenade' in sub:
                lx = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
                lz = float(parts[6]) if len(parts) > 6 and parts[6] else 0.0
                radius = parts[7] if len(parts) > 7 else ''
                labels.append({'ts': ts, 'label_type': 'Grenade', 'pid': pid,
                                'land_x': lx, 'land_z': lz, 'radius': radius})

            elif '搜' in sub or 'Looting' in sub:
                quality = parts[3] if len(parts) > 3 else ''
                source  = parts[4] if len(parts) > 4 else ''
                labels.append({'ts': ts, 'label_type': 'Looting', 'pid': pid,
                                'quality': quality, 'source': source})

            elif '救援' in sub or 'BeingRescue' in sub:
                rescued   = parts[3].replace('玩家', '') if len(parts) > 3 else ''
                from_dead = parts[4] if len(parts) > 4 else '0'
                labels.append({'ts': ts, 'label_type': 'BeingRescue', 'pid': pid,
                                'rescued': rescued, 'from_dead': from_dead})

            else:
                # sub 为空（测试集）或 Action 类 → label_type 先设为 None，
                # 后续由文件名推断补全
                labels.append({'ts': ts, 'label_type': None, 'pid': pid,
                                'raw_etype': etype})

    # 计算帧数据中出现频率最高的玩家
    pid_counts = defaultdict(int)
    for ts, f_data in frames.items():
        for p in f_data.keys():
            pid_counts[p] += 1
    most_freq_pid = max(pid_counts, key=pid_counts.get) if pid_counts else '0'

    # 当决策行或动作行中没有正确提取出 main_pid，则回退/设定为出现频率最高的玩家
    for lb in labels:
        if not lb.get('pid'):
            lb['pid'] = most_freq_pid
    for da in decision_actions:
        if not da.get('pid'):
            da['pid'] = most_freq_pid

    sorted_ts = sorted(frames.keys())
    events.sort(key=lambda e: e['ts'])

    return {
        'players_info':     players_info,
        'frames':           frames,
        'events':           events,
        'labels':           labels,
        'decision_actions': decision_actions,
        'sorted_ts':        sorted_ts,
        'most_freq_pid':    most_freq_pid,
    }


def infer_label_from_filename(filename: str) -> str | None:
    """从文件名（小写）推断决策类型"""
    lower = filename.lower()
    for keyword, label_type in FILENAME_LABEL_MAP.items():
        if keyword in lower:
            # 返回标准化 label_type（首字母大写匹配 LABEL_META key）
            return label_type.capitalize() if label_type.lower() in \
                   [k.lower() for k in LABEL_META] else label_type
    return None


def infer_label_from_filename(filename: str):
    """从文件名推断决策类型，返回 LABEL_META 中的标准 key"""
    lower = filename.lower()
    # 优先长匹配（beingrescue > rescure，skillstart > skill）
    candidates = sorted(FILENAME_LABEL_MAP.keys(), key=len, reverse=True)
    for keyword in candidates:
        if keyword in lower:
            raw = FILENAME_LABEL_MAP[keyword]
            # 找到 LABEL_META 中大小写匹配的 key
            for k in LABEL_META:
                if k.lower() == raw.lower():
                    return k
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2  战术特征计算
# ═══════════════════════════════════════════════════════════════════════════════

def get_nearest_enemy(ts, frames, players_info, main_pid, main_team):
    if main_pid not in frames.get(ts, {}):
        return None, 9999.0
    mp = frames[ts][main_pid]
    best_pid, best_dist = None, 9999.0
    for pid, pos in frames[ts].items():
        if pid == main_pid or pid not in players_info:
            continue
        if players_info[pid]['team'] == main_team:
            continue
        d = dist_2d(mp['x'], mp['z'], pos['x'], pos['z'])
        if d < best_dist:
            best_dist, best_pid = d, pid
    return best_pid, best_dist


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3  生成 input_text（20s 上文）
# ═══════════════════════════════════════════════════════════════════════════════

def generate_input_text(data: dict, main_pid: str, label_ts: float) -> str:
    frames       = data['frames']
    events       = data['events']
    players_info = data['players_info']
    sorted_ts    = data['sorted_ts']

    main_team = players_info.get(main_pid, {}).get('team', '')
    main_name = players_info.get(main_pid, {}).get('name', '?')

    ctx_start  = label_ts - 20.0
    ctx_ts     = [t for t in sorted_ts if ctx_start <= t < label_ts]   # 严格截止到决策时刻之前
    late_start = label_ts - LATE_WINDOW_SEC
    init_ts    = ctx_ts[0] if ctx_ts else label_ts

    lines = []

    # ── 全局态势快照 ──────────────────────────────────────────────────────────
    allies  = [p for p, i in players_info.items()
               if i['team'] == main_team and p != main_pid]
    enemies = [p for p, i in players_info.items()
               if i['team'] != main_team]
    # 找主玩家最早有数据的帧，避免 (0,0)
    init_frame_ts = next((t for t in sorted_ts if main_pid in frames.get(t, {})), init_ts)
    minit  = frames.get(init_frame_ts, {}).get(main_pid, {})
    eid, edist = get_nearest_enemy(init_frame_ts, frames, players_info,
                                   main_pid, main_team)
    ename  = players_info.get(eid, {}).get('name', '?') if eid else '暂无'

    lines += [
        "【全局态势】",
        (f"主玩家: {main_name}(ID={main_pid}, 队伍{main_team}) | "
         f"共{len(players_info)}名干员 | 我方{len(allies)+1}人 / 敌方{len(enemies)}人"),
        (f"初始位置: ({minit.get('x',0):.0f},{minit.get('z',0):.0f}) | "
         f"初始朝向: {yaw_to_dir(minit.get('yaw',0))} | "
         f"开局最近敌方: {ename}({eid}) 距{edist:.0f}m"),
        "",
    ]

    # ── 战斗事件（全窗口高优先级） ────────────────────────────────────────────
    combat = []
    for ev in events:
        if not (ctx_start <= ev['ts'] < label_ts):   # 不含决策时刻本身
            continue
        t = ev['ts']
        if ev['type'] == '伤害':
            atag = player_tag(ev['attacker'], players_info, main_pid)
            vtag = player_tag(ev['victim'],   players_info, main_pid)
            down = " 【击倒】" if ev.get('is_down', '0') != '0' else ''
            combat.append(f"  {t:.2f}s: {atag} → {vtag} "
                          f"造成{ev['hp_dmg']:.0f}伤害(剩余{ev['hp_left']:.0f}){down}")
        elif ev['type'] == '击倒':
            combat.append(f"  {t:.2f}s: {player_tag(ev['attacker'],players_info,main_pid)}"
                          f" 击倒 {player_tag(ev['victim'],players_info,main_pid)}")
        elif ev['type'] == '死亡':
            ktag = player_tag(ev['killer'], players_info, main_pid) if ev['killer'] else '未知'
            combat.append(f"  {t:.2f}s: {player_tag(ev['dead'],players_info,main_pid)}"
                          f" 死亡 (杀手: {ktag})")
    if combat:
        lines += ["【战斗事件】"] + combat + [""]

    # ── 前段摘要 ──────────────────────────────────────────────────────────────
    early_main = [(t, frames[t][main_pid])
                  for t in ctx_ts if t < late_start and main_pid in frames.get(t, {})]

    if early_main:
        lines.append(f"【前段摘要 ({ctx_start:.1f}s ~ {late_start:.1f}s)】")
        speeds = [f['speed'] for _, f in early_main]
        still_ratio = sum(1 for s in speeds if s < 0.3) / max(len(speeds), 1)

        if still_ratio > 0.85:
            f0, f1 = early_main[0][1], early_main[-1][1]
            total_d = dist_2d(f0['x'], f0['z'], f1['x'], f1['z'])
            lines.append(f"  主玩家长时间静止潜伏(静止率{still_ratio*100:.0f}%)，"
                         f"累计位移{total_d:.1f}m，持续朝{yaw_to_dir(f0['yaw'])}方向警戒")
        else:
            seg = ctx_start
            while seg < late_start:
                seg_end    = seg + EARLY_SUMMARY_INTERVAL
                seg_frames = [frames[t][main_pid] for t in ctx_ts
                              if seg <= t < seg_end and main_pid in frames.get(t, {})]
                if len(seg_frames) >= 2:
                    f0, f1   = seg_frames[0], seg_frames[-1]
                    seg_d    = dist_2d(f0['x'], f0['z'], f1['x'], f1['z'])
                    seg_spd  = sum(f['speed'] for f in seg_frames) / len(seg_frames)
                    if seg_d > DISP_THRESHOLD_EARLY or seg_spd > 1.0:
                        # 找该段末尾最近敌方
                        seg_t_last = [t for t in ctx_ts if seg <= t < seg_end]
                        if seg_t_last:
                            e2id, e2d = get_nearest_enemy(seg_t_last[-1], frames,
                                                          players_info, main_pid, main_team)
                            e2n = players_info.get(e2id, {}).get('name', '?') if e2id else '无'
                            lines.append(
                                f"  {seg:.1f}~{seg_end:.1f}s: 朝{yaw_to_dir(f1['yaw'])}移动"
                                f"{seg_d:.1f}m 均速{seg_spd:.1f}m/s | 最近敌[{e2n}]{e2d:.0f}m")
                seg = seg_end

        # 前段离散事件（动作+重要技能）
        for ev in events:
            if not (ctx_start <= ev['ts'] < late_start):
                continue
            t = ev['ts']
            if ev['type'] == '动作' and ev.get('pid') == main_pid:
                lines.append(f"  {t:.2f}s: 主玩家执行动作 [{ev['action']}]")
            elif ev['type'] == '技能生效':
                buf = ev.get('buff', '')
                cid = ev.get('caster', '')
                if is_noise_buff(buf):
                    continue
                if cid == main_pid or is_important_buff(buf):
                    tag = "主玩家" if cid == main_pid else f"玩家{cid}"
                    lines.append(f"  {t:.2f}s: {tag} 激活 [{buf}]")
        lines.append("")

    # ── 关键窗口（高密度变化触发） ────────────────────────────────────────────
    key_ts = [t for t in ctx_ts if t >= late_start]
    lines.append(f"【关键窗口 ({late_start:.1f}s ~ {label_ts:.2f}s)】")

    prev_feat = None
    for t in key_ts:
        if main_pid not in frames.get(t, {}):
            continue
        cur = frames[t][main_pid]
        eid2, edist2 = get_nearest_enemy(t, frames, players_info, main_pid, main_team)
        ename2 = players_info.get(eid2, {}).get('name', '?') if eid2 else '无'

        should_output = prev_feat is None
        notes = []

        if prev_feat:
            # 位移
            disp = dist_2d(cur['x'], cur['z'], prev_feat['x'], prev_feat['z'])
            if disp > DISP_THRESHOLD_LATE:
                should_output = True
                move_angle = math.degrees(
                    math.atan2(cur['x'] - prev_feat['x'],
                               -(cur['z'] - prev_feat['z']))) % 360
                notes.append(f"→{yaw_to_dir(move_angle)}位移{disp:.1f}m")
            # 朝向
            yd = yaw_delta(cur['yaw'], prev_feat['yaw'])
            if yd > YAW_THRESHOLD_LATE:
                should_output = True
                notes.append(f"转向{yaw_to_dir(cur['yaw'])}({cur['yaw']:.0f}°)")
            # 开镜
            if cur['scope'] != prev_feat['scope']:
                should_output = True
                notes.append(f"★{cur['scope']}")
            # 速度突变
            sdelta = abs(cur['speed'] - prev_feat['speed'])
            if sdelta > SPEED_JUMP_THRESHOLD:
                should_output = True
                notes.append(f"速度{prev_feat['speed']:.1f}→{cur['speed']:.1f}m/s")
            # 敌距跨档
            if eid2 and crossed_threshold(prev_feat.get('edist', 9999), edist2,
                                          ENEMY_DIST_THRESHOLDS):
                should_output = True
                notes.append(f"敌距跨档→{edist2:.0f}m")
            # 兜底：每 KEY_WINDOW_INTERVAL 秒必须出一帧
            if (t - prev_feat['ts']) >= KEY_WINDOW_INTERVAL:
                should_output = True

        if not should_output:
            prev_feat = {**cur, 'ts': t, 'edist': edist2}
            continue

        note_str = f" [{', '.join(notes)}]" if notes else ""
        lines.append(
            f"  {t:.2f}s: ({cur['x']:.1f},{cur['z']:.1f}) "
            f"朝{yaw_to_dir(cur['yaw'])} {cur['scope']} "
            f"{classify_speed(cur['speed'])}{note_str} | 最近敌[{ename2}]{edist2:.0f}m")

        prev_feat = {**cur, 'ts': t, 'edist': edist2}

    # 关键窗口离散事件
    kw_evts = []
    for ev in events:
        if not (late_start <= ev['ts'] < label_ts):   # 不含决策时刻本身
            continue
        t = ev['ts']
        if ev['type'] == '动作':
            tag = "主玩家" if ev['pid'] == main_pid else f"玩家{ev['pid']}"
            kw_evts.append(f"  {t:.2f}s: {tag} [{ev['action']}]")
        elif ev['type'] == '技能生效':
            buf = ev.get('buff', '')
            if is_noise_buff(buf):
                continue
            cid = ev.get('caster', '')
            tag = "主玩家" if cid == main_pid else f"玩家{cid}"
            kw_evts.append(f"  {t:.2f}s: {tag} 激活 [{buf}]")
    if kw_evts:
        lines += ["  --- 事件 ---"] + kw_evts

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4  生成 label_text（后5秒续写）
# 格式: 主玩家…… 随后…… 最后……
# ═══════════════════════════════════════════════════════════════════════════════

def generate_label_text(label: dict, data: dict, main_pid: str, cut_ts: float | None = None, label_end_ts: float | None = None) -> str:
    frames       = data['frames']
    players_info = data['players_info']
    sorted_ts    = data['sorted_ts']
    events       = data['events']

    label_type = label['label_type']
    label_ts   = cut_ts if cut_ts is not None else label['ts']
    label_end_ts = label_end_ts if label_end_ts is not None else label.get('label_end_ts', label['ts'])
    main_name  = players_info.get(main_pid, {}).get('name', '无名')
    main_team  = players_info.get(main_pid, {}).get('team', '')
    meta       = LABEL_META.get(label_type, {'camp': '未知', 'action': '未知'})

    # 最终帧状态
    final_ts   = min(sorted_ts, key=lambda t: abs(t - label_ts)) if sorted_ts else label_ts
    mf         = frames.get(final_ts, {}).get(main_pid, {})
    eid, edist = get_nearest_enemy(final_ts, frames, players_info, main_pid, main_team)
    ename      = players_info.get(eid, {}).get('name', '?') if eid else '未知敌方'

    # 关键窗口速度 & 前段行为特征
    ctx_start  = label_ts - 20.0
    kw_frames  = [frames[t][main_pid]
                  for t in sorted_ts
                  if (label_ts - 5) <= t <= label_ts and main_pid in frames.get(t, {})]
    early_fs   = [frames[t][main_pid]
                  for t in sorted_ts
                  if ctx_start <= t <= label_ts - 5 and main_pid in frames.get(t, {})]
    avg_kw_spd = sum(f['speed'] for f in kw_frames) / max(len(kw_frames), 1)
    still_pct  = (sum(1 for f in early_fs if f['speed'] < 0.3) / max(len(early_fs), 1))

    # 前置状态描述（段首）
    if still_pct > 0.8:
        pre = f"主玩家{main_name}长时间潜伏观察"
    elif avg_kw_spd > 7:
        pre = f"主玩家{main_name}高速冲刺机动"
    elif avg_kw_spd > 3:
        pre = f"主玩家{main_name}持续跑步推进"
    else:
        pre = f"主玩家{main_name}缓速移动"

    # 朝向 & 开镜描述
    scope_state = mf.get('scope', '关镜')
    direction   = yaw_to_dir(mf.get('yaw', 0))

    # ── 按决策类型生成三段式续写 ────────────────────────────────────────────
    if label_type == 'Fire':
        fire_yaw   = label.get('yaw', mf.get('yaw', 0))
        fire_pitch = label.get('pitch', mf.get('pitch', 0))
        fire_dir   = yaw_to_dir(fire_yaw)
        speed_desc = ("快速压缩交战距离" if avg_kw_spd > 5 else "缓慢接近敌方")
        return (
            f"{pre}，{speed_desc}，朝向{direction}；"
            f"随后进入{scope_state}状态，锁定{ename}（距约{edist:.0f}m）；"
            f"最后向{fire_dir}方向（偏航{fire_yaw:.1f}°/俯仰{fire_pitch:.1f}°）主动开火，发起【交战-开火】。"
        )

    elif label_type == 'SkillStart':
        skill_id = label.get('skill_id', '')
        # 尝试从事件里找该玩家在label附近的技能buff
        skill_name = ''
        for ev in events:
            if (ev['type'] == '技能生效' and ev.get('caster') == main_pid
                    and abs(ev['ts'] - label_ts) <= 2.0):
                buf = ev.get('buff', '')
                if not is_noise_buff(buf):
                    skill_name = buf
                    break
        skill_desc = f"[{skill_name}]" if skill_name else f"[ID:{skill_id}]"
        return (
            f"{pre}，观察战场态势，当前最近敌方为{ename}（距约{edist:.0f}m）；"
            f"随后判断释放技能时机，选择对队伍实施战术支援；"
            f"最后主动释放技能{skill_desc}，执行【交战-放技能】。"
        )

    elif label_type == 'Grenade':
        lx = label.get('land_x', 0)
        lz = label.get('land_z', 0)
        mx = mf.get('x', 0)
        mz = mf.get('z', 0)
        throw_d   = dist_2d(mx, mz, lx, lz)
        throw_dir = yaw_to_dir(math.degrees(math.atan2(lx-mx, -(lz-mz))) % 360)
        radius    = label.get('radius', '?')
        return (
            f"{pre}，观察到敌方{ename}（距约{edist:.0f}m）聚集或处于掩体后方；"
            f"随后调整投掷站位，保持{scope_state}状态；"
            f"最后向{throw_dir}方向投出投掷物，落点约{throw_d:.0f}m外（落点({lx:.0f},{lz:.0f})），影响范围{radius}m，执行【交战-丢雷】。"
        )

    elif label_type == 'Looting':
        quality = label.get('quality', '?')
        source  = label.get('source', '?')
        return (
            f"{pre}，确认周边{ename}（距约{edist:.0f}m）暂无直接威胁；"
            f"随后调整位置接近物资点，保持警戒姿态；"
            f"最后在{source}处搜取品质[{quality}]物资，执行【避战-搜索】。"
        )

    elif label_type == 'BeingRescue':
        rescued_pid  = label.get('rescued', '')
        from_dead    = label.get('from_dead', '0')
        rtype        = "从倒地救援" if from_dead != '0' else "救援"
        # 修复：rescued_pid 为空或等于 main_pid 时不展示角色名，避免"救援自己"
        if rescued_pid and rescued_pid != main_pid and rescued_pid in players_info:
            rescued_name = players_info[rescued_pid].get('name', '队友')
            rescue_target = f"队友{rescued_name}"
        else:
            rescue_target = "受伤队友"
        speed_desc = '高速' if avg_kw_spd > 5 else '谨慎'
        return (
            f"{pre}，发现{rescue_target}需要救援，评估周围威胁（最近敌{ename}距约{edist:.0f}m）；"
            f"随后{speed_desc}接近队友位置；"
            f"最后对{rescue_target}实施{rtype}，执行【避战-救援】。"
        )

    elif label_type == 'Action':
        # 找label时刻前后主玩家的动作序列
        future_action_evts = data.get('decision_actions', [])
        action_evts = [ev for ev in future_action_evts
                       if ev.get('pid') == main_pid and label_ts <= ev['ts'] <= label_end_ts]
        if not action_evts:
            action_evts = [ev for ev in events
                           if ev['type'] == '动作' and ev.get('pid') == main_pid
                           and max(0.0, label_ts - 1.0) <= ev['ts'] < label_ts]
        if action_evts:
            acts = [ev['action'] for ev in action_evts[-3:]]
            act_chain = "→".join(acts)
            # 判断是否是战术性动作
            is_combat_act = any(a in ACTION_COMBAT_KEYWORDS for a in acts)
            camp_desc = "准备交战" if is_combat_act else "保持机动"
            return (
                f"{pre}，{camp_desc}，面朝{direction}（最近敌{ename}距约{edist:.0f}m）；"
                f"随后执行一系列动作指令（{act_chain}）调整作战姿态；"
                f"最后完成动作链，以【动作-{acts[-1]}】作为结果动作。"
            )
        else:
            return (
                f"{pre}，面朝{direction}，当前{scope_state}状态（最近敌{ename}距约{edist:.0f}m）；"
                f"随后调整身体姿态与位置；"
                f"最后完成动作调整，执行【动作指令】。"
            )

    else:
        return (
            f"{pre}，当前朝{direction}，最近敌{ename}距约{edist:.0f}m；"
            f"随后完成战术准备；"
            f"最后执行决策[{label_type}]。"
        )
