# compress_jsonl_v4.py
import json, re
from pathlib import Path

INPUT_DIR  = "./数据/output_jsonl"   # 原始文件目录
OUTPUT_DIR = "./数据/compressed_jsonl" # 压缩后文件目录

MAX_KEY_LINES        = 35   # 关键窗口帧行上限（合并后计数）
MAX_EVENT_LINES      = 12   # 关键窗口事件行上限
MAX_SUMMARY_LINES    = 6    # 前段摘要移动行上限
MAX_SUMMARY_EVENTS   = 6    # 前段摘要事件行上限
MAX_COMBAT_LINES     = 10    # 战斗事件行上限

# 连续帧合并：最近敌距离变化容忍值（米）
MERGE_DIST_TOL = 5
# 连续帧合并：至少连续 N 帧才合并（避免把2帧也合并掉丢失细节）
MERGE_MIN_FRAMES = 3

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 噪声过滤规则
# ═══════════════════════════════════════════════════════════════════════════════

NOISE_BUFF_PATTERNS = [
    r"高效治疗SOL",
    r"侦察标记敌人被动",
    r"屏蔽侦察兵标记buff",
    r"被侦察伤害标记",
    r"入水",
    r"止痛\d+秒",
    r"高级体力针",
    r"初级体力针",
    r"医疗箱止痛\d+秒",
    r"初级负重针",
    r"动能辅助系统.*?特效",
    r"动力推进特效",
    r"动力推进音效",
    r"【弃用】",
    r"（弃用）",
]
_NOISE_RE = re.compile("|".join(NOISE_BUFF_PATTERNS))
_MASS_STAND_RE = re.compile(r"玩家\S+\s+\[站\]")

def is_noise_buff_line(line: str) -> bool:
    if "激活" not in line:
        return False
    m = re.search(r"激活\s+\[(.+?)\]", line)
    return bool(m and _NOISE_RE.search(m.group(1)))

def is_junk_line(line: str) -> bool:
    s = line.strip()
    if re.search(r"开局最近敌方.*?None|距9999m", s):
        return True
    if re.search(r"锁定未知敌方|距约9999m", s):
        return True
    if re.search(r"造成[01]伤害", s):
        return True
    if re.search(r"\| 最近敌\[.*?\]9999m", s):
        return True
    m = re.match(r"\s+\d+\.?\d*~\d+\.?\d*s: 朝.*?移动(\d+\.?\d*)m 均速(\d+\.?\d*)m/s", s)
    if m and float(m.group(1)) < 0.2:
        return True
    return False

def clean_global_line(line: str) -> str:
    # 暂无(None) 距9999m → 暂无
    line = re.sub(r": 暂无\(None\) 距9999m", ": 暂无", line)
    # (0,0) + 暂无 + 9999 三者同时出现才是缺失占位符，整行已被 is_junk_line 过滤
    return line

# ═══════════════════════════════════════════════════════════════════════════════
# 帧行解析 & note 精简
# ═══════════════════════════════════════════════════════════════════════════════

# 解析一行关键窗口帧，返回结构化字典，解析失败返回 None
# 格式：  15.00s: (567.5,-439.7) 朝东南 关镜 跑步 [note,...] | 最近敌[无名]85m
_FRAME_RE = re.compile(
    r"^\s*"
    r"(?P<ts>\d+\.\d+)s:\s+"
    r"\((?P<x>-?\d+\.?\d*),(?P<y>-?\d+\.?\d*)\)\s+"
    r"朝(?P<dir>\S+)\s+"
    r"(?P<scope>开镜|关镜)\s+"
    r"(?P<move>跑步|慢走|冲刺|静止)"
    r"(?:\s+\[(?P<notes>[^\]]*)\])?"
    r"\s+\|\s+最近敌\[(?P<enemy>[^\]]+)\](?P<dist>\d+)m"
)

def parse_frame(line: str):
    m = _FRAME_RE.match(line)
    if not m:
        return None
    notes_raw = m.group("notes") or ""
    notes = [n.strip() for n in notes_raw.split(",") if n.strip()]
    return {
        "ts":    float(m.group("ts")),
        "x":     float(m.group("x")),
        "y":     float(m.group("y")),
        "dir":   m.group("dir"),
        "scope": m.group("scope"),
        "move":  m.group("move"),
        "notes": notes,
        "enemy": m.group("enemy"),
        "dist":  int(m.group("dist")),
        "raw":   line,
    }

def simplify_notes(notes: list) -> list:
    """
    精简帧 note（合并判断 & 渲染共用）：
    - 过滤极小横向漂移（≤ 0.5m）
    - 过滤纯速度波动（无论幅度），因为跑步帧率采样导致速度来回波动，
      只要没有转向，就不是有效信息
    - 保留：转向、开/关镜状态变化、冲刺等
    """
    filtered = []
    for n in notes:
        if re.match(r"→.+位移0\.[0-4]\d*m$", n):
            continue
        if re.match(r"速度\d+\.?\d*→\d+\.?\d*m/s$", n):
            continue
        filtered.append(n)
    return filtered

def render_frame(f: dict) -> str:
    """将解析后的帧字典重新渲染为行文本"""
    notes = simplify_notes(f["notes"])
    note_str = f" [{', '.join(notes)}]" if notes else ""
    return (f"  {f['ts']:.2f}s: ({f['x']:.1f},{f['y']:.1f})"
            f" 朝{f['dir']} {f['scope']} {f['move']}"
            f"{note_str}"
            f" | 最近敌[{f['enemy']}]{f['dist']}m")

# ═══════════════════════════════════════════════════════════════════════════════
# 连续帧合并
# ═══════════════════════════════════════════════════════════════════════════════

def _can_merge(a: dict, b: dict) -> bool:
    """判断相邻两帧是否可以合入同一段"""
    if a["dir"] != b["dir"]:
        return False
    if a["scope"] != b["scope"]:
        return False
    if a["move"] != b["move"]:
        return False
    if abs(a["dist"] - b["dist"]) > MERGE_DIST_TOL:
        return False
    # b 帧有有效 note（转向、速度大变等）则不合并，保留原帧
    if simplify_notes(b["notes"]):
        return False
    return True

def render_merged(group: list) -> str:
    """将一组可合并帧渲染为单行摘要"""
    first, last = group[0], group[-1]
    n = len(group)
    dist_str = (f"{first['dist']}m"
                if first["dist"] == last["dist"]
                else f"{first['dist']}→{last['dist']}m")
    return (f"  {first['ts']:.2f}s~{last['ts']:.2f}s:"
            f" ({first['x']:.1f},{first['y']:.1f})→({last['x']:.1f},{last['y']:.1f})"
            f" 朝{first['dir']} {first['scope']} {first['move']}"
            f" [×{n}帧]"
            f" | 最近敌[{last['enemy']}]{dist_str}")

def merge_key_frames(lines: list) -> list:
    """
    输入：已过滤但未合并的关键窗口帧行列表（纯文本）
    输出：合并后的行列表
    """
    # 解析所有可解析帧；不可解析的直接透传
    parsed = []  # list of (frame_dict | None, original_line)
    for line in lines:
        f = parse_frame(line)
        parsed.append((f, line))

    result = []
    i = 0
    while i < len(parsed):
        f, raw = parsed[i]
        if f is None:
            # 无法解析的行（如静止行格式不同）直接输出
            result.append(simplify_frame_line(raw))
            i += 1
            continue

        # 尝试向后延伸当前合并段
        group = [f]
        j = i + 1
        while j < len(parsed):
            nf, nraw = parsed[j]
            if nf is None:
                break
            if _can_merge(group[-1], nf):
                group.append(nf)
                j += 1
            else:
                break

        if len(group) >= MERGE_MIN_FRAMES:
            result.append(render_merged(group))
        else:
            # 不满足合并条件，逐帧输出（仍做 note 精简）
            for gf in group:
                result.append(render_frame(gf))
        i += len(group)

    return result

def simplify_frame_line(line: str) -> str:
    """对无法用 parse_frame 解析的行做基础 note 精简"""
    m = re.match(r"^(\s+\d+\.\d+s:.*?)\s+\[(.+?)\](\s+\| .+)$", line)
    if not m:
        return line
    prefix, notes_str, suffix = m.group(1), m.group(2), m.group(3)
    notes = [n.strip() for n in notes_str.split(",")]
    filtered = simplify_notes(notes)
    if not filtered:
        return f"{prefix}{suffix}"
    return f"{prefix} [{', '.join(filtered)}]{suffix}"

# ═══════════════════════════════════════════════════════════════════════════════
# 事件行处理
# ═══════════════════════════════════════════════════════════════════════════════

def dedup_and_filter_events(lines: list) -> list:
    seen_buffs = set()
    ts_stand = {}
    for line in lines:
        if _MASS_STAND_RE.search(line) and "主玩家" not in line:
            m = re.match(r"\s+(\d+\.\d+)s:", line)
            if m:
                ts_stand.setdefault(m.group(1), []).append(line)
    bulk_stand_ts = {ts for ts, ls in ts_stand.items() if len(ls) >= 3}

    out = []
    bulk_added = set()
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if is_noise_buff_line(line):
            continue
        m_ts = re.match(r"\s+(\d+\.\d+)s:", line)
        if m_ts and m_ts.group(1) in bulk_stand_ts:
            if _MASS_STAND_RE.search(line) and "主玩家" not in line:
                ts_key = m_ts.group(1)
                if ts_key not in bulk_added:
                    cnt = len(ts_stand[ts_key])
                    out.append(f"  {ts_key}s: （{cnt}名玩家同时[站]）")
                    bulk_added.add(ts_key)
                continue
        bm = re.search(r"(玩家\S+|主玩家)\s+激活\s+\[(.+?)\]", line)
        if bm:
            key = (bm.group(1), bm.group(2))
            if key in seen_buffs:
                continue
            seen_buffs.add(key)
        out.append(line)
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# 采样工具
# ═══════════════════════════════════════════════════════════════════════════════

def uniform_sample(lines, n):
    if len(lines) <= n:
        return lines
    step = len(lines) / n
    return [lines[int(i * step)] for i in range(n)]

def keep_first_last_sample(lines, n):
    if len(lines) <= n:
        return lines
    middle = lines[1:-1]
    sampled = uniform_sample(middle, max(n - 2, 0))
    return [lines[0]] + sampled + [lines[-1]]

# ═══════════════════════════════════════════════════════════════════════════════
# 主压缩函数
# ═══════════════════════════════════════════════════════════════════════════════

def compress_input(text: str) -> str:
    lines = text.split("\n")

    header_lines   = []
    combat_lines   = []
    summary_header = ""
    summary_move   = []
    summary_events = []
    key_header     = ""
    key_frames_raw = []   # 原始帧行，合并前
    key_events     = []

    section   = None
    in_events = False

    for raw in lines:
        line     = raw
        stripped = line.strip()

        if re.match(r"【全局态势】", stripped):
            section = "header"; in_events = False
            header_lines.append(line)
            continue
        if re.match(r"【战斗事件】", stripped):
            section = "combat"; in_events = False
            continue
        if re.match(r"【前段摘要", stripped):
            section = "summary"; in_events = False
            summary_header = line
            continue
        if re.match(r"【关键窗口", stripped):
            section = "key"; in_events = False
            key_header = line
            continue
        if stripped == "--- 事件 ---":
            in_events = True
            continue

        if section == "header":
            cleaned = clean_global_line(line)
            if not is_junk_line(cleaned) and stripped:
                header_lines.append(cleaned)

        elif section == "combat":
            if stripped and not is_junk_line(line):
                combat_lines.append(line)

        elif section == "summary":
            if not stripped or is_junk_line(line):
                continue
            if in_events:
                if not is_noise_buff_line(line):
                    summary_events.append(line)
            else:
                summary_move.append(line)

        elif section == "key":
            if not stripped or is_junk_line(line):
                continue
            if in_events:
                key_events.append(line)
            else:
                key_frames_raw.append(line)

    # ── 连续帧合并 ────────────────────────────────────────────────────────────
    key_frames_merged = merge_key_frames(key_frames_raw)

    # ── 组装输出 ──────────────────────────────────────────────────────────────
    out = []

    out.extend(header_lines)
    if header_lines:
        out.append("")

    if combat_lines:
        out.append("【战斗事件】")
        out.extend(combat_lines[:MAX_COMBAT_LINES])
        out.append("")

    if summary_header:
        out.append(summary_header)
        kept_move = uniform_sample(summary_move, MAX_SUMMARY_LINES)
        out.extend(kept_move)
        if not kept_move:
            out.append("  (前段无显著移动)")
        deduped_se = dedup_and_filter_events(summary_events)
        out.extend(deduped_se[:MAX_SUMMARY_EVENTS])
        out.append("")

    if key_header:
        out.append(key_header)
    # 合并后再按行上限采样
    kept_frames = keep_first_last_sample(key_frames_merged, MAX_KEY_LINES)
    out.extend(kept_frames)

    if key_events:
        deduped_ke = dedup_and_filter_events(key_events)
        kept_ke = deduped_ke[:MAX_EVENT_LINES]
        if kept_ke:
            out.append("  --- 事件 ---")
            out.extend(kept_ke)

    return "\n".join(out)

# ═══════════════════════════════════════════════════════════════════════════════
# 统计 & 批量处理
# ═══════════════════════════════════════════════════════════════════════════════

def process_file(src: Path, dst: Path):
    total = saved = 0
    with open(src, encoding="utf-8") as fin, \
         open(dst, "w", encoding="utf-8") as fout:
        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            sample = json.loads(raw_line)
            orig = len(sample["input"])
            sample["input"] = compress_input(sample["input"])
            new  = len(sample["input"])
            total += orig
            saved += orig - new
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return total, saved

all_total = all_saved = 0
for src in sorted(Path(INPUT_DIR).glob("*.jsonl")):
    dst = Path(OUTPUT_DIR) / src.name
    t, s = process_file(src, dst)
    all_total += t
    all_saved  += s
    ratio = s / t * 100 if t else 0
    print(f"{src.name:<35s}: {t:>8,} chars → 节省 {s:>7,} ({ratio:.1f}%)")

if all_total:
    print(f"\n合计: {all_total:,} chars → 节省 {all_saved:,} "
          f"({all_saved/all_total*100:.1f}%)")
