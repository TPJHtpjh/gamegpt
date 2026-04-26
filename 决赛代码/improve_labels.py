#!/usr/bin/env python3
# =============================================================================
# improve_labels.py  —  用 LLM 对后5秒行为生成高质量 label
#
# 用法:
#   python improve_labels.py                        # 处理 ./improve/ 下所有 txt
#   python improve_labels.py --input_dir ./improve --output ./improved_labels.jsonl
#   python improve_labels.py --model gpt-4o-mini    # 指定模型
#   python improve_labels.py --dry_run              # 不调用 API，只打印 prompt
#
# 输出:
#   improved_labels.jsonl  每行 {"input", "label", "meta"}，label 由 LLM 生成
#
# 依赖: openai>=1.0（推荐，pip install openai）或 requests（自动降级）
# API Key: 环境变量 OPENAI_API_KEY 或 --api_key 参数
# =============================================================================


import os, sys, json, argparse, textwrap, time
from pathlib import Path
from collections import defaultdict


# 把同目录下的 processor/config 引入
sys.path.insert(0, str(Path(__file__).parent))
from processor import (
    parse_file, infer_label_from_filename,
    generate_input_text,
    get_nearest_enemy, yaw_to_dir, classify_speed,
    dist_2d, yaw_delta, crossed_threshold,
    is_noise_buff, player_tag,
)
from config import (
    DISP_THRESHOLD_LATE, YAW_THRESHOLD_LATE,
    SPEED_JUMP_THRESHOLD, ENEMY_DIST_THRESHOLDS,
    KEY_WINDOW_INTERVAL, LABEL_META,
)


# ─────────────────────────────────────────────────────────────────────────────
# 默认参数
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = "./improve/SkillStart"
DEFAULT_OUTPUT    = "./output_jsonl/improved_SkillStart.jsonl"
DEFAULT_MODEL     = "gpt-4o"  # 可替换为其他兼容模型
RETRY_TIMES       = 1
RETRY_DELAY       = 10   # 秒

class AllAPIsFailedError(Exception):
    pass

API_CONFIGS = [
    
    # 可以在这里添加更多 API key、base_url 和 model 组合，例如：
    # {
    #     "api_key": "sk-xxxxxx",
    #     "base_url": "https://api.openai.com/v1",
    #     "model": "gpt-4o"
    # }
]

# ─────────────────────────────────────────────────────────────────────────────
# 后5秒关键窗口文本生成
# 与 generate_input_text 里关键窗口逻辑完全一致，只是时间窗口不同
# ─────────────────────────────────────────────────────────────────────────────


def generate_future_window_text(data: dict, main_pid: str,
                                 cut_ts: float, end_ts: float) -> str:
    """把 cut_ts ~ end_ts 这5秒的帧+事件，用与关键窗口相同格式输出"""
    frames       = data['frames']
    events       = data['events']
    players_info = data['players_info']
    sorted_ts    = data['sorted_ts']
    main_team    = players_info.get(main_pid, {}).get('team', '')


    window_ts = [t for t in sorted_ts if cut_ts <= t < end_ts]
    lines = [f"【后5秒行为窗口 ({cut_ts:.1f}s ~ {end_ts:.2f}s)】"]


    prev_feat = None
    for t in window_ts:
        if main_pid not in frames.get(t, {}):
            continue
        cur = frames[t][main_pid]
        eid2, edist2 = get_nearest_enemy(t, frames, players_info, main_pid, main_team)
        ename2 = players_info.get(eid2, {}).get('name', '?') if eid2 else '暂无'


        should_output = (prev_feat is None)
        notes = []
        if prev_feat:
            disp = dist_2d(cur['x'], cur['z'], prev_feat['x'], prev_feat['z'])
            if disp > DISP_THRESHOLD_LATE:
                should_output = True
                import math
                move_angle = math.degrees(
                    math.atan2(cur['x'] - prev_feat['x'],
                               -(cur['z'] - prev_feat['z']))) % 360
                notes.append(f"→{yaw_to_dir(move_angle)}位移{disp:.1f}m")
            yd = yaw_delta(cur['yaw'], prev_feat['yaw'])
            if yd > YAW_THRESHOLD_LATE:
                should_output = True
                notes.append(f"转向{yaw_to_dir(cur['yaw'])}({cur['yaw']:.0f}°)")
            if cur['scope'] != prev_feat['scope']:
                should_output = True
                notes.append(f"★{cur['scope']}")
            sdelta = abs(cur['speed'] - prev_feat['speed'])
            if sdelta > SPEED_JUMP_THRESHOLD:
                should_output = True
                notes.append(f"速度{prev_feat['speed']:.1f}→{cur['speed']:.1f}m/s")
            if eid2 and crossed_threshold(prev_feat.get('edist', 9999), edist2,
                                          ENEMY_DIST_THRESHOLDS):
                should_output = True
                notes.append(f"敌距跨档→{edist2:.0f}m")
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


    # 后5秒离散事件（主玩家动作 + 决策动作 + 重要技能）
    fut_evts = []
    decision_actions = data.get('decision_actions', [])
    for ev in events:
        if not (cut_ts <= ev['ts'] < end_ts):
            continue
        t = ev['ts']
        if ev['type'] == '动作':
            tag = "主玩家" if ev.get('pid') == main_pid else f"玩家{ev['pid']}"
            fut_evts.append(f"  {t:.2f}s: {tag} [{ev['action']}]")
        elif ev['type'] == '技能生效':
            buf = ev.get('buff', '')
            if not is_noise_buff(buf):
                cid = ev.get('caster', '')
                tag = "主玩家" if cid == main_pid else f"玩家{cid}"
                fut_evts.append(f"  {t:.2f}s: {tag} 激活 [{buf}]")
        elif ev['type'] == '伤害':
            atag = player_tag(ev['attacker'], players_info, main_pid)
            vtag = player_tag(ev['victim'],   players_info, main_pid)
            down = " 【击倒】" if ev.get('is_down', '0') != '0' else ''
            fut_evts.append(f"  {t:.2f}s: {atag} → {vtag} 造成{ev['hp_dmg']:.0f}伤害{down}")
        elif ev['type'] in ('击倒', '死亡'):
            pass  # 可按需补充


    for da in decision_actions:
        if cut_ts <= da['ts'] < end_ts and da.get('pid') == main_pid:
            fut_evts.append(f"  {da['ts']:.2f}s: 主玩家 [决策动作:{da['action']}]")


    if fut_evts:
        lines += ["  --- 后5秒事件 ---"] + sorted(fut_evts)


    return '\n'.join(lines)



# ─────────────────────────────────────────────────────────────────────────────
# Prompt 组装
# ─────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = textwrap.dedent("""\
    你是三角洲行动游戏的战术行为分析专家。
    我会给你提供：
    ① 主玩家5秒内的帧数据与事件（原始提取，尚未转化为自然语言）
    ② 决策类型标签

    你的任务是：根据这5秒数据，用自然语言描述主玩家在这5秒内的具体行为过程。

    输出格式要求（严格遵守）：
    - 只输出一段话，不超过150字
    - 必须包含三个逻辑段，以"……随后……最后……"串联
    - 第一段("主玩家……")：描述初始状态或进入动作前的姿态
    - 第二段("随后……")：描述过渡动作或战术调整
    - 第三段("最后……")：描述最终完成的行为及决策类型
    - 禁止出现任何具体时间（如"于20.00s"）
    - 禁止编造数据中没有的行为
    - 用简练的战术语言，不要废话
""")


def build_prompt(future_text: str, label_type: str, main_name: str) -> list:
    meta   = LABEL_META.get(label_type, {})
    camp   = meta.get('camp',   '未知')
    action = meta.get('action', '未知')
    user_msg = (
        f"【决策标签】{label_type}（{camp} - {action}）\n\n"
        f"{future_text}\n\n"
        f"请根据以上5秒的数据，用【主玩家{main_name}...随后...最后...】格式输出行为描述。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]



# ─────────────────────────────────────────────────────────────────────────────
# LLM 调用（支持多个组合回退）
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(messages: list, primary_config_idx: int = 0) -> tuple:
    """
    逐个尝试 API_CONFIGS 中的组合，优先使用 primary_config_idx 指定的组合。
    每个组合连续调用失败 RETRY_TIMES (5次) 后自动切换下一个。
    全部尝试均失败时抛出 AllAPIsFailedError 异常。
    返回 (模型返回内容, 实际使用的modelName)
    """
    config_indices = [primary_config_idx] + [i for i in range(len(API_CONFIGS)) if i != primary_config_idx]
    for config_idx in config_indices:
        api_conf = API_CONFIGS[config_idx]
        api_key = api_conf["api_key"]
        base_url = api_conf["base_url"]
        model = api_conf["model"]
        
        # 自动补全 /v1 后缀
        url = base_url.rstrip('/')
        if not url.endswith('/v1'):
            url += '/v1'
        chat_url = url + '/chat/completions'
        
        for attempt in range(RETRY_TIMES):
            try:
                # ── 优先尝试导入 openai SDK ────────────────────────
                try:
                    from openai import OpenAI as _OpenAI
                    client = _OpenAI(api_key=api_key, base_url=base_url)
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=2000,
                    )
                    return resp.choices[0].message.content.strip(), model
                except ImportError:
                    # ── 纯 requests 降级 ───────────────────────────
                    import requests as _req
                    headers = {
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type':  'application/json',
                    }
                    payload = {
                        'model':       model,
                        'messages':    messages,
                        'temperature': 0.3,
                        'max_tokens':  2000,
                    }
                    resp = _req.post(chat_url, headers=headers, json=payload, timeout=30)
                    resp.raise_for_status()
                    return resp.json()['choices'][0]['message']['content'].strip(), model
            except Exception as e:
                print(f"    [API错误] 配置{config_idx+1}/{len(API_CONFIGS)}(模型 {model}) 尝试 {attempt+1}/{RETRY_TIMES}: {e}")
                if attempt < RETRY_TIMES - 1:
                    time.sleep(RETRY_DELAY)
                    
    # 如果运行到这里，说明所有的配置均尝试且失败了
    raise AllAPIsFailedError("所有API配置均已尝试并连续失败")


# ─────────────────────────────────────────────────────────────────────────────
# 时间边界解析（与 main.py 中 resolve_time_boundary 保持一致逻辑）
# ─────────────────────────────────────────────────────────────────────────────


def resolve_boundary(data: dict, lb: dict):
    sorted_ts = data.get('sorted_ts', [])
    max_ts    = sorted_ts[-1] if sorted_ts else lb.get('ts', 20.0)
    das = sorted([x for x in data.get('decision_actions', []) if x.get('pid') == lb.get('pid')],
                 key=lambda x: x['ts'])
    if max_ts >= 24.5:
        if das:
            return das[0]['ts'], max_ts
        raw_ts = lb.get('ts', max_ts)
        if raw_ts <= max_ts - 4.5:
            return raw_ts, max_ts
        return max_ts - 5.0, max_ts
    raw_ts = lb.get('ts', max_ts)
    return raw_ts, raw_ts



# ─────────────────────────────────────────────────────────────────────────────
# 主处理逻辑
# ─────────────────────────────────────────────────────────────────────────────


def collect_txt(input_dir: str):
    result = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in sorted(files):
            if fname.lower().endswith('.txt'):
                result.append(os.path.join(root, fname))
    return result



def process_file(filepath: str, config_queue=None,
                 dry_run: bool = False, debug: bool = False) -> list:
    filename = os.path.basename(filepath)
    data     = parse_file(filepath)
    labels   = data['labels']

    inferred = infer_label_from_filename(filename)
    if not labels:
        if not inferred:
            print(f"  ⚠ {filename}: 无决策行且无法推断类型，跳过")
            return []
        sorted_ts = data.get('sorted_ts', [])
        label_ts  = sorted_ts[-1] if sorted_ts else 20.0
        main_pid  = data.get('most_freq_pid', '0')
        labels    = [{'ts': label_ts, 'label_type': inferred, 'pid': main_pid}]
    else:
        seen, unique = set(), []
        for lb in labels:
            if lb['label_type'] is None:
                lb['label_type'] = inferred
            key = (lb['ts'], lb['label_type'], lb['pid'])
            if key not in seen:
                seen.add(key)
                unique.append(lb)
        labels = unique

    samples = []
    for lb in labels:
        if not lb.get('label_type'):
            continue
        main_pid = lb['pid']
        if main_pid not in data['players_info']:
            continue

        cut_ts, label_end_ts = resolve_boundary(data, lb)

        # 后5秒必须有帧数据
        future_ts = [t for t in data['sorted_ts'] if cut_ts <= t < label_end_ts]
        if len(future_ts) < 3:
            print(f"  ⚠ {filename}: 后5秒帧数不足({len(future_ts)})，跳过")
            continue

        ctx_start  = max(0.0, cut_ts - 20.0)
        ctx_ts     = [t for t in data['sorted_ts'] if ctx_start <= t < cut_ts]
        if len(ctx_ts) < 5 and debug:
            print(f'  ⚠ {filename}: 上文帧数较少({len(ctx_ts)})，仅影响 input 字段')

        try:
            input_text  = generate_input_text(data, main_pid, cut_ts)
            future_text = generate_future_window_text(data, main_pid, cut_ts, label_end_ts)
        except Exception as e:
            print(f"  ✗ {filename}: 文本生成失败 [{e}]")
            continue

        main_name = data['players_info'].get(main_pid, {}).get('name', '')
        messages = build_prompt(future_text, lb['label_type'], main_name)

        if dry_run:
            print(f"\n{'='*60}")
            print(f"[DRY RUN] {filename}  label_type={lb['label_type']}")
            print(f"[SYSTEM]\n{messages[0]['content'][:300]}...")
            print(f"[USER]\n{messages[1]['content'][:600]}...")
            label_text = "<DRY_RUN>"
            used_model = "dry_run"
        else:
            if config_queue:
                primary_config_idx = config_queue.get()
            else:
                primary_config_idx = 0
                
            try:
                if debug:
                    print(f"    准备调用 LLM (优先组合 {primary_config_idx+1})...")
                label_text, used_model = call_llm(messages, primary_config_idx)
            finally:
                if config_queue:
                    config_queue.put(primary_config_idx)
                
            if not label_text:
                print(f"  ✗ {filename}: API 返回为空，跳过")
                continue
            if debug:
                print(f"    → {label_text}")

        samples.append({
            "input": input_text,
            "label": label_text,
            "meta": {
                "source_file":  filename,
                "label_type":   lb['label_type'],
                "label_camp":   LABEL_META.get(lb['label_type'], {}).get('camp', '?'),
                "main_pid":     main_pid,
                "label_ts":     cut_ts,
                "label_end_ts": label_end_ts,
                "llm_model":    used_model,
                "input_chars":  len(input_text),
            }
        })
    return samples



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='GameGPT LLM 高质量 Label 生成脚本')
    parser.add_argument('--input_dir', default=DEFAULT_INPUT_DIR,
                        help='原始 txt 目录（默认 ./improve）')
    parser.add_argument('--output',    default=DEFAULT_OUTPUT,
                        help='输出 jsonl 路径')
    parser.add_argument('--model',     default=DEFAULT_MODEL,
                        help=f'OpenAI 模型名（默认 {DEFAULT_MODEL}）')
    parser.add_argument('--api_key',   default=None,
                        help='API Key（也可通过 OPENAI_API_KEY 环境变量设置）')
    parser.add_argument('--base_url',  default=None,
                        help='API base_url，用于第三方兼容接口')
    parser.add_argument('--dry_run',   action='store_true',
                        help='不调用 API，只打印 prompt（用于调试）')
    parser.add_argument('--debug',     action='store_true',
                        help='打印每条 LLM 返回内容')
    parser.add_argument('--workers',   type=int, default=5,
                        help='并发处理数量（默认 3）')
    parser.add_argument('--file',      default=None,
                        help='只处理单个文件（相对于 input_dir 或绝对路径）')
    args = parser.parse_args()

    # 参数校验与初始化默认配置
    # 注意：原命令行传入的 api_key、base_url、model 目前被当做第一个组合覆盖写入到 API_CONFIGS
    if args.api_key:
        API_CONFIGS[0]["api_key"] = args.api_key
    if args.base_url:
        API_CONFIGS[0]["base_url"] = args.base_url
    if args.model:
        API_CONFIGS[0]["model"] = args.model
        
    if not args.dry_run and not API_CONFIGS[0]["api_key"]:
        print("[错误] 未设置 API Key，请传 --api_key 或在代码中配置 API_CONFIGS")
        sys.exit(1)

    # 收集文件
    if args.file:
        txt_files = [args.file if os.path.isabs(args.file)
                     else os.path.join(args.input_dir, args.file)]
    else:
        txt_files = collect_txt(args.input_dir)

    if not txt_files:
        print(f"[错误] 在 {args.input_dir} 下未找到任何 .txt 文件")
        return

    import threading
    processed_files = set()
    if os.path.exists(args.output):
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    if 'meta' in data and 'source_file' in data['meta']:
                        processed_files.add(data['meta']['source_file'])
            print(f"[防护机制] 已找到 {len(processed_files)} 个并已处理文件，自动跳过。")
        except Exception as e:
            print(f"[防护机制] 警告: 读取已有输出文件失败: {e}")

    pending_files = []
    for f in txt_files:
        if os.path.basename(f) not in processed_files:
            pending_files.append(f)

    if not pending_files:
        print("[完成] 所有文件均已处理完毕。")
        return

    print(f"[LLM Label 生成] 共 {len(txt_files)} 个文件，待处理 {len(pending_files)} 个 | 模型: {args.model} | 并发数: {args.workers}")

    all_samples = []
    stats = defaultdict(int)

    total = len(pending_files)
    import math as _math
    import concurrent.futures
    import queue
    log_every = max(1, 10 ** max(0, _math.floor(_math.log10(total)) - 1)) if total > 100 else 1

    config_queue = queue.Queue()
    for i in range(args.workers):
        if i < 3:
            config_queue.put(0)  # 前3个工作线程优先分配第一个组合
        elif i < 5 and len(API_CONFIGS) > 1:
            config_queue.put(1)  # 接着的2个工作线程优先分配第二个组合
        else:
            config_queue.put(0)  # 其余的默认分配第一个组合（如果 workers > 5）

    write_lock = threading.Lock()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, 'a', encoding='utf-8') as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_fpath = {
                executor.submit(process_file, fpath, config_queue, args.dry_run, args.debug): fpath
                for fpath in pending_files
            }

            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_fpath):
                completed_count += 1
                fpath = future_to_fpath[future]
                try:
                    samples = future.result()
                    for s in samples:
                        stats[s['meta']['label_type']] += 1
                    
                    if samples:
                        with write_lock:
                            for s in samples:
                                out_f.write(json.dumps(s, ensure_ascii=False) + '\n')
                            out_f.flush()
                        all_samples.extend(samples)
                except AllAPIsFailedError as exc:
                    print(f"\n[致命错误] {os.path.basename(fpath)}: {exc}。停止分配新任务并保存已完成数据。")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                except Exception as exc:
                    print(f"  ✗ {os.path.basename(fpath)}: 处理引发异常: {exc}")

                if completed_count % log_every == 0 or completed_count == total:
                    print(f'  [{completed_count:6d}/{total}] 新增样本: {len(all_samples)} | 新增分布: {dict(stats)}')

    print(f"\n[完成] 本次新增样本: {len(all_samples)}")
    print(f"  新增决策分布: {dict(stats)}")
    print(f"  输出（即时追加）: {args.output}")


if __name__ == '__main__':
    main()
