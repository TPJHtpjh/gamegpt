#!/usr/bin/env python3
# =============================================================================
# main.py  —  批量处理入口
#
# 用法:
#   python main.py                                         # 处理 ./data/ 下所有 txt（递归）
#   python main.py --input_dir /path --output_dir ./out   # 自定义路径
#   python main.py --file SkillStart115.txt               # 单文件调试（自动打印样本）
#   python main.py --file SkillStart115.txt --no_debug    # 单文件静默输出
#
# 输出:
#   output_jsonl/train_sft.jsonl    每行一个 {"input","label","meta"} 样本
#   output_jsonl/stats.json         数据统计报告
# =============================================================================

import os, sys, json, argparse, traceback
from collections import defaultdict
from processor import parse_file, infer_label_from_filename, generate_input_text, generate_label_text
from config    import INPUT_DIR, OUTPUT_DIR, OUTPUT_FILE, LABEL_META


# ─────────────────────────────────────────────────────────────────────────────
# 单文件处理
# ─────────────────────────────────────────────────────────────────────────────

def resolve_time_boundary(data: dict, lb: dict, filename: str):
    sorted_ts = data.get('sorted_ts', [])
    max_ts = sorted_ts[-1] if sorted_ts else lb.get('ts', 20.0)
    decision_actions = [x for x in data.get('decision_actions', []) if x.get('pid') == lb.get('pid')]
    decision_actions = sorted(decision_actions, key=lambda x: x['ts'])

    # 训练集常见情况：总长25s，前20s是input，后5s是label窗口
    if max_ts >= 24.5:
        # Action 类优先用第一条“（决策）动作”作为切分点
        if decision_actions:
            cut_ts = decision_actions[0]['ts']
            return cut_ts, max_ts
        # Fire/SkillStart/Grenade 等，如果决策行已经落在20s附近，就直接用它
        raw_ts = lb.get('ts', max_ts)
        if raw_ts <= max_ts - 4.5:
            return raw_ts, max_ts
        # 否则兜底按 max_ts-5 切分
        return max_ts - 5.0, max_ts

    # 测试集通常只有20s，没有future 5s
    raw_ts = lb.get('ts', max_ts)
    return raw_ts, raw_ts


def process_one(filepath: str, debug: bool = False) -> list:
    filename  = os.path.basename(filepath)
    data      = parse_file(filepath)
    labels    = data['labels']

    # ── 决策行补全：从文件名推断缺失的 label_type ───────────────────────────
    inferred_type = infer_label_from_filename(filename)

    if not labels:
        # 测试集：无任何决策行 → 用文件名构造占位 label
        if inferred_type is None:
            print(f"  ⚠  {filename}: 无决策行且无法从文件名推断类型，跳过")
            return []
        # 找最后一个帧时间戳作为 label_ts，找主玩家 pid（取第一个玩家）
        label_ts = data['sorted_ts'][-1] if data['sorted_ts'] else 20.0
        # 无法从文件中得知主玩家，用出现频率最高的玩家
        main_pid = data.get('most_freq_pid', '0')
        labels   = [{'ts': label_ts, 'label_type': inferred_type, 'pid': main_pid}]
        if debug:
            print(f"  ℹ  {filename}: 测试集，从文件名推断 label={inferred_type}, pid={main_pid}")
    else:
        # 去重，同时补全 label_type=None 的行（如 20.00|（决策）|玩家xxx）
        seen, unique = set(), []
        for lb in labels:
            if lb['label_type'] is None:
                lb['label_type'] = inferred_type  # 可能仍为 None
            key = (lb['ts'], lb['label_type'], lb['pid'])
            if key not in seen:
                seen.add(key)
                unique.append(lb)
        labels = unique

    samples = []
    for lb in labels:
        if lb['label_type'] is None:
            print(f"  ⚠  {filename}: 决策类型无法确定，跳过此条")
            continue

        main_pid   = lb['pid']
        raw_label_ts = lb['ts']
        label_type = lb['label_type']
        cut_ts, label_end_ts = resolve_time_boundary(data, lb, filename)

        if main_pid not in data['players_info']:
            print(f"  ⚠  {filename}: 主玩家 {main_pid} 不在游戏开始名单")
            continue

        ctx_start = max(0.0, cut_ts - 20.0)
        ctx_ts = [t for t in data['sorted_ts'] if ctx_start <= t < cut_ts]
        if len(ctx_ts) < 5:
            print(f"  ⚠  {filename}: 上文帧数不足({len(ctx_ts)})，跳过")
            continue

        try:
            input_text = generate_input_text(data, main_pid, cut_ts)
            label_text = generate_label_text(lb, data, main_pid, cut_ts=cut_ts, label_end_ts=label_end_ts)
        except Exception as e:
            print(f"  ✗  {filename}: 生成失败 [{e}]")
            if debug:
                traceback.print_exc()
            continue

        # 输出格式：不含 system 提示词，训练时统一添加
        sample = {
            "input":  input_text,
            "label":  label_text,
            "meta": {
                "source_file": filename,
                "label_type":  label_type,
                "label_camp":  LABEL_META.get(label_type, {}).get('camp', '?'),
                "main_pid":    main_pid,
                "label_ts":    cut_ts,
                "label_end_ts": label_end_ts,
                "raw_label_ts": raw_label_ts,
                "input_chars": len(input_text),
                "label_chars": len(label_text),
            }
        }
        samples.append(sample)

        if debug:
            print_sample(sample)

    return samples


def print_sample(sample: dict, width: int = 72, truncate: int = 1000):
    sep = "=" * width
    print(sep)
    m = sample.get('meta', {})
    print(f"[FILE] {m.get('source_file')}  "
          f"[LABEL] {m.get('label_type')} / {m.get('label_camp')}  "
          f"[PID] {m.get('main_pid')}  "
          f"[INPUT] {m.get('input_chars')}字  [LABEL] {m.get('label_chars')}字")
    print()
    inp = sample['input']
    lbl = sample['label']
    print("── INPUT ──")
    print(inp[:truncate] + (f"\n  ...(共{len(inp)}字)" if len(inp) > truncate else ""))
    print()
    print("── LABEL ──")
    print(lbl)
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 批量处理（递归扫描子目录）
# ─────────────────────────────────────────────────────────────────────────────

def collect_txt_files(input_dir: str) -> list:
    """递归收集 input_dir 下所有 .txt 文件"""
    result = []
    for root, dirs, files in os.walk(input_dir):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in sorted(files):
            if fname.lower().endswith('.txt'):
                result.append(os.path.join(root, fname))
    return result


def process_batch(input_dir: str, output_dir: str,
                  output_file: str, debug: bool = False):
    txt_files = collect_txt_files(input_dir)
    if not txt_files:
        print(f"[错误] 在 {input_dir} 下未找到任何 .txt 文件（含子目录）")
        return

    print(f"[批量处理] 共发现 {len(txt_files)} 个 txt 文件")
    all_samples   = []
    stats         = defaultdict(int)
    failed_files  = []
    skipped_files = []

    total = len(txt_files)
    # 动态日志频率：文件数<=100每条都打, <=1000每10条, <=10000每100条, 以此类推
    import math
    log_every = max(1, 10 ** max(0, math.floor(math.log10(total)) - 1)) if total > 100 else 1

    for idx, fpath in enumerate(txt_files, 1):
        rel = os.path.relpath(fpath, input_dir)
        try:
            samples = process_one(fpath, debug=debug)
            if not samples:
                skipped_files.append(rel)
            else:
                for s in samples:
                    stats[s['meta']['label_type']] += 1
                all_samples.extend(samples)
        except Exception as e:
            failed_files.append(rel)
            if debug:
                print(f"  ✗ {rel}: {e}")
                traceback.print_exc()

        # 按频率打印进度
        if idx % log_every == 0 or idx == total:
            done    = idx
            skipped = len(skipped_files)
            failed  = len(failed_files)
            ok      = done - skipped - failed
            print(f"  [{done:6d}/{total}] 已处理: {ok} 成功 / {skipped} 跳过 / {failed} 失败 | 累计样本: {len(all_samples)}")

    # 写出 jsonl
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_file)
    with open(out_path, 'w', encoding='utf-8') as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    # 写出统计
    stats_data = {
        'total_samples':  len(all_samples),
        'by_label_type':  dict(stats),
        'total_files':    len(txt_files),
        'skipped_files':  len(skipped_files),
        'failed_files':   len(failed_files),
        'failed_list':    failed_files,
    }
    stats_path = os.path.join(output_dir, 'stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)

    print(f"\n[完成]")
    print(f"  总样本: {len(all_samples)}")
    print(f"  决策分布: {dict(stats)}")
    print(f"  跳过: {len(skipped_files)} | 失败: {len(failed_files)}")
    print(f"  训练集: {out_path}")
    print(f"  统计:   {stats_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='GameGPT SFT 数据处理脚本')
    parser.add_argument('--input_dir',   default=INPUT_DIR)
    parser.add_argument('--output_dir',  default=OUTPUT_DIR)
    parser.add_argument('--output_file', default=OUTPUT_FILE)
    parser.add_argument('--file',        default=None,
                        help='只处理单个文件（路径相对于 input_dir 或绝对路径）')
    parser.add_argument('--no_debug',    action='store_true',
                        help='单文件模式不打印详细内容')
    args = parser.parse_args()

    if args.file:
        fpath = args.file if os.path.isabs(args.file) \
                else os.path.join(args.input_dir, args.file)
        print(f"[单文件] {fpath}")
        samples = process_one(fpath, debug=not args.no_debug)
        if samples:
            os.makedirs(args.output_dir, exist_ok=True)
            out = os.path.join(args.output_dir, 'debug_sample.jsonl')
            with open(out, 'w', encoding='utf-8') as f:
                for s in samples:
                    f.write(json.dumps(s, ensure_ascii=False) + '\n')
            print(f"\n[已保存] {len(samples)} 条 → {out}")
    else:
        process_batch(args.input_dir, args.output_dir,
                      args.output_file, debug=args.debug if hasattr(args,'debug') else False)


if __name__ == '__main__':
    main()
