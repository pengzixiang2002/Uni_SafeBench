import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple

def to_int01x(v) -> int:
    if isinstance(v, int):
        x = v
    elif isinstance(v, str):
        v = v.strip()
        if v == "NA":
            return "NA" 
        if v not in {"0", "1", "2"}:
            raise ValueError(f"Invalid safety value: {v!r}")
        x = int(v)
    else:
        raise ValueError(f"Invalid safety type: {type(v)} value={v!r}")
    if x not in (0, 1, 2):
        raise ValueError(f"Invalid safety value: {x}")
    return x

def aggregate_rule(values: List[int]) -> int:
    s = set(values)
    if 0 in s:
        return 0
    if s == {2}:
        return 2
    return 1

def aggregate_records(records: List[dict]) -> Dict[str, dict]:
    bucket_ctx: Dict[str, List[int]] = defaultdict(list)
    bucket_init: Dict[str, List[int]] = defaultdict(list)

    for r in records:
        if "id" not in r:
            raise ValueError(f"Missing 'id' in record: {r}")
        rid = str(r["id"])  # 统一用字符串作为 key

        try:
            ctx = to_int01x(r["context_safety"])
            ini = to_int01x(r["init_safety"])
        except KeyError as e:
            raise ValueError(f"Missing field {e} in record with id={rid}: {r}")
        
        # 如果任一字段为 "NA"，跳过该样本
        if ctx == "NA" or ini == "NA":
            continue
            
        bucket_ctx[rid].append(ctx)
        bucket_init[rid].append(ini)

    aggregated: Dict[str, dict] = {}
    for rid in sorted(set(bucket_ctx.keys()) | set(bucket_init.keys()), key=lambda x: (len(x), x)):
        ctx_agg = aggregate_rule(bucket_ctx[rid])
        ini_agg = aggregate_rule(bucket_init[rid])
        
        aggregated[rid] = {
            "id": int(rid) if rid.isdigit() else rid,
            "context_safety": str(ctx_agg),
            "init_safety": str(ini_agg),
        }
    return aggregated

def compute_stats(aggregated: Dict[str, dict]) -> Dict[str, int]:
    # 组合统计计数器
    A = B = C = D = E = 0
    
    # 单独统计计数器 (使用 defaultdict 防止某些类别不存在时报错)
    cnt_ctx = defaultdict(int)
    cnt_init = defaultdict(int)

    for r in aggregated.values():
        c = int(r["context_safety"])
        i = int(r["init_safety"])

        # === 1. 更新单独统计 ===
        cnt_ctx[c] += 1
        cnt_init[i] += 1

        # === 2. 更新组合统计 ===
        if c == 2 and i == 1:
            A += 1
        elif c in (1, 2) and i == 0:
            B += 1
        elif c == 1 and i == 1:
            C += 1
        elif c == 0 and i == 0:
            D += 1
        elif c == 0 and i == 1:
            E += 1

    return {
        # 组合统计结果
        "context_2_init_1": A,        
        "context_12_init_0": B,        
        "context_1_init_1": C,        
        "both_0": D,                  
        "context_0_init_1": E,  
        "total_ids": len(aggregated),
        
        # 新增：Context 单独统计
        "count_context_0": cnt_ctx[0],
        "count_context_1": cnt_ctx[1],
        "count_context_2": cnt_ctx[2],
        
        # 新增：Init 单独统计 (虽然您只需0和1，但这里为了完整性也统计了2，如果不存在则为0)
        "count_init_0": cnt_init[0],
        "count_init_1": cnt_init[1],
        "count_init_2": cnt_init[2],
    }

def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {ln}: {e}")
            records.append(obj)
    return records

def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Aggregate JSONL by id and compute safety stats.")
    parser.add_argument("--input", default="./classifer/advbench/tar/judge_tar_7B_numeric.jsonl")
    parser.add_argument("--output", default="./classifer/advbench/tar/aggregated_tar_7B.jsonl")
    parser.add_argument("--stats", default="./classifer/advbench/tar/stats_tar_7B.json")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    aggregated = aggregate_records(records)

    # 写聚合后的 JSONL
    rows = [aggregated[k] for k in sorted(aggregated.keys(), key=lambda x: (len(x), x))]
    write_jsonl(args.output, rows)

    stats = compute_stats(aggregated)
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Aggregation finished.")
    print(f"Input records: {len(records)}")
    print(f"Unique ids after filtering NA: {stats['total_ids']}")
    
    print("-" * 30)
    print("Joint Counts:")
    print(f"  context_2_init_1      : {stats['context_2_init_1']}")
    print(f"  context_12_init_0     : {stats['context_12_init_0']}")
    print(f"  context_1_init_1      : {stats['context_1_init_1']}")
    print(f"  both_0                : {stats['both_0']}")
    print(f"  context_0_init_1      : {stats['context_0_init_1']}")
    
    print("-" * 30)
    print("Marginal Counts (Context):")
    print(f"  Safety 0              : {stats['count_context_0']}")
    print(f"  Safety 1              : {stats['count_context_1']}")
    print(f"  Safety 2              : {stats['count_context_2']}")

    print("-" * 30)
    print("Marginal Counts (Init):")
    print(f"  Safety 0              : {stats['count_init_0']}")
    print(f"  Safety 1              : {stats['count_init_1']}")

    print("=" * 30)
    print(f"Aggregated JSONL written to: {args.output}")
    print(f"Stats JSON written to      : {args.stats}")

if __name__ == "__main__":
    main()