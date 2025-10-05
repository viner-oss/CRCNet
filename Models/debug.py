import gc, sys, collections
from functools import partial
import psutil, os

def top_object_types(n=20):
    gc.collect()
    objs = gc.get_objects()
    type_count = {}
    for o in objs:
        t = type(o).__name__
        type_count.setdefault(t, 0)
        type_count[t] += 1
    items = sorted(type_count.items(), key=lambda x: x[1], reverse=True)[:n]
    print("Top object types:", items)

def torch_tensor_summary():
    gc.collect()
    total = 0
    cnt = 0
    import torch
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) or (hasattr(o, "data") and torch.is_tensor(o.data)):
                cnt += 1
                total += o.element_size() * o.nelement()
        except Exception:
            pass
    print(f"Tensor count: {cnt}, total bytes: {total/1024**2:.1f} MB")

def print_mem_debug(step):
    p = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    print(f"[mem_dbg step={step}] pid_rss={p.memory_info().rss/1024**2:.1f}MB cpu_used={(vm.total-vm.available)/1024**2:.1f}MB cpu_avail={vm.available/1024**2:.1f}MB")
    top_object_types(10)
    torch_tensor_summary()

# ==============================================================================================================================================
# 放入训练脚本（调试专用）
import gc, torch, math, types

def safe_repr(o, maxlen=200):
    try:
        r = repr(o)
        return r[:maxlen] + ('...' if len(r)>maxlen else '')
    except Exception:
        return f"<{type(o).__name__} repr error>"

def find_tensor_holders(limit=30):
    gc.collect()
    objs = gc.get_objects()
    holders = []
    for o in objs:
        try:
            # 只检查容器类型，避免检查每个对象（快一点）
            if isinstance(o, (list, tuple, dict, set)):
                cnt = 0
                total_bytes = 0
                # iterate elements safely
                if isinstance(o, dict):
                    iterator = o.values()
                else:
                    iterator = o
                # don't iterate huge containers fully: sample first 1000
                sampled = 0
                for item in iterator:
                    sampled += 1
                    if sampled > 1000:
                        break
                    if torch.is_tensor(item) or (hasattr(item, 'data') and torch.is_tensor(item.data)):
                        try:
                            cnt += 1
                            total_bytes += item.element_size() * item.nelement()
                        except Exception:
                            cnt += 1
                    # also consider numpy arrays
                    import numpy as np
                    if isinstance(item, np.ndarray):
                        cnt += 1
                        total_bytes += item.nbytes
                if cnt > 0:
                    # get some light info about container
                    snippet = None
                    try:
                        if hasattr(o, '__len__'):
                            snippet = f'len={len(o)}'
                    except Exception:
                        snippet = None
                    holders.append((type(o).__name__, cnt, total_bytes, snippet, id(o)))
        except Exception:
            pass
    holders.sort(key=lambda x: x[2], reverse=True)
    print("=== Top tensor holders ===")
    for i, (tname, cnt, tb, snippet, oid) in enumerate(holders[:limit]):
        print(f"{i+1}. {tname} cnt_tensors={cnt} approx_bytes={tb//1024}KB {snippet} id={oid}")
    return holders

# 用法示例：每 50 步打印一次并保存到日志
# if step % 50 == 0:
#     holders = find_tensor_holders(20)
#     # 可选：把 holders 保存到文件，便于离线分析
#     with open("holders.log","a") as f:
#         f.write(f"step={step} holders={holders}\n")

# =============================================================================================================================================
# paste into your training script (debug)
import gc, inspect, types, torch, pprint

def inspect_holder_by_id(oid, max_refs=50):
    gc.collect()
    objs = gc.get_objects()
    target = None
    for o in objs:
        if id(o) == oid:
            target = o
            break
    if target is None:
        print("Holder object not found")
        return
    print("Found target:", type(target), "len(if any):", getattr(target, "__len__", lambda: None)())
    # show first few elements summary
    print("Summary of first 10 elements:")
    for i, el in enumerate(list(target)[:10]):
        try:
            if torch.is_tensor(el):
                print(f"  [{i}] Tensor shape={tuple(el.shape)} dtype={el.dtype} device={el.device} bytes={el.element_size()*el.nelement()//1024}KB")
            else:
                print(f"  [{i}] type={type(el)} repr={repr(el)[:200]}")
        except Exception as e:
            print(f"  [{i}] repr error: {e}")
    # find referrers
    refs = gc.get_referrers(target)
    print(f"Number of referrers: {len(refs)} (showing up to {max_refs})")
    for j, r in enumerate(refs[:max_refs]):
        print(f"Referrer {j}: type={type(r)}")
        # if it's a dict, show keys that point to target
        if isinstance(r, dict):
            for k,v in r.items():
                if id(v) == oid:
                    print("  -> dict key pointing to it:", k)
        # if it's a module globals, try match name
        try:
            if hasattr(r, "__dict__"):
                # search names in its attributes
                for name, val in vars(r).items():
                    if id(val) == oid:
                        print("  -> attribute", name, "in", r)
        except Exception:
            pass
    return target, refs

# =============================================================================================================================================
# diagnostic_inspect_logger.py
import torch, numpy as np, gc
from collections import Counter

def inspect_logger_history(logger, n=5):
    gc.collect()
    hist = getattr(logger, "history", None)
    if hist is None:
        print("logger has no attribute 'history'")
        return
    print("logger.history type:", type(hist), "len:", len(hist) if hasattr(hist, "__len__") else "unknown")
    # sample first n entries
    for i, entry in enumerate(list(hist)[:n]):
        print(f"--- entry {i} type {type(entry)} ---")
        if isinstance(entry, dict):
            for k, v in list(entry.items())[:20]:
                t = type(v).__name__
                info = ""
                try:
                    if torch.is_tensor(v):
                        info = f"tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device} numel={v.numel()}"
                    elif isinstance(v, np.ndarray):
                        info = f"ndarray shape={v.shape} dtype={v.dtype} nbytes={v.nbytes}"
                    else:
                        info = repr(v)[:200]
                except Exception as e:
                    info = f"repr error {e}"
                print(f"  {k!r}: ({t}) {info}")
        else:
            print("entry repr:", repr(entry)[:500])

# 用法（在 run_loop 某处）
# inspect_logger_history(self.logger, n=3)

# =============================================================================================================================================
import gc, torch, sys
def list_cuda_tensors(limit=20):
    objs = gc.get_objects()
    cnt = 0
    for o in objs:
        try:
            if torch.is_tensor(o) and o.is_cuda:
                print(type(o), o.size(), o.dtype, "id=", id(o), "refcount=", sys.getrefcount(o))
                cnt += 1
                if cnt >= limit: break
        except Exception:
            pass
    print("total cuda tensors found:", cnt)


# =============================================================================================================================================
import gc, torch, sys

def snapshot_cuda_tensors():
    """返回当前所有 CUDA tensor 的快照：id -> (shape, dtype, device)"""
    gc.collect()
    snap = {}
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                snap[id(o)] = (tuple(o.size()), o.dtype, str(o.device))
        except Exception:
            pass
    return snap

def compare_snapshots(s0, s1, s2, show_limit=20, model=None, optimizer=None):
    """比较三个快照并打印结果。
       s0: 训练前， s1: 训练后（但未 gc）， s2: gc + empty_cache 后
    """
    new_after_steps = set(s1.keys()) - set(s0.keys())
    still_after_gc = set(s2.keys()) - set(s0.keys())

    print("== Summary ==")
    print("新创建的 tensors (训练期间):", len(new_after_steps))
    print("训练后并在 gc/empty_cache 后仍然存在的 tensors:", len(still_after_gc))
    print()

    # 构造 id->obj map 以便获取更多信息（小心：可能异常）
    id2obj = {}
    for o in gc.get_objects():
        try:
            if torch.is_tensor(o) and o.is_cuda:
                id2obj[id(o)] = o
        except Exception:
            pass

    # helpers: build model/optimizer id maps if provided
    param_ids = {}
    buf_ids = {}
    opt_state_ids = {}
    if model is not None:
        for name, p in model.named_parameters():
            param_ids[id(p)] = name
        for name, b in model.named_buffers():
            buf_ids[id(b)] = name
    if optimizer is not None:
        for pid, state in optimizer.state.items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    opt_state_ids[id(v)] = (pid, k)

    def describe_tid(tid):
        obj = id2obj.get(tid, None)
        if obj is None:
            return f"{tid} -> NOT FOUND"
        s = f"id={tid} shape={tuple(obj.size())} dtype={obj.dtype} device={obj.device} refcount={sys.getrefcount(obj)}"
        if tid in param_ids:
            s += f"  <-- model.param: {param_ids[tid]}"
        if tid in buf_ids:
            s += f"  <-- model.buffer: {buf_ids[tid]}"
        if tid in opt_state_ids:
            s += f"  <-- optimizer.state: param_id={opt_state_ids[tid][0]} key={opt_state_ids[tid][1]}"
        return s

    if new_after_steps:
        print("---- 新创建的 tensors（训练期间）示例 ----")
        for tid in list(new_after_steps)[:show_limit]:
            print(describe_tid(tid))
        print()

    if still_after_gc:
        print("---- 训练后并在 gc/empty_cache 后仍然存在的 tensors（疑似泄漏）示例 ----")
        for tid in list(still_after_gc)[:show_limit]:
            print(describe_tid(tid))
        print()

    return {
        "new_after_steps": new_after_steps,
        "still_after_gc": still_after_gc,
        "id2obj": id2obj,
        "param_ids": param_ids,
        "buf_ids": buf_ids,
        "opt_state_ids": opt_state_ids
    }

def show_referrers_of_id(tid, max_out=20):
    """查看某个对象的 referrers（可能需要 objgraph 来可视化）"""
    gc.collect()
    target = None
    for o in gc.get_objects():
        if id(o) == tid:
            target = o
            break
    if target is None:
        print("object id not found in gc")
        return
    refs = gc.get_referrers(target)
    print("referrers count:", len(refs))
    for r in refs[:max_out]:
        try:
            print(" -", type(r), repr(r)[:300])
        except Exception:
            print(" -", type(r))

