import json
import os
import hashlib
import time
import torch
import torch.nn.functional as F

"""
最简批量帧文本编码（仅正向），增强兼容性：
- 解决：不同 prompt 生成的 cond 长度不一致导致 torch.cat 报错 -> 统一 padding
- 解决：某些 clip.encode_from_tokens 在 return_pooled=True 时返回 (cond, None) 或根本不支持 pooled 的情况
- 若 pooled 全部为 None，则不放 pooled_output 或放占位空字符串
"""

def parse_frame_prompts(raw_text: str):
    """
    解析输入为 {frame_index: prompt} 的升序字典。
    允许缺少最外层 {}；允许末尾多余逗号。
    """
    txt = raw_text.strip()
    if not txt:
        return {}
    if not txt.startswith("{"):
        txt = "{" + txt
    if not txt.endswith("}"):
        txt = txt + "}"
    txt = txt.replace(",}", "}")
    data = json.loads(txt)
    out = {}
    for k, v in data.items():
        try:
            idx = int(k)
        except:
            continue
            # 非数字 key 忽略
        out[idx] = str(v)
    return dict(sorted(out.items(), key=lambda x: x[0]))


def encode_prompts_sequential(clip, ordered_items):
    """
    逐条编码；对 cond 做长度统一；pooled 可能为 None。
    ordered_items: List[(frame_index, prompt)]
    返回: cond_final, pooled_final(或 None), has_pooled(bool)
    """
    cond_list = []
    pooled_list = []
    pooled_available = True  # 先假设有 pooled
    max_tokens = 0

    with torch.no_grad():
        for frame_idx, prompt in ordered_items:
            tokens = clip.tokenize(prompt)
            # 兼容：有的实现如果 return_pooled=True 仍返回单值
            enc = clip.encode_from_tokens(tokens, return_pooled=True)
            if isinstance(enc, (list, tuple)) and len(enc) == 2:
                cond, pooled = enc
            else:
                cond, pooled = enc, None

            if cond.dim() != 3:
                # 期望 shape: [B(=1), T, C]; 若是 [T, C] 补一个 batch 维
                if cond.dim() == 2:
                    cond = cond.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected cond tensor shape: {cond.shape}")

            if cond.shape[1] > max_tokens:
                max_tokens = cond.shape[1]

            cond_list.append(cond)

            if pooled is None:
                pooled_available = False
                pooled_list.append(None)
            else:
                # 期望 [B(=1), D] 或 [D]; 若 [D] -> [1,D]
                if pooled.dim() == 1:
                    pooled = pooled.unsqueeze(0)
                pooled_list.append(pooled)

    if not cond_list:
        dummy = torch.zeros(1, 1)
        return dummy, None, False

    # 对 cond 做 padding
    padded = []
    for c in cond_list:
        if c.shape[1] < max_tokens:
            pad_len = max_tokens - c.shape[1]
            # shape [1, T, C]，pad 第二维 (tokens)：(last_dim_left, last_dim_right, second_dim_left, second_dim_right)
            c = F.pad(c, (0, 0, 0, pad_len))
        padded.append(c)
    cond_final = torch.cat(padded, dim=0)  # [N, max_tokens, C]

    pooled_final = None
    if pooled_available:
        # 校验 pooled 维度并 cat
        processed = []
        for p in pooled_list:
            if p is None:
                pooled_available = False
                break
            if p.dim() == 1:
                p = p.unsqueeze(0)
            processed.append(p)
        if pooled_available and processed:
            try:
                pooled_final = torch.cat(processed, dim=0)  # [N, D]
            except Exception:
                # 如果维度不一致，放弃 pooled
                pooled_final = None
                pooled_available = False

    return cond_final, pooled_final, pooled_available


## 删除了 BatchPrompt 节点，保留 PromptQueue


def _read_lines_from_source(multiline_text: str, file_path: str = "", use_file: bool = False):
    """
    从多行文本或文件读取，每行一个提示词；自动去除空行和首尾空白。
    """
    lines = []
    if use_file and file_path:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            # 回退到多行文本
            lines = multiline_text.splitlines()
    else:
        lines = multiline_text.splitlines()

    cleaned = []
    for raw in lines:
        s = str(raw).strip()
        if not s:
            continue
        cleaned.append(s)
    return cleaned


def _apply_template_to_lines(lines, template: str):
    """
    将每行替换进模板。若模板为空或不含 {p}，则直接返回行本身。
    """
    if not template or "{p}" not in template:
        return lines
    return [template.replace("{p}", line) for line in lines]


class PromptQueue:
    """
    将多行提示词视为队列，或从 txt 文件读取（逐行）。
    输出按批次堆叠的 CONDITIONING，便于一次性批量生成。
    可选模板，将每行填充到模板中的 {prompt} 占位符。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "multiline_text": ("STRING", {"multiline": True, "default": "A cat\nA dog\nA bird"}),
                "use_file": ("BOOLEAN", {"default": False}),
                "file_path": ("STRING", {"multiline": False, "default": ""}),
                "template": ("STRING", {"multiline": False, "default": "{p}"}),
                "mode": (["all", "incremental"],),
                "label": ("STRING", {"multiline": False, "default": "Prompt Queue 001"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("COND",)
    FUNCTION = "run"
    CATEGORY = "SimpleBatch"

    def run(self, clip, multiline_text, use_file, file_path, template, mode="all", label="Prompt Queue 001"):
        lines = _read_lines_from_source(multiline_text, file_path, use_file)
        lines = _apply_template_to_lines(lines, template)

        if not lines:
            dummy = torch.zeros(1, 1)
            return ([[dummy, {"pooled_output": ""}]],)

        # Mode: incremental consumes one line per call using a persistent counter per label
        if mode == "incremental":
            idx = _pq_get_next_index(label, _pq_total=len(lines), source_descriptor=_pq_source_descriptor(use_file, file_path, multiline_text))
            line = lines[idx]
            ordered = [(0, line)]
            cond, pooled, has_pooled = encode_prompts_sequential(clip, ordered)
            meta = {"pooled_output": pooled if has_pooled and pooled is not None else ""}
            return ([[cond, meta]],)

        # Default: all lines batched together
        ordered = list(enumerate(lines))
        cond, pooled, has_pooled = encode_prompts_sequential(clip, ordered)
        meta = {"pooled_output": pooled if has_pooled and pooled is not None else ""}
        return ([[cond, meta]],)

    @classmethod
    def IS_CHANGED(cls, clip, multiline_text, use_file, file_path, template, mode="all", label="Prompt Queue 001"):
        # Ensure incremental mode re-executes each time to advance the queue
        if mode == "incremental":
            return str(time.time())
        # For deterministic modes, hash inputs so caching works as expected
        try:
            desc = _pq_source_descriptor(bool(use_file), str(file_path), str(multiline_text))
            key = f"{desc}|{template}"
            return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return ""


# --- Persistent queue state (per label) ---
class _JsonKVStore:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            else:
                self.data = {}
        except Exception:
            self.data = {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def get(self, category: str, key: str, default=None):
        return self.data.get(category, {}).get(key, default)

    def put(self, category: str, key: str, value) -> None:
        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value
        self._save()


_pq_state_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_prompt_queue_state")
_pq_store = _JsonKVStore(os.path.join(_pq_state_dir, "state.json"))


def _pq_source_descriptor(use_file: bool, file_path: str, multiline_text: str) -> str:
    if use_file and file_path:
        try:
            return "FILE:" + os.path.abspath(file_path)
        except Exception:
            return "FILE:" + file_path
    # Text hash as descriptor; content changes won't reset index to allow continuing when appending
    h = hashlib.sha1(multiline_text.encode("utf-8", errors="ignore")).hexdigest()
    return "TEXT:" + h


def _pq_get_next_index(label: str, _pq_total: int, source_descriptor: str) -> int:
    if _pq_total <= 0:
        return 0
    # Reset only when source type switches between FILE and TEXT, or file path changes
    stored_src = _pq_store.get("PromptQueue Sources", label, None)
    if stored_src != source_descriptor:
        _pq_store.put("PromptQueue Sources", label, source_descriptor)
        _pq_store.put("PromptQueue Counters", label, 0)

    current = _pq_store.get("PromptQueue Counters", label, 0) or 0
    next_index = current % _pq_total
    _pq_store.put("PromptQueue Counters", label, (current + 1) % _pq_total)
    return next_index


# 注册新节点
NODE_CLASS_MAPPINGS.update({
    "PromptQueue": PromptQueue,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "PromptQueue": "Prompt Queue",
})