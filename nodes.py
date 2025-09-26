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

# ComfyUI 节点注册字典（在文件末尾进行 update 之前，必须先定义）
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 参数标签翻译映射
PARAMETER_LABELS = {
    "multiline_text": "多行文本",
    "use_file": "使用文件", 
    "file_path": "文件路径",
    "mode": "运行模式",
    "label": "队列标识",
    "preset": "预设模板",
    "template": "模板",
    "save_preset": "保存预设",
    "delete_preset": "删除预设"
}

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
                    raise ValueError(f"意外的 cond 张量形状: {cond.shape}")

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
        # 创建正确维度的 dummy pooled tensor (SDXL 需要 2816 维)
        dummy_pooled = torch.zeros(1, 2816)
        return dummy, dummy_pooled, False

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

    # 如果 pooled 不可用，创建 dummy pooled tensor
    if pooled_final is None:
        # 创建正确维度的 dummy pooled tensor (SDXL 需要 2816 维)
        pooled_final = torch.zeros(cond_final.shape[0], 2816)  # [N, 2816]

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


class SimplePromptQueue:
    """
    简化版提示词队列节点，支持从文件读取和预设模板功能。
    将多行提示词视为队列，从 txt 文件读取（逐行）。
    输出按批次堆叠的 CONDITIONING，便于一次性批量生成。
    """
    @classmethod
    def INPUT_TYPES(cls):
        # 动态获取预设列表（正面）
        presets = _pq_load_positive_template_presets()
        preset_list = ["None"] + list(presets.keys())

        return {
            "required": {
                "clip": ("CLIP",),
                "文件路径": ("STRING", {"multiline": False, "default": ""}),
                "运行模式": (["all", "incremental"], {"default": "all"}),
                "队列标识": ("STRING", {"multiline": False, "default": "Simple Queue 001"}),
                "预设模板": (preset_list, {"default": "None"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "run"
    CATEGORY = "PromptQueue"

    def run(self, clip, 文件路径, 运行模式="all", 队列标识="Simple Queue 001", 预设模板="None"):
        # 选择预设（若为 None 则不应用模板）
        模板 = ""
        if 预设模板 != "None":
            presets = _pq_load_positive_template_presets()
            if 预设模板 in presets:
                模板 = presets[预设模板]
                print(f"[Simple Prompt Queue] 使用预设 '{预设模板}': {模板}")
            else:
                print(f"[Simple Prompt Queue] 预设 '{预设模板}' 不存在，忽略模板")
        
        if not 文件路径.strip():
            dummy = torch.zeros(1, 1)
            # 创建正确维度的 dummy pooled tensor (SDXL 需要 2816 维)
            dummy_pooled = torch.zeros(1, 2816)
            return ([[dummy, {"pooled_output": dummy_pooled}]],)
        
        # 从文件读取提示词
        lines = _read_lines_from_source("", 文件路径, True)
        lines = _apply_template_to_lines(lines, 模板)
        
        if not lines:
            dummy = torch.zeros(1, 1)
            # 创建正确维度的 dummy pooled tensor (SDXL 需要 2816 维)
            dummy_pooled = torch.zeros(1, 2816)
            return ([[dummy, {"pooled_output": dummy_pooled}]],)

        # 模式：incremental 每次调用消费一行，使用每个标签的持久计数器
        if 运行模式 == "incremental":
            idx = _pq_get_next_index(队列标识, _pq_total=len(lines), source_descriptor=_pq_source_descriptor(True, 文件路径, ""))
            line = lines[idx]
            ordered = [(0, line)]
            cond, pooled, has_pooled = encode_prompts_sequential(clip, ordered)
            meta = {"pooled_output": pooled}
            return ([[cond, meta]],)

        # 默认：所有行一起批处理
        ordered = list(enumerate(lines))
        cond, pooled, has_pooled = encode_prompts_sequential(clip, ordered)
        meta = {"pooled_output": pooled}
        return ([[cond, meta]],)

    @classmethod
    def IS_CHANGED(cls, clip, 文件路径, 运行模式="all", 队列标识="Simple Queue 001", 预设模板="None"):
        # 确保增量模式每次重新执行以推进队列
        if 运行模式 == "incremental":
            return str(time.time())
        # 当预设更改时强制重新执行以实时更新模板
        if 预设模板 != "None":
            return str(time.time())
        # 对于确定性模式，对输入进行哈希以便缓存按预期工作
        try:
            desc = _pq_source_descriptor(True, str(文件路径), "")
            key = f"{desc}|{运行模式}|{队列标识}|{预设模板}"
            return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return ""


class NegativePromptQueue:
    """
    负面提示词队列节点，专门用于处理负面提示词。
    支持多行文本输入和预设模板功能，模板配置方式与PromptQueue一致。
    """
    @classmethod
    def INPUT_TYPES(cls):
        # 动态获取预设列表（负面）
        presets = _pq_load_negative_template_presets()
        preset_list = ["None"] + list(presets.keys())

        return {
            "required": {
                "clip": ("CLIP",),
                "多行文本": ("STRING", {"multiline": True, "default": "blurry, low quality, distorted"}),
                "预设模板": (preset_list, {"default": "None"}),
                "模板": ("STRING", {"multiline": False, "default": "{p}"}),
                "保存预设": ("BOOLEAN", {"default": False}),
                "删除预设": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("negative",)
    FUNCTION = "run"
    CATEGORY = "PromptQueue"

    def run(self, clip, 多行文本, 预设模板="None", 模板="{p}", 保存预设=False, 删除预设=False):
        # 处理删除预设 - 拥有最高优先级
        if 删除预设 and 预设模板 != "None":
            _pq_delete_negative_template_preset(预设模板)
            print(f"[Negative Prompt Queue] 已删除预设 '{预设模板}'")
            print(f"[Negative Prompt Queue] 使用 template 内容（删除预设后）: {模板}")
        # 处理保存预设 - 第二优先级，强制使用 template 内容
        elif 保存预设 and 模板.strip():
            _pq_save_negative_template_preset(模板.strip(), 模板.strip())
            print(f"[Negative Prompt Queue] 已保存预设 '{模板.strip()}': {模板.strip()}")
            print(f"[Negative Prompt Queue] 使用 template 内容（save_preset 优先级）: {模板}")
        # 处理预设选择 - 当选择非"None"预设且未开启保存/删除时，使用预设内容
        elif 预设模板 != "None":
            presets = _pq_load_negative_template_presets()
            if 预设模板 in presets:
                模板 = presets[预设模板]
                print(f"[Negative Prompt Queue] 使用预设 '{预设模板}': {模板}")
            else:
                print(f"[Negative Prompt Queue] 预设 '{预设模板}' 不存在，使用 template 内容")
        else:
            print(f"[Negative Prompt Queue] 使用 template 内容: {模板}")
        
        lines = _read_lines_from_source(多行文本, "", False)
        lines = _apply_template_to_lines(lines, 模板)

        if not lines:
            # 使用标准的 CLIP 编码方式处理空输入
            tokens = clip.tokenize("")
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            if pooled is None:
                pooled = torch.zeros(1, 2816)  # SDXL 需要 2816 维
            return ([[cond, {"pooled_output": pooled}]],)

        # 使用标准的 CLIP 编码方式处理每一行
        cond_list = []
        pooled_list = []
        
        with torch.no_grad():
            for line in lines:
                tokens = clip.tokenize(line)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                
                # 确保 cond 是 3D 张量 [B, T, C]
                if cond.dim() == 2:
                    cond = cond.unsqueeze(0)
                
                # 确保 pooled 是 2D 张量 [B, D]
                if pooled is None:
                    pooled = torch.zeros(1, 2816)  # SDXL 需要 2816 维
                elif pooled.dim() == 1:
                    pooled = pooled.unsqueeze(0)
                
                cond_list.append(cond)
                pooled_list.append(pooled)
        
        # 将所有条件连接起来
        if cond_list:
            # 对 cond 做 padding 以确保长度一致
            max_tokens = max(c.shape[1] for c in cond_list)
            padded_conds = []
            for c in cond_list:
                if c.shape[1] < max_tokens:
                    pad_len = max_tokens - c.shape[1]
                    c = F.pad(c, (0, 0, 0, pad_len))
                padded_conds.append(c)
            cond_final = torch.cat(padded_conds, dim=0)
            pooled_final = torch.cat(pooled_list, dim=0)
        else:
            # 空情况
            cond_final = torch.zeros(1, 1)
            pooled_final = torch.zeros(1, 2816)
        
        return ([[cond_final, {"pooled_output": pooled_final}]],)

    @classmethod
    def IS_CHANGED(cls, clip, 多行文本, 预设模板="None", 模板="{p}", 保存预设=False, 删除预设=False):
        # 当预设更改时强制重新执行以实时更新模板
        if 预设模板 != "None":
            return str(time.time())
        # 对于确定性模式，对输入进行哈希以便缓存按预期工作
        try:
            desc = _pq_source_descriptor(False, "", str(多行文本))
            key = f"{desc}|{模板}|{预设模板}"
            return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return ""


class PromptQueue:
    """
    将多行提示词视为队列，或从 txt 文件读取（逐行）。
    输出按批次堆叠的 CONDITIONING，便于一次性批量生成。
    可选模板，将每行填充到模板中的 {prompt} 占位符。
    """
    @classmethod
    def INPUT_TYPES(cls):
        # 动态获取预设列表（正面）
        presets = _pq_load_positive_template_presets()
        preset_list = ["None"] + list(presets.keys())
        
        return {
            "required": {
                "clip": ("CLIP",),
                "多行文本": ("STRING", {"multiline": True, "default": "A cat\nA dog\nA bird"}),
                "使用文件": ("BOOLEAN", {"default": False}),
                "文件路径": ("STRING", {"multiline": False, "default": ""}),
                "运行模式": (["all", "incremental"], {"default": "all"}),
                "队列标识": ("STRING", {"multiline": False, "default": "Prompt Queue 001"}),
                "预设模板": (preset_list, {"default": "None"}),
                "模板": ("STRING", {"multiline": False, "default": "{p}"}),
                "保存预设": ("BOOLEAN", {"default": False}),
                "删除预设": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "run"
    CATEGORY = "PromptQueue"

    def run(self, clip, 多行文本, 使用文件, 文件路径, 模板, 运行模式="all", 队列标识="Prompt Queue 001", 预设模板="None", 保存预设=False, 删除预设=False):
        # 处理删除预设 - 拥有最高优先级
        if 删除预设 and 预设模板 != "None":
            _pq_delete_positive_template_preset(预设模板)
            print(f"[Prompt Queue] 已删除预设 '{预设模板}'")
            print(f"[Prompt Queue] 使用 template 内容（删除预设后）: {模板}")
        # 处理保存预设 - 第二优先级，强制使用 template 内容
        elif 保存预设 and 模板.strip():
            _pq_save_positive_template_preset(模板.strip(), 模板.strip())
            print(f"[Prompt Queue] 已保存预设 '{模板.strip()}': {模板.strip()}")
            print(f"[Prompt Queue] 使用 template 内容（save_preset 优先级）: {模板}")
        # 处理预设选择 - 当选择非"None"预设且未开启保存/删除时，使用预设内容
        elif 预设模板 != "None":
            presets = _pq_load_positive_template_presets()
            if 预设模板 in presets:
                模板 = presets[预设模板]
                print(f"[Prompt Queue] 使用预设 '{预设模板}': {模板}")
            else:
                print(f"[Prompt Queue] 预设 '{预设模板}' 不存在，使用 template 内容")
        else:
            print(f"[Prompt Queue] 使用 template 内容: {模板}")
        
        lines = _read_lines_from_source(多行文本, 文件路径, 使用文件)
        lines = _apply_template_to_lines(lines, 模板)

        if not lines:
            dummy = torch.zeros(1, 1)
            # 创建正确维度的 dummy pooled tensor (SDXL 需要 2816 维)
            dummy_pooled = torch.zeros(1, 2816)
            return ([[dummy, {"pooled_output": dummy_pooled}]],)

        # 模式：incremental 每次调用消费一行，使用每个标签的持久计数器
        if 运行模式 == "incremental":
            idx = _pq_get_next_index(队列标识, _pq_total=len(lines), source_descriptor=_pq_source_descriptor(使用文件, 文件路径, 多行文本))
            line = lines[idx]
            ordered = [(0, line)]
            cond, pooled, has_pooled = encode_prompts_sequential(clip, ordered)
            meta = {"pooled_output": pooled}
            return ([[cond, meta]],)

        # 默认：所有行一起批处理
        ordered = list(enumerate(lines))
        cond, pooled, has_pooled = encode_prompts_sequential(clip, ordered)
        meta = {"pooled_output": pooled}
        return ([[cond, meta]],)

    @classmethod
    def IS_CHANGED(cls, clip, 多行文本, 使用文件, 文件路径, 模板, 运行模式="all", 队列标识="Prompt Queue 001", 预设模板="None", 保存预设=False, 删除预设=False):
        # 确保增量模式每次重新执行以推进队列
        if 运行模式 == "incremental":
            return str(time.time())
        # 当预设更改时强制重新执行以实时更新模板
        if 预设模板 != "None":
            return str(time.time())
        # 对于确定性模式，对输入进行哈希以便缓存按预期工作
        try:
            desc = _pq_source_descriptor(bool(使用文件), str(文件路径), str(多行文本))
            key = f"{desc}|{模板}|{预设模板}"
            return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return ""


# --- 持久队列状态（每个标签）---
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
    # 文本哈希作为描述符；内容更改不会重置索引以允许在追加时继续
    h = hashlib.sha1(multiline_text.encode("utf-8", errors="ignore")).hexdigest()
    return "TEXT:" + h


def _pq_get_next_index(label: str, _pq_total: int, source_descriptor: str) -> int:
    if _pq_total <= 0:
        return 0
    # 仅在源类型在 FILE 和 TEXT 之间切换或文件路径更改时重置
    stored_src = _pq_store.get("PromptQueue Sources", label, None)
    if stored_src != source_descriptor:
        _pq_store.put("PromptQueue Sources", label, source_descriptor)
        _pq_store.put("PromptQueue Counters", label, 0)

    current = _pq_store.get("PromptQueue Counters", label, 0) or 0
    next_index = current % _pq_total
    _pq_store.put("PromptQueue Counters", label, (current + 1) % _pq_total)
    return next_index


def _pq_save_positive_template_preset(name: str, template: str) -> None:
    """保存正面模板预设"""
    _pq_store.put("Template Presets Positive", name, template)


def _pq_load_positive_template_presets() -> dict:
    """加载所有正面模板预设"""
    return _pq_store.data.get("Template Presets Positive", {})


def _pq_delete_positive_template_preset(name: str) -> None:
    """删除正面模板预设"""
    if "Template Presets Positive" in _pq_store.data and name in _pq_store.data["Template Presets Positive"]:
        del _pq_store.data["Template Presets Positive"][name]
        _pq_store._save()


def _pq_save_negative_template_preset(name: str, template: str) -> None:
    """保存负面模板预设"""
    _pq_store.put("Template Presets Negative", name, template)


def _pq_load_negative_template_presets() -> dict:
    """加载所有负面模板预设"""
    return _pq_store.data.get("Template Presets Negative", {})


def _pq_delete_negative_template_preset(name: str) -> None:
    """删除负面模板预设"""
    if "Template Presets Negative" in _pq_store.data and name in _pq_store.data["Template Presets Negative"]:
        del _pq_store.data["Template Presets Negative"][name]
        _pq_store._save()


# 注册新节点
NODE_CLASS_MAPPINGS.update({
    "PromptQueue": PromptQueue,
    "SimplePromptQueue": SimplePromptQueue,
    "NegativePromptQueue": NegativePromptQueue,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "PromptQueue": "Prompt Queue / 提示词队列",
    "SimplePromptQueue": "Simple Prompt Queue / 简化提示词队列",
    "NegativePromptQueue": "Negative Prompt Queue / 负面提示词队列",
})