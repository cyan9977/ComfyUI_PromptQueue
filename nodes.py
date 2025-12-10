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
    "delete_preset": "删除预设",
    "artist_style_prompt": "画师风格提示词",
    "prompt_suffix": "提示词后缀",
    "preset_name_for_save": "保存预设名称",
    "index_setting": "索引设置"
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


def _parse_index_setting(index_str: str) -> list:
    """
    解析索引设置字符串。
    支持格式:
    - "-1" 或 "": 记忆模式 (Auto/Resume) -> 返回 [-1]
    - "0": 重置模式 (Reset to 0) -> 返回 [0]
    - "5": 单行模式 -> 返回 [5] (第6行)
    - "2,8,35": 列表模式 -> 返回 [2, 8, 35]
    """
    s = str(index_str).strip()
    if not s:
        return [-1]
    
    # 尝试分割逗号
    parts = s.replace("，", ",").split(",")
    indices = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            val = int(p)
            indices.append(val)
        except ValueError:
            continue
            
    if not indices:
        return [-1]
    
    return indices


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


class StylePromptQueue:
    """
    [增量模式] 风格化提示词队列节点。
    - 每次运行仅输出一行提示词。
    - 即使 Batch Size > 1，也全部使用这一行（用于提高容错率）。
    - 索引设置：支持 "-1"(记忆/自动), "0"(从0开始), "5"(从5开始), "2,8,35"(列表循环)
    """
    @classmethod
    def INPUT_TYPES(cls):
        # 动态获取预设列表（正面）
        presets = _pq_load_positive_template_presets()
        preset_list = ["None"] + list(presets.keys())
        
        return {
            "required": {
                "clip": ("CLIP",),
                "画师风格提示词": ("STRING", {"multiline": False, "default": ""}),
                "多行文本": ("STRING", {"multiline": True, "default": "A cat\nA dog"}),
                "提示词后缀": ("STRING", {"multiline": False, "default": ""}),
                "使用文件": ("BOOLEAN", {"default": False}),
                "文件路径": ("STRING", {"multiline": False, "default": ""}),
                "索引设置": ("STRING", {"multiline": False, "default": "-1"}),
                "预设模板": (preset_list, {"default": "None"}),
                "保存预设名称": ("STRING", {"multiline": False, "default": ""}),
                "保存预设": ("BOOLEAN", {"default": False}),
                "删除预设": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "prompt_text")
    FUNCTION = "run"
    CATEGORY = "PromptQueue"

    def run(self, clip, 画师风格提示词, 多行文本, 提示词后缀, 使用文件, 文件路径, 索引设置, 预设模板, 保存预设名称, 保存预设, 删除预设, unique_id=None):
        prefix = 画师风格提示词
        suffix = 提示词后缀

        # 处理预设
        if 删除预设 and 预设模板 != "None":
            _pq_delete_positive_template_preset(预设模板)
            print(f"[Style Prompt Queue] 已删除预设 '{预设模板}'")
        
        elif 保存预设 and 保存预设名称.strip():
            preset_name = 保存预设名称.strip()
            
            parts = []
            p = prefix.strip()
            s = suffix.strip()
            if p:
                parts.append(p)
            parts.append("{p}")
            if s:
                parts.append(s)
            
            preset_data = ",".join(parts)
            _pq_save_positive_template_preset(preset_name, preset_data)
            print(f"[Style Prompt Queue] 已保存预设 '{preset_name}': {preset_data}")

        elif 预设模板 != "None":
            presets = _pq_load_positive_template_presets()
            if 预设模板 in presets:
                preset_content = presets[预设模板]
                
                is_legacy_json = False
                try:
                    data = json.loads(preset_content)
                    if isinstance(data, dict) and ("prefix" in data or "suffix" in data):
                        prefix = data.get("prefix", "")
                        suffix = data.get("suffix", "")
                        print(f"[Style Prompt Queue] 已从旧版JSON格式加载预设 '{预设模板}'")
                        is_legacy_json = True
                except (json.JSONDecodeError, TypeError):
                    pass

                if not is_legacy_json:
                    if "{p}" in preset_content:
                        parts = preset_content.split("{p}", 1)
                        prefix = parts[0].rstrip(',').strip()
                        if len(parts) > 1:
                            suffix = parts[1].lstrip(',').strip()
                        else:
                            suffix = ""
                        print(f"[Style Prompt Queue] 已加载预设 '{预设模板}'")
                    else:
                        prefix = preset_content
                        suffix = ""
                        print(f"[Style Prompt Queue] 无法解析预设 '{预设模板}'，已作为前缀处理")
        
        lines = _read_lines_from_source(多行文本, 文件路径, 使用文件)
        
        # 应用前缀和后缀
        final_lines = []
        for line in lines:
            parts = [p for p in [prefix, line, suffix] if p and p.strip()]
            final_lines.append(", ".join(parts))

        if not final_lines:
            dummy = torch.zeros(1, 1)
            dummy_pooled = torch.zeros(1, 2816)
            return ([[dummy, {"pooled_output": dummy_pooled}]], "")

        # --- 核心索引控制逻辑 ---
        indices = _parse_index_setting(索引设置)
        idx_mode = indices[0]
        
        target_idx = 0
        desc = _pq_source_descriptor(使用文件, 文件路径, 多行文本)
        
        # 扩展逻辑：支持 "Start From X" 且自动递增
        # 检测 索引设置变更 或 源内容变更
        
        queue_label = "Auto_Queue"
        if unique_id:
            queue_label = f"{unique_id}_Auto_Queue"
            
        current_setting_str = str(索引设置).strip()
        # 索引设置状态也需要隔离，否则不同节点使用相同文本但不同设置时会互相干扰
        setting_key = f"{queue_label}_Settings_{desc}"
        last_setting = _pq_store.get("Index Settings", setting_key, "")
        
        setting_changed = (current_setting_str != last_setting)
        if setting_changed:
             _pq_store.put("Index Settings", setting_key, current_setting_str)

        # 获取 Auto_Queue 的最后一次源，用于判断是否发生源变更
        last_source = _pq_store.get("PromptQueue Sources", queue_label, None)
        source_changed = (last_source != desc)
        
        should_reset_counter = setting_changed or source_changed

        if idx_mode == -1:
            # 记忆模式 (Auto/Resume)
            target_idx = _pq_get_next_index(queue_label, len(final_lines), desc)
            
        elif len(indices) == 1 and idx_mode >= 0:
            # 0: Auto from beginning (Start from 0 + Auto Increment)
            # N > 0: Fixed single line N (1-based index) -> Index N-1
            
            if idx_mode == 0:
                 # Auto Mode (0)
                 if should_reset_counter:
                     _pq_store.put("PromptQueue Counters", queue_label, 0)
                     _pq_store.put("PromptQueue Sources", queue_label, desc)
                 
                 target_idx = _pq_get_next_index(queue_label, len(final_lines), desc)
            else:
                 # Fixed Line Mode (e.g. 1 -> 1st line -> index 0)
                 # Non-incremental
                 target_idx = idx_mode - 1
            
        else:
            # 列表模式 (2,8,35) -> 视为 1-based index
            # 使用一个基于列表内容的特殊 key
            list_key = f"List_{索引设置}"
            # 这里的 total 是列表长度
            list_idx = _pq_get_next_index(list_key, len(indices), list_key)
            val = indices[list_idx]
            # 转换为 0-based
            target_idx = val - 1
            if target_idx < 0:
                target_idx = 0

        # 越界保护
        if target_idx >= len(final_lines):
            target_idx = target_idx % len(final_lines)
        if target_idx < 0:
             target_idx = 0

        # Debug print
        print(f"[PromptQueue] Index Setting: '{索引设置}' -> Mode: {idx_mode} -> Target Line: {target_idx + 1}")

        line_to_process = final_lines[target_idx]
        
        ordered = [(0, line_to_process)]
        cond, pooled, has_pooled = encode_prompts_sequential(clip, ordered)
        meta = {"pooled_output": pooled}
        return ([[cond, meta]], line_to_process)

    @classmethod
    def IS_CHANGED(cls, clip, 画师风格提示词, 多行文本, 提示词后缀, 使用文件, 文件路径, 索引设置, 预设模板, 保存预设名称, 保存预设, 删除预设):
        # 始终返回时间戳以强制重新执行
        return str(time.time())


class SimpleStylePromptQueue(StylePromptQueue):
    """
    [增量模式] 简化版提示词队列节点 (无预设管理)。
    继承自 StylePromptQueue，但去除了预设管理相关的输入。
    """
    @classmethod
    def INPUT_TYPES(cls):
        presets = _pq_load_positive_template_presets()
        preset_list = ["None"] + list(presets.keys())
        return {
            "required": {
                "clip": ("CLIP",),
                "画师风格提示词": ("STRING", {"multiline": False, "default": ""}),
                "多行文本": ("STRING", {"multiline": True, "default": "A cat\nA dog"}),
                "提示词后缀": ("STRING", {"multiline": False, "default": ""}),
                "使用文件": ("BOOLEAN", {"default": False}),
                "文件路径": ("STRING", {"multiline": False, "default": ""}),
                "索引设置": ("STRING", {"multiline": False, "default": "-1"}),
                "预设模板": (preset_list, {"default": "None"}),
            }
        }
        
    def run(self, clip, 画师风格提示词, 多行文本, 提示词后缀, 使用文件, 文件路径, 索引设置, 预设模板, unique_id=None):
        # 调用父类的 run，传入 None/False 作为预设参数
        return super().run(clip, 画师风格提示词, 多行文本, 提示词后缀, 使用文件, 文件路径, 索引设置, 预设模板, "", False, False, unique_id=unique_id)
        
    @classmethod
    def IS_CHANGED(cls, clip, 画师风格提示词, 多行文本, 提示词后缀, 使用文件, 文件路径, 索引设置, 预设模板):
        return str(time.time())


class StylePromptQueueBatch:
    """
    [批量模式] 风格化提示词队列节点。
    - 每次运行输出所有行（或指定范围）。
    - 这里的索引设置主要用于指定 "从哪一行开始" 或 "只输出哪几行"。
    """
    @classmethod
    def INPUT_TYPES(cls):
        presets = _pq_load_positive_template_presets()
        preset_list = ["None"] + list(presets.keys())
        
        return {
            "required": {
                "clip": ("CLIP",),
                "画师风格提示词": ("STRING", {"multiline": False, "default": ""}),
                "多行文本": ("STRING", {"multiline": True, "default": "A cat\nA dog"}),
                "提示词后缀": ("STRING", {"multiline": False, "default": ""}),
                "使用文件": ("BOOLEAN", {"default": False}),
                "文件路径": ("STRING", {"multiline": False, "default": ""}),
                "索引设置": ("STRING", {"multiline": False, "default": "0"}),
                "预设模板": (preset_list, {"default": "None"}),
                "保存预设名称": ("STRING", {"multiline": False, "default": ""}),
                "保存预设": ("BOOLEAN", {"default": False}),
                "删除预设": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "prompt_text_list")
    FUNCTION = "run"
    CATEGORY = "PromptQueue"

    def run(self, clip, 画师风格提示词, 多行文本, 提示词后缀, 使用文件, 文件路径, 索引设置, 预设模板, 保存预设名称, 保存预设, 删除预设):
        prefix = 画师风格提示词
        suffix = 提示词后缀
        
        # 处理预设
        if 删除预设 and 预设模板 != "None":
            _pq_delete_positive_template_preset(预设模板)
            print(f"[Prompt Queue Batch] 已删除预设 '{预设模板}'")
        
        elif 保存预设 and 保存预设名称.strip():
            preset_name = 保存预设名称.strip()
            
            parts = []
            p = prefix.strip()
            s = suffix.strip()
            if p:
                parts.append(p)
            parts.append("{p}")
            if s:
                parts.append(s)
            
            preset_data = ",".join(parts)
            _pq_save_positive_template_preset(preset_name, preset_data)
            print(f"[Prompt Queue Batch] 已保存预设 '{preset_name}': {preset_data}")

        elif 预设模板 != "None":
            presets = _pq_load_positive_template_presets()
            if 预设模板 in presets:
                preset_content = presets[预设模板]
                # 简化处理，仅支持新版 {p} 格式或纯前缀
                if "{p}" in preset_content:
                    parts = preset_content.split("{p}", 1)
                    prefix = parts[0].rstrip(',').strip()
                    if len(parts) > 1:
                        suffix = parts[1].lstrip(',').strip()
                    else:
                        suffix = ""
                else:
                    prefix = preset_content
                    suffix = ""
                    
        lines = _read_lines_from_source(多行文本, 文件路径, 使用文件)
        
        final_lines = []
        for line in lines:
            parts = [p for p in [prefix, line, suffix] if p and p.strip()]
            final_lines.append(", ".join(parts))

        if not final_lines:
            dummy = torch.zeros(1, 1)
            dummy_pooled = torch.zeros(1, 2816)
            return ([[dummy, {"pooled_output": dummy_pooled}]], "")

        # 索引处理
        # 批量模式下，索引设置通常意味着：
        # "0": 输出所有，从 0 开始
        # "5": 输出所有，从 5 开始 (5, 6, 7...)
        # "2,8,35": 只输出这几行
        
        indices = _parse_index_setting(索引设置)
        selected_lines = []
        
        if indices[0] == -1:
            # 默认全部
            selected_lines = list(enumerate(final_lines))
        elif len(indices) == 1 and indices[0] >= 0:
            # 从指定行开始到结束
            start_idx = indices[0]
            if start_idx >= len(final_lines):
                start_idx = 0 # 越界回退
            
            # 创建切片
            subset = final_lines[start_idx:]
            # 重新编号，为了 encode_prompts_sequential 里的排序（其实顺序已经定了）
            # 注意：这里的 idx 仅仅用于排序，只要是升序即可
            for i, line in enumerate(subset):
                selected_lines.append((i, line))
        else:
            # 指定列表
            for i, idx in enumerate(indices):
                if 0 <= idx < len(final_lines):
                    selected_lines.append((i, final_lines[idx]))

        if not selected_lines:
             # 如果选完是空的，回退到全部
             selected_lines = list(enumerate(final_lines))

        cond, pooled, has_pooled = encode_prompts_sequential(clip, selected_lines)
        meta = {"pooled_output": pooled}
        
        # 批量模式下返回所有选中的提示词文本，用换行符分隔
        output_text = "\n".join([item[1] for item in selected_lines])
        return ([[cond, meta]], output_text)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 批量模式通常是确定性的，除非输入变了
        # 但为了避免缓存导致修改索引无效，我们可以简单 hash 输入
        return float("nan") # 总是更新


class SimpleStylePromptQueueBatch(StylePromptQueueBatch):
    """
    [批量模式] 简化版提示词队列节点 (无预设管理)。
    继承自 StylePromptQueueBatch，但去除了预设管理相关的输入。
    """
    @classmethod
    def INPUT_TYPES(cls):
        presets = _pq_load_positive_template_presets()
        preset_list = ["None"] + list(presets.keys())
        return {
            "required": {
                "clip": ("CLIP",),
                "画师风格提示词": ("STRING", {"multiline": False, "default": ""}),
                "多行文本": ("STRING", {"multiline": True, "default": "A cat\nA dog"}),
                "提示词后缀": ("STRING", {"multiline": False, "default": ""}),
                "使用文件": ("BOOLEAN", {"default": False}),
                "文件路径": ("STRING", {"multiline": False, "default": ""}),
                "索引设置": ("STRING", {"multiline": False, "default": "0"}),
                "预设模板": (preset_list, {"default": "None"}),
            }
        }
        
    def run(self, clip, 画师风格提示词, 多行文本, 提示词后缀, 使用文件, 文件路径, 索引设置, 预设模板):
        # 调用父类的 run，传入 None 作为预设参数
        return super().run(clip, 画师风格提示词, 多行文本, 提示词后缀, 使用文件, 文件路径, 索引设置, 预设模板, "", False, False)
        
    @classmethod
    def IS_CHANGED(cls, clip, 画师风格提示词, 多行文本, 提示词后缀, 使用文件, 文件路径, 索引设置, 预设模板):
        return float("nan")


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
        
    def remove(self, category: str, key: str) -> None:
        """从存储中移除一个键"""
        if category in self.data and key in self.data[category]:
            del self.data[category][key]
            self._save()


_pq_state_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_prompt_queue_state")
_pq_store = _JsonKVStore(os.path.join(_pq_state_dir, "state.json"))
_preset_store = _JsonKVStore(os.path.join(_pq_state_dir, "presets.json"))

# --- 数据迁移逻辑 ---
# 检查旧 store 中是否有预设数据，如果有则迁移到新 store 并从旧 store 删除
_migration_occurred = False
for category in ["Template Presets Positive", "Template Presets Negative"]:
    if category in _pq_store.data:
        # 复制数据到 preset store
        if category not in _preset_store.data:
             _preset_store.data[category] = {}
        
        # 合并数据（优先保留 preset store 现有的，或者覆盖？这里假设 preset store 为空或应该包含旧数据）
        for k, v in _pq_store.data[category].items():
            _preset_store.data[category][k] = v
        
        # 从旧 store 删除 category
        del _pq_store.data[category]
        _migration_occurred = True

if _migration_occurred:
    _preset_store._save()
    _pq_store._save()
    print("[PromptQueue] Migrated presets from state.json to presets.json")


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
    _preset_store.put("Template Presets Positive", name, template)


def _pq_load_positive_template_presets() -> dict:
    """加载所有正面模板预设"""
    return _preset_store.data.get("Template Presets Positive", {})


def _pq_delete_positive_template_preset(name: str) -> None:
    """删除正面模板预设"""
    _preset_store.remove("Template Presets Positive", name)


def _pq_save_negative_template_preset(name: str, template: str) -> None:
    """保存负面模板预设"""
    _preset_store.put("Template Presets Negative", name, template)


def _pq_load_negative_template_presets() -> dict:
    """加载所有负面模板预设"""
    return _preset_store.data.get("Template Presets Negative", {})


def _pq_delete_negative_template_preset(name: str) -> None:
    """删除负面模板预设"""
    _preset_store.remove("Template Presets Negative", name)


# 注册新节点
NODE_CLASS_MAPPINGS.update({
    "NegativePromptQueue": NegativePromptQueue,
    "StylePromptQueue": StylePromptQueue,
    "SimpleStylePromptQueue": SimpleStylePromptQueue,
    "StylePromptQueueBatch": StylePromptQueueBatch,
    "SimpleStylePromptQueueBatch": SimpleStylePromptQueueBatch,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "NegativePromptQueue": "Negative PQ / 负面队列",
    "StylePromptQueue": "PQ (Incremental) / 增量队列",
    "SimpleStylePromptQueue": "PQ Simple (Incr.) / 增量简易",
    "StylePromptQueueBatch": "PQ (Batch) / 批量队列",
    "SimpleStylePromptQueueBatch": "PQ Simple (Batch) / 批量简易",
})