## 安装

将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录，然后重启 ComfyUI。

```
cd <ComfyUI 根目录>/custom_nodes
git clone https://github.com/cyan9977/ComfyUI_PromptQueue.git
```


## Nodes

### Prompt Queue
- 将多行提示词视为队列，或从 txt 文件按行读取，批量输出到 `CONDITIONING`。
- 参数
  - `clip`: 连接到任意 CLIP 节点
  - `multiline_text`: 多行输入，每行一个提示词
  - `use_file`: 勾选后从文件读取
  - `file_path`: 指向 `.txt` 文件，按行读取（UTF-8，忽略无法解码字符）
  - `mode`: 运行模式
    - `all`: 批量处理所有行
    - `incremental`: 每次运行消费下一行（按队列顺序）
  - `label`: 队列标识符，用于区分不同队列的进度
  - `preset`: 选择已保存的模板预设（选择非"None"时自动使用预设，忽略 template）
  - `template`: 可选模板，使用 `{p}` 作为占位符，例如：`"best quality, {p}, 4k"`
  - `save_preset`: 勾选后保存当前 template 内容为预设（拥有第二优先级，强制使用 template 内容）
  - `delete_preset`: 勾选后删除当前选中的预设（拥有最高优先级）
- 输出
  - `COND`: 已按行批量堆叠好的 conditioning，可直接连接到采样器正向提示

示例：

```
multiline_text:
  A cat on the beach
  A dog in the park
  A bird on a tree

 template:
   best quality, {p}, 4k
```


## Quick Start

1) 放置 `CLIP Text Encode`（或任意提供 `CLIP` 的上游）
2) 放置 `Prompt Queue` 并连接 `clip`
3) 在 `multiline_text` 输入多行提示词，或勾选 `use_file` 并选择 `file_path`
4) 选择运行模式：
   - `all`: 一次性处理所有行
   - `incremental`: 每次运行处理下一行（需要多次运行）
5) 可选：设置 `template`（例如 `best quality, {p}, 4k`）或选择预设
6) 将 `Prompt Queue` 的 `COND` 输出连接到采样器（正向提示）
7) 运行，即可对每一行提示词批量出图


## 使用说明与注意事项

- 批量合并：内部对 `cond` 做了 padding，解决不同提示词长度导致的合并错误
- 兼容性：若上游 `clip.encode_from_tokens(..., return_pooled=True)` 不返回 pooled，会自动降级并以空字符串占位，保证下游不出错
- 文件编码：从 `file_path` 读取 `.txt` 时使用 UTF-8 并忽略无法解码字符；读取失败会回退到 `multiline_text`
- 空输入：当没有有效行时会输出占位 `COND`
- 模板占位：`template` 中使用 `{p}` 作为单行替换占位符
- 队列模式：
  - `all` 模式：一次性处理所有行，适合批量生成
  - `incremental` 模式：每次运行消费下一行，适合逐张生成
- 预设功能：
  - 保存预设：在 `template` 输入内容，填写 `save_preset_name`，勾选 `save_preset` 后运行
  - 使用预设：从 `preset` 下拉菜单选择已保存的预设
  - 预设持久化：保存在 `_prompt_queue_state/state.json` 中


## 示例（从文件读取）

- 准备 `prompts.txt`：

```
A cat on the beach
A dog in the park
A bird on a tree
```

- 节点设置：
  - `use_file`: 勾选
  - `file_path`: 指向 `prompts.txt`
  - `template`: `best quality, {p}, 4k`（可选）

- 连接采样器并运行，即可一次性按行批量生成。


## FAQ

- 为什么只改了模板就全部生效？
  - 模板会对每一行 `{p}` 替换后再统一编码，因此能批量生效。

- `incremental` 模式和 `all` 模式的区别？
  - `all` 模式：一次运行处理所有行，输出包含所有提示词的批量 conditioning
  - `incremental` 模式：每次运行只处理一行，按队列顺序逐行消费，适合逐张生成

- 如何保存、使用和删除预设？
  - 保存：在 `template` 输入完整提示词，勾选 `save_preset` 后运行（预设名称为提示词内容）
  - 使用：从 `preset` 下拉菜单选择预设（非"None"），自动使用预设内容
  - 删除：选择要删除的预设，勾选 `delete_preset` 后运行
  - 切换：选择 `preset="None"`，直接使用 `template` 字段内容
  - 优先级：`delete_preset` > `save_preset` > `preset`，高优先级操作会忽略低优先级

- `label` 的作用是什么？
  - 用于区分不同的队列，每个 `label` 维护独立的计数器，避免不同工作流的队列互相影响