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
  - `template`: 可选模板，使用 `{p}` 作为占位符，例如：`"best quality, {p}, 4k"`
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
 4) 可选：设置 `template`（例如 `best quality, {p}, 4k`）
5) 将 `Prompt Queue` 的 `COND` 输出连接到采样器（正向提示）
6) 运行，即可对每一行提示词批量出图


## 使用说明与注意事项

- 批量合并：内部对 `cond` 做了 padding，解决不同提示词长度导致的合并错误
- 兼容性：若上游 `clip.encode_from_tokens(..., return_pooled=True)` 不返回 pooled，会自动降级并以空字符串占位，保证下游不出错
- 文件编码：从 `file_path` 读取 `.txt` 时使用 UTF-8 并忽略无法解码字符；读取失败会回退到 `multiline_text`
- 空输入：当没有有效行时会输出占位 `COND`
- 模板占位：`template` 中使用 `{p}` 作为单行替换占位符


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