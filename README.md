## ComfyUI_PromptQueue

这是一个为 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 设计的自定义节点，旨在简化和自动化批量生成图像时的工作流。它允许你通过多行文本、外部文件以及可复用的模板来管理和序列化你的提示词。

---

## 节点介绍

### 1. Style Prompt Queue / 风格化提示词队列 🆕

这是一个功能强大的节点，专门用于**序列化生成**。它将“画师/风格”部分和“通用后缀”从核心提示词中分离出来，让你能够轻松地对一个主体列表应用不同的风格组合。

- **核心功能**:
  - **画师/风格提示词**: 在此输入艺术家、画风等前缀。
  - **提示词序列**: 在此输入你的核心主体，每行一个，例如“一只猫”、“一条狗”。
  - **提示词后缀**: 在此输入通用后缀，例如“杰作, 8k”。
  - **预设系统**: 将常用的“画师/风格”和“后缀”组合保存为预设，方便一键切换。
  - **文件支持**: 可以通过外部 `.txt` 文件加载提示词序列。
- **运行模式**:
  - **固定为增量模式**: 每次执行只处理序列中的一个提示词，非常适合长序列的自动化生成。

![Style Prompt Queue 示例](https://raw.githubusercontent.com/cyan9977/ComfyUI_PromptQueue/main/screenshots/style-prompt-queue-cn.png)

### 2. Prompt Queue / 提示词队列

一个通用的正向提示词队列节点，支持文本、文件和模板三种输入方式。

- **核心功能**:
  - **多行文本**: 直接在节点中输入多个提示词。
  - **文件支持**: 通过 `.txt` 文件加载提示词。
  - **模板功能**: 使用 `{p}` 作为占位符，将模板应用于每一行提示词。
- **运行模式**:
  - `all`: 一次性处理所有提示词，生成一个批次。
  - `incremental`: 每次执行处理一个提示词。

### 3. Negative Prompt Queue / 负面提示词队列

专门用于管理和应用负面提示词的节点。

- **核心功能**:
  - **多行文本**: 输入所有负面提示词。
  - **模板预设**: 可以将常用的负面提示词组合保存为预设。

### 4. Simple Prompt Queue / 简化提示词队列

一个精简版的正向提示词队列，仅支持从文件加载，并可以选择预设，但不提供预设的保存和删除功能。

---

## 安装

1.  打开命令行工具。
2.  进入 ComfyUI 的 `custom_nodes` 目录:
    ```bash
    cd <你的ComfyUI根目录>/custom_nodes
    ```
3.  克隆本仓库:
    ```bash
    git clone https://github.com/cyan9977/ComfyUI_PromptQueue.git
    ```
4.  **重启 ComfyUI**。

---

## 快速上手

### 使用 `Style Prompt Queue`

1.  **添加节点**: 在 ComfyUI 中添加 `Style Prompt Queue` 节点。
2.  **连接 CLIP**: 将上游的 `CLIP` 输出连接到此节点的 `clip` 输入。
3.  **填写提示词**:
    - 在 **`artist_style_prompt`** 中填入风格，例如 `by Makoto Shinkai, cinematic lighting`。
    - 在 **`multiline_text`** 中填入主体，每行一个，例如:
      ```
      a girl on the beach
      a boy in the park
      a robot on a tree
      ```
    - 在 **`prompt_suffix`** 中填入后缀，例如 `best quality, masterpiece, 8k`。
4.  **连接输出**: 将此节点的 `positive` 输出连接到采样器（KSampler）的 `positive` 输入。
5.  **运行**: 点击 `Queue Prompt`。每次执行，它会自动处理 `multiline_text` 中的下一行，并与前后缀组合成最终的提示词。

### 保存和使用预设 (`Style Prompt Queue`)

1.  **填写内容**: 在 `artist_style_prompt` 和 `prompt_suffix` 中填入你想要保存的组合。
2.  **命名预设**: 在 `preset_name_for_save` 中为你的预设命名，例如 `电影感-通用`。
3.  **保存**: 勾选 `save_preset` 并运行一次工作流。预设即被保存。
4.  **使用预设**: 从 `preset` 下拉菜单中选择你刚才保存的预设，节点会自动填入对应的前后缀。

---

## 常见问题 (FAQ)

- **`队列标识 (label)` 的作用是什么？**
  - 它为每个队列实例提供一个独立的ID。如果你在同一个工作流中使用了多个队列节点，请确保它们的 `label` 是不同的，这样它们的执行进度才不会互相干扰。

- **`incremental` 和 `all` 模式有什么区别？**
  - `all`: 一次性处理所有提示词，将它们打包成一个批次输出。适合需要一次性生成所有结果的场景。
  - `incremental`: 每次执行只处理队列中的下一个提示词。适合自动化、长序列的生成，例如动画帧。

- **预设保存在哪里？**
  - 所有的预设数据都保存在 `ComfyUI/custom_nodes/ComfyUI_PromptQueue/_prompt_queue_state/state.json` 文件中。