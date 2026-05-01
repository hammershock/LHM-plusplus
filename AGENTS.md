# AGENTS.md

本文件是 LHM++ 仓库的项目级协作规范。全局规则仍然适用；若与本文件冲突，以本文件中更具体的项目约定为准。

## 工作分工

- 本地 Mac 是主工作区：代码编辑、文档维护、git 提交、模型/数据长期备份都优先在本地完成。
- 华为云远程主机只用于需要 CUDA/GPU 的调试、验证和长时间推理，不作为唯一代码仓库或唯一数据存储。
- 远程调试前先确认 GPU/CPU/磁盘资源，避免占用共享主机上的他人进程或目录。
- 长时间任务使用 remote ssh 后台会话运行，并记录命令、输出目录和结论。

## 代码同步

- 本地到远程的源码同步使用 git：本地提交或暂存明确变更后，在远程通过 `git pull`、`git fetch`、`git checkout` 等方式同步。
- 不用 remote ssh 直接覆盖源码文件来同步代码，除非是在处理一次性临时实验且不会替代 git 历史。
- 远程仓库只写 repo-level git config，不写共享账号的全局 git config、SSH config 或 credential helper。

## 模型、数据与大文件

- 模型、数据、motion assets、实验输入输出等大文件不提交到 git。
- 大文件传输使用 remote ssh 工具：小文件可用 `ssh_upload`/`ssh_download`，大文件或目录用 `ssh_upload_bg`/`ssh_download_bg` 并通过 `transfer_status` 跟踪。
- 大文件不能作为远程唯一副本。远程可能随时被抢占，所有重要模型和数据必须在本地 Mac 留存备份。
- 远程需要保留的副本放在 `/temp/hanmo/`；临时加速、可重建缓存、短期中间产物放在 `/cache/hanmo/`。
- 如远程环境本身搭建成本较高，尽量在本地备份可复现材料，例如 conda env 导出、wheel 文件清单、安装命令、补丁说明和失败日志。
- 新增任何大文件目录前先检查 `.gitignore`。若目录可能出现在项目树内，必须先加入 `.gitignore` 或放到项目目录外。
- 本项目常见大文件目录包括 `pretrained_models/`、`motion_video/`、`exps/`、`train_data/`、`benchmark/`、`debug/`，应保持未跟踪状态。

## 部署与调试记录

- 部署、依赖安装、远程调试、模型/数据同步的经验统一维护在 `docs/deployment.md`。
- 不要只把关键部署经验留在 shell history、临时聊天记录或远程机器上。
- 每次解决新的安装问题、CUDA/PyTorch 兼容问题、模型路径问题或数据同步问题后，补充到 `docs/deployment.md`。
- 记录应包含日期、主机、环境版本、执行命令、失败现象、解决方式、相关本地备份路径和远程路径。

## 项目特定注意事项

- 依赖安装以 README/INSTALL 文档为基础，尤其注意 Python 3.10、CUDA 12.1、PyTorch 2.3.0 及相关 CUDA wheel 的匹配。
- `pretrained_models/` 和 `motion_video/` 是运行 demo/测试的关键资产目录，但不进入 git。
- 推理和测试输出优先放在已忽略的 `exps/`、`debug/` 或明确的临时目录中。
- 修改代码前先检查当前工作树，避免覆盖用户未提交的改动。
