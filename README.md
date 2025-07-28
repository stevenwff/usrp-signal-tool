# USRP 信号录制与发送工具

这是一个基于 Flask 和 UHD 的 USRP 设备控制工具，支持通过 Web 界面进行信号的录制、发送和文件管理，同时提供 API 接口便于集成到前端应用中。工具采用分块处理机制优化大文件操作，并支持通过 C 程序调用提升性能。

## 功能特点

- **设备管理**：自动发现并选择连接的 USRP 设备，支持多设备切换
- **信号录制**：
  - 可配置频率、采样率、增益和录制时长
  - 录制前自动检查存储空间是否充足（含 20% 安全余量）
  - 分块保存 IQ 数据，降低内存占用
  - 支持通过 C 程序进行高性能录制
- **信号发送**：
  - 发送预录制的 IQ 数据文件（.iq, .bin, .dat 格式）
  - 实时监控发送进度
  - 支持通过 C 程序进行高性能发送
- **文件管理**：
  - 上传本地 IQ 数据文件并自动验证格式
  - 下载、删除录制或上传的文件
  - 区分录制文件和上传文件存储目录
- **状态监控**：实时获取设备状态、录制/发送进度和存储空间信息

## 技术栈

- 后端框架：Flask
- USRP 控制：UHD (UHD Software Driver)
- 数据处理：NumPy
- 并发处理：Python 线程、subprocess（调用 C 程序）
- 文件格式：INT16 格式 IQ 数据（每个样本含 2 个 int16 值，共 4 字节）

## 部署方法

### 前置要求

- 支持 USRP 设备的 Linux 系统 (推荐 Ubuntu 20.04+)
- USRP 设备及 UHD 驱动
- Python 3.7+

### 依赖安装

1. **安装 UHD 驱动**：

```bash
# Ubuntu 系统示例
sudo apt update
sudo apt install libuhd-dev uhd-host
sudo uhd_images_downloader  # 下载 USRP 固件
```

2. **安装 Python 依赖**：

```bash
# 创建虚拟环境（可选但推荐）
python3 -m venv venv
source venv/bin/activate

# 安装依赖包
pip install flask flask-cors numpy uhd-python shutil
```

### 运行程序

1. 克隆仓库：

```bash
git clone https://github.com/stevenwff/usrp-signal-tool.git
cd usrp-signal-tool
```

2. 启动应用：

```bash
python app.py
```

3. 访问 Web 界面：
   打开浏览器，访问 `http://localhost:5000`

## 操作说明

### 设备管理

1. 程序启动后会自动扫描并列出连接的 USRP 设备
2. 在设备列表中点击选择要使用的设备
3. 可通过 `/api/devices` 接口获取设备列表及当前选中设备

### 信号录制

1. 点击"开始录制"按钮
2. 设置录制参数：
   - 文件名：保存的文件名（无需扩展名，自动添加 .iq）
   - 频率：录制的中心频率（Hz）
   - 采样率：信号采样率（Hz）
   - 增益：接收增益（dB）
   - 时长：录制时长（秒）
3. 系统会自动检查存储空间是否充足
4. 点击确认开始录制，可实时查看进度（每 10% 更新一次）
5. 如需中断，点击"停止录制"按钮

### 信号发送

1. 点击"开始发送"按钮
2. 设置发送参数：
   - 文件名：从文件列表选择要发送的 IQ 数据文件
   - 频率：发送的中心频率（Hz）
   - 采样率：发送采样率（Hz）
   - 增益：发送增益（dB）
3. 点击确认开始发送，可实时查看进度
4. 如需中断，点击"停止发送"按钮

### 文件管理

- **上传文件**：通过"上传文件"功能选择本地 IQ 数据文件，系统会自动验证文件格式（大小需为 4 的倍数）
- **下载文件**：点击文件列表中的文件名即可下载
- **删除文件**：点击文件后的删除按钮可删除不需要的文件
- 文件分类存储：录制文件保存在 `records` 目录，上传文件保存在 `uploads` 目录

## API 接口说明

### 设备相关

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/devices` | GET | 获取设备列表及当前选中设备 |
| `/api/select-device` | POST | 选择设备（参数：`device_id`） |
| `/api/status` | GET | 获取当前系统状态（录制/发送状态、选中设备） |

### 录制相关

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/start-recording` | POST | 开始录制（参数：`filename`, `frequency`, `sample_rate`, `gain`, `duration`） |
| `/api/stop-recording` | POST | 停止录制 |
| `/api/storage-status` | GET | 获取存储空间状态（总容量、已用、可用） |
| `/api/default-params` | GET | 获取默认录制参数 |

### 发送相关

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/start-transmission` | POST | 开始发送（参数：`filename`, `frequency`, `sample_rate`, `gain`） |
| `/api/stop-transmission` | POST | 停止发送 |
| `/api/transmission-progress` | GET | 获取发送进度（百分比） |

### 文件相关

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/files` | GET | 获取文件列表（区分录制和上传文件） |
| `/api/upload` | POST | 上传文件（表单字段：`file`） |
| `/api/delete-file/<folder>/<filename>` | DELETE | 删除文件（`folder` 为 `uploads` 或 `records`） |
| `/download/<folder>/<filename>` | GET | 下载文件（`folder` 为 `uploads` 或 `records`） |

## 技术细节

- **数据格式**：所有 IQ 数据均采用 INT16 格式存储，每个样本包含 I 分量和 Q 分量（各 2 字节）
- **性能优化**：
  - 大文件采用分块读写（默认 1MB/块），降低内存占用
  - 支持调用 UHD 官方 C 示例程序（`rx_samples_to_file` 和 `tx_samples_from_file`）进行高性能处理
  - 发送缓冲区优化（默认 32768 样本/包）
- **错误处理**：
  - 录制/发送过程中实时捕获并记录错误
  - 存储空间不足时提前预警
  - 无效文件格式（非 4 字节倍数）自动识别并提示

## 注意事项

1. 确保 USRP 设备已正确连接并被系统识别（可通过 `uhd_usrp_probe` 验证）
2. 录制和发送操作不能同时进行
3. 大文件（>1GB）的发送可能需要较长时间，请耐心等待
4. 网络连接的 USRP 设备需保证网络稳定
5. 程序运行时会在当前目录创建 `records` 和 `uploads` 文件夹用于存储文件

## 许可证

[MIT](LICENSE)
