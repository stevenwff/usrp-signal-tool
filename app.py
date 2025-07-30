import os
import sys
import threading
import time
import numpy as np
import subprocess
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import uhd
import shutil
from uhd import usrp
from scipy.fft import fft, fftshift
import math

# 创建Flask应用实例
app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
RECORD_FOLDER = 'records'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECORD_FOLDER, exist_ok=True)

# C程序路径
RX_C_PROGRAM = "/usr/lib/uhd/examples/rx_samples_to_file"
TX_C_PROGRAM = "/usr/lib/uhd/examples/tx_samples_from_file"

# 默认参数设置
DEFAULT_FREQUENCY = 634000000  # 634 MHz
DEFAULT_SAMPLE_RATE = 7000000  # 7 MHz
DEFAULT_GAIN = 30.0
DEFAULT_CHANNEL = 0

# 发送配置
MAX_SAMPLES_PER_PACKET = 32768  # 每包最大样本数
NUM_BUFFERS = 16  # 缓冲区数量
PACKET_SIZE = MAX_SAMPLES_PER_PACKET * 4  # 每包字节数
FILE_READ_CHUNK = NUM_BUFFERS * PACKET_SIZE  # 一次读取的文件块大小
STREAM_TIMEOUT = 0.1  # 100ms 流超时
PROGRESS_LOG_INTERVAL = 10  # 每10%进度记录一次日志

# 全局状态
app_state = {
    'recording': False,
    'transmitting': False,
    'selected_device': None,
    'record_thread': None,
    'transmit_thread': None,
    'stop_event': threading.Event(),
    'transmission_progress': 0,
    'record_process': None,
    'transmit_process': None,
    'transmission_time': {  # 新增：时间跟踪状态
        'elapsed': 0,
        'total': 0,
        'sent_samples': 0
    },
    'transmission_timer': None,  # 计时器线程
    'transmission_start_time': 0  # 发送开始时间
}

# 设备地址模式
common_addresses = [
    " ",  # 默认地址
]

# 设备发现
def discover_devices():
    devices = []
    tested_addrs = set()
    
    for addr_str in common_addresses:
        if addr_str in tested_addrs:
            continue
        tested_addrs.add(addr_str)
        
        try:
            usrp_device = usrp.MultiUSRP(addr_str)
            
            dev_type = "USRP Device"
            dev_identifier = addr_str if addr_str else "default"
            
            if addr_str.startswith("type="):
                dev_type = addr_str.split("=")[1].upper()
            elif addr_str.startswith("ip="):
                dev_type = f"USRP at {addr_str.split('=')[1]}"
                
            # 获取设备的详细信息
            pp_string = usrp_device.get_pp_string()
            
            # 提取主板型号
            model = "Unknown"
            for line in pp_string.split('\n'):
                if "Mboard 0:" in line:
                    model = line.split(":")[1].strip()
                    break
            
            device = {
                "id": dev_identifier,
                "type": dev_type,
                "serial": model,
                "ip": addr_str.split("=")[1] if addr_str.startswith("ip=") else "unknown"
            }
            
            devices.append(device)
            app.logger.info(f"Found USRP device: {device['type']}")
            
            del usrp_device
            
        except Exception as e:
            app.logger.debug(f"Device not found at {addr_str}: {str(e)}")
            continue
    
    return devices

# 保存IQ数据
def save_iq_data(samples, filename, chunk_size=1048576):
    if not filename.endswith(('.iq', '.bin', '.dat')):
        filename += '.iq'
    
    full_path = os.path.join(RECORD_FOLDER, filename)
    
    total_samples = len(samples)
    samples_per_chunk = chunk_size // 4
    
    with open(full_path, 'wb', buffering=8*1024*1024) as f:
        app.logger.info(f"开始分块保存数据，总样本数: {total_samples}, 分块大小: {samples_per_chunk}")
        
        for i in range(0, total_samples, samples_per_chunk):
            end_idx = min(i + samples_per_chunk, total_samples)
            chunk = samples[i:end_idx]
            
            iq_data = np.empty(2 * len(chunk), dtype=np.int16)
            iq_data[0::2] = (chunk.real * 32767).astype(np.int16)
            iq_data[1::2] = (chunk.imag * 32767).astype(np.int16)
            
            f.write(iq_data.tobytes())
            
            progress = int((end_idx / total_samples) * 100)
            if progress % 10 == 0 and progress > (int((i / total_samples) * 100) // 10) * 10:
                app.logger.info(f"保存进度: {progress}%")
    
    app.logger.info(f"已将 {total_samples} 个样本保存为INT16格式IQ数据，文件大小: {os.path.getsize(full_path)} 字节")
    return full_path

# 原有录制线程函数
def record_thread_func(params):
    try:
        usrp_args = params['device_id'] if params['device_id'] else ""
        usrp_device = usrp.MultiUSRP(usrp_args)
        
        channel = params.get('channel', DEFAULT_CHANNEL)
        sample_rate = params.get('sample_rate', DEFAULT_SAMPLE_RATE)
        frequency = params.get('frequency', DEFAULT_FREQUENCY)
        gain = params.get('gain', DEFAULT_GAIN)
        
        usrp_device.set_rx_rate(sample_rate, channel)
        usrp_device.set_rx_freq(uhd.types.TuneRequest(frequency), channel)
        usrp_device.set_rx_gain(gain, channel)
        
        actual_rate = usrp_device.get_rx_rate(channel)
        actual_freq = usrp_device.get_rx_freq(channel)
        app.logger.info(f"实际采样率: {actual_rate}, 实际频率: {actual_freq}")
        
        num_samples = int(sample_rate * params['duration'])
        app.logger.info(f"开始录制, 样本数: {num_samples}, 频率: {frequency}Hz, 采样率: {sample_rate}Hz")
        
        st_args = usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [channel]
        rx_streamer = usrp_device.get_rx_stream(st_args)
        
        metadata = uhd.types.RXMetadata()
        buffer = np.zeros((1, num_samples), dtype=np.complex64)
        recv_offset = 0
        total_recv = 0
        
        time.sleep(0.5)
        
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        rx_streamer.issue_stream_cmd(stream_cmd)
        
        last_logged_progress = -1
        while total_recv < num_samples and not app_state['stop_event'].is_set():
            recv_samps = rx_streamer.recv(
                buffer[:, recv_offset:],
                metadata
            )
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                app.logger.warning(f"接收错误: {metadata.strerror()}")
                if metadata.error_code == uhd.types.RXMetadataErrorCode.fatal:
                    break
                continue
                
            total_recv += recv_samps
            recv_offset += recv_samps
            
            progress = min(100, int((total_recv / num_samples) * 100))
            if progress % PROGRESS_LOG_INTERVAL == 0 and progress != last_logged_progress:
                app.logger.info(f"录制进度: {progress}%")
                last_logged_progress = progress
        
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_streamer.issue_stream_cmd(stream_cmd)
        
        if total_recv > 0:
            try:
                file_path = save_iq_data(buffer[0, :total_recv], params['filename'], chunk_size=524288)
                app.logger.info(f"录制完成. 保存至 {file_path}, 实际接收样本: {total_recv}")
            except IOError as e:
                if "No space left on device" in str(e):
                    app.logger.error(f"保存文件失败：存储空间不足")
                else:
                    app.logger.error(f"保存文件失败: {str(e)}")
        else:
            app.logger.warning("未接收到任何样本数据")
        
    except Exception as e:
        app.logger.error(f"录制错误: {str(e)}")
    finally:
        app_state['recording'] = False
        app_state['stop_event'].clear()
        app_state['record_thread'] = None

# 调用C程序的录制线程函数
def c_record_thread_func(params):
    try:
        # 构建输出文件路径
        filename = params['filename']
        if not filename.endswith(('.iq', '.bin', '.dat')):
            filename += '.iq'
        full_path = os.path.join(RECORD_FOLDER, filename)

        # 构建C程序命令参数
        args = [
            RX_C_PROGRAM,
            f"--args={params['device_id'] or ''}",
            f"--freq={params['frequency']}",
            f"--rate={params['sample_rate']}",
            f"--gain={params['gain']}",
            f"--duration={params['duration']}",
            f"--channel={params.get('channel', DEFAULT_CHANNEL)}",
            f"--file={full_path}"
        ]

        app.logger.info(f"启动C录制程序: {' '.join(args)}")

        # ✅ 启动C程序进程（无缓冲 + 实时读取）
        app_state['record_process'] = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # 行缓冲
        )

        # ✅ 实时读取输出
        # 每隔100ms读取一次，避免阻塞
        while True:
            if app_state['stop_event'].is_set():
                break
            try:
                chunk = app_state['record_process'].stdout.read(128)
                if not chunk:
                    break
                app.logger.info(f"[RX_C] {chunk.strip()}")
            except:
                break


        # ✅ 等待进程结束
        app_state['record_process'].wait()

        if app_state['record_process'].returncode == 0:
            app.logger.info(f"C程序录制完成，文件保存至: {full_path}")
        else:
            app.logger.error(f"C程序录制失败，返回码: {app_state['record_process'].returncode}")
            if os.path.exists(full_path):
                os.remove(full_path)  # 删除不完整文件

    except Exception as e:
        app.logger.error(f"C程序录制错误: {str(e)}")
    finally:
        app_state['recording'] = False
        app_state['stop_event'].clear()
        app_state['record_thread'] = None
        app_state['record_process'] = None


# 原有发送线程函数
def transmit_thread_func(params):
    app_state['transmission_progress'] = 0
    
    try:
        filename = params['filename']
        full_path = None
        
        possible_paths = [
            os.path.join(RECORD_FOLDER, filename),
            os.path.join(UPLOAD_FOLDER, filename)
        ]
        
        if not any(filename.endswith(ext) for ext in ('.iq', '.bin', '.dat')):
            for ext in ('.iq', '.bin', '.dat'):
                possible_paths.insert(0, os.path.join(RECORD_FOLDER, filename + ext))
                possible_paths.append(os.path.join(UPLOAD_FOLDER, filename + ext))
        
        app.logger.debug(f"查找文件 {filename}，检查路径: {possible_paths}")
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                full_path = path
                app.logger.debug(f"找到文件: {full_path}")
                break
        
        if not full_path or not os.path.exists(full_path):
            app.logger.error(f"文件未找到: {filename}")
            return
        
        file_size = os.path.getsize(full_path)
        if file_size % 4 != 0:
            app.logger.error(f"文件大小无效，不是INT16 IQ数据格式 (大小: {file_size} 字节，应为4的倍数)")
            return
            
        if file_size == 0:
            app.logger.error(f"文件为空: {full_path}")
            return
        
        num_samples = file_size // 4
        app.logger.info(f"文件大小: {file_size} 字节，包含 {num_samples} 个INT16格式IQ样本，准备发送...")
        
        usrp_args = params['device_id'] if params['device_id'] else ""
        app.logger.info(f"使用设备: {usrp_args}")
        
        try:
            usrp_device = usrp.MultiUSRP(usrp_args)
            app.logger.info("USRP设备初始化成功")
        except Exception as e:
            app.logger.error(f"USRP设备初始化失败: {str(e)}")
            return
        
        channel = params.get('channel', DEFAULT_CHANNEL)
        sample_rate = params.get('sample_rate', DEFAULT_SAMPLE_RATE)
        frequency = params.get('frequency', DEFAULT_FREQUENCY)
        gain = params.get('gain', DEFAULT_GAIN)
        
        try:
            usrp_device.set_clock_source("internal")
            app.logger.info("设置时钟源为内部时钟")
        except Exception as e:
            app.logger.warning(f"无法设置时钟源: {str(e)}")
        
        try:
            usrp_device.set_tx_rate(sample_rate, channel)
            actual_rate = usrp_device.get_tx_rate(channel)
            app.logger.info(f"设置发送采样率: {sample_rate}, 实际: {actual_rate}")
            
            tune_request = uhd.types.TuneRequest(frequency)
            usrp_device.set_tx_freq(tune_request, channel)
            actual_freq = usrp_device.get_tx_freq(channel)
            app.logger.info(f"设置发送频率: {frequency}, 实际: {actual_freq}")
            
            usrp_device.set_tx_gain(gain, channel)
            actual_gain = usrp_device.get_tx_gain(channel)
            app.logger.info(f"设置发送增益: {gain}, 实际: {actual_gain}")
            
            try:
                usrp_device.set_tx_antenna("TX/RX", channel)
                app.logger.info("设置发送天线为TX/RX")
            except Exception as e:
                app.logger.warning(f"无法设置发送天线: {str(e)}")
                
        except Exception as e:
            app.logger.error(f"设置USRP参数失败: {str(e)}")
            return
        
        try:
            st_args = usrp.StreamArgs("fc32", "sc16")
            st_args.channels = [channel]
            tx_streamer = usrp_device.get_tx_stream(st_args)
            
            max_num_samps = 32768
            app.logger.info(f"使用固定最大发送样本数: {max_num_samps}")
            
            global MAX_SAMPLES_PER_PACKET
            MAX_SAMPLES_PER_PACKET = max_num_samps
                
        except Exception as e:
            app.logger.error(f"创建发送流失败: {str(e)}")
            return
        
        send_time = usrp_device.get_time_now() + uhd.types.TimeSpec(0.1)
        
        metadata = uhd.types.TXMetadata()
        metadata.start_of_burst = True
        metadata.end_of_burst = False
        metadata.time_spec = send_time
        
        app.logger.info("等待设备就绪...")
        time.sleep(0.5)
        
        total_num_samps = num_samples
        samples_sent = 0
        last_logged_progress = -1
        
        with open(full_path, 'rb') as f:
            app.logger.info("开始发送数据...")
            
            buffer = np.zeros(MAX_SAMPLES_PER_PACKET, dtype=np.complex64)
            
            while samples_sent < total_num_samps and not app_state['stop_event'].is_set():
                remaining = total_num_samps - samples_sent
                samps_to_send = min(remaining, MAX_SAMPLES_PER_PACKET)
                
                data = f.read(samps_to_send * 4)
                if not data:
                    break
                    
                iq_int16 = np.frombuffer(data, dtype=np.int16)
                i_components = iq_int16[0::2].astype(np.float32)
                q_components = iq_int16[1::2].astype(np.float32)
                
                max_val = 32768
                buffer[:samps_to_send] = (i_components / max_val) + 1j * (q_components / max_val)
                
                offset = 0
                while offset < samps_to_send:
                    remaining_in_buffer = samps_to_send - offset
                    
                    try:
                        sent = tx_streamer.send(
                            buffer[offset:offset+remaining_in_buffer],
                            metadata,
                            STREAM_TIMEOUT
                        )
                        
                        if sent == 0:
                            app.logger.warning("发送超时，没有样本被发送")
                            time.sleep(0.001)
                            continue
                            
                        offset += sent
                        samples_sent += sent
                        
                        metadata.start_of_burst = False
                        metadata.time_spec = uhd.types.TimeSpec(0.0)
                        
                    except Exception as e:
                        app.logger.error(f"发送错误: {str(e)}")
                        time.sleep(0.1)
                        continue
                
                progress = min(100, int((samples_sent / total_num_samps) * 100))
                app_state['transmission_progress'] = progress
                
                if progress % PROGRESS_LOG_INTERVAL == 0 and progress != last_logged_progress:
                    app.logger.info(f"发送进度: {progress}%")
                    last_logged_progress = progress
        
        metadata.end_of_burst = True
        tx_streamer.send(np.array([], dtype=np.complex64), metadata)
        
        app.logger.info(f"发送完成，总发送样本: {samples_sent}/{total_num_samps} ({samples_sent/total_num_samps*100:.1f}%)")
        
        if samples_sent < total_num_samps:
            app.logger.warning(f"发送不完整，缺少 {total_num_samps - samples_sent} 个样本")
        
    except Exception as e:
        app.logger.error(f"发送错误: {str(e)}")
    finally:
        app_state['transmitting'] = False
        app_state['stop_event'].clear()
        app_state['transmit_thread'] = None
        app_state['transmission_progress'] = 0

# 调用 C 程序的发送线程函数（实时读取 stdout）
def c_transmit_thread_func(params):
    app_state['transmission_progress'] = 0
    # 初始化时间跟踪
    app_state['transmission_time'] = {
        'elapsed': 0,
        'total': 0,
        'sent_samples': 0
    }

    try:
        filename = params['filename']
        full_path = None

        # 1. 查找文件（保持原有逻辑）
        possible_paths = [
            os.path.join(RECORD_FOLDER, filename),
            os.path.join(UPLOAD_FOLDER, filename)
        ]
        if not any(filename.endswith(ext) for ext in ('.iq', '.bin', '.dat')):
            for ext in ('.iq', '.bin', '.dat'):
                possible_paths.insert(0, os.path.join(RECORD_FOLDER, filename + ext))
                possible_paths.append(os.path.join(UPLOAD_FOLDER, filename + ext))

        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                full_path = path
                break
        if not full_path:
            app.logger.error(f"文件未找到: {filename}")
            return

        # 2. 计算总时长（保持原有逻辑）
        file_size = os.path.getsize(full_path)
        total_samples = file_size // 4
        sample_rate = params.get('sample_rate', DEFAULT_SAMPLE_RATE)
        total_duration = total_samples / sample_rate
        app_state['transmission_time']['total'] = total_duration

        # 3. 构建 C 程序命令参数
        args = [
            TX_C_PROGRAM,
            f"--args={params['device_id'] or ''}",
            f"--freq={params['frequency']}",
            f"--rate={params['sample_rate']}",
            f"--gain={params['gain']}",
            f"--channel={params.get('channel', DEFAULT_CHANNEL)}",
            f"--file={full_path}"
        ]
        app.logger.info(f"启动C发送程序: {' '.join(args)}")

        # 4. 启动进程（无缓冲 + 实时读取）
        app_state['transmit_process'] = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1          # 行缓冲
        )

        # 5. 启动后台进度计时器（保持原有逻辑）
        app_state['transmission_start_time'] = time.time()

        def update_progress_periodically():
            while app_state['transmitting'] and not app_state['stop_event'].is_set():
                elapsed = time.time() - app_state['transmission_start_time']
                progress = min(100, (elapsed / total_duration) * 100) if total_duration else 0
                app_state['transmission_progress'] = progress
                app_state['transmission_time']['elapsed'] = elapsed
                app_state['transmission_time']['sent_samples'] = min(
                    total_samples,
                    int((elapsed / total_duration) * total_samples)
                )
                time.sleep(1)

        app_state['transmission_timer'] = threading.Thread(target=update_progress_periodically, daemon=True)
        app_state['transmission_timer'].start()

        # 每隔100ms读取一次，避免阻塞
        while True:
            if app_state['stop_event'].is_set():
                break
            try:
                chunk = app_state['transmit_process'].stdout.read(128)
                if not chunk:
                    break
                app.logger.info(f"[TX_C] {chunk.strip()}")
            except:
                break            

        # 7. 等待进程结束
        app_state['transmit_process'].wait()

        if app_state['transmit_process'].returncode == 0:
            app.logger.info(f"C程序发送完成，文件: {full_path}")
            app_state['transmission_progress'] = 100
        else:
            app.logger.error(f"C程序发送失败，返回码: {app_state['transmit_process'].returncode}")

    except Exception as e:
        app.logger.error(f"C程序发送错误: {str(e)}")
    finally:
        # 清理
        if app_state['transmission_timer']:
            app_state['transmission_timer'] = None
        app_state['transmitting'] = False
        app_state['stop_event'].clear()
        app_state['transmit_thread'] = None
        app_state['transmit_process'] = None


# API 路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/devices')
def get_devices():
    devices = discover_devices()
    return jsonify({
        'devices': devices,
        'selected': app_state['selected_device']
    })

@app.route('/api/select-device', methods=['POST'])
def select_device():
    data = request.json
    device_id = data.get('device_id')
    app_state['selected_device'] = device_id
    return jsonify({
        'status': 'success',
        'selected': device_id
    })

@app.route('/api/status')
def get_status():
    return jsonify({
        'recording': app_state['recording'],
        'transmitting': app_state['transmitting'],
        'selected_device': app_state['selected_device'],
        'transmission_progress': app_state['transmission_progress']
    })

# 新增：发送时间进度API
@app.route('/api/transmission-time')
def get_transmission_time():
    return jsonify({
        'elapsed': app_state['transmission_time'].get('elapsed', 0),
        'total': app_state['transmission_time'].get('total', 0),
        'sent_samples': app_state['transmission_time'].get('sent_samples', 0),
        'progress': app_state['transmission_progress']
    })

@app.route('/api/stop-recording', methods=['POST'])
def stop_recording():
    app_state['stop_event'].set()
    if app_state['record_process']:
        try:
            app_state['record_process'].terminate()
        except Exception as e:
            app.logger.error(f"Error terminating record process: {str(e)}")
    return jsonify({'status': 'success', 'message': 'Recording stopped'})

@app.route('/api/start-transmission', methods=['POST'])
def start_transmission():
    if app_state['transmitting'] or app_state['recording']:
        return jsonify({'status': 'error', 'message': 'Already transmitting or recording'})
    
    data = request.json
    app_state['transmitting'] = True
    app_state['stop_event'].clear()
    app_state['transmission_progress'] = 0
    
    params = {
        'device_id': app_state['selected_device'],
        'filename': data.get('filename'),
        'frequency': data.get('frequency', DEFAULT_FREQUENCY),
        'sample_rate': data.get('sample_rate', DEFAULT_SAMPLE_RATE),
        'gain': data.get('gain', DEFAULT_GAIN),
        'channel': data.get('channel', DEFAULT_CHANNEL)
    }
    
    # 使用C程序进行发送
    app_state['transmit_thread'] = threading.Thread(target=c_transmit_thread_func, args=(params,))
    app_state['transmit_thread'].start()
    
    return jsonify({'status': 'success', 'message': 'Transmission started'})

@app.route('/api/stop-transmission', methods=['POST'])
def stop_transmission():
    app_state['stop_event'].set()
    if app_state['transmit_process']:
        try:
            app_state['transmit_process'].terminate()
        except Exception as e:
            app.logger.error(f"Error terminating transmit process: {str(e)}")
    return jsonify({'status': 'success', 'message': 'Transmission stopped'})

@app.route('/api/files')
def get_files():
    uploads = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    records = [f for f in os.listdir(RECORD_FOLDER) if os.path.isfile(os.path.join(RECORD_FOLDER, f))]
    return jsonify({
        'uploads': uploads,
        'records': records
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return jsonify({'status': 'success', 'filename': filename})
    
    return jsonify({'status': 'error', 'message': 'File upload failed'})

@app.route('/api/delete-file/<folder>/<filename>', methods=['DELETE'])
def delete_file(folder, filename):
    if folder not in ['uploads', 'records']:
        return jsonify({'status': 'error', 'message': 'Invalid folder'})
    
    folder_path = UPLOAD_FOLDER if folder == 'uploads' else RECORD_FOLDER
    filepath = os.path.join(folder_path, filename)
    
    if os.path.exists(filepath) and os.path.isfile(filepath):
        os.remove(filepath)
        return jsonify({'status': 'success', 'message': 'File deleted'})
    
    return jsonify({'status': 'error', 'message': 'File not found'})

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    if folder not in ['uploads', 'records']:
        return jsonify({'status': 'error', 'message': 'Invalid folder'}), 400
    
    folder_path = UPLOAD_FOLDER if folder == 'uploads' else RECORD_FOLDER
    return send_from_directory(folder_path, filename, as_attachment=True)

@app.route('/api/storage-status')
def get_storage_status():
    disk = shutil.disk_usage('/')
    total_mb = disk.total // (1024 * 1024)
    used_mb = disk.used // (1024 * 1024)
    available_mb = disk.free // (1024 * 1024)
    
    # 估算所需空间（基于最大可能录制时长和采样率）
    max_duration = 5  # 1小时
    max_sample_rate = 15.36e6  # 20 MS/s
    bytes_per_second = max_sample_rate * 4  # 每个样本4字节（IQ各16位）
    required_mb = (bytes_per_second * max_duration) // (1024 * 1024)
    
    enough = available_mb > required_mb * 1.2  # 留出20%余量
    
    return jsonify({
        'total_mb': total_mb,
        'used_mb': used_mb,
        'available_mb': available_mb,
        'required_mb': required_mb,
        'enough': enough
    })


# 添加存储检查接口
@app.route('/api/check-storage', methods=['POST'])
def check_storage():
    data = request.json
    required_mb = float(data.get('required_mb', 0))
    
    # 获取存储路径的磁盘信息
    disk = shutil.disk_usage(RECORD_FOLDER)
    
    # 计算可用空间 (MB)
    available_mb = disk.free / (1024 * 1024)
    # 至少保留1GB (1024MB) 空间
    required_reserve_mb = 1024
    # 实际可用空间 = 总可用空间 - 必须保留的1GB
    actual_available_mb = available_mb - required_reserve_mb
    
    # 检查是否足够
    enough = actual_available_mb >= required_mb
    
    return jsonify({
        'enough': enough,
        'required_mb': required_mb,
        'available_mb': available_mb,
        'actual_available_mb': actual_available_mb,
        'required_reserve_mb': required_reserve_mb
    })

# 增强录制接口的存储检查（双重保险）
@app.route('/api/start-recording', methods=['POST'])
def start_recording():
    if app_state['recording'] or app_state['transmitting']:
        return jsonify({'success': False, 'message': 'Device is busy'})
    
    if not app_state['selected_device']:
        return jsonify({'success': False, 'message': 'No device selected'})
    
    data = request.json
    
    # 计算所需空间
    duration = float(data.get('duration', 0))
    sample_rate = int(data.get('sample_rate', 0))
    required_mb = (duration * sample_rate * 8) / (1024 * 1024)  # 8字节/样本
    
    # 检查存储（双重保险）
    disk = shutil.disk_usage(RECORD_FOLDER)
    available_mb = disk.free / (1024 * 1024)
    if available_mb - 1024 < required_mb:  # 保留1GB
        return jsonify({
            'success': False,
            'message': f'存储空间不足！需要 {required_mb:.2f} MB，可用 {available_mb:.2f} MB（需保留1GB）'
        })
    
    # 原有录制逻辑...
    params = {
        'device_id': app_state['selected_device'],
        'filename': data.get('filename', f'record_{int(time.time())}'),
        'frequency': float(data.get('frequency', DEFAULT_FREQUENCY)),
        'duration': float(data.get('duration', 5)),
        'sample_rate': int(data.get('sample_rate', DEFAULT_SAMPLE_RATE)),
        'gain': float(data.get('gain', DEFAULT_GAIN)),
        'channel': int(data.get('channel', DEFAULT_CHANNEL))
    }
    
    app_state['recording'] = True
    app_state['stop_event'].clear()
    app_state['record_thread'] = threading.Thread(target=c_record_thread_func, args=(params,))
    app_state['record_thread'].start()
    
    return jsonify({'success': True, 'message': 'Recording started'})    

@app.route('/api/generate-spectrum', methods=['POST'])
def generate_spectrum():
    try:
        data = request.json
        filename = data.get('filename')
        sample_rate = int(data.get('sample_rate', 7680000))
        
        if not filename:
            return jsonify({'status': 'error', 'message': 'No filename provided'})
        
        # 查找文件
        full_path = None
        possible_paths = [
            os.path.join(RECORD_FOLDER, filename),
            os.path.join(UPLOAD_FOLDER, filename)
        ]
        
        if not any(filename.endswith(ext) for ext in ('.iq', '.bin', '.dat')):
            for ext in ('.iq', '.bin', '.dat'):
                possible_paths.insert(0, os.path.join(RECORD_FOLDER, filename + ext))
                possible_paths.append(os.path.join(UPLOAD_FOLDER, filename + ext))
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                full_path = path
                break
        
        if not full_path:
            return jsonify({'status': 'error', 'message': f'File {filename} not found'})
        
        # 读取IQ文件（只读取前250ms数据用于频谱分析，避免内存占用过大）
        max_bytes = sample_rate  # 250ms
        file_size = os.path.getsize(full_path)
        read_bytes = min(max_bytes, file_size)
        
        with open(full_path, 'rb') as f:
            iq_data = np.fromfile(f, dtype=np.int16, count=read_bytes // 2)
        
        # 转换为复数IQ信号
        iq_samples = iq_data[0::2] + 1j * iq_data[1::2]
        iq_samples = iq_samples.astype(np.complex64) / 32768.0  # 归一化
        
        # 计算FFT
        n = len(iq_samples)        
        fft_result = fft(iq_samples, n)
        fft_result = fftshift(fft_result) 
        
        nsamples_display = n
        while nsamples_display>1000:
            nsamples_display = nsamples_display // 2
        frequencies = np.linspace(-sample_rate/2, sample_rate/2, nsamples_display).tolist()
        
        # 计算幅度（dB）
        amplitudes = 20 * np.log10(np.abs(fft_result) + 1e-10)  # 加小值避免log(0)
        
        # 降采样以减少数据点数量，加快前端渲染
        downsample_factor = max(1, n // nsamples_display)  # 最多1000个点        
        amplitudes = amplitudes[::downsample_factor].tolist()
        
        return jsonify({
            'status': 'success',
            'frequencies': frequencies,
            'amplitudes': amplitudes
        })
        
    except Exception as e:
        app.logger.error(f"Error generating spectrum: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)