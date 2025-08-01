import os
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
import socket
import select
import re

# -------------------- Flask 应用初始化 --------------------
app = Flask(__name__)
CORS(app)

# -------------------- 路径与参数配置 --------------------
UPLOAD_FOLDER = 'uploads'
RECORD_FOLDER = 'records'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECORD_FOLDER, exist_ok=True)

RX_UDP_C_PROGRAM = "/usr/lib/uhd/examples/rx_samples_to_udp"
RX_C_PROGRAM = "/usr/lib/uhd/examples/rx_samples_to_file"
TX_C_PROGRAM = "/usr/lib/uhd/examples/tx_samples_from_file"

# -------------------- 实时频谱采集配置 --------------------
CAPTURE_SERVER = '127.0.0.1'
CAPTURE_UDP_PORT = 12345
CAPTURE_MAX_PKT = 8192
CAPTURE_NFFT = 2048

DEFAULT_FREQUENCY = 634000000  # 634 MHz
DEFAULT_SAMPLE_RATE = 7000000  # 7 MHz
DEFAULT_GAIN = 70.0
DEFAULT_CHANNEL = 0

MAX_SAMPLES_PER_PACKET = 32768
NUM_BUFFERS = 16
PACKET_SIZE = MAX_SAMPLES_PER_PACKET * 4
FILE_READ_CHUNK = NUM_BUFFERS * PACKET_SIZE
STREAM_TIMEOUT = 0.1
PROGRESS_LOG_INTERVAL = 10

# -------------------- 全局状态 --------------------
app_state = {
    'recording': False,
    'transmitting': False,
    'capturing': False,      # 运行标志

    'selected_device': None,
    'stop_event': threading.Event(),
    'transmission_progress': 0,

    'record_process': None,
    'transmit_process': None,
    'capture_process': None, 

    'record_thread': None,
    'transmit_thread': None,
    'capture_thread': None,  

    'transmission_time': {'elapsed': 0, 'total': 0, 'sent_samples': 0},
    'transmission_timer': None,
    'transmission_start_time': 0,

    'usrp_status': 'OK',
    'usrp_status_msg': '',   
    
    
    'rt_last_fft': None      # 缓存最近一次 FFT 结果
}

# USRP 增益限制表
USRP_GAIN_LIMITS = {
    "B200":   {"tx": 89.8, "rx": 73.0},
    "B210":   {"tx": 89.8, "rx": 73.0},
    "B200mini": {"tx": 89.8, "rx": 73.0},
    "B205mini": {"tx": 89.8, "rx": 73.0},
    "N200":   {"tx": 31.5, "rx": 31.5},
    "N210":   {"tx": 31.5, "rx": 31.5},
    "N300":   {"tx": 31.5, "rx": 31.5},
    "N310":   {"tx": 31.5, "rx": 31.5},
    "N320":   {"tx": 31.5, "rx": 31.5},
    "N321":   {"tx": 31.5, "rx": 31.5},
    "X300":   {"tx": 31.5, "rx": 31.5},
    "X310":   {"tx": 31.5, "rx": 31.5},
    "X410":   {"tx": 60.0, "rx": 60.0},
    "X440":   {"tx": 60.0, "rx": 60.0},
    "E310":   {"tx": 89.8, "rx": 73.0},
    "E312":   {"tx": 89.8, "rx": 73.0},
    "E313":   {"tx": 89.8, "rx": 73.0},
    "E320":   {"tx": 60.0, "rx": 60.0},
    "USRP-2974": {"tx": 31.5, "rx": 31.5}
}

# -------------------- 设备发现 --------------------
def discover_devices():
    """调用 uhd_find_devices 发现USRP设备"""
    try:
        raw = subprocess.check_output(
            ["uhd_find_devices"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5
        )
    except Exception:
        return []

    devices = []
    pattern = re.compile(
        r'UHD Device\s*\d+\s*\n'
        r'(?P<block>.*?)'
        r'(?=\nUHD Device\s*\d+|\Z)',
        re.DOTALL
    )
    for match in pattern.finditer(raw):
        blk = match.group('block').strip()
        if not blk:
            continue
        info = {}
        for line in blk.splitlines():
            if ':' in line:
                key, val = map(str.strip, line.split(':', 1))
                info[key.lower()] = val
        if any(info.values()):
            max_tx_gain = USRP_GAIN_LIMITS.get(info.get("product", ""), {}).get("tx", 80)
            max_rx_gain = USRP_GAIN_LIMITS.get(info.get("product", ""), {}).get("rx", 70)
            devices.append({
                "serial":  info.get("serial", ""),
                "name":    info.get("name", ""),
                "product": info.get("product", ""),
                "type":    info.get("type", ""),
                "addr":    info.get("addr", ""),
                "max_tx_gain": max_tx_gain,
                "max_rx_gain": max_rx_gain
            })
    return devices

# -------------------- IQ数据保存 --------------------
def save_iq_data(samples, filename, chunk_size=1048576):
    """保存IQ数据为INT16格式文件，分块写入"""
    if not filename.endswith(('.iq', '.bin', '.dat')):
        filename += '.iq'
    full_path = os.path.join(RECORD_FOLDER, filename)
    total_samples = len(samples)
    samples_per_chunk = chunk_size // 4
    with open(full_path, 'wb', buffering=8*1024*1024) as f:
        for i in range(0, total_samples, samples_per_chunk):
            end_idx = min(i + samples_per_chunk, total_samples)
            chunk = samples[i:end_idx]
            iq_data = np.empty(2 * len(chunk), dtype=np.int16)
            iq_data[0::2] = (chunk.real * 32767).astype(np.int16)
            iq_data[1::2] = (chunk.imag * 32767).astype(np.int16)
            f.write(iq_data.tobytes())
    return full_path

# -------------------- 录制线程（C程序） --------------------
def c_record_thread_func(params):
    """调用 UHD C 示例程序进行录制，支持实时状态监控"""
    try:
        filename = params['filename']
        if not filename.endswith(('.iq', '.bin', '.dat')):
            filename += '.iq'
        full_path = os.path.join(RECORD_FOLDER, filename)
        args = [
            RX_C_PROGRAM,
            f"--args=serial={app_state['selected_device']}",
            f"--freq={params['frequency']}",
            f"--rate={params['sample_rate']}",
            f"--gain={params['gain']}",
            f"--duration={params['duration']}",
            f"--channel={params.get('channel', DEFAULT_CHANNEL)}",
            f"--file={full_path}"
        ]
        app_state['record_process'] = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        fd = app_state['record_process'].stdout.fileno()
        os.set_blocking(fd, False)
        while True:
            if app_state['stop_event'].is_set():
                break
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(fd, 128).decode('utf-8', errors='ignore')
                    if not chunk:
                        break
                    if 'O' in chunk:
                        app_state['usrp_status'] = 'O'
                        app_state['usrp_status_msg'] = 'Overrun'
                    if 'U' in chunk:
                        app_state['usrp_status'] = 'U'
                        app_state['usrp_status_msg'] = 'Underrun'
                except BlockingIOError:
                    pass
            else:
                app_state['usrp_status'] = 'OK'
                app_state['usrp_status_msg'] = ''
        app_state['record_process'].wait()
    except Exception as e:
        app.logger.error(f"C程序录制错误: {str(e)}")
    finally:
        if app_state['record_process']:
            try:
                app_state['record_process'].terminate()
                app_state['record_process'].wait(timeout=2)
            except:
                pass
        app_state['usrp_status'] = 'OK'
        app_state['usrp_status_msg'] = ''
        app_state['recording'] = False
        app_state['stop_event'].clear()
        app_state['record_thread'] = None
        app_state['record_process'] = None

# -------------------- 发送线程（C程序） --------------------
def c_transmit_thread_func(params):
    """调用 UHD C 示例程序进行发送，支持实时状态监控和进度估算"""
    app_state['transmission_progress'] = 0
    app_state['transmission_time'] = {'elapsed': 0, 'total': 0, 'sent_samples': 0}
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
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                full_path = path
                break
        if not full_path:
            return
        file_size = os.path.getsize(full_path)
        total_samples = file_size // 4
        sample_rate = params.get('sample_rate', DEFAULT_SAMPLE_RATE)
        total_duration = total_samples / sample_rate
        app_state['transmission_time']['total'] = total_duration
        args = [
            TX_C_PROGRAM,
            f"--args=serial={app_state['selected_device']}",
            f"--freq={params['frequency']}",
            f"--rate={params['sample_rate']}",
            f"--gain={params['gain']}",
            f"--channel={params.get('channel', DEFAULT_CHANNEL)}",
            f"--file={full_path}"
        ]
        app_state['transmit_process'] = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
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
        fd = app_state['transmit_process'].stdout.fileno()
        os.set_blocking(fd, False)
        while True:
            if app_state['stop_event'].is_set():
                break
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(fd, 128).decode('utf-8', errors='ignore')
                    if not chunk:
                        break
                    if 'O' in chunk:
                        app_state['usrp_status'] = 'O'
                        app_state['usrp_status_msg'] = 'Overrun'
                    if 'U' in chunk:
                        app_state['usrp_status'] = 'U'
                        app_state['usrp_status_msg'] = 'Underrun'
                except BlockingIOError:
                    pass
            else:
                app_state['usrp_status'] = 'OK'
                app_state['usrp_status_msg'] = ''
        app_state['transmit_process'].wait()
        app_state['transmission_progress'] = 100
    except Exception as e:
        app.logger.error(f"C程序发送错误: {str(e)}")
    finally:
        if app_state['transmit_process']:
            try:
                app_state['transmit_process'].terminate()
                app_state['transmit_process'].wait(timeout=2)
            except:
                pass
        app_state['usrp_status'] = 'OK'
        app_state['usrp_status_msg'] = ''
        app_state['transmission_timer'] = None
        app_state['transmitting'] = False
        app_state['stop_event'].clear()
        app_state['transmit_thread'] = None
        app_state['transmit_process'] = None


# -------------------- 实时捕获线程（C程序） --------------------
def c_capture_thread_func(params):
    """实时频谱捕获线程，UDP收包并做FFT"""
    # 若有旧进程，先安全终止
    if app_state['capture_process']:
        try:
            app_state['capture_process'].terminate()
            app_state['capture_process'].wait(timeout=2)
        except Exception as e:
            app.logger.warning(f"Terminate old capture_process failed: {e}")
    subdev_map = ["A:A", "A:B"]
    subdev = subdev_map[int(params['channel']) % 2]
    cmd = [
        RX_UDP_C_PROGRAM,
        f"--args=serial={app_state['selected_device']}",
        f"--freq={params['freq']}",
        f"--rate={params['rate']}",
        f"--gain={params['gain']}",
        f"--bw={params['bw']}",
        f"--subdev={subdev}",
        f"--addr={CAPTURE_SERVER}",
        f"--port={CAPTURE_UDP_PORT}",
        "--nsamps=200000000"  # 接近最大值，避免过早退出
    ]
    app_state['capture_process'] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((CAPTURE_SERVER, CAPTURE_UDP_PORT))
    sock.settimeout(0.5)
    try:
        while app_state['capturing'] and app_state['capture_process'].poll() is None:
            try:
                data, _ = sock.recvfrom(CAPTURE_MAX_PKT)
            except socket.timeout:
                continue
            n = len(data) // 8
            if n == 0:
                continue
            iq = np.frombuffer(data, dtype=np.complex64, count=n)
            center_freq = params['freq']
            fft_data = np.fft.fftshift(np.fft.fft(iq, n=CAPTURE_NFFT))
            psd = 20 * np.log10(np.abs(fft_data) + 1e-12)
            freqs = (np.arange(-CAPTURE_NFFT//2, CAPTURE_NFFT//2) * (float(params['rate']) / CAPTURE_NFFT))
            freqs = (freqs + center_freq).tolist()
            app_state['rt_last_fft'] = {'freqs': freqs, 'psd': psd.tolist()}
    finally:
        sock.close()
        if app_state['capture_process']:
            try:
                app_state['capture_process'].terminate()
                app_state['capture_process'].wait()
            except Exception as e:
                app.logger.warning(f"Terminate capture_process failed: {e}")
        app_state['capture_process'] = None
        app_state['capture_thread'] = None
        app_state['capturing'] = False


# -------------------- API 路由 --------------------
# -------------------- 设备管理相关 API（设备选择卡） --------------------
@app.route('/api/devices')
def get_devices():
    devices = discover_devices()
    return jsonify({'devices': devices, 'selected': app_state['selected_device']})


@app.route('/api/select-device', methods=['POST'])
def select_device():
    data = request.json
    device_serial = data.get('device_serial')
    device_product = data.get('device_product')
    app_state['selected_device'] = device_serial
    max_tx_gain = USRP_GAIN_LIMITS.get(device_product, {}).get("tx", 80)
    max_rx_gain = USRP_GAIN_LIMITS.get(device_product, {}).get("rx", 70)
    return jsonify({
        'status': 'success',
        'selected': device_serial,
        'max_tx_gain': max_tx_gain,
        'max_rx_gain': max_rx_gain
    })


@app.route('/api/usrp-status')
def get_usrp_status():
    return jsonify({
        'status': app_state['usrp_status'],
        'msg':    app_state['usrp_status_msg']
    })


@app.route('/api/status')
def get_status():
    return jsonify({
        'recording': app_state['recording'],
        'transmitting': app_state['transmitting'],
        'capturing': app_state.get('capturing', False),
        'selected_device': app_state['selected_device'],
        'transmission_progress': app_state['transmission_progress']
    })



# -------------------- 录制相关 API（Record Signal Tab） --------------------
@app.route('/api/start-recording', methods=['POST'])
def start_recording():
    if app_state['recording'] or app_state['transmitting']:
        return jsonify({'success': False, 'message': 'Device is busy'})
    if not app_state['selected_device']:
        return jsonify({'success': False, 'message': 'No device selected'})
    data = request.json
    duration = float(data.get('duration', 0))
    sample_rate = int(data.get('sample_rate', 0))
    required_mb = (duration * sample_rate * 8) / (1024 * 1024)
    disk = shutil.disk_usage(RECORD_FOLDER)
    available_mb = disk.free / (1024 * 1024)
    if available_mb - 1024 < required_mb:
        return jsonify({
            'success': False,
            'message': f'存储空间不足！需要 {required_mb:.2f} MB，可用 {available_mb:.2f} MB（需保留1GB）'
        })
    app_state['recording'] = True
    app_state['stop_event'].clear()
    app_state['record_thread'] = threading.Thread(target=c_record_thread_func, args=(data,))
    app_state['record_thread'].start()
    return jsonify({'success': True, 'message': 'Recording started'})


@app.route('/api/stop-recording', methods=['POST'])
def stop_recording():
    app_state['stop_event'].set()
    if app_state['record_process']:
        try:
            app_state['record_process'].terminate()
        except Exception as e:
            app.logger.error(f"Error terminating record process: {str(e)}")
    return jsonify({'status': 'success', 'message': 'Recording stopped'})


@app.route('/api/storage-status')
def get_storage_status():
    disk = shutil.disk_usage(RECORD_FOLDER)
    total_mb = disk.total // (1024 * 1024)
    used_mb = disk.used // (1024 * 1024)
    available_mb = disk.free // (1024 * 1024)
    max_duration = 5
    max_sample_rate = 15.36e6
    bytes_per_second = max_sample_rate * 4
    required_mb = (bytes_per_second * max_duration) // (1024 * 1024)
    enough = available_mb > required_mb * 1.2
    return jsonify({
        'total_mb': total_mb,
        'used_mb': used_mb,
        'available_mb': available_mb,
        'required_mb': required_mb,
        'enough': enough
    })


@app.route('/api/check-storage', methods=['POST'])
def check_storage():
    data = request.json
    required_mb = float(data.get('required_mb', 0))
    disk = shutil.disk_usage(RECORD_FOLDER)
    available_mb = disk.free / (1024 * 1024)
    required_reserve_mb = 1024
    actual_available_mb = available_mb - required_reserve_mb
    enough = actual_available_mb >= required_mb
    return jsonify({
        'enough': enough,
        'required_mb': required_mb,
        'available_mb': available_mb,
        'actual_available_mb': actual_available_mb,
        'required_reserve_mb': required_reserve_mb
    })



# -------------------- 发送相关 API（Transmit Signal Tab） --------------------
@app.route('/api/start-transmission', methods=['POST'])
def start_transmission():
    if app_state['transmitting'] or app_state['recording']:
        return jsonify({'status': 'error', 'message': 'Already transmitting or recording'})
    data = request.json
    app_state['transmitting'] = True
    app_state['stop_event'].clear()
    app_state['transmission_progress'] = 0
    app_state['transmit_thread'] = threading.Thread(target=c_transmit_thread_func, args=(data,))
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


@app.route('/api/transmission-time')
def get_transmission_time():
    return jsonify({
        'elapsed': app_state['transmission_time'].get('elapsed', 0),
        'total': app_state['transmission_time'].get('total', 0),
        'sent_samples': app_state['transmission_time'].get('sent_samples', 0),
        'progress': app_state['transmission_progress']
    })



# -------------------- 实时频谱采集相关 API（Realtime Capture Tab） --------------------
@app.route('/api/start-capturing', methods=['POST'])
def start_capturing():
    if app_state['capturing'] or app_state['recording'] or app_state['transmitting']:
        return jsonify({'status': 'error', 'msg': 'Device is busy'})
    if not app_state['selected_device']:
        return jsonify({'status': 'error', 'msg': 'No device selected'})
    data = request.json
    param = dict(data)
    if not param.get('device_serial'):
        param['device_serial'] = app_state['selected_device']
    app_state['capturing'] = True
    app_state['stop_event'].clear()
    app_state['capture_thread'] = threading.Thread(target=c_capture_thread_func, args=(param,))
    app_state['capture_thread'].start()
    return jsonify({'status': 'ok'})

@app.route('/api/stop-capturing', methods=['POST'])
def stop_capturing():
    app_state['capturing'] = False
    if app_state['capture_process']:
        try:
            app_state['capture_process'].terminate()
        except Exception as e:
            app.logger.warning(f"Terminate capture_process failed: {e}")
    return jsonify({'status': 'ok'})

@app.route('/api/capture-spectrum')
def capture_spectrum():
    if app_state['rt_last_fft'] is None:
        return jsonify({'status': 'no-data'})
    return jsonify(app_state['rt_last_fft'])

# -------------------- 文件管理相关 API（File Management Tab） --------------------
@app.route('/api/files')
def get_files():
    uploads = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    records = [
        {'name': f, 'time': os.path.getmtime(os.path.join(RECORD_FOLDER, f))}
        for f in os.listdir(RECORD_FOLDER)
        if os.path.isfile(os.path.join(RECORD_FOLDER, f))
    ]
    records.sort(key=lambda x: x['time'], reverse=True)
    sorted_records = [item['name'] for item in records]
    return jsonify({'uploads': uploads, 'records': sorted_records})

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

# -------------------- 频谱分析相关 API（Spectrum Analyzer Tab） --------------------
@app.route('/api/generate-spectrum', methods=['POST'])
def generate_spectrum():
    try:
        data = request.json
        filename = data.get('filename')
        sample_rate = int(data.get('sample_rate', 7680000))
        if not filename:
            return jsonify({'status': 'error', 'message': 'No filename provided'})
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
        max_bytes = sample_rate
        file_size = os.path.getsize(full_path)
        read_bytes = min(max_bytes, file_size)
        with open(full_path, 'rb') as f:
            iq_data = np.fromfile(f, dtype=np.int16, count=read_bytes // 2)
        iq_samples = iq_data[0::2] + 1j * iq_data[1::2]
        iq_samples = iq_samples.astype(np.complex64) / 32768.0
        n = len(iq_samples)
        fft_result = fftshift(fft(iq_samples, n))
        nsamples_display = n
        while nsamples_display > 1000:
            nsamples_display = nsamples_display // 2
        frequencies = np.linspace(-sample_rate/2, sample_rate/2, nsamples_display).tolist()
        amplitudes = 20 * np.log10(np.abs(fft_result) + 1e-10)
        downsample_factor = max(1, n // nsamples_display)
        amplitudes = amplitudes[::downsample_factor].tolist()
        return jsonify({
            'status': 'success',
            'frequencies': frequencies,
            'amplitudes': amplitudes
        })
    except Exception as e:
        app.logger.error(f"Error generating spectrum: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

# -------------------- 首页 --------------------
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)