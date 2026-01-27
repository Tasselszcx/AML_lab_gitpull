import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time
import csv
from pathlib import Path
from datetime import datetime

# ================= 配置区域 =================
# 1. 修改为你的 Arduino 端口号 (Windows: 'COM3', Mac: '/dev/cu.usbmodem...')
SERIAL_PORT = 'COM5' 

# 2. 波特率必须与 Arduino 代码一致
BAUD_RATE = 115200

# 3. 想要在屏幕上显示多少个点 (调整窗口宽度)
MAX_POINTS = 200  

# 4. 采集时长：启动后自动运行多少秒，然后关闭并保存CSV
RECORD_SECONDS = 10

# 4a. 预热时长（秒）：启动监视后前N秒不写入CSV/PNG
WARMUP_SECONDS = 2.0

# 4b. 目标记录频率（Hz）：CSV里会按该频率输出等间隔时间轴
TARGET_HZ = 50

# 5. CSV输出目录（相对本脚本目录）
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# 6. shell里打印采集时间的频率（秒）；设为0可每条数据都打印（不建议）
CONSOLE_TIME_INTERVAL = 0.5
# ===========================================

# 初始化数据容器 (使用 deque 实现固定长度的滚动队列)
data_ch1 = deque([0] * MAX_POINTS, maxlen=MAX_POINTS)
data_ch2 = deque([0] * MAX_POINTS, maxlen=MAX_POINTS)

# 全局变量
labels = ["Channel 1", "Channel 2"] # 默认标签，会自动更新
is_running = True

# 记录数据（用于导出CSV）
# 注意：时间以“收到第一条有效数据”为t=0
start_mono = None
record_base_t = WARMUP_SECONDS
next_record_t = WARMUP_SECONDS
timer_started = False
global_fig = None
record_lock = threading.Lock()
record_times = []
record_ch1 = []
record_ch2 = []

# 尝试连接串口
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"成功连接到 {SERIAL_PORT}")
    ser.reset_input_buffer() # 清空缓存，防止读取积压数据
except Exception as e:
    print(f"错误: 无法打开串口 {SERIAL_PORT}。请检查端口号或关闭 Arduino IDE 的监视器。")
    print(f"详细错误: {e}")
    exit()

# ---- 后台线程：专门负责读取串口数据 ----
def read_serial_data():
    global labels, start_mono, record_base_t, next_record_t, timer_started
    last_console_print = -1.0
    sample_period = 1.0 / float(TARGET_HZ)
    max_records = int(RECORD_SECONDS * TARGET_HZ)
    while is_running:
        try:
            if ser.in_waiting:
                # 读取一行数据，解码并去除首尾空格
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # 数据格式预期: "Label1:Value1 \t Label2:Value2"
                if line and ":" in line:
                    parts = line.split('\t')
                    
                    if len(parts) >= 2:
                        # 解析第一列
                        label1, val1 = parts[0].split(':')
                        # 解析第二列
                        label2, val2 = parts[1].split(':')
                        
                        # 更新数据
                        v1 = float(val1)
                        v2 = float(val2)

                        data_ch1.append(v1)
                        data_ch2.append(v2)

                        # 初始化时间基准（首个有效样本时刻 = t=0）
                        if start_mono is None:
                            start_mono = time.perf_counter()
                            record_base_t = float(WARMUP_SECONDS)
                            next_record_t = float(WARMUP_SECONDS)
                            # 自动停止：预热 + 记录时长，从第一条有效数据开始计时
                            if (not timer_started) and (global_fig is not None):
                                timer = threading.Timer(WARMUP_SECONDS + RECORD_SECONDS, stop_program, kwargs={"fig": global_fig})
                                timer.daemon = True
                                timer.start()
                                timer_started = True

                        # 记录到CSV：跳过预热段，并按固定50Hz时间轴输出（用最新数据填充）
                        t_now = time.perf_counter() - start_mono
                        if t_now >= record_base_t:
                            with record_lock:
                                while t_now >= next_record_t:
                                    if len(record_times) >= max_records:
                                        break
                                    record_times.append(next_record_t - record_base_t)
                                    record_ch1.append(v1)
                                    record_ch2.append(v2)
                                    next_record_t += sample_period

                        # 实时输出采集时间（限频，避免刷屏）
                        if start_mono is not None:
                            t_print = time.perf_counter() - start_mono
                            if CONSOLE_TIME_INTERVAL <= 0 or (t_print - last_console_print) >= CONSOLE_TIME_INTERVAL:
                                if t_print < record_base_t:
                                    print(f"采集时间: {t_print:7.2f}s (预热中...)", end='\r', flush=True)
                                else:
                                    print(f"采集时间: {t_print:7.2f}s", end='\r', flush=True)
                                last_console_print = t_print
                        
                        # 更新图例标签 (只做一次)
                        if labels[0] == "Channel 1":
                            labels[0] = label1
                            labels[1] = label2
                            print(f"检测到数据源: {label1} 和 {label2}")

        except ValueError:
            pass # 忽略解析错误的行（比如启动时的乱码）
        except Exception as e:
            print(f"读取错误: {e}")
            break

    # 补一个换行，避免\r覆盖导致最后一行粘连
    print()

# ---- 绘图动画函数 ----
def animate(i):
    ax.clear()
    
    # 绘制两条线
    ax.plot(data_ch1, label=labels[0], color='blue', linewidth=1.5)
    ax.plot(data_ch2, label=labels[1], color='red', linewidth=1.5, alpha=0.8)
    
    # 图表美化
    ax.legend(loc='upper right')
    ax.set_title(f'Real-time Sensor Data: {labels[0]} & {labels[1]}')
    ax.set_xlabel('Time (Samples)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Y轴范围可以自动适应，或者手动固定 (如果波形跳动太厉害，可以取消下面注释固定范围)
    # ax.set_ylim(-1000, 1000) 


def stop_program(fig=None):
    global is_running
    if not is_running:
        return
    is_running = False
    try:
        if ser and ser.is_open:
            ser.close()
    except Exception:
        pass
    try:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close('all')
    except Exception:
        pass


def write_csv():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"eog_capture_{ts}.csv"

    with record_lock:
        rows = list(zip(record_times, record_ch1, record_ch2))
        label1, label2 = labels[0], labels[1]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", label1, label2])
        writer.writerows(rows)

    print(f"已保存CSV: {out_path}")
    return ts


def write_png(ts: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"eog_capture_{ts}.png"

    with record_lock:
        times = list(record_times)
        ch1 = list(record_ch1)
        ch2 = list(record_ch2)
        label1, label2 = labels[0], labels[1]

    if not times:
        print("没有可保存的波形数据，跳过PNG导出")
        return


    fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=150)
    ax2.plot(times, ch1, label=label1, color='blue', linewidth=1.2)
    ax2.plot(times, ch2, label=label2, color='red', linewidth=1.2, alpha=0.85)
    ax2.set_title(f"EOG Capture ({times[-1]:.1f}s)\n{label1} & {label2}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    # 固定y轴范围为中心±100单位，总范围200单位
    all_y = ch1 + ch2
    if all_y:
        y_min = min(all_y)
        y_max = max(all_y)
        y_center = (y_min + y_max) / 2
        y_min_tick = int((y_center - 150) // 50) * 50
        y_max_tick = int((y_center + 150) // 50) * 50
        ax2.set_ylim(y_min_tick, y_max_tick)
        ax2.set_yticks(list(range(y_min_tick, y_max_tick + 1, 50)))
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right')
    fig2.tight_layout()
    fig2.savefig(out_path)
    plt.close(fig2)

    print(f"已保存PNG: {out_path}")

# ---- 主程序 ----
if __name__ == "__main__":
    # 设置绘图窗口
    fig, ax = plt.subplots(figsize=(10, 6))

    # 提供给采集线程用于到点关闭
    global_fig = fig

    # 启动后台线程读取数据（放到fig创建之后，确保能启动自动停止计时器）
    thread = threading.Thread(target=read_serial_data)
    thread.daemon = True
    thread.start()
    
    # 启动动画 (interval=50 表示每10ms刷新一次画面)
    ani = animation.FuncAnimation(fig, animate, interval=10)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        stop_program(fig)
        print("程序已停止")
    finally:
        # 确保线程退出 & 落盘
        is_running = False
        try:
            thread.join(timeout=2)
        except Exception:
            pass
        try:
            ts = write_csv()
            write_png(ts)
        except Exception as e:
            print(f"保存CSV失败: {e}")