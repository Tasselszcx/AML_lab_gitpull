"""
MuMu Bridge — ESP32 串口信号 → ADB 控制 MuMu 安卓虚拟机
=========================================================
读取 ESP32 的串口输出（EOG 命令 + IMU 陀螺仪数据），
通过 ADB 控制 MuMu 模拟器。

映射关系：
  EOG 上看 → 页面上滑 (scroll up)
  EOG 下看 → 页面下滑 (scroll down)
  EOG 左看 → 点击当前光标位置（光标在边缘时变为翻页）
  EOG 右看 → 返回 (Android Back)
  IMU 扭头 → 虚拟光标移动
  Blink   → 不使用（预留）

使用方法：
  python mumu_bridge.py                    # 默认 COM3
  python mumu_bridge.py --port COM5        # 指定串口
  python mumu_bridge.py --sim              # 无 ESP32 时用键盘模拟测试
"""

import subprocess
import time
import sys
import os
import argparse
import threading

# ============================================================
# 配置
# ============================================================
SCREEN_W = 1080
SCREEN_H = 1920
ADB_PATH = r"D:\Program Files\Netease\MuMu\nx_main\adb.exe"
ADB_DEVICE = "127.0.0.1:7555"  # MuMu 默认设备，多设备时必须指定

SERIAL_PORT = 'COM3'
BAUD_RATE = 115200

# 滑动参数
SWIPE_DURATION = 300       # 滑动持续毫秒
SCROLL_MARGIN_RATIO = 0.25 # 上下滑动距离占屏幕的比例

# 翻页边缘区域（左右 5%）
EDGE_RATIO = 0.05

# IMU 光标参数
IMU_SENSITIVITY = 0.02     # 与 .ino 中的 SENSITIVITY 一致
IMU_DEADZONE = 250         # 与 .ino 中的 DEADZONE 一致
CURSOR_SPEED = 0.5         # 陀螺仪原始值 → 像素的缩放系数

# 冷却时间（秒）—— 防止连续触发（全局锁定）
COOLDOWN_SEC = 1.0


# ============================================================
# ADB 控制函数
# ============================================================
def check_adb():
    """检查 ADB 连接状态。"""
    try:
        result = subprocess.run(
            [ADB_PATH, "devices"], capture_output=True, text=True, timeout=5
        )
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        devices = [l for l in lines[1:] if 'device' in l and 'offline' not in l]
        if devices:
            print(f"  ADB 已连接 {len(devices)} 个设备")
            return True
        else:
            print("  [警告] 未检测到 ADB 设备")
            return False
    except FileNotFoundError:
        print(f"  [错误] 未找到 adb: {ADB_PATH}")
        return False


def adb_shell(cmd):
    """异步执行 adb shell 命令。"""
    subprocess.Popen(
        [ADB_PATH, "-s", ADB_DEVICE, "shell", cmd],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def tap_at(x, y):
    """点击指定坐标。"""
    x, y = int(x), int(y)
    print(f"  >> 点击 ({x}, {y})")
    adb_shell(f"input tap {x} {y}")


def swipe(x1, y1, x2, y2, duration=SWIPE_DURATION):
    """滑动。"""
    print(f"  >> 滑动 ({x1},{y1}) → ({x2},{y2})")
    adb_shell(f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {duration}")


def adb_back():
    """Android 返回键。"""
    print("  >> 返回")
    adb_shell("input keyevent 4")


# ============================================================
# 虚拟光标控制器
# ============================================================
class CursorController:
    """根据 IMU 陀螺仪数据维护虚拟光标位置。"""

    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.x = screen_w / 2.0
        self.y = screen_h / 2.0

    def update(self, gx, gy):
        """
        用陀螺仪原始值更新光标位置。
        gx/gy 是 .ino 中经过死区过滤后的 int16 值。
        """
        self.x += gx * CURSOR_SPEED
        self.y += gy * CURSOR_SPEED
        self.x = max(0, min(self.screen_w - 1, self.x))
        self.y = max(0, min(self.screen_h - 1, self.y))
        return int(self.x), int(self.y)

    def recenter(self):
        self.x = self.screen_w / 2.0
        self.y = self.screen_h / 2.0


# ============================================================
# EOG 命令执行器（含边缘翻页逻辑）
# ============================================================
class ActionExecutor:
    """根据 EOG 命令 + 光标位置执行 ADB 操作。"""

    def __init__(self, screen_w, screen_h, cursor: CursorController):
        self.w = screen_w
        self.h = screen_h
        self.cursor = cursor
        self.cx = screen_w // 2
        self.cy = screen_h // 2
        self.scroll_dist = int(screen_h * SCROLL_MARGIN_RATIO)
        self.edge_left = int(screen_w * EDGE_RATIO)
        self.edge_right = int(screen_w * (1 - EDGE_RATIO))
        self.swipe_margin_x = screen_w * 3 // 8
        self._last_time = None  # 全局冷却计时器

    def _cooled(self, action):
        now = time.time()
        if self._last_time is not None:
            if now - self._last_time < COOLDOWN_SEC:
                return False
        self._last_time = now
        return True

    def execute(self, cmd):
        """执行 EOG 命令。返回描述字符串或 None。"""
        if cmd == "Rest" or cmd == "Blink":
            return None

        if not self._cooled(cmd):
            return None

        if cmd == "Up":
            # 页面上滑（手指从下往上）
            swipe(self.cx, self.cy + self.scroll_dist,
                  self.cx, self.cy - self.scroll_dist)
            return "上滑"

        elif cmd == "Down":
            # 页面下滑（手指从上往下）
            swipe(self.cx, self.cy - self.scroll_dist,
                  self.cx, self.cy + self.scroll_dist)
            return "下滑"

        elif cmd == "Left":
            # 点击 —— 但光标在边缘时变为翻页
            cur_x = int(self.cursor.x)
            cur_y = int(self.cursor.y)

            if cur_x < self.edge_left:
                # 左边缘 → 左翻页（从左往右滑）
                swipe(self.cx - self.swipe_margin_x, self.cy,
                      self.cx + self.swipe_margin_x, self.cy)
                return "左翻页"
            elif cur_x > self.edge_right:
                # 右边缘 → 右翻页（从右往左滑）
                swipe(self.cx + self.swipe_margin_x, self.cy,
                      self.cx - self.swipe_margin_x, self.cy)
                return "右翻页"
            else:
                # 正常点击
                tap_at(cur_x, cur_y)
                return f"点击({cur_x},{cur_y})"

        elif cmd == "Right":
            adb_back()
            return "返回"

        return None


# ============================================================
# 串口解析
# ============================================================
def parse_serial_line(line):
    """
    解析串口一行数据。
    格式: H:512.0\tV:510.0\tCMD:Rest\tGX:0\tGY:0
    返回 (cmd, gx, gy) 或 None。
    """
    try:
        parts = line.split('\t')
        data = {}
        for p in parts:
            if ':' in p:
                k, v = p.split(':', 1)
                data[k.strip()] = v.strip()

        cmd = data.get('CMD', 'Rest')
        gx = int(float(data.get('GX', '0')))
        gy = int(float(data.get('GY', '0')))
        return cmd, gx, gy
    except (ValueError, KeyError):
        return None


# ============================================================
# 主循环 — 串口实时模式
# ============================================================
def serial_mode(port):
    """连接 ESP32 串口，实时控制 MuMu。"""
    import serial

    cursor = CursorController(SCREEN_W, SCREEN_H)
    executor = ActionExecutor(SCREEN_W, SCREEN_H, cursor)

    print(f"\n  连接串口: {port} @ {BAUD_RATE}")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"  [错误] 串口打开失败: {e}")
        return
    ser.reset_input_buffer()

    print("  开始控制 MuMu... (Ctrl+C 退出)")
    print(f"  边缘翻页区域: 左<{int(SCREEN_W * EDGE_RATIO)}px, 右>{int(SCREEN_W * (1 - EDGE_RATIO))}px\n")

    try:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                parsed = parse_serial_line(line)
                if parsed is None:
                    continue

                cmd, gx, gy = parsed

                # 更新光标
                cx, cy = cursor.update(gx, gy)

                # 执行 EOG 命令
                desc = executor.execute(cmd)
                if desc:
                    print(f"  [{cmd}] → {desc}  光标({cx},{cy})")
    except KeyboardInterrupt:
        print("\n  退出。")
    finally:
        ser.close()


# ============================================================
# 串口模式 + Tkinter 可视化
# ============================================================
def visual_mode(port):
    """带截屏镜像的串口控制模式。"""
    import serial
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw
    from io import BytesIO

    CANVAS_SCALE = 4
    cw = SCREEN_W // CANVAS_SCALE
    ch = SCREEN_H // CANVAS_SCALE

    cursor = CursorController(SCREEN_W, SCREEN_H)
    executor = ActionExecutor(SCREEN_W, SCREEN_H, cursor)
    running = [True]
    latest_bg = [None]
    status_text = ["等待数据..."]

    # 串口连接
    print(f"\n  连接串口: {port} @ {BAUD_RATE}")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
    except Exception as e:
        print(f"  [错误] 串口打开失败: {e}")
        return

    # 截屏线程
    def bg_capture():
        while running[0]:
            try:
                result = subprocess.run(
                    [ADB_PATH, "-s", ADB_DEVICE, "exec-out", "screencap", "-p"],
                    capture_output=True, timeout=3
                )
                if result.stdout:
                    img = Image.open(BytesIO(result.stdout))
                    latest_bg[0] = img.resize((cw, ch), Image.LANCZOS)
            except Exception:
                pass
            time.sleep(0.15)

    # 串口读取线程
    def serial_read():
        while running[0]:
            try:
                if ser.in_waiting:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    parsed = parse_serial_line(line)
                    if parsed is None:
                        continue
                    cmd, gx, gy = parsed
                    cx, cy = cursor.update(gx, gy)
                    desc = executor.execute(cmd)
                    if desc:
                        status_text[0] = f"[{cmd}] → {desc}"
                        print(f"  {status_text[0]}  光标({cx},{cy})")
                else:
                    time.sleep(0.01)
            except Exception as e:
                if running[0]:
                    print(f"  串口错误: {e}")
                break

    threading.Thread(target=bg_capture, daemon=True).start()
    threading.Thread(target=serial_read, daemon=True).start()

    # Tkinter UI
    root = tk.Tk()
    root.title("MuMu Bridge — EOG + IMU 控制")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=cw, height=ch, bg="#1a1a2e")
    canvas.pack()

    status_var = tk.StringVar(value="等待信号...")
    tk.Label(root, textvariable=status_var, font=("Consolas", 10),
             bg="#16213e", fg="white", anchor="w", padx=5).pack(fill="x")

    coord_var = tk.StringVar(value=f"光标: ({SCREEN_W // 2}, {SCREEN_H // 2})")
    tk.Label(root, textvariable=coord_var, font=("Consolas", 10),
             bg="#0f3460", fg="#e94560", anchor="w", padx=5).pack(fill="x")

    tk_img_ref = [None]
    bg_image_id = canvas.create_image(0, 0, anchor="nw")

    # 绘制边缘区域标记
    edge_left_px = int(SCREEN_W * EDGE_RATIO) // CANVAS_SCALE
    edge_right_px = int(SCREEN_W * (1 - EDGE_RATIO)) // CANVAS_SCALE

    def refresh():
        bg = latest_bg[0]
        if bg:
            img = bg.copy()
            draw = ImageDraw.Draw(img)

            # 绘制边缘翻页区域（半透明蓝色竖条）
            draw.rectangle([(0, 0), (edge_left_px, ch)], fill=(0, 100, 255, 40))
            draw.rectangle([(edge_right_px, 0), (cw, ch)], fill=(0, 100, 255, 40))

            # 绘制光标十字
            cx_s = int(cursor.x) // CANVAS_SCALE
            cy_s = int(cursor.y) // CANVAS_SCALE
            r = 12
            draw.line([(cx_s - r, cy_s), (cx_s + r, cy_s)], fill="red", width=2)
            draw.line([(cx_s, cy_s - r), (cx_s, cy_s + r)], fill="red", width=2)
            draw.ellipse([(cx_s - 5, cy_s - 5), (cx_s + 5, cy_s + 5)],
                         outline="red", width=2)

            tk_img_ref[0] = ImageTk.PhotoImage(img)
            canvas.itemconfig(bg_image_id, image=tk_img_ref[0])

        cx_val, cy_val = int(cursor.x), int(cursor.y)
        coord_var.set(f"光标: ({cx_val}, {cy_val})")
        status_var.set(status_text[0])
        root.after(80, refresh)

    refresh()

    def on_close():
        running[0] = False
        ser.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


# ============================================================
# 键盘模拟模式（截屏镜像 + 全局热键）
# ============================================================
def sim_mode():
    """全局键盘 + 截屏镜像：无需 ESP32，用键盘模拟信号并可视化光标。"""
    from pynput import keyboard as kb
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw
    from io import BytesIO

    CANVAS_SCALE = 4
    cw = SCREEN_W // CANVAS_SCALE
    ch = SCREEN_H // CANVAS_SCALE
    MOVE_STEP = 80

    cursor = CursorController(SCREEN_W, SCREEN_H)
    executor = ActionExecutor(SCREEN_W, SCREEN_H, cursor)
    running = [True]
    latest_bg = [None]
    status_text = ["就绪 — WASD=眼动 IJKL=移动光标"]

    # 截屏线程
    def bg_capture():
        while running[0]:
            try:
                result = subprocess.run(
                    [ADB_PATH, "-s", ADB_DEVICE, "exec-out", "screencap", "-p"],
                    capture_output=True, timeout=3
                )
                if result.stdout:
                    img = Image.open(BytesIO(result.stdout))
                    latest_bg[0] = img.resize((cw, ch), Image.LANCZOS)
            except Exception:
                pass
            time.sleep(0.15)

    threading.Thread(target=bg_capture, daemon=True).start()

    # 全局键盘监听线程
    def on_press(key):
        try:
            ch_key = key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            if key == kb.Key.esc:
                running[0] = False
                return False
            return

        if ch_key is None:
            return

        desc = None
        if ch_key == 'q':
            running[0] = False
            return False
        elif ch_key == 'w':
            desc = executor.execute("Up")
        elif ch_key == 's':
            desc = executor.execute("Down")
        elif ch_key == 'a':
            desc = executor.execute("Left")
        elif ch_key == 'd':
            desc = executor.execute("Right")
        elif ch_key == 'i':
            cursor.y = max(0, cursor.y - MOVE_STEP)
        elif ch_key == 'k':
            cursor.y = min(SCREEN_H - 1, cursor.y + MOVE_STEP)
        elif ch_key == 'j':
            cursor.x = max(0, cursor.x - MOVE_STEP)
        elif ch_key == 'l':
            cursor.x = min(SCREEN_W - 1, cursor.x + MOVE_STEP)
        elif ch_key == 'c':
            cursor.recenter()
        else:
            return

        if desc:
            status_text[0] = f"[{ch_key.upper()}] → {desc}"
            print(f"  {status_text[0]}  光标({int(cursor.x)},{int(cursor.y)})")

    listener = kb.Listener(on_press=on_press)
    listener.start()

    # Tkinter UI
    root = tk.Tk()
    root.title("MuMu Bridge — 键盘模拟模式")
    root.resizable(False, False)

    canvas = tk.Canvas(root, width=cw, height=ch, bg="#1a1a2e")
    canvas.pack()

    status_var = tk.StringVar(value="WASD=眼动 | IJKL=移动光标 | C=归中 | Q=退出")
    tk.Label(root, textvariable=status_var, font=("Consolas", 10),
             bg="#16213e", fg="white", anchor="w", padx=5).pack(fill="x")

    coord_var = tk.StringVar(value=f"光标: ({SCREEN_W // 2}, {SCREEN_H // 2})")
    tk.Label(root, textvariable=coord_var, font=("Consolas", 10),
             bg="#0f3460", fg="#e94560", anchor="w", padx=5).pack(fill="x")

    tk_img_ref = [None]
    bg_image_id = canvas.create_image(0, 0, anchor="nw")

    edge_left_px = int(SCREEN_W * EDGE_RATIO) // CANVAS_SCALE
    edge_right_px = int(SCREEN_W * (1 - EDGE_RATIO)) // CANVAS_SCALE

    def refresh():
        if not running[0]:
            root.destroy()
            return

        bg = latest_bg[0]
        if bg:
            img = bg.copy()
            draw = ImageDraw.Draw(img)

            # 边缘翻页区域
            draw.rectangle([(0, 0), (edge_left_px, ch)], fill=(0, 100, 255, 40))
            draw.rectangle([(edge_right_px, 0), (cw, ch)], fill=(0, 100, 255, 40))

            # 光标十字
            cx_s = int(cursor.x) // CANVAS_SCALE
            cy_s = int(cursor.y) // CANVAS_SCALE
            r = 12
            draw.line([(cx_s - r, cy_s), (cx_s + r, cy_s)], fill="red", width=2)
            draw.line([(cx_s, cy_s - r), (cx_s, cy_s + r)], fill="red", width=2)
            draw.ellipse([(cx_s - 5, cy_s - 5), (cx_s + 5, cy_s + 5)],
                         outline="red", width=2)

            tk_img_ref[0] = ImageTk.PhotoImage(img)
            canvas.itemconfig(bg_image_id, image=tk_img_ref[0])

        cx_val, cy_val = int(cursor.x), int(cursor.y)
        zone = "左翻页区" if cx_val < int(SCREEN_W * EDGE_RATIO) else \
               "右翻页区" if cx_val > int(SCREEN_W * (1 - EDGE_RATIO)) else ""
        coord_var.set(f"光标: ({cx_val}, {cy_val}) {zone}")
        status_var.set(status_text[0])
        root.after(80, refresh)

    refresh()

    print("\n" + "=" * 50)
    print("  可视化键盘模拟模式（全局热键）")
    print("=" * 50)
    print("  W/S = 上下滑动 | A = 点击/翻页 | D = 返回")
    print("  I/J/K/L = 移动光标 | C = 归中 | Q/Esc = 退出")
    print("=" * 50)

    def on_close():
        running[0] = False
        listener.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    listener.stop()


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MuMu Bridge: ESP32 → ADB 控制")
    parser.add_argument('--port', type=str, default=SERIAL_PORT, help='串口号 (如 COM3)')
    parser.add_argument('--sim', action='store_true', help='键盘模拟模式（无需 ESP32）')
    parser.add_argument('--no-visual', action='store_true', help='无 UI 纯串口模式')
    args = parser.parse_args()

    print("MuMu Bridge — EOG + IMU → ADB")
    print(f"  屏幕: {SCREEN_W}x{SCREEN_H}")
    print(f"  ADB:  {ADB_PATH}")

    print("\n检查 ADB 连接...")
    if not check_adb():
        print("  继续运行（ADB 命令可能失败）\n")

    if args.sim:
        sim_mode()
    elif args.no_visual:
        serial_mode(args.port)
    else:
        visual_mode(args.port)


if __name__ == "__main__":
    main()
