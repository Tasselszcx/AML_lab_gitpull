import serial
import time
import pyautogui
import keyboard  # 用于键盘辅助控制（点击、校准）

# ==========================================
# 核心控制器：IMU 姿态到光标坐标的转换
# ==========================================
class HeadMouseController:
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cursor_x = screen_w / 2
        self.cursor_y = screen_h / 2
        self.last_yaw = 0.0
        self.last_pitch = 0.0
        
        # 【调参区】你可以根据实际体验修改这两个值
        self.sensitivity_x = 35.0  # X轴（左右摇头）灵敏度
        self.sensitivity_y = 35.0  # Y轴（上下点头）灵敏度
        self.deadzone = 0.2        # 死区(度)：小于0.2度的晃动将被忽略，防抖动
        
        self.is_initialized = False

    def recenter(self, current_yaw, current_pitch):
        """居中校准：将当前头部姿态设定为屏幕中心点"""
        self.last_yaw = current_yaw
        self.last_pitch = current_pitch
        self.cursor_x = self.screen_w / 2
        self.cursor_y = self.screen_h / 2
        self.is_initialized = True
        print("\n[系统] 🎯 已校准！光标复位至屏幕中心。")

    def update(self, current_yaw, current_pitch):
        # 第一次接收数据时自动校准
        if not self.is_initialized:
            self.recenter(current_yaw, current_pitch)
            return int(self.cursor_x), int(self.cursor_y)

        # 1. 计算角度变化量
        delta_yaw = current_yaw - self.last_yaw
        delta_pitch = current_pitch - self.last_pitch

        # 2. 死区过滤（过滤生理性呼吸带来的微小抖动）
        if abs(delta_yaw) < self.deadzone: delta_yaw = 0
        if abs(delta_pitch) < self.deadzone: delta_pitch = 0

        # 3. 将角度变化量映射为像素位移
        # 【重要】如果发现光标移动方向与头部动作反了，请将 -= 改为 +=
        self.cursor_x -= delta_yaw * self.sensitivity_x 
        self.cursor_y += delta_pitch * self.sensitivity_y

        # 4. 边界限制（防止光标飞出屏幕）
        self.cursor_x = max(0, min(self.screen_w, self.cursor_x))
        self.cursor_y = max(0, min(self.screen_h, self.cursor_y))

        # 5. 更新历史角度
        if delta_yaw != 0: self.last_yaw = current_yaw
        if delta_pitch != 0: self.last_pitch = current_pitch

        return int(self.cursor_x), int(self.cursor_y)

# ==========================================
# 主程序
# ==========================================
def main():
    COM_PORT = 'COM3'      # <<< 请修改为你的实际串口号
    BAUD_RATE = 115200     # <<< 请修改为你的单片机波特率
    
    pyautogui.FAILSAFE = False 
    screen_w, screen_h = pyautogui.size()
    controller = HeadMouseController(screen_w, screen_h)

    print(f"正在连接串口 {COM_PORT}...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print("✅ 连接成功！")
        print("-----------------------------------------")
        print("操作说明：")
        print("1. 戴好 IMU，正视屏幕。")
        print("2. 按下键盘【C键】：重新校准光标到中心。")
        print("3. 按下键盘【空格键】：触发鼠标左键点击。")
        print("4. 向左歪头 (Roll < -30°)：触发模拟器返回 (ESC)。")
        print("-----------------------------------------")
    except Exception as e:
        print(f"❌ 串口连接失败: {e}")
        return

    last_roll_time = time.time()
    cooldown = 1.0 # 歪头动作的冷却时间

    try:
        while True:
            # --- 键盘快捷键监听 ---
            if keyboard.is_pressed('c'):
                # 触发重新校准，为了防止连续触发，稍作延时
                controller.is_initialized = False 
                time.sleep(0.3)
                
            if keyboard.is_pressed('space'):
                print("👉 [键盘触发] 鼠标点击")
                pyautogui.click()
                time.sleep(0.3)

            # --- 串口数据读取与解析 ---
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                
                try:
                    # 解析 IMU 数据：Yaw, Pitch, Roll
                    parts = line.split(',')
                    yaw = float(parts[0])
                    pitch = float(parts[1])
                    roll = float(parts[2])
                except (ValueError, IndexError):
                    continue # 忽略格式错误的数据

                # --- 更新光标坐标 ---
                cx, cy = controller.update(yaw, pitch)
                pyautogui.moveTo(cx, cy, _pause=False)

                # --- 歪头返回逻辑 ---
                current_time = time.time()
                if roll < -30.0 and (current_time - last_roll_time > cooldown):
                    print("↩️ [IMU动作] 左歪头 -> 触发返回 (ESC)")
                    pyautogui.press('esc')
                    last_roll_time = current_time

    except KeyboardInterrupt:
        print("\n程序已手动停止。")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()