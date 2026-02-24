import time

class HeadMouseController:
    def __init__(self, screen_w=1080, screen_h=1920):
        # 1. 屏幕分辨率（根据你的安卓虚拟机实际分辨率修改）
        self.screen_w = screen_w
        self.screen_h = screen_h

        # 2. 光标当前位置（初始设置在屏幕正中心）
        self.cursor_x = screen_w / 2
        self.cursor_y = screen_h / 2

        # 3. 历史姿态角记录（用于计算相对增量）
        self.last_yaw = 0.0
        self.last_pitch = 0.0

        # 4. 【关键调参区】
        # 灵敏度系数：转动 1 度，光标移动多少像素？（值越大，光标跑得越快）
        self.sensitivity_x = 25.0  
        self.sensitivity_y = 25.0  
        
        # 死区阈值（度）：小于这个角度的变化将被忽略，防止呼吸或脉搏引起的光标抖动
        self.deadzone = 0.3  

        self.is_initialized = False

    def recenter(self, current_yaw, current_pitch):
        """一键归零：将当前的头部姿态强制映射为屏幕正中心"""
        self.last_yaw = current_yaw
        self.last_pitch = current_pitch
        self.cursor_x = self.screen_w / 2
        self.cursor_y = self.screen_h / 2
        self.is_initialized = True
        print("\n>>> [系统提示] 已触发归零，光标复位至屏幕中心 <<<")

    def update(self, current_yaw, current_pitch):
        """主控函数：传入实时 IMU 角度，返回计算后的光标 (X, Y) 坐标"""
        # 如果是第一次接收数据，自动执行一次归零
        if not self.is_initialized:
            self.recenter(current_yaw, current_pitch)
            return int(self.cursor_x), int(self.cursor_y)

        # Step 1: 计算角度增量 (Delta)
        delta_yaw = current_yaw - self.last_yaw
        delta_pitch = current_pitch - self.last_pitch

        # Step 2: 死区过滤 (Deadzone filter)
        if abs(delta_yaw) < self.deadzone: 
            delta_yaw = 0
        if abs(delta_pitch) < self.deadzone: 
            delta_pitch = 0

        # Step 3: 将角度增量转化为像素位移
        # 注意：这里的加减号可能需要根据你 IMU 坐标系的定义进行反转（例如改成 -= ）
        self.cursor_x += delta_yaw * self.sensitivity_x
        self.cursor_y += delta_pitch * self.sensitivity_y

        # Step 4: 边界限制 (Clamping) - 防止光标飞出虚拟机屏幕
        self.cursor_x = max(0, min(self.screen_w, self.cursor_x))
        self.cursor_y = max(0, min(self.screen_h, self.cursor_y))

        # Step 5: 更新历史角度（仅当有有效运动时更新，防止死区误差累积）
        if delta_yaw != 0: 
            self.last_yaw = current_yaw
        if delta_pitch != 0: 
            self.last_pitch = current_pitch

        return int(self.cursor_x), int(self.cursor_y)

# ==========================================
# 模拟测试：假设这是你的主程序循环
# ==========================================
if __name__ == "__main__":
    controller = HeadMouseController(screen_w=1080, screen_h=1920)

    # 模拟 IMU 传来的连续数据 (Yaw, Pitch)
    # 假设用户一开始戴歪了，初始角度是 (45度, 10度)
    mock_sensor_data = [
        (45.0, 10.0),   # 第1帧：系统自动以此为基准归零 (输出: 540, 960)
        (45.2, 10.1),   # 第2帧：微小晃动，被【死区】过滤 (输出: 540, 960)
        (46.5, 10.0),   # 第3帧：向右转头 1.5 度，X 轴增加 (输出: 577, 960)
        (48.0, 12.0),   # 第4帧：继续向右 1.5 度，同时低头 2 度，Y 轴增加
    ]

    print("开始模拟接收 IMU 数据...")
    for yaw, pitch in mock_sensor_data:
        cx, cy = controller.update(yaw, pitch)
        print(f"输入姿态: Yaw={yaw:5.1f}°, Pitch={pitch:5.1f}°  |  输出光标: X={cx:4d}, Y={cy:4d}")
        time.sleep(0.5)
        
    # 模拟用户按下键盘或通过长眨眼触发“重新校准”
    controller.recenter(48.0, 12.0) 
    cx, cy = controller.update(48.0, 12.0)
    print(f"输入姿态: Yaw={48.0:5.1f}°, Pitch={12.0:5.1f}°  |  输出光标: X={cx:4d}, Y={cy:4d}")