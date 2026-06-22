"""
window_utils.py
Windows / macOS 跨平台窗口管理工具库

使用方法：
    from window_utils import activate_window, minimize_window, close_window, is_running

进程名获取方式：
    Windows：打开任务管理器 → 详细信息 → 看"名称"列，去掉.exe即可
        例：cloudmusic、WeChat、chrome、notepad
    macOS：打开活动监视器 → 进程名列，或直接用应用名
        例：Music、WeChat、Google Chrome、TextEdit
"""

import psutil
import subprocess
import time
import os
import sys

_IS_WINDOWS = sys.platform == "win32"
_IS_MAC = sys.platform == "darwin"

# ── Windows专有导入 ────────────────────────────────────────────────────────────
if _IS_WINDOWS:
    import win32gui
    import win32con
    import win32process
    import win32api


# ══════════════════════════════════════════════════════════════════════════════
# 内部辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _run_applescript(script: str) -> str:
    """执行AppleScript，返回stdout字符串，出错返回空字符串"""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def _get_app_name_for_process(process_name: str) -> str | None:
    """
    macOS：通过进程名找到对应的应用名（AppleScript需要应用名，不是进程名）
    先尝试psutil找到进程，再用proc.name()推导
    返回可用于AppleScript的应用名，找不到返回None
    """
    for proc in psutil.process_iter(['name', 'pid']):
        if process_name.lower() in proc.info['name'].lower():
            # 大部分Mac应用进程名就是应用名，直接可用
            return proc.info['name']
    return None


def _find_hwnd_windows(process_name: str) -> list:
    """Windows：通过进程名找到主窗口句柄列表"""
    result = []

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        if not win32gui.GetWindowText(hwnd):
            return
        rect = win32gui.GetWindowRect(hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if width <= 300 or height <= 300:
            return
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            process = psutil.Process(pid)
            if process_name.lower() in process.name().lower():
                result.append(hwnd)
        except Exception:
            pass

    win32gui.EnumWindows(callback, None)
    return result


def _force_activate_windows(hwnd):
    """Windows：Alt Hack + TOPMOST技巧强制置顶"""
    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)

    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
    )
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
    )
    win32gui.SetForegroundWindow(hwnd)


# ══════════════════════════════════════════════════════════════════════════════
# 公开接口
# ══════════════════════════════════════════════════════════════════════════════

def activate_window(process_name: str) -> bool:
    """
    激活并置顶窗口，如果窗口最小化则先还原。
    process_name: 进程名
        Windows 例：'cloudmusic'、'WeChat'
        macOS   例：'Music'、'WeChat'
    返回：成功True，失败False

    示例：activate_window('cloudmusic')
    """
    if _IS_WINDOWS:
        hwnds = _find_hwnd_windows(process_name)
        if not hwnds:
            print(f'找不到进程：{process_name}')
            return False
        hwnd = hwnds[0]
        for attempt in range(3):
            _force_activate_windows(hwnd)
            time.sleep(0.3)
            if win32gui.GetForegroundWindow() == hwnd:
                return True
            print(f'置顶重试 {attempt + 1}/3')
        print(f'置顶失败：{process_name}')
        return False

    elif _IS_MAC:
        app_name = _get_app_name_for_process(process_name)
        if not app_name:
            print(f'找不到进程：{process_name}')
            return False
        # 先取消最小化（如果有），再激活
        _run_applescript(f'''
            tell application "{app_name}"
                activate
            end tell
            tell application "System Events"
                tell process "{app_name}"
                    set miniaturized of every window to false
                end tell
            end tell
        ''')
        return True

    else:
        print(f'不支持的操作系统：{sys.platform}')
        return False


def minimize_window(process_name: str) -> bool:
    """
    最小化窗口。
    返回：成功True，失败False

    示例：minimize_window('cloudmusic')
    """
    if _IS_WINDOWS:
        hwnds = _find_hwnd_windows(process_name)
        if hwnds:
            win32gui.ShowWindow(hwnds[0], win32con.SW_MINIMIZE)
            return True
        print(f'找不到进程：{process_name}')
        return False

    elif _IS_MAC:
        app_name = _get_app_name_for_process(process_name)
        if not app_name:
            print(f'找不到进程：{process_name}')
            return False
        _run_applescript(f'''
            tell application "System Events"
                tell process "{app_name}"
                    set miniaturized of every window to true
                end tell
            end tell
        ''')
        return True

    else:
        print(f'不支持的操作系统：{sys.platform}')
        return False


def maximize_window(process_name: str) -> bool:
    """
    最大化窗口。
    macOS注意：Mac没有真正的"最大化"，这里等价于全屏（zoom）。
    返回：成功True，失败False

    示例：maximize_window('cloudmusic')
    """
    if _IS_WINDOWS:
        hwnds = _find_hwnd_windows(process_name)
        if hwnds:
            win32gui.ShowWindow(hwnds[0], win32con.SW_MAXIMIZE)
            return True
        print(f'找不到进程：{process_name}')
        return False

    elif _IS_MAC:
        app_name = _get_app_name_for_process(process_name)
        if not app_name:
            print(f'找不到进程：{process_name}')
            return False
        # zoom相当于点击绿色按钮（非全屏模式的最大化）
        _run_applescript(f'''
            tell application "System Events"
                tell process "{app_name}"
                    perform action "AXZoom" of (first window whose value of attribute "AXZoomed" is false)
                end tell
            end tell
        ''')
        return True

    else:
        print(f'不支持的操作系统：{sys.platform}')
        return False


def close_window(process_name: str) -> bool:
    """
    关闭窗口（发送关闭信号，相当于点击X，不是强制终止进程）。
    返回：成功True，失败False

    示例：close_window('cloudmusic')
    """
    if _IS_WINDOWS:
        hwnds = _find_hwnd_windows(process_name)
        if hwnds:
            win32gui.PostMessage(hwnds[0], win32con.WM_CLOSE, 0, 0)
            return True
        print(f'找不到进程：{process_name}')
        return False

    elif _IS_MAC:
        app_name = _get_app_name_for_process(process_name)
        if not app_name:
            print(f'找不到进程：{process_name}')
            return False
        # 关闭所有窗口，但不退出进程（等价于Cmd+W）
        _run_applescript(f'''
            tell application "System Events"
                tell process "{app_name}"
                    keystroke "w" using command down
                end tell
            end tell
        ''')
        return True

    else:
        print(f'不支持的操作系统：{sys.platform}')
        return False


def is_running(process_name: str) -> bool:
    """
    检测某个程序是否正在运行。
    返回：运行中True，未运行False

    示例：
        if not is_running('cloudmusic'):
            subprocess.Popen('cloudmusic.exe')
        else:
            activate_window('cloudmusic')
    """
    for process in psutil.process_iter(['name']):
        if process_name.lower() in process.info['name'].lower():
            return True
    return False


def kill_window(process_name: str):
    """
    强制终止程序，相当于任务管理器/活动监视器里的"强制退出"。
    比close_window更强硬，不会弹确认框。
    """
    for process in psutil.process_iter(['name']):
        if process_name.lower() in process.info['name'].lower():
            process.kill()


def open_or_activate(process_name: str, exe_path: str, args: list = None, wait: int = 2):
    """
    如果程序未运行则打开，已运行则激活置顶。
    process_name: 进程名
    exe_path:
        Windows：程序路径，比如 'notepad.exe' 或 'C:/Program Files/xxx.exe'
        macOS：  应用路径，比如 '/Applications/TextEdit.app' 或直接用应用名 'TextEdit'
    args: 额外参数列表，比如 [r'C:/video.mp4']（主要用于Windows）
    wait: 打开程序后等待几秒，默认2秒

    示例：
        # Windows
        open_or_activate('notepad', 'notepad.exe')
        open_or_activate('potplayer', r'C:/potplayer/potplayer.exe', args=[r'C:/video.mp4'])
        # macOS
        open_or_activate('TextEdit', '/Applications/TextEdit.app')
        open_or_activate('vlc', '/Applications/VLC.app', args=['/Users/me/video.mp4'])
    """
    if not is_running(process_name):
        if _IS_WINDOWS:
            if args:
                subprocess.Popen([exe_path] + args)
            else:
                os.startfile(exe_path)
        elif _IS_MAC:
            cmd = ["open", exe_path]
            if args:
                cmd += ["--args"] + args
            subprocess.Popen(cmd)
        else:
            subprocess.Popen([exe_path] + (args or []))
        time.sleep(wait)

    time.sleep(0.3)
    activate_window(process_name)
    time.sleep(0.5)


def get_window_rect(process_name: str) -> tuple | None:
    """
    获取窗口的位置和大小。
    返回：(left, top, right, bottom) 或 None

    示例：
        rect = get_window_rect('cloudmusic')
        print(rect)  # (100, 50, 900, 700)
    """
    if _IS_WINDOWS:
        hwnds = _find_hwnd_windows(process_name)
        if hwnds:
            return win32gui.GetWindowRect(hwnds[0])
        print(f'找不到进程：{process_name}')
        return None

    elif _IS_MAC:
        app_name = _get_app_name_for_process(process_name)
        if not app_name:
            print(f'找不到进程：{process_name}')
            return None
        # AppleScript返回 "x, y, width, height" 格式
        raw = _run_applescript(f'''
            tell application "System Events"
                tell process "{app_name}"
                    set p to position of front window
                    set s to size of front window
                    return (item 1 of p) & "," & (item 2 of p) & "," & (item 1 of s) & "," & (item 2 of s)
                end tell
            end tell
        ''')
        try:
            x, y, w, h = (int(v.strip()) for v in raw.split(","))
            return (x, y, x + w, y + h)  # 转换为(left, top, right, bottom)，与Windows保持一致
        except Exception:
            print(f'获取窗口位置失败：{process_name}')
            return None

    else:
        print(f'不支持的操作系统：{sys.platform}')
        return None