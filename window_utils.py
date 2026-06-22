"""
window_utils.py
Windows窗口管理工具库

使用方法：
    from window_utils import activate_window, minimize_window, close_window, is_running

进程名获取方式：打开任务管理器 → 详细信息 → 看"名称"列，去掉.exe即可
    例：cloudmusic、WeChat、chrome、notepad
"""

import win32gui
import win32con
import win32process
import win32api
import psutil
import subprocess
import time
import os
import pygetwindow as gw


def _find_hwnd(process_name):
    """
    内部函数：通过进程名找到主窗口句柄（hwnd）
    过滤掉不可见、无标题、尺寸过小的窗口
    返回hwnd列表，找不到返回空列表
    """
    result = []

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        if not win32gui.GetWindowText(hwnd):
            return
        # 过滤尺寸过小的隐藏/消息窗口
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
        except:
            pass

    win32gui.EnumWindows(callback, None)
    return result


def force_activate(hwnd):
    """
    强制置顶窗口，使用 Alt Hack + TOPMOST 技巧提高成功率
    """
    # Alt Hack：模拟按下Alt，让Windows允许切换前台窗口
    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)

    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)

    # 先临时置顶再取消，Windows会认为用户在操作这个窗口
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
    )
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
    )
    win32gui.SetForegroundWindow(hwnd)


def activate_window(process_name):
    """
    激活并置顶窗口，如果窗口最小化则先还原。
    验证是否真的置顶，失败重试3次。
    process_name: 进程名，比如 'cloudmusic'、'WeChat'
    返回：成功True，失败False

    示例：activate_window('cloudmusic')
    """
    hwnds = _find_hwnd(process_name)
    if not hwnds:
        print(f'找不到进程：{process_name}')
        return False

    hwnd = hwnds[0]
    for attempt in range(3):
        force_activate(hwnd)
        time.sleep(0.3)
        if win32gui.GetForegroundWindow() == hwnd:
            return True
        print(f'置顶重试 {attempt + 1}/3')

    print(f'置顶失败：{process_name}')
    return False


def minimize_window(process_name):
    """
    最小化窗口。
    process_name: 进程名，比如 'cloudmusic'、'WeChat'
    返回：成功True，失败False

    示例：minimize_window('cloudmusic')
    """
    hwnds = _find_hwnd(process_name)
    if hwnds:
        win32gui.ShowWindow(hwnds[0], win32con.SW_MINIMIZE)
        return True
    print(f'找不到进程：{process_name}')
    return False


def maximize_window(process_name):
    """
    最大化窗口。
    process_name: 进程名，比如 'cloudmusic'、'WeChat'
    返回：成功True，失败False

    示例：maximize_window('cloudmusic')
    """
    hwnds = _find_hwnd(process_name)
    if hwnds:
        win32gui.ShowWindow(hwnds[0], win32con.SW_MAXIMIZE)
        return True
    print(f'找不到进程：{process_name}')
    return False


def close_window(process_name):
    """
    关闭窗口（发送关闭信号，相当于点击X）。
    process_name: 进程名，比如 'cloudmusic'、'WeChat'
    返回：成功True，失败False

    示例：close_window('cloudmusic')
    """
    hwnds = _find_hwnd(process_name)
    if hwnds:
        win32gui.PostMessage(hwnds[0], win32con.WM_CLOSE, 0, 0)
        return True
    print(f'找不到进程：{process_name}')
    return False


def is_running(process_name):
    """
    检测某个程序是否正在运行。
    process_name: 进程名，比如 'cloudmusic'、'WeChat'
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


def kill_window(process_name):
    """
    强制终止程序，相当于任务管理器里的'结束任务'。
    比close_window更强硬，不会弹确认框。
    """
    for process in psutil.process_iter(['name']):
        if process_name.lower() in process.info['name'].lower():
            process.kill()


def open_or_activate(process_name, exe_path, args=None, wait=2):
    """
    如果程序未运行则打开，已运行则激活置顶。
    process_name: 进程名，比如 'cloudmusic'
    exe_path: 程序路径，比如 'notepad.exe' 或 'C:/Program Files/xxx.exe'
    args: 额外参数列表，比如 [r'C:/video.mp4']
    wait: 打开程序后等待几秒，默认2秒

    示例：
        open_or_activate('notepad', 'notepad.exe')
        open_or_activate('potplayer', r'C:/potplayer/potplayer.exe', args=[r'C:/video.mp4'])
    """
    if not is_running(process_name):
        if args:
            subprocess.Popen([exe_path] + args)
        else:
            os.startfile(exe_path)
        time.sleep(wait)
    time.sleep(0.3)
    activate_window(process_name)
    time.sleep(0.5)


def get_window_rect(process_name):
    """
    获取窗口的位置和大小。
    返回：(left, top, right, bottom) 或 None

    示例：
        rect = get_window_rect('cloudmusic')
        print(rect)  # (100, 50, 900, 700)
    """
    hwnds = _find_hwnd(process_name)
    if hwnds:
        return win32gui.GetWindowRect(hwnds[0])
    print(f'找不到进程：{process_name}')
    return None