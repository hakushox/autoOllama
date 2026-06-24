#角色 — 告诉模型它是谁（你是电脑控制助手）
# 任务 — 告诉模型要做什么（把指令转成JSON数组）
# 约束 — 告诉模型不能做什么（不能输出其他文字）
# 格式 — 用例子说明输出结构
# 边界 — 告诉模型遇到不懂的怎么办（输出unknown）

import ollama
import json
import subprocess
import pyautogui as pg
import pyperclip
import re
from window_utils import open_or_activate, is_running
import time
from test_ollama_pic import visual_locate
import platform
import os
import types as _types
_os = _types.ModuleType('os')
_os.__dict__.update(os.__dict__)
if not hasattr(_os, 'startfile'):
    def _startfile(path, operation=None):
        subprocess.run(['open', str(path)])
    _os.startfile = _startfile
from pathlib import Path
import numpy as np
from collections import deque
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import sounddevice as sd
import mlx_whisper
import threading
from pynput import keyboard
from prompt_toolkit import prompt
import soundfile as sf
from json import JSONDecoder
from API_KEY import GROQ_API_KEY, CEREBRAS_API_KEY
from openai import OpenAI
import edge_tts
import asyncio
import io
import datetime
from openwakeword.model import Model

olla_models = {
    'gemma': 'gemma4:e4b-mlx', 'qwen3.5': 'qwen3.5:4b-mlx', 'qwen3.5:9':'qwen3.5:9b-mlx',
}
# ollama.chat(model='qwen3:8b', messages=[...], options={'temperature': 0},think=False)
# model='gemma4',options={'temperature': 0.7, 'top_p': 0.9}
wake_event = threading.Event()
quit_event = threading.Event()

is_active = threading.Event()

import openwakeword

LOCAL_MODEL_DIR = Path(__file__).parent / 'models'
PACKAGE_MODEL_DIR = Path(openwakeword.__file__).parent / 'resources' / 'models'

def _wake_model_path(filename):
    local_path = LOCAL_MODEL_DIR / filename
    if local_path.exists():
        return local_path
    return PACKAGE_MODEL_DIR / filename

oww_model = Model(
    wakeword_models=[
        str(_wake_model_path(os.getenv('QUIT_MODEL_FILE', 'alexa_v0.1.onnx'))),
        str(_wake_model_path(os.getenv('WAKE_MODEL_FILE', 'hello_mercy.onnx'))),
    ],
    inference_framework='onnx'
)
WAKE_WORD = Path(os.getenv('WAKE_MODEL_FILE', 'hello_mercy.onnx')).stem
QUIT_WORD = Path(os.getenv('QUIT_MODEL_FILE', 'alexa_v0.1.onnx')).stem
WAKE_THRESHOLD = float(os.getenv('WAKE_THRESHOLD', '0.5'))
QUIT_THRESHOLD = float(os.getenv('QUIT_THRESHOLD', '0.9'))
WAKE_FRAME_SIZE = 1280
WAKE_DEBUG = os.getenv('WAKE_DEBUG') == '1'
WAKE_TEST = os.getenv('WAKE_TEST') == '1'
WAKE_DIAG_SECONDS = float(os.getenv('WAKE_DIAG_SECONDS', '20'))
WAKE_DEVICE = os.getenv('WAKE_DEVICE')
WAKE_SAMPLERATE = int(os.getenv('WAKE_SAMPLERATE', '16000'))
WAKE_RECORD_SECONDS = float(os.getenv('WAKE_RECORD_SECONDS', '0'))
AUDIO_LOCK = threading.Lock()
current_output_stream = None

def get_wake_input_device():
    input_device = int(WAKE_DEVICE) if WAKE_DEVICE is not None and WAKE_DEVICE.isdigit() else WAKE_DEVICE
    if input_device is None:
        device_info = sd.query_devices(kind='input')
    else:
        device_info = sd.query_devices(input_device, kind='input')
    return input_device, device_info

last_wake_time = datetime.datetime.min

def wake_word_listener():
    last_debug_time = 0
    listener_start_time = datetime.datetime.now().timestamp()
    print(f'Wake models loaded: {list(oww_model.models.keys())}')
    if WAKE_DEBUG or WAKE_TEST:
        print(sd.query_devices())
    try:
        input_device, device_info = get_wake_input_device()
        print(f'Wake input device: {device_info["name"]}')
    except Exception as e:
        print(f'Cannot read input device info: {e}')
        input_device = None

    def callback(indata, frames, time, status):
        nonlocal last_debug_time
        if status:
            print(f'Wake-word audio status: {status}')

        try:
            audio = (indata[:, 0] * 32767).astype(np.int16)
            prediction = oww_model.predict(audio)
        except Exception as e:
            print(f'Wake-word listener error: {e}')
            return

        wake_score = prediction.get(WAKE_WORD, 0)
        quit_score = prediction.get(QUIT_WORD, 0)

        should_print_diag = (
            WAKE_DEBUG
            or WAKE_TEST
            or datetime.datetime.now().timestamp() - listener_start_time <= WAKE_DIAG_SECONDS
        )
        if should_print_diag:
            now = datetime.datetime.now().timestamp()
            if now - last_debug_time >= 0.5:
                rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
                # print(f'\rwake={wake_score:.3f} quit={quit_score:.3f} rms={rms:.0f}', end='', flush=True)
                last_debug_time = now

        global last_wake_time
        if wake_score > WAKE_THRESHOLD and not is_active.is_set():
            last_wake_time = datetime.datetime.now()
            print(f'\n检测到唤醒词，得分 {wake_score:.2f}')
            wake_event.set()
        elif quit_score > QUIT_THRESHOLD:
            diff = (datetime.datetime.now() - last_wake_time).total_seconds()
            print(f'\n退出词得分触发，得分 {quit_score:.2f}，距唤醒{diff:.1f}秒')
            if diff > 3:
                print('检测到退出词')
                quit_event.set()
            # if (datetime.datetime.now() - last_wake_time).total_seconds() > 3:
            #     print('检测到退出词')
            #     quit_event.set()

    try:
        with sd.InputStream(
            samplerate=WAKE_SAMPLERATE,
            channels=1,
            dtype='float32',
            blocksize=WAKE_FRAME_SIZE,
            device=input_device,
            callback=callback,
        ):
            print(f'Wake listener started at {WAKE_SAMPLERATE} Hz')
            while True:
                sd.sleep(100)
    except Exception as e:
        print(f'Wake listener failed to start: {e}')
        quit_event.set()

def record_wake_sample(seconds=5):
    input_device, device_info = get_wake_input_device()
    print(f'Recording {seconds:.1f}s from: {device_info["name"]}')
    frames = []

    def callback(indata, frame_count, time_info, status):
        if status:
            print(f'Record status: {status}')
        frames.append(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=WAKE_SAMPLERATE,
        channels=1,
        dtype='float32',
        blocksize=WAKE_FRAME_SIZE,
        device=input_device,
        callback=callback,
    ):
        sd.sleep(int(seconds * 1000))

    audio_float = np.concatenate(frames)
    audio_int16 = (audio_float * 32767).astype(np.int16)
    rms = float(np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2)))
    out_path = Path('/tmp/wake_debug.wav')
    sf.write(out_path, audio_int16, WAKE_SAMPLERATE, subtype='PCM_16')
    print(f'Saved {out_path}, rms={rms:.0f}, samples={len(audio_int16)}')

    oww_model.reset()
    predictions = oww_model.predict_clip(audio_int16, chunk_size=WAKE_FRAME_SIZE)
    max_scores = {
        key: max((float(p.get(key, 0)) for p in predictions), default=0)
        for key in [WAKE_WORD, QUIT_WORD]
    }
    print(f'Offline max scores: {max_scores}')
    return out_path, max_scores

async def _speak(text, rate='+15%', voice='zh-CN-XiaoyiNeural'):
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk['type'] == 'audio':
            buffer.write(chunk['data'])
    buffer.seek(0)
    return buffer

def tts(text, rate='+15%'):
    text = text.replace('*','~')
    def _run():
        buffer = asyncio.run(_speak(text, rate=rate))
        data, samplerate = sf.read(buffer)
        play_audio(data, samplerate)
    threading.Thread(target=_run, daemon=True).start()

async def _preload():
    responses = {
        'activated': 'Hello, 我在',
        'ready': '助手已就绪，按F8手动输入',
        'waiting': '等待唤醒',
        'typing': '请输入你的指令, 使用semicolon来分割多组命令',
        'speaking': '请说出你的指令',
        'bye': 'bye~~~!',
        'executing': '执行中...',
        'canceling': '已取消执行！',
        'execute_inquiry': '是否执行此次代码？',
    }
    audio_cache = {}
    for key, text in responses.items():
        buffer = await _speak(text)
        data, samplerate = sf.read(buffer)
        audio_cache[key] = (data, samplerate)
    return audio_cache

def play_cached(key):
    """直接播放预生成的音频，无网络延迟"""
    data, samplerate = audio_cache[key]
    play_audio(data, samplerate)

def play_audio(data, samplerate):
    global current_output_stream
    with AUDIO_LOCK:
        audio = np.asarray(data)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        audio = audio.astype(np.float32, copy=False)

        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=audio.shape[1],
            dtype='float32',
        )
        current_output_stream = stream
        try:
            with stream:
                stream.write(audio)
        finally:
            current_output_stream = None

def stop_output_audio():
    with AUDIO_LOCK:
        if current_output_stream:
            current_output_stream.abort()

CLIENT_INDEX = 0
MODEL_INDEX = 0

PROVIDERS = [
        {
        'client': OpenAI(api_key=CEREBRAS_API_KEY, base_url='https://api.cerebras.ai/v1'),
        'models': ['gpt-oss-120b']
    },
    {
        'client': OpenAI(api_key=GROQ_API_KEY, base_url='https://api.groq.com/openai/v1'),
        'models': ['llama-3.3-70b-versatile', 'qwen-qwen3-32b']
    },

]

def get_current_client():
    return PROVIDERS[CLIENT_INDEX]['client']

def get_current_model():
    return PROVIDERS[CLIENT_INDEX]['models'][MODEL_INDEX]

def switch_model():
    global CLIENT_INDEX, MODEL_INDEX
    current_models = PROVIDERS[CLIENT_INDEX]['models']
    if MODEL_INDEX < len(current_models) - 1:
        MODEL_INDEX += 1
    elif CLIENT_INDEX < len(PROVIDERS) - 1:
        CLIENT_INDEX += 1
        MODEL_INDEX = 0
    else:
        reset_provider()
        # raise Exception('所有模型已耗尽')
    print(f'切换到：{PROVIDERS[CLIENT_INDEX]["client"].base_url} / {get_current_model()}')

_last_reset_time = None

def reset_provider():
    global CLIENT_INDEX, MODEL_INDEX, _last_reset_time
    now = datetime.datetime.now()
    if _last_reset_time is None or (now - _last_reset_time).total_seconds() >= 90 * 60:
        CLIENT_INDEX = 0
        MODEL_INDEX = 0
        _last_reset_time = now
        print(f'''已重置到{PROVIDERS[CLIENT_INDEX]['client'].base_url} 
              || {PROVIDERS[CLIENT_INDEX]['models'][MODEL_INDEX]}''')

_IS_MAC = platform.system() == 'Darwin'
_MODIFIER = 'cmd' if _IS_MAC else 'ctrl'
_OPEN_FILE_RULE = (
    "- 打开文件/文件夹必须用subprocess.run(['open', path])，禁止os.startfile()"
    if _IS_MAC else
    "- 打开文件/文件夹用os.startfile(path)"
)
_FIND_APP_EXAMPLE = (
    '''用户说"打开欧路词典"，输出：
[{{"action":"code","script":[
  "import subprocess",
  "result = subprocess.run(['mdfind', '-name', 'Eudic'], capture_output=True, text=True)",
  "paths = [p for p in result.stdout.strip().splitlines() if p.endswith('.app')]",
  "if paths:",
  "    subprocess.run(['open', paths[0]])",
  "else:",
  "    print('未找到')"
]}}]'''
    if _IS_MAC else
    '''用户说"打开欧路词典"，输出：
[{{"action":"code","script":[
  "from pathlib import Path",
  "import os",
  "results = list(Path('C:/').glob('**/Eudic.exe')) + list(Path('D:/').glob('**/Eudic.exe'))",
  "if results:",
  "    os.startfile(str(results[0]))",
  "else:",
  "    print('未找到')"
]}}]'''
)
_FOLDER_EXAMPLE = (
    '''用户说"打开下载文件夹，问我要打开哪个"，输出：
[{{"action":"code","script":[
  "import os, subprocess",
  "path = os.path.expanduser('~/Downloads')",
  "subprocess.run(['open', path])",
  "files = os.listdir(path)",
  "for i, f in enumerate(files):",
  "    print(f'{{i}}. {{f}}')",
  "choice = input('请输入序号或文件名：')",
  "if choice.isdigit():",
  "    target = files[int(choice)]",
  "else:",
  "    matches = [f for f in files if choice.lower() in f.lower()]",
  "    target = matches[0] if matches else None",
  "if target:",
  "    subprocess.run(['open', os.path.join(path, target)])",
  "else:",
  "    print('未找到匹配文件')"
]}}]'''
    if _IS_MAC else
    '''用户说"打开E盘video文件夹，问我要打开哪个"，输出：
[{{"action":"code","script":[
  "import os",
  "path = 'E:/video'",
  "os.startfile(path)",
  "files = os.listdir(path)",
  "for i, f in enumerate(files):",
  "    print(f'{{i}}. {{f}}')",
  "choice = input('请输入序号或文件名：')",
  "if choice.isdigit():",
  "    target = files[int(choice)]",
  "else:",
  "    matches = [f for f in files if choice.lower() in f.lower()]",
  "    target = matches[0] if matches else None",
  "if target:",
  "    os.startfile(os.path.join(path, target))",
  "else:",
  "    print('未找到匹配文件')"
]}}]'''
)
_RUN_EXAMPLES = (
    f'''run：打开程序（macOS用app名称，不带.exe）
{{"action":"run","program":"textedit"}}
chrome必须带profile：{{"action":"run","program":"chrome"}}
搜索引擎搜索关键词，搜索URL中文关键词禁止URL编码！
{{"action":"run","program":"chrome","file":"https://www.baidu.com/s?wd=关键词"}}
系统文件夹（垃圾箱/下载/桌面）：{{"action":"run","program":"trash"}}'''
    if _IS_MAC else
    f'''run：打开程序
{{"action":"run","program":"notepad"}}
chrome必须带profile：{{"action":"run","program":"chrome"}}
搜索引擎搜索关键词，搜索URL中文关键词禁止URL编码！
{{"action":"run","program":"chrome","file":"https://www.baidu.com/s?wd=关键词"}}
word空白文档：{{"action":"run","program":"winword","file":"/w"}}
视频播放器：{{"action":"run","program":"potplayer","file":"路径"}}
系统文件夹（回收站/下载/桌面/此电脑）：{{"action":"run","program":"回收站"}}'''
)
_CHROME_TAB = (
    f'''切换Chrome已有标签时,禁止cmd+t开新标签，切换已有标签必须用cmd+shift+a,规则：
[{{"action":"activate","program":"chrome"}},
{{"action":"hotkey","keys":["cmd","shift","a"]}},
{{"action":"sleep","seconds":1}},
{{"action":"type","text":"关键词"}},
{{"action":"sleep","seconds":1}},
{{"action":"hotkey","keys":["enter"]}},{{"action":"sleep","seconds":1}}]
显示桌面：["cmd","mission_control"]（或用三指上划手势）'''
    if _IS_MAC else
    f'''切换Chrome已有标签时,禁止ctrl+t开新标签切换已有标签必须用ctrl+shift+a,规则：
[{{"action":"activate","program":"chrome"}},
{{"action":"hotkey","keys":["ctrl","shift","a"]}},
{{"action":"sleep","seconds":1}},
{{"action":"type","text":"关键词"}},
{{"action":"sleep","seconds":1}},
{{"action":"hotkey","keys":["enter"]}},{{"action":"sleep","seconds":1}}]
显示桌面：["win","d"]'''
)
_FINAL_EXAMPLE = (
    f'''示例：
用户说"打开文本编辑然后输入hello"，输出：
[{{"action":"run","program":"textedit"}},{{"action":"sleep","seconds":2}},{{"action":"type","text":"hello"}}]

单条指令也必须用数组包裹：
[{{"action":"run","program":"textedit"}}]'''
    if _IS_MAC else
    f'''示例：
用户说"打开记事本然后输入hello"，输出：
[{{"action":"run","program":"notepad"}},{{"action":"sleep","seconds":2}},{{"action":"type","text":"hello"}}]

单条指令也必须用数组包裹：
[{{"action":"run","program":"notepad"}}]'''
)

system_prompt = f'''你是控制助手，当前系统：{platform.system()}，用户名：{os.getlogin()}
给你操作指令中可能会有多个步骤。你要将每个操作步骤输出到一个JSON数组中。
JSON数组中每个元素是一条操作，不能有任何其他文字！
## Action类型
{_RUN_EXAMPLES}
activate：激活已有窗口（不新建），置顶指定程序
{{"action":"activate","program":"chrome"}}
操作已打开的程序前必须先activate
type：输入文字
{{"action":"type","text":"hello"}}
hotkey：快捷键（macOS用cmd代替ctrl）
{{"action":"hotkey","keys":["{_MODIFIER}","s"]}}
{_CHROME_TAB}
mouseclick：鼠标点击，坐标未知时x/y填null
{{"action":"mouseclick","button":"left","clicks":1,"x":100,"y":200}}
mouseclick x/y为null时禁止生成mouseclick ，改用hotkey或type直接操作
mousescroll：滚动，正数向上负数向下，默认-200
{{"action":"mousescroll","amount":-200}}
sleep：等待
{{"action":"sleep","seconds":2}}

code：执行Python代码，script必须是字符串数组，每个元素是一行代码
{{"action":"code","script":["import subprocess","subprocess.run(['open', '/path/to/file'])"]}}

unknown：无法完成的指令
{{"action":"unknown"}}

## 重要规则
文件操作：
- 新建任何文件必须用code，不能用run的file参数
- 桌面路径：os.path.expanduser('~/Desktop')
- 下载路径：os.path.expanduser('~/Downloads')
- 新建docx必须用python-docx的Document()不带参数，再save(path)
- 搜索文件有明确条件时直接匹配打开，无需列出让用户选
{_OPEN_FILE_RULE}
打开路径不确定的程序时，先搜索再打开：
{_FIND_APP_EXAMPLE}

代码规范：
- script是字符串数组，每个元素是一行，缩进用空格写在字符串里
- 多步操作能合并时写成一个code
- if/for/while等块语句缩进必须正确
- next()查找时必须提供默认值：next((x for x in ...), None)，禁止直接next(generator)

网页操作（两步走）：
第一步：sync_playwright打开页面，遍历frame抓取input/button/a元素赋值给_result，禁止browser.close(),script里禁止import sync_playwright，它已在执行环境中可直接使用
{{"action":"code","script":[
"pw = sync_playwright().start()",
"browser = pw.chromium.connect_over_cdp('http://localhost:9222')",
"pages = browser.contexts[0].pages",
"page = next((p for p in pages if '目标域名关键词' in p.url), None)",
"if not page:",
"    page = browser.contexts[0].new_page()",
"    page.goto('目标网址')",
"    page.wait_for_load_state('networkidle')",
"page.bring_to_front()",
  "all_elements = []",
  "for frame in page.frames:",
  "    html = frame.content()",
  "    soup = BeautifulSoup(html, 'html.parser')",
  "    for el in soup.find_all(['input','button','a']):",
  "        el_id = el.get('id','')",
  "        el_name = el.get('name','')",
  "        el_label = el.get('aria-label','') or el.get('placeholder','') or el.get_text(strip=True)",
  "        el_type = el.get('type','')",
  "        if el_type == 'hidden': continue",
  "        if el_id or el_name:",
  "            selector = '#' + el_id if el_id else '[name=' + el_name + ']'",
  "            all_elements.append(f'{{selector}} type={{el_type}} ({{el_label}}) frame={{frame.url}}')",
  "_result = '\\n'.join(all_elements)"
]}}
第二步：主程序传回_result，根据真实元素生成fill/click操作
禁止使用async_playwright和asyncio，禁止用requests抓取网页

需要问用户输入才能继续的操作，用input()写在script里。

{_FOLDER_EXAMPLE}

{_FINAL_EXAMPLE}
'''

EXAMPLES_FILE = Path(__file__).parent / 'ollama/examples.json'
EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_examples():
    if EXAMPLES_FILE.exists():
        try:
            return json.loads(EXAMPLES_FILE.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            print('examples.json格式错误')
            return []
    return []

examples_cache = load_examples()

def save_examples(user_input, datas, direct_execute=False):
    embedding = get_embedding(user_input)
    new_example ={
        'input': user_input,
        'output': json.dumps(datas, ensure_ascii=False),
        'embedding': embedding,
        'direct_execute': direct_execute
    }
    examples_cache.append(new_example)
    EXAMPLES_FILE.write_text(json.dumps(examples_cache, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已保存：成功执行案例')

def get_embedding(text):
    response = ollama.embed(model='qwen3-embedding:0.6b', input=text)
    return response['embeddings'][0]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_similar(user_input, threshold=0.9):
    examples = load_examples()
    if not examples:
        return None

    new_embedding = get_embedding(user_input)
    best = None 
    best_score = 0
    for example in examples_cache:
        if not example['embedding']:
            continue
        score = cosine_similarity(new_embedding, example['embedding'])
        if score > best_score:
            best_score = score
            best = example
            
    if best_score >= threshold:
        print(f'找到相似案例:{best["input"][:5]}，相似度{best_score:.2f}')
        best['score'] = best_score
        return best
    return None


def generate_actions(content, extra_messages=None, similar=None, max_attempts=3):
    if similar:
        print(f"direct_execute: {similar.get('direct_execute')}, score: {similar.get('score', 0)}")
    if similar and similar.get('direct_execute') and similar.get('score', 0) >= 0.75:
        tts(f'找到曾经执行的案例： 评分{similar.get("score"):.1f}')
        output = similar['output']
        return json.loads(output), output
 
    messages = [{'role': 'system', 'content': system_prompt}]

    if extra_messages:
        messages.extend(extra_messages)
    elif similar:
        messages.append({'role': 'user', 'content': similar['input']})
        messages.append({'role': 'assistant', 'content': similar['output']})
    messages.append({'role': 'user', 'content': content})

    for attempt in range(max_attempts):
        try:
            response = get_current_client().chat.completions.create(
                model=get_current_model(),
                messages=messages, temperature= 0.4,
        )            
        except Exception as e:
            if '429' in str(e):
                switch_model()
                continue
            raise

        print('=' *30 +response.model+'=' *30)
        output = response.choices[0].message.content
        output = output.replace('"hot,', '"hotkey",')
        print(f'第{attempt+1}次原始输出：\n{repr(output)}\n')
        try:
            decoder = json.JSONDecoder()
            text = output.strip()
            # 第一种情况：
            # 模型直接输出JSON数组
            datas, end = decoder.raw_decode(text)

        except json.JSONDecodeError as e:

            print(f'JSON解析失败1：{e}')
            start = output.find('[')
            if start == -1:
                messages.extend([
                    {
                        'role': 'assistant',
                        'content': output
                    },
                    {
                        'role': 'user',
                        'content': f'''输出不是JSON数组： {e}
请重新输出。不要解释。不要Markdown。只输出JSON数组。'''
                    }
                ])
                continue

            try:
                datas, end = decoder.raw_decode(output[start:])

            except json.JSONDecodeError as e:
                print(f'JSON解析失败2：{e}')

                messages.extend([
                    {'role': 'assistant', 'content': output},
                    {'role': 'user', 'content': f'''JSON格式错误：{e}
    请重新输出合法JSON数组。不要解释。不要Markdown。只输出JSON数组'''
                    }])
                continue
            if not isinstance(datas, list):
                print('顶层不是数组')
                messages.extend([
                    {
                        'role': 'assistant',
                        'content': output
                    },
                    {
                        'role': 'user',
                        'content': '''
            顶层必须是JSON数组。
            正确：
            [{"action":"run","program":"notepad"}]
            请重新输出'''
                    }])
                continue

        # 校验所有code action
        valid = True
        for d in datas:
            if d['action'] == 'code':
                # 校验script字段存在且为list
                if 'script' not in d:
                    print(f'code action缺少script字段，第{attempt+1}次重试')
                    messages.extend([
                        {'role': 'assistant', 'content': output},
                        {'role': 'user', 'content': '''code action缺少script字段。script必须是字符串数组。
示例：{"action":"code", "script":["import os","print('hello')"]}'''}])
                    valid = False
                    break
                if not isinstance(d['script'], list):
                    print(f'script不是数组，第{attempt+1}次重试')
                    messages.extend([
                        {'role': 'assistant', 'content': output},
                        {'role': 'user', 'content': f'script必须是字符串数组，例如["import os","os.startfile(path)"]，不能是字符串'}
                    ])
                    valid = False
                    break
                # 拼接并编译校验语法
                script = '\n'.join(d['script'])
                try:
                    compile(script, '<string>', 'exec')
                except SyntaxError as e:
                    print(f'语法错误，第{attempt+1}次重试：{e}')
                    messages.extend([
                        {'role': 'assistant', 'content': output},
                        {'role': 'user', 'content': f'Python代码语法错误：{e}\n保持JSON结构不变，只修复script数组中的代码缩进和语法。'}
                    ])
                    valid = False
                    break
        if valid:
            return datas, output
    return None, None

waiting_input = threading.Event()
waiting_input.clear()

def on_press(key):
    if key == keyboard.Key.space:
        stop_output_audio()
    if key == keyboard.Key.f8:
        if not waiting_input.is_set():
            print('正在切换为手动输入模式...')
            waiting_input.set() #is_set() 返回 True
            stop_output_audio()


def listen():
    while not start_recording.is_set():  # 等F9按下
        time.sleep(0.05)
    frames =[]
    stop_event = threading.Event()

    def transcribe_loop():
        start = time.time()
        while not stop_event.is_set():
            time.sleep(1.5)
            if not frames:
                continue
            audio = np.concatenate(frames).squeeze()
            result = mlx_whisper.transcribe(audio, language='zh',
                path_or_hf_repo='mlx-community/whisper-large-v3-turbo',
                initial_prompt='E盘, 海豹, 网易云，百度, baidu, gmail, 桌面...')
            text = ''.join(seg['text'] for seg in result['segments'])
            elapsed = time.time() - start
            print(f'\rRecording...{elapsed:.1f}秒。识别中：{text}', end='', flush=True)

            for keyword in ['发送', 'over', 'send', '完成', 'ok', 'OK']:
                pos = text.rfind(keyword)
                if pos != -1 and len(text) - pos <= 5:
                    start_recording.clear()
                    stop_event.set()
                    break

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())
    
    t = threading.Thread(target=transcribe_loop, daemon=True)
    t.start()

    with sd.InputStream(samplerate=16000, channels=1, dtype='float32', callback=callback):
        
        while start_recording.is_set():
            time.sleep(0.05)

    stop_event.set()
    t.join(timeout=2)
    print()
    if not frames:
        return ''
    
    audio = np.concatenate(frames).squeeze()
    result = mlx_whisper.transcribe(audio, language='zh',
        path_or_hf_repo='mlx-community/whisper-large-v3-turbo',
        initial_prompt='E盘, 海豹, 网易云，百度, baidu, gmail, 桌面...')
    text = ''.join(seg['text'] for seg in result['segments'])

    # 去掉结束关键词
    for keyword in ['发送', 'over', 'send', '完成', 'ok']:
        pos = text.rfind(keyword)
        if pos != -1 and len(text) - pos <= 5:
            text = text[:pos].strip()
            break

    print(f'\n最终识别：{text}')
    return text

start_recording = threading.Event()  #模块级别的变量

def correct_text(text):
    if not text:
        return text
    print(f'correcting text...MODEL: {olla_models["qwen3.5"]}')
    response = ollama.chat(
            model=olla_models["qwen3.5"],
            messages=[{'role': 'system',                                
                                    'content':f'''你是语音识别结果的校正器。
输入是Whisper的识别文本，可能有同音字错误、漏字、专有名词识别失误。
这些文本是用来对电脑自动化操作的，比如打开某文件夹，登录某网站, 网址格式等。里面会含有英文。
E盘可能会被识别成一盘，类似这种谐音识别错误，你需要修正。
规则：
1. 只修正明显的识别错误，不改变用户意图
2. 结合上下文判断，这是一条电脑操作指令
3. 如果整句话已经清晰，原样返回，不要改动
4. 只输出修正后的文本，不要任何解释
5. 你只校正，禁止回答任何问题'''}, 
                        {'role': 'user', 'content': text}],
                        options={'temperature': 0.3}, think=False)
    print('=' * 30 + f'当前模型:{response.model}' + '='* 30)
    return response.message.content.strip()

def edit_text(text):
    result = prompt('指令(可修改，按enter确认)：', default=text).strip()
    return result

chat_history = []
MAX_HISTORY = 20

def classify_content(content):
    print(f'Classifying intent...MODEL: {olla_models["qwen3.5"]}')
    response = ollama.chat(
        model=olla_models['qwen3.5'],
        messages=[{'role': 'user', 'content': f'''判断意图，只输出一个词：automation/chat/review
如果开头明确写了chat:或review:或automation:，直接给与相应判断！
automation：执行电脑操作指令
chat：普通聊天或问问题
review：讨论或修改刚才执行过的自动化操作
除非十分明确要chat或review,否则都是automation!
"{content}"'''}],
        options={'temperature': 0}, think=False,
    )

    result = response.message.content.strip().lower()
    for intent in ['automation', 'chat', 'review']:
        if intent in result:
            print(f'Classified result: {intent}')
            return intent
    return 'automation'

def chat_mode(content, with_context=False):
    global chat_history
    # 第一次进入，初始化system prompt
    if not chat_history:
        chat_history.append({
            'role': 'system',
            'content': '''
你是一个风趣幽默的助手，自然语言风格聊天，废话较少，直入主题。
禁止用**等符号。
回复可以中英混用，要求地道英文
看到我的英文，如果有语法问题，你要先纠正语法问题，再回答其他问题'''
        })

    # 需要背景且还没加过
    if with_context and not any('自动化系统背景' in m.get('content', '') for m in chat_history):
        chat_history.append({
            'role': 'user',
            'content': f'以下是我们的自动化系统背景，供你参考：\n{system_prompt}'
        })
        chat_history.append({
            'role': 'assistant',
            'content': '收到，我了解了你的自动化系统，有什么想聊的？'
        })

    chat_history.append({'role': 'user', 'content': content})
    if len(chat_history) > MAX_HISTORY:
        chat_history = chat_history[:1] + chat_history[-(MAX_HISTORY-1):]
    for _ in range(2):
        try:
            response = get_current_client().chat.completions.create(
                model=get_current_model(),
                messages=chat_history,
                temperature=0.7,
            )
        except Exception as e:
            if '429' in str(e):
                print('429...切换下一个')
                switch_model()
                continue
            raise
        break
    else:
        tts('模型两次失败，等会吧')
        return

    reply = response.choices[0].message.content
    chat_history.append({'role': 'assistant', 'content': reply})
    print('=' *20 +f'{PROVIDERS[CLIENT_INDEX]["client"].base_url} / {get_current_model()}'+'=' *20 + response.model+'=' *30 )
    print(f'助手：{reply}')
    tts(reply)

PROGRAMS = {
    'chrome': 'Google Chrome',
    '网易云音乐': 'NeteaseMusic',
    'qq音乐': 'QQMusic',
    'word': 'Microsoft Word',
    'powerpoint': 'Microsoft PowerPoint',
    '废纸篓': os.path.expanduser('~/.Trash'),
    '下载': os.path.expanduser('~/Downloads'),
    '桌面': os.path.expanduser('~/Desktop'),
    '文档': os.path.expanduser('~/Documents'),
    '图片': os.path.expanduser('~/Pictures'),
}
ACCOUNTS = {
     '126': {
        'keywords': ['126', '126邮箱', '网易邮箱'],
        'url': 'https://mail.126.com',
        'email': 'xshaw1987',
        'password': 'Duskreaper123',
        'frame': 'passport.126.com',
        'email_selector': '[name="email"]',
        'password_selector': '[name="password"]',
        'submit_selector': '#dologin',
    },
    'icloud': {
        'keywords': ['icloud', '苹果云', 'apple'],
        'url': 'https://www.icloud.com',
        'email': 'xxx@icloud.com',
        'password': 'xxx',
        'frame': None  # 待调试后填入
    },
}

if __name__ == '__main__':
    if WAKE_RECORD_SECONDS > 0:
        print('Wake record mode. Say the wake word during the recording window.')
        record_wake_sample(WAKE_RECORD_SECONDS)
        exit()

    if WAKE_TEST:
        print('Wake test mode. Say the wake word, or press Ctrl+C to stop.')
        wake_word_listener()

    print('正在加载Whisper model...')
    mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo='mlx-community/whisper-large-v3-turbo')

    # 启动时调用一次
    audio_cache = asyncio.run(_preload())

    threading.Thread(target=wake_word_listener, daemon=True).start()

    listener = keyboard.Listener(on_press=on_press, on_release=None)
    listener.start()
    play_cached('ready')

    last_active_time = time.time()

    while True:
        if not is_active.is_set() and not waiting_input.is_set():
            print('等待唤醒词...')
            play_cached('waiting')
            while True:
                if waiting_input.is_set():
                    break
                if wake_event.is_set():
                    wake_event.wait()
                    wake_event.clear()
                    is_active.set()
                    last_active_time = time.time()
                    play_cached('activated')
                    start_recording.set()
                    break
                if quit_event.is_set() or time.time() - last_active_time > 600:
                    play_cached('bye')
                    exit()
                time.sleep(0.1)

        commands = []

        if time.time() - last_active_time > 600:
            tts('长时间未操作，自动休眠')
            is_active.clear()
            continue
        if quit_event.is_set():
            is_active.clear()
            play_cached('bye')
            break
        if is_active.is_set():
            play_cached('speaking')
            print('请说出你的指令')
            start_recording.set()
            raw_content = listen()
            start_recording.clear() 
            is_active.clear()

            if waiting_input.is_set():
                is_active.clear()
                continue

            if not raw_content.strip():
                continue

            corrected = correct_text(raw_content)
            content = edit_text(corrected)
            if content.strip().endswith(('exit', 'quit', '退出')):
                wake_event.clear()
                is_active.clear()
                continue
            commands = [content]

        elif waiting_input.is_set():
            if wake_event.is_set():
                wake_event.clear()
                waiting_input.clear()
                is_active.set()
                last_active_time = time.time()
                play_cached('activated')
                start_recording.set()
                continue     

            play_cached('typing')
            raw = input('请输入指令(使用;来分割多组命令)： ')
            commands = [c.strip() for c in raw.replace('；', ';').split(';') if c.strip()]
            last_active_time = time.time()

        if not commands:
            continue
        elif any(c in ('exit', 'quit', '退出') for c in commands):
            break

        last_active_time = time.time()

        for content in commands:
            intent = classify_content(content)
            if intent == 'review':
                tts('刚才有什么问题？')
                chat_mode(content, with_context=True)
                continue
            elif intent == 'chat':
                tts('你想聊点什么呢？')
                chat_mode(content, with_context=False)
                continue

            similar = find_similar(content, threshold=0.75)
            datas, output = generate_actions(content, similar=similar)
            print(f'{content} -- {datas}')
            if datas is None:
                continue

            last_coords = None
            last_program = None

            persistent_namespace = {'user_input': content, '__file__': __file__,
                                    'BeautifulSoup': BeautifulSoup,
                                    'sync_playwright': sync_playwright,
                                    'ACCOUNTS': ACCOUNTS,
                                    'os': _os,
                                    'Path': Path,
                                    'subprocess': subprocess,
                                    'time': time,
                                    'json': json,
                                    're': re,
                                    }
            queue = deque(datas)
            exec_retry = 0

            while queue:
                data = queue.popleft()
                if data['action'] == 'run':
                    app_name = data['program'].replace('.exe', '').lower().strip()
                    program = PROGRAMS.get(app_name)
                    for key in PROGRAMS:
                        if key in app_name or app_name in key:
                            program = PROGRAMS[key]
                            break

                    args = data.get('file', '')
                    if app_name == 'chrome':
                        chrome_args = ['open', '-a', 'Google Chrome', '--args',
                                       '--remote-debugging-port=9222',
                                       '--profile-directory=Default']
                        if args:
                            chrome_args.append(args)
                        subprocess.Popen(chrome_args)
                    elif args:
                        subprocess.run(['open', args])
                    elif program and program.startswith('/'):
                        subprocess.run(['open', program])
                    elif program:
                        subprocess.run(['open', '-a', program])
                    else:
                        subprocess.run(['open', '-a', app_name])
                    last_program = program or app_name
                    print(last_program)
                elif data['action'] == 'type':
                    if any('\u4e00' <= c <= '\u9fff' for c in data['text']):
                        pyperclip.copy(data['text'])
                        pg.hotkey('cmd', 'v')
                    else:
                        pg.write(data['text'], interval=0.1)
                elif data['action'] == 'activate':
                    app_name = data['program'].lower()
                    program = PROGRAMS.get(app_name, app_name)
                    open_or_activate(Path(program).name, program)
                    time.sleep(1)
                    last_program = program
                elif data['action'] == 'hotkey':
                    pg.hotkey(*data['keys'])

                elif data['action'] == 'visual_locate':
                    coordinates = visual_locate([data['target']])
                    last_coords = coordinates.get(data['target'])
                    print(last_coords)

                elif data['action'] == 'mouseclick':
                    if last_coords:
                        x, y = last_coords
                    else:
                        x=data['x'] if data['x'] is not None else pg.position().x
                        y = data['y'] if data['y'] is not None else pg.position().y
                    pg.click(x, y,
                            button=data['button'], clicks=data.get('clicks', 1),
                            interval=data.get('interval', 0.1), duration= 1)
                elif data['action'] == 'mousescroll':
                    pg.scroll(data.get('amount', 200))
                elif data['action'] == 'unknown':
                    print(f'识别失败已记录{content}')
                elif data['action'] == 'sleep':
                    time.sleep(data['seconds'])
                elif data['action'] == 'code':
                    script = '\n'.join(data['script'])
                    print(f'准备执行代码： \n{script}')
                    play_cached('execute_inquiry')
                    confirm = prompt('是否执行此次代码(y/n)？')
                    if confirm.lower() == 'y':
                        try:
                            play_cached('executing')
                            exec(script, persistent_namespace)
                        except Exception as e:
                            if 'connect_over_cdp' in str(e) or 'Connection refused' in str(e):
                                print('Chrome未启动调试端口，正在重启...')
                                tts('Chrome未启动调试端口，正在重启...')
                                subprocess.Popen(
                                    ['open', '-a', 'Google Chrome', '--args',
                                     '--remote-debugging-port=9222', '--profile-directory=Default']
                                )
                                time.sleep(3)  # 等Chrome启动
                                queue.extendleft(reversed([data]))
                            else:
                                exec_retry += 1
                                if exec_retry >= 2:
                                    print(f'已重试{exec_retry}次，放弃了')
                                    tts(f'已重试{exec_retry}次，放弃了')
                                    queue.clear()
                                else:
                                    print(f'执行错误： {e}\n 转交模型再次处理！')
                                    
                                    tts('执行错误，转交模型再次处理！')
                                    extra = [
                                        {'role': 'user', 'content': content},
                                        {'role': 'assistant', 'content': output},
                                        {'role': 'user', 'content': f'执行报错：{e}\n请修复代码，只输出JSON数组'},
                                    ]
                                    datas, output = generate_actions(content, extra_messages=extra)
                                    if datas:
                                        queue.extendleft(reversed(datas))
                        else:
                            result = persistent_namespace.pop('_result', None)
                            if result and result != 'unknown_site':
                                print(result[:3000])
                                next_prompt = f'''用户原始指令：{content}
        页面元素列表（格式为 选择器 type=类型 (说明) frame=所在frame地址）：
        {result}
        规则：
        1. 浏览器已打开，page对象已存在，不要重新launch浏览器
        2. frame地址不是主页面的元素，必须用 next(f for f in page.frames if "子域名" in f.url) 获取frame后再操作
        3. id以auto-id开头的绝对不能用，必须改用name属性
        4. 所有操作合并到一个code action，不能发明新action类型，不要赋值_result
        5. 账号密码从user_input里提取，user_input = '{content}'
        6. 填写用户名后如果密码框disabled，需要先等待：page.wait_for_selector("#password:enabled")'''
                                new_datas, _ = generate_actions(next_prompt)
                                if new_datas:
                                    queue.extendleft(reversed(new_datas))
                            elif result == 'unknown_site':
                                next_prompt = f'''用户原始指令：{content}
        page对象已存在但页面为空，绝对不能调用sync_playwright()或launch()。
        请根据指令判断目标网站登录页url，生成Playwright脚本：
        1. page.goto(目标登录页url)
        2. page.wait_for_load_state('networkidle')
        3. 遍历所有frame抓取表单元素赋值给_result
        格式参考system prompt里的网页操作示例。'''
                                new_datas, _ = generate_actions(next_prompt)
                                if new_datas:
                                    queue.extendleft(reversed(new_datas))
                    else:
                        print('已取消')
                        play_cached('canceling')
            if not similar:
                tts('想保存本次执行结果吗？')
                save = prompt('保存本次结果？（y/n）')
                if save.lower() == 'y':
                    save_examples(content, datas)
                    tts('已保存本次执行结果')
            if intent == 'automation' and datas:
                chat_history.append({
                                'role': 'system',
                                'content': f'用户执行了指令：{content}\n生成的动作：{json.dumps(datas, ensure_ascii=False)}'
                            })
