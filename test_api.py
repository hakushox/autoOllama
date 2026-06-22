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
from window_utils import open_or_activate, is_running, _find_hwnd
import time
from test_ollama_pic import visual_locate
import platform
import os
from pathlib import Path
import numpy as np
from collections import deque
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import sounddevice as sd
from faster_whisper import WhisperModel
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
    'gemma': 'gemma4', 'qwen3.5': 'qwen3.5:4b', 'qwen2.5':'qwen2.5vl:7b',
}
# ollama.chat(model='qwen3:8b', messages=[...], options={'temperature': 0},think=False)
# model='gemma4',options={'temperature': 0.7, 'top_p': 0.9}
wake_event = threading.Event()
quit_event = threading.Event()

is_active = threading.Event()

oww_model = Model(wakeword_models=[r'D:\Program Files\Lib\site-packages\openwakeword\resources\models\hello_mercy.onnx',
        'alexa_v0.1.onnx', 'alexa_v0.1.onnx'], inference_framework='onnx')
WAKE_WORD = 'hello_mercy' 
QUIT_WORD = 'alexa_v0.1.onnx'    

last_wake_time = datetime.datetime.min

def wake_word_listener():
    def callback(indata, frames, time, status):
        audio = (indata[:,0] * 32768).astype(np.int16)
        prediction = oww_model.predict(audio)
        global last_wake_time
        if prediction[WAKE_WORD] > 0.7 and not is_active.is_set():
            last_wake_time = datetime.datetime.now()
            print('检测到唤醒词')
            wake_event.set()
        elif prediction[QUIT_WORD] > 0.9:
            diff = (datetime.datetime.now() - last_wake_time).total_seconds()
            print(f'退出词得分触发，距唤醒{diff:.1f}秒')
            if diff > 3:
                print('检测到退出词')
                quit_event.set()
            # if (datetime.datetime.now() - last_wake_time).total_seconds() > 3:
            #     print('检测到退出词')
            #     quit_event.set()

    with sd.InputStream(samplerate=16000, channels=1, dtype='float32',
                        blocksize=8000, callback=callback):
        while True:
            sd.sleep(100)

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
        sd.stop()
        buffer = asyncio.run(_speak(text, rate=rate))
        data, samplerate = sf.read(buffer)
        sd.play(data, samplerate)
        sd.wait()
    threading.Thread(target=_run, daemon=True).start()

async def _preload():
    responses = {
        'activated': 'Hello, 我在',
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
    sd.stop()
    sd.play(data, samplerate)
    sd.wait()

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

system_prompt = f'''你是控制助手，当前系统：{platform.system()}，用户名：{os.getlogin()}
给你操作指令中可能会有多个步骤。你要将每个操作步骤输出到一个JSON数组中。
JSON数组中每个元素是一条操作，不能有任何其他文字！
## Action类型
run：打开程序
{{"action":"run","program":"notepad"}}
chrome必须带profile：{{"action":"run","program":"chrome"}}
搜索引擎搜索关键词，搜索URL中文关键词禁止URL编码！
{{"action":"run","program":"chrome","file":"https://www.baidu.com/s?wd=关键词"}}
word空白文档：{{"action":"run","program":"winword","file":"/w"}}
视频播放器：{{"action":"run","program":"potplayer","file":"路径"}}
系统文件夹（回收站/下载/桌面/此电脑）：{{"action":"run","program":"回收站"}}
activate：激活已有窗口（不新建），置顶指定程序
{{"action":"activate","program":"chrome"}}
操作已打开的程序前必须先activate
type：输入文字
{{"action":"type","text":"hello"}}
hotkey：快捷键
{{"action":"hotkey","keys":["ctrl","s"]}}
切换Chrome已有标签时,禁止ctrl+t开新标签切换已有标签必须用ctrl+shift+a,规则：
[{{"action":"activate","program":"chrome"}},
{{"action":"hotkey","keys":["ctrl","shift","a"]}},
{{"action":"sleep","seconds":1}},
{{"action":"type","text":"关键词"}},
{{"action":"sleep","seconds":1}},
{{"action":"hotkey","keys":["enter"]}},{{"action":"sleep","seconds":1}}]
显示桌面：["win","d"]
mouseclick：鼠标点击，坐标未知时x/y填null
{{"action":"mouseclick","button":"left","clicks":1,"x":100,"y":200}}
mouseclick x/y为null时禁止生成mouseclick ，改用hotkey或type直接操作
mousescroll：滚动，正数向上负数向下，默认-200
{{"action":"mousescroll","amount":-200}}
sleep：等待
{{"action":"sleep","seconds":2}}
 
code：执行Python代码，script必须是字符串数组，每个元素是一行代码
{{"action":"code","script":["import os","path = 'D:/'","os.startfile(path)"]}}
 
unknown：无法完成的指令
{{"action":"unknown"}}
 
## 重要规则
文件操作：
- 新建任何文件必须用code，不能用run的file参数
- 桌面路径：os.path.expanduser('~/Desktop')
- 新建docx必须用python-docx的Document()不带参数，再save(path)
- 搜索文件有明确条件时直接匹配打开，无需列出让用户选
- Windows路径用'D:/'或'D:\\\\'，禁止用r'D:\\'
打开程序路径不确定时，先用Path.glob搜索C盘和D盘，找到再打开：
用户说"打开欧路词典"，输出：
[{{"action":"code","script":[
  "from pathlib import Path",
  "import os",
  "results = list(Path('C:/').glob('**/Eudic.exe')) + list(Path('D:/').glob('**/Eudic.exe'))",
  "if results:",
  "    os.startfile(str(results[0]))",
  "else:",
  "    print('未找到')"
]}}]
 
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
 
用户说"打开E盘video文件夹，问我要打开哪个"，输出：
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
]}}]
 
示例：
用户说"打开记事本然后输入hello"，输出：
[{{"action":"run","program":"notepad"}},{{"action":"sleep","seconds":2}},{{"action":"type","text":"hello"}}]
 
单条指令也必须用数组包裹：
[{{"action":"run","program":"notepad"}}]
 
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
        sd.stop()
    if key == keyboard.Key.f8:
        if not waiting_input.is_set():
            print('正在切换为手动输入模式...')
            waiting_input.set() #is_set() 返回 True
            sd.stop()


# def on_release(key):
#     if key == keyboard.Key.f9:
#         waiting_input.clear()  #is_set() 返回 False

def listen():
    while not start_recording.is_set():  # 等F9按下
        time.sleep(0.05)
    frames =[]
    display_text = ['']
    stop_event = threading.Event()

    def transcribe_loop():
        start = time.time()
        while not stop_event.is_set():
            time.sleep(1.5)
            if not frames:
                continue
            audio = np.concatenate(frames).squeeze()
            segments, _ = model.transcribe(audio, language='zh', vad_filter=True,
                initial_prompt='E盘, 海豹, 网易云，百度, baidu, gmail, 桌面...')
            text = ''.join(seg.text for seg in segments)
            display_text[0] = text
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
    segments, _ = model.transcribe(audio, language='zh', vad_filter=True,
        initial_prompt='E盘, 海豹, 网易云，百度, baidu, gmail, 桌面...')
    text = ''.join(seg.text for seg in segments)

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
                        options={'temperature': 0.3})
    print('=' * 30 + f'当前模型:{response.model}' + '='* 30)
    return response.message.content.strip()

def edit_text(text):
    result = prompt('指令(可修改，按enter确认)：', default=text).strip()
    return result

chat_history = []
MAX_HISTORY = 20

def classify_content(content):
    print(f'Classifying intent...MODEL: {olla_models["qwen2.5"]}')
    response = ollama.chat(
        model=olla_models['qwen3.5'],
        messages=[{'role': 'user', 'content': f'''判断意图，只输出一个词：automation/chat/review
如果开头明确写了chat:或review:或automation:，直接给与相应判断！
automation：执行电脑操作指令
chat：普通聊天或问问题
review：讨论或修改刚才执行过的自动化操作
除非十分明确要chat或review,否则都是automation!
"{content}"'''}],
        options={'temperature': 0,'think': False},
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
    '网易云音乐': r'"D:\Program Files (x86)\网易云音乐PC版\cloudmusic.exe"',
    'chrome':  r'C:\Program Files\Google\Chrome\Application\chrome.exe',  
    'potplayer':  r"D:\Program Files\DAUM\PotPlayer\PotPlayerMini64.exe", 
    'winword': r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE",
    'powerpoint': r"C:\Program Files (x86)\Microsoft Office\root\Office16\POWERPNT.EXE",
    'qq音乐': r"E:\Program Files (x86)\tencent\qqmusic\QQMusic.exe",
    '此电脑': 'explorer.exe',
    '回收站': 'explorer.exe shell:RecycleBinFolder',
    '下载': 'explorer.exe shell:Downloads',
    '桌面': 'explorer.exe shell:Desktop',
    '文档': 'explorer.exe shell:Personal',
    '图片': 'explorer.exe shell:My Pictures',
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
    print('正在加载Whisper model...')
    model = WhisperModel('large-v3-turbo', device='cuda', compute_type='float16')

    # 启动时调用一次
    audio_cache = asyncio.run(_preload())

    threading.Thread(target=wake_word_listener, daemon=True).start()

    listener = keyboard.Listener(on_press=on_press, on_release=None)
    listener.start()
    tts('助手已就绪，按F9手动输入', rate='+50%')

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
                                        'os': os,
                                    'Path': Path,
                                    'subprocess': subprocess,
                                    'time': time,
                                    'json': json,
                                    're': re,
                                    }
            auto_execute = False

            queue = deque(datas)
            exec_retry = 0

            while queue:
                data = queue.popleft()
                if data['action'] == 'run':
                    app_name = data['program'].replace('.exe', '').lower().strip()
                    program = app_name
                    for key in PROGRAMS:
                        if key in app_name or app_name in key:
                            program = PROGRAMS[key]
                            break
                                                
                    args = data.get('file', '')
                    if args:
                        subprocess.Popen(f'"{program}" "{args}"', shell=True)
                    elif app_name == 'explorer' or '此电脑' in app_name:
                        subprocess.Popen(program, shell=True)
                    elif app_name == 'chrome':
                        # --remote-debugging-port=9222 允许Playwright通过connect_over_cdp连接已有Chrome
                        # --new-window 每次新开窗口而不是新标签
                        # --profile-directory=Default 使用默认用户配置，保持登录状态
                        subprocess.Popen(
                            f'"{PROGRAMS["chrome"]}" --remote-debugging-port=9222 --new-window --profile-directory=Default',
                            shell=True
                        )
                    elif is_running(app_name):
                        open_or_activate(Path(program).name, program)
                    else:
                        subprocess.Popen(program, shell=True)
                    last_program = program
                    timeout = 10
                    start = time.time()
                    while not _find_hwnd(app_name):
                        if time.time() - start > timeout:
                            break
                        time.sleep(0.3)
                    print(last_program)
                elif data['action'] == 'type':
                    # if last_program:
                    #     time.sleep(0.3)
                    #     open_or_activate(Path(last_program).name, last_program)
                    if any('\u4e00' <= c <= '\u9fff' for c in data['text']):
                        pyperclip.copy(data['text'])
                        pg.hotkey('ctrl', 'v')
                    else:
                        pg.write(data['text'], interval=0.1)
                elif data['action'] == 'activate':
                    app_name = data['program'].lower()
                    program = PROGRAMS.get(app_name, app_name)
                    open_or_activate(Path(program).name, program)
                    time.sleep(1)
                    last_program = program
                elif data['action'] == 'hotkey':
                    # if last_program:
                    #     open_or_activate(Path(last_program).name, last_program)
                    #     time.sleep(0.3)
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
                    # print('是否执行？按F9说yes或no')
                    # confirm = listen()
                    # confirm = 'y' if 'yes' in confirm.lower() or '是' in confirm or '确认' in confirm else 'n'
                    if confirm.lower() == 'y':
                        try:
                            play_cached('executing')
                            exec(script, persistent_namespace)
                        except Exception as e:
                            if 'connect_over_cdp' in str(e) or 'Connection refused' in str(e):
                                print('Chrome未启动调试端口，正在重启...')
                                tts('Chrome未启动调试端口，正在重启...')
                                subprocess.Popen(
                                    f'"{PROGRAMS["chrome"]}" --remote-debugging-port=9222 --profile-directory=Default',
                                    shell=True
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
                                auto_execute = True
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
                                auto_execute = True
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
                # print('是否保存本次执行？按F9说yes或no')
                # save = listen()
                # save = 'y' if 'yes' in save.lower() or '是' in confirm or '确认' in confirm else 'n'
                if save.lower() == 'y':
                    save_examples(content, datas)
                    tts('已保存本次执行结果')
            if intent == 'automation' and datas:
                chat_history.append({
                                'role': 'system',
                                'content': f'用户执行了指令：{content}\n生成的动作：{json.dumps(datas, ensure_ascii=False)}'
                            })

