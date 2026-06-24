#

import ollama
import pyautogui as pg
import io
import base64
import re
import json
# from plyer import notification


def visual_locate(targets):
    screenshot = pg.screenshot()
    w, h = screenshot.size
    screenshot = screenshot.resize((w // 2, h // 2))

    buf = io.BytesIO()
    screenshot.save(buf, format='PNG')
    print('已截图保存，正在上传model...')
    image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    targets_str = ','.join(targets)

    response = ollama.chat(
        model='qwen2.5vl:7b',
        messages=[{
            'role': 'user',
            'content':
            f'''在截图中，找到以下元素的中心点坐标:{targets_str}，不要遗漏！
            输出真实坐标，不要使用示例中的数字。
            以JSON数组格式输出，不要任何其他文字！
            单组数据格式：{{"name": "地址栏", "x": 111, "y": 111}}
            找不到的元素：{{"name": "unknown", "x": null, "y": null}}

            Json数组示例格式：[
            {{"name": "地址栏", "x": 111, "y": 111}},
            {{"name": "关闭按钮", "x": 111, "y": 111}}
            ]
            单条指令也必须用数组包裹：
            [{{"name": "地址栏", "x": 111, "y": 111}}]
            ''',
            'images': [image_b64]
        }]
    )
    print(response.message.content)

    return parse_visual_location(response.message.content)


def parse_visual_location(raw_output, scale=2):
    try:
        datas = json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if match:
            try:
                datas = json.loads(match.group().replace("'", '"'))
            except json.JSONDecodeError:
                notification.notify(
                    title='解析json失败', message=match.group()[:100], timeout=1
                )
                return {}
        else:
            notification.notify(
                title='解析json失败', message=raw_output[:100], timeout=1
            )
            return {}

    centers = {}
    for item in datas:
        if item['name'] == 'unknown' or item['x'] is None:
            continue
        centers[item['name']] = (item['x'] * scale, item['y'] * scale)
        print(f'{item["name"]} 坐标已准备好。')
    return centers

