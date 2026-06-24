import json
from pathlib import Path
from test_api import get_embedding, save_examples


EXAMPLES_FILE = Path(__file__).parent / 'ollama/examples.json'

def add_example(user_input, script, direct_execute=False):
    datas = [{"action": "code", "script": script}]
    save_examples(user_input, datas, direct_execute)
    print(f'已添加：{user_input}')

def update_example(user_input, script):
    examples = json.loads(EXAMPLES_FILE.read_text(encoding='utf-8'))
    found = False
    # 找到要修改的条目
    for ex in examples:
        if ex['input'] == user_input:
            datas = json.loads(ex['output'])
            datas[0]['script'] = script
            ex['output'] = json.dumps(datas, ensure_ascii=False)
            found = True
            break
    if found:
        EXAMPLES_FILE.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'已更新{EXAMPLES_FILE}')
    else:
        print(f'未找到：{user_input}')


user_input = '删除桌面上的 gameinstaller.dmg 文件'
script = [{"action":"code","script":["import os, subprocess","desktop = os.path.expanduser('~/Desktop')","target = os.path.join(desktop, 'gameinstaller.dmg')","if os.path.isfile(target):","    trash_path = os.path.expanduser('~/.Trash')","    subprocess.run(['mv', target, trash_path])","    print('已移动到垃圾箱')","else:","    print('文件未找到')"]}]

add_example(user_input, script)