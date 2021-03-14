import redis
import hashlib

import sys
# 项目根文件夹 绝对路径
PROJECT_PATH = "d:\\Code\\Deep_Learning\\Project\\NLP_Project\\Doctor"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from serve.conv_serve.manager import Manager
from flask import Flask, render_template, request
from util.tools import get_logger


logger = get_logger()

app = Flask(__name__, static_folder='templates', static_url_path='')


# 因为在使用sklearn tf-idf模型使用时自定义了分词方法， pickle load时需要该方法在__main__空间中
def split_func(s): return s.split()


bot = Manager()


@app.route('/api/chat', methods=['GET'])
def chat():
    query = request.args.get('message')
    logger.info(f"用户：{query}")

    md5 = hashlib.md5()
    md5.update(str(request.remote_addr).encode('utf-8'))
    uid = md5.hexdigest()

    r = redis.StrictRedis(connection_pool=bot.redis_conn_pool)

    prev_text = r.hget(str(uid), "prev_text")
    r.hset(str(uid), "prev_text", query)  # 存入redis, 下一次访问的"上一句话"

    if prev_text:
        res = bot.main_process(query, prev_text, r, uid)
    else:
        res = bot.init_process(query, r, uid)

    logger.info(f"回答：{res}")

    return res


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)