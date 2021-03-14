import redis
from flask import Flask
from flask import request

from .manager import Manager, logger

app = Flask(__name__)


manager = Manager()


@app.route('/v1/main_serve/', methods=["POST"])
def main_serve():
    logger.info("进入对话逻辑服务, 正常运行...")

    # 接收来自werobot请求, uid: 用户唯一标识, text: 用户输入的文本信息
    uid = request.form['uid']
    text = request.form['text']

    # 从redis连接池中获得一个活跃的连接
    r = redis.StrictRedis(connection_pool=manager.redis_conn_pool)

    prev_text = r.hget(str(uid), "prev_text")
    r.hset(str(uid), "prev_text", text)  # 存入redis, 下一次访问的"上一句话"

    if prev_text:
        return manager.main_process(text, prev_text, r, uid)
    else:
        return manager.init_process(text, r, uid)