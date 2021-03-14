import json
import random
import pickle
import requests
import redis
import ahocorasick
import itertools as it
from neo4j import GraphDatabase
from util.tools import get_logger
from .config import Config


logger = get_logger()


class Manager:
    def __init__(self):
        self.n4j_driver = GraphDatabase.driver( **Config.NEO4J_CONFIG)
        self.redis_conn_pool = redis.ConnectionPool(**Config.REDIS_CONFIG)
        logger.info("Redis 连接正常   neo4j 连接正常")

        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s" \
                % (Config.client_id, Config.client_secret)
        res = requests.get(url)
        access_token = eval(res.text)["access_token"]
        # 根据 access_token 获取聊天机器人接口数据
        self.unit_chatbot_url = "https://aip.baidubce.com/rpc/2.0/unit/service/chat?access_token=" + access_token

        with open(Config.ac_bin_path, "rb") as f:
            self.symptom_AC = pickle.load(f)


    def init_process(self, inputs, redis_conn, uid):
        "首次对话启动过程"
        logger.info("该用户近期首次发言, 不进行句对匹配检测, 准备请求neo4j查询服务...")

        neo_res = self.query_neo4j(inputs)
        logger.info(f"neo4j查询服务请求成功, 返回结果: {neo_res}")

        # 判断如果结果为空列表, 访问百度机器人
        if not neo_res:
            return self.unit_chat(inputs)

        # redis缓存为 previous，计算下一句的匹配程度
        self.redis_conn.hset(str(uid), "prev_disease", str(neo_res))
        self.redis_conn.expire(str(uid), Config.ex_time)  # 过期时间

        # 使用规则模板生成回复
        res = "，".join(neo_res)
        logger.info("使用规则对话模板进行返回对话的生成: ")
        return Config.reply_pattern["2"]%res


    def main_process(self, inputs, previous, redis_conn, uid):
        "对话状态检测，回复内容生成"
        logger.info("开始检测是否新对话...")
        try:
            data = {"text1": previous, "text2": inputs}
            result = requests.post(Config.model_serve_url, data=data, timeout=Config.TIMEOUT)
            if not result.text:
                return self.unit_chat(inputs)
            logger.info(f"对话状态检测结果为: {result.text}")
        except Exception as e:
            logger.info(f"对话状态模型服务异常: {e}")
            return self.unit_chat(inputs)

        syms_of_input = self.find_symptom_words(inputs)
        if len(syms_of_input) > 0:
            logger.info("准备请求neo4j查询服务")
            neo_res = self.query_neo4j(syms_of_input)
            if len(neo_res) == 0:
                return "抱歉，未能得出有关的诊断结果。请尝试重新描述症状吧"
        # 没有结果，调用外部聊天接口
        else:
            return self.unit_chat(inputs)

        logger.info(f"neo4j返回结果: {neo_res}")

        old_disease = redis_conn.hget(str(uid), "prev_disease")  # 上一次得到的疾病名称
        if old_disease:
            res = list(set(neo_res) & set(eval(old_disease)))
            if len(res) == 0:  # 不同病症，重启记录
                res = list(set(neo_res))
            disease_record = res
        else:
            res = list(set(neo_res))
            disease_record = res

        redis_conn.hset(str(uid), "prev_disease", str(disease_record))  # 更新 disease_record
        redis_conn.expire(str(uid), Config.ex_time)  # 重置 缓存时间

        logger.info("使用规则对话模板进行返回对话的生成: ")
        if not res:
            return Config.reply_pattern["4"]
        else:
            if len(res) <= 5:
                res = " | ".join(res)
            else:
                res = " | ".join(res[: 5])
            return Config.reply_pattern["2"]%res


    def query_neo4j(self, sym_list):
        ''''
        根据用户对话文本中疾病症状, 返回对应的疾病名称
        sym_list: 疾病症状
        return: 用户描述的症状所对应的的疾病名称列表
        '''
        with self.n4j_driver.session() as session:
            # 匹配句子中存在的所有症状节点
            cypher = ""
            for sym in sym_list:
                cypher += "MATCH (d:Disease)-[*1]-(:Symptom {name: %r}) " %sym
            cypher += "RETURN d.name"
            record = session.run(cypher)
            result = list(map(lambda x: x[0], record.values()))
        return result


    def unit_chat(self, inputs, user_id="9527"):
        "调用百度UNIT接口，回复聊天内容"

        # 请求数据，主要是填充 query 值
        post_data = {
                    "log_id": str(random.random()),
                    "request": {
                        "query": inputs,
                        "user_id": user_id
                    },
                    "session_id": "",
                    "service_id": "S48867",
                    "version": "2.0"
                }
        res = requests.post(url=self.unit_chatbot_url, json=post_data)

        # result
        unit_chat_obj = json.loads(res.content)
        if unit_chat_obj["error_code"] != 0:  # 错误
            return Config.reply_pattern["5"]

        # 解析聊天接口返回数据，找到返回文本内容 result -> response_list -> schema -> intent_confidence(>0) -> action_list -> say
        unit_chat_response_list = unit_chat_obj["result"]["response_list"]
        # 随机选取一个"意图置信度"不为0的回答
        unit_chat_response_obj = random.choice(
            [unit_chat_response for unit_chat_response in unit_chat_response_list if unit_chat_response["schema"]["intent_confidence"] > 0.0])
        result = random.choice(unit_chat_response_obj["action_list"])["say"]

        return result


    def find_symptom_words(self, string):
        "匹配出症状字符串，返回 list"
        res = []
        for end_index, (order, sym) in self.symptom_AC.iter(string):
            res.append(sym)
        return res


