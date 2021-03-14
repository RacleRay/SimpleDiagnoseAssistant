
class Config:
    REDIS_CONFIG = {
        "host": "127.0.0.1",
        "port": 6379
    }


    # redis中用户对话信息保存的过期时间
    ex_time = 36000


    NEO4J_CONFIG = {
        "uri": "bolt://127.0.0.1:7687",
        "auth": ("neo4j", "123456"),
        "encrypted": False
    }


    model_serve_url = "http://127.0.0.1:5001/v1/recognition/"


    # 设置服务的超时时间 s
    TIMEOUT = 5


    # Unit 服务请求
    client_id = "NG51OKH7YSO5LjL2gvQRVSlX"
    client_secret = "OXG0whpjXTW1kpPg29w5TGIkPXtCIf0X"


    reply_pattern = {
                    "1": "亲爱的用户, 在线医患问答机器人为您服务，请您说明一些当前的症状。",
                    "2": "根据您当前的症状描述, 您可能患有以下疾病: 【%s】。还有其他的症状吗?",
                    "3": "对不起, 您所说的内容超出了在线医生的知识范围. 请尝试换一些描述方式！",
                    "4": "您的这次描述并没有给我带来更多信息，请您继续描述您的症状.",
                    "5": "不好意思，相关内容正在学习中，请问问其他的问题吧"
                    }

    ac_bin_path = "../conv_serve/weights/AC.bin"
