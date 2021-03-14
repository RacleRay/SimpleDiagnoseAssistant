import requests


# 本机开启测试
# flask run -p 5001


if __name__ == "__main__":
    url = "http://0.0.0.0:5001/v1/recognition/"

    data = {"text1":"可以咨询问题吗", "text2":"今夕是何年"}

    res = requests.post(url, data=data)

    print("预测样本:", data["text1"], "|", data["text2"])
    print("预测结果:", res.text)