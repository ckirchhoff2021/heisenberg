import os
import json
from datasets import Dataset
from PIL import Image

FILES_ROOT = "datas/aiops/files"
DEFAULT_PROMPT_FILE = "datas/aiops/files/prompts/argos-dp-prompt.md"


class TRLDatasets:
    """把 argos JSON 标签转成 TRL 会话式（多模态）数据集。

    输出列：
        prompt        : [{"role": "user", "content": "<image> <prompt>"} ]
        curve         : 曲线趋势真值，供 reward 函数使用
        recover       : 恢复状态真值，供 reward 函数使用
        images        : [图像绝对路径]
    """

    def __init__(
        self,
        data_files: str,
        prompt_file: str = DEFAULT_PROMPT_FILE,
        files_root: str = FILES_ROOT,
        check_images: bool = False,
    ):
        self.data_path = data_files
        with open(prompt_file, encoding="utf-8") as f:
            self.prompt_text = f.read().strip()
        self.files_root = files_root
        self.check_images = check_images

    def _image_path(self, record: dict) -> str:
        image_path = os.path.join(self.files_root, record["from"], record["image"])
        if self.check_images and not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        return image_path

    def _to_example(self, record: dict) -> dict:
        '''
        record: {
            "from": "gen",
            "image": "1.png",
            "curve": "normal",
            "recover": "recovered",
        }
        '''
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": f"<image> {self.prompt_text}"
                }
            ],
            # trl 1.15 server 模式需要 PIL Image（转 base64 发 vllm server），不能是路径字符串
            "images": [Image.open(self._image_path(record)).convert("RGB")],
            "curve": record["curve"],
            "recover": record["recover"],
        }

    def build(self) -> Dataset:
        with open(self.data_path, encoding="utf-8") as f:
            records = json.load(f)
        return Dataset.from_list([self._to_example(r) for r in records])


if __name__ == "__main__":
    ds = TRLDatasets(os.path.join(FILES_ROOT, "jsons/argos-trains.json")).build()
    print(ds)