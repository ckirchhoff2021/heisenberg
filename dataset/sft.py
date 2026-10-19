import os
import json
from typing import List, Dict, Any

from datasets import Dataset
from PIL import Image

FILES_ROOT = "datas/aiops/files"
DEFAULT_PROMPT_FILE = "datas/aiops/files/prompts/argos-dp-prompt-noreason.md"


class SFTDatasets:
    """把 argos JSON 转成 TRL SFT 会话式（多模态）数据集。

    数据级调整（与 GRPO 的关键差异）：
    - <trend>/<recover> 标签一律以 JSON 的 curve/recover 字段为准（100% 正确，
      规避 output 中 ~10% 的 recover 标注错误）。
    - 不输出 <reason>：去掉长文本 reason 后 completion 只剩标签，loss 不再被
      reason 复述稀释（原方案 reason 占 completion 84%，导致坍缩到固定输出）。

    输出列（prompt-completion 型，供 SFTTrainer 使用）：
        prompt     : [{"role": "user", "content": "<image> <prompt>"}]
        completion : [{"role": "assistant", "content": "<trend>...</trend><recover>...</recover>"}]
        images     : [PIL Image]
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

    def _build_output(self, record: dict) -> str:
        """构造 assistant 的完整输出文本（纯 <trend>/<recover> 标签，无 reason）。

        标签一律以 JSON 的 curve/recover 字段为准（100% 正确）。
        去掉 reason 是为了避免 completion loss 被长文本 reason 稀释（约占 84%），
        导致模型只学复述 reason 而学不到「图片→标签」映射、坍缩到固定输出。
        """
        curve = record["curve"]
        recover = record["recover"]
        return (
            f"<trend>{curve}</trend>\n"
            f"<recover>{recover}</recover>"
        )

    def _to_example(self, record: dict) -> dict:
        return {
            "prompt": [
                {"role": "user", "content": f"<image> {self.prompt_text}"}
            ],
            "completion": [
                {"role": "assistant", "content": self._build_output(record)}
            ],
            "images": [Image.open(self._image_path(record)).convert("RGB")],
        }

    def build(self) -> Dataset:
        with open(self.data_path, encoding="utf-8") as f:
            records = json.load(f)
        return Dataset.from_list([self._to_example(r) for r in records])


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FILES_ROOT, "jsons/argos-trains.json")
    ds = SFTDatasets(path).build()
    print(ds)
    print("\n--- 示例 ---")
    for i in range(2):
        print(json.dumps(ds[i]["completion"][0]["content"], ensure_ascii=False)[:300])
        print()
