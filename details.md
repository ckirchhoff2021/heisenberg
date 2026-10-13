# 支持的任务执行明细
- grpo训练
- grpo_opd训练
- jev训练
- 模型单次推理测试
- 模型评测

## 模型评测脚本
- `evaluate/grpo_eval_vllm.py`：用于评估 argos 数据集上的性能
- 在线推理测试如下：
```bash
python -m evaluate.grpo_eval_vllm --api-base http://localhost:8000/v1 --model-name qwen3.5-2b
```

## 模型单次推理测试
- `tests/test_vllm_inference.py`：用于测试 模型在 argos 数据集上的推理效果
- 在线推理测试如下：
```bash
python  -m tests.test_vllm_inference --api-base http://localhost:8000/v1 --model-name qwen3.5-2b
```





