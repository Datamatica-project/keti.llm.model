from transformers import AutoProcessor

base_model = "google/gemma-3-4b-it" # 너가 쓴 베이스 모델
target_dir = r"C:\Users\dm_ohminchan\RAGLLM-Feature-model-train\models\trainer_output\gemma_final"

print("원본 Gemma3 프로세서 로드 중...")
processor = AutoProcessor.from_pretrained(base_model)

print("checkpoint-2352 폴더에 preprocessor 저장 중...")
processor.save_pretrained(target_dir)

print("완료! 이제 checkpoint-2352 안에 preprocessor_config.json 이 생겼을 거야.")
