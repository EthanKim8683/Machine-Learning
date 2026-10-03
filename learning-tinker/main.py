import tinker

service_client = tinker.ServiceClient()

training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B", rank=16
)

tokenizer = training_client.get_tokenizer()