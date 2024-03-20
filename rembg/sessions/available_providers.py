import onnxruntime

providers = onnxruntime.get_available_providers()
print(providers)