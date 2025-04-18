import torch

print(torch.cuda.is_available())  # 返回True表示PyTorch可以访问CUDA

print("="*40)
print("CUDA基本检查:")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"当前设备: {torch.cuda.current_device()}")
print(f"设备名称: {torch.cuda.get_device_name(0)}")

print("\n" + "="*40)
print("详细设备信息:")
print(f"设备数量: {torch.cuda.device_count()}")
print(f"设备能力: {torch.cuda.get_device_capability(0)}")
print(f"设备属性: {torch.cuda.get_device_properties(0)}")

print("\n" + "="*40)
print("内存信息:")
print(f"已分配内存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
print(f"保留内存: {torch.cuda.memory_reserved()/1024**2:.2f} MB")