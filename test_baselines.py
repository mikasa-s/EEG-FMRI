"""测试修改后的 baseline 模型列表"""
import torch
from mmcontrast.baselines.eeg_baseline import EEGBaselineModel, VALID_MODEL_NAMES

print('=== 验证模型列表 ===')
print(f'支持的模型：{VALID_MODEL_NAMES}')
print(f'总数：{len(VALID_MODEL_NAMES)}')

print('\n=== 测试模型实例化 ===')
x = torch.randn(2, 62, 30, 200)
models_to_test = ['svm', 'labram', 'cbramod', 'eeg_deformer', 'eegnet', 'conformer']
errors = []

for name in models_to_test:
    try:
        m = EEGBaselineModel(model_name=name)
        out = m(x)
        print(f'{name}: {out.shape} ✓')
    except Exception as e:
        errors.append((name, str(e)))
        print(f'{name}: FAILED - {e}')

if errors:
    print(f'\n失败的模型：{errors}')
else:
    print('\n✓ 所有模型测试通过')

# 测试强制 model_name 参数
print('\n=== 测试强制 model_name 参数 ===')
try:
    m = EEGBaselineModel()  # 应该报错
    print('ERROR: 应该报错但没有')
except TypeError as e:
    print(f'✓ 正确报错：{e}')

# 测试无效的 model_name
print('\n=== 测试无效的 model_name ===')
try:
    m = EEGBaselineModel(model_name='invalid_model')
    print('ERROR: 应该报错但没有')
except ValueError as e:
    print(f'✓ 正确报错：{e}')

# 测试别名不支持
print('\n=== 测试别名不支持 ===')
try:
    m = EEGBaselineModel(model_name='LaBraM')  # 应该转换为小写
    print(f'✓ 自动转换为小写：{m.model_name}')
except Exception as e:
    print(f'报错：{e}')
