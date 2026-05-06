import xgboost as xgb
try:
    # Try to create a small matrix on the GPU
    xgb.XGBClassifier(tree_method='hist', device='cuda').fit([[1]], [1])
    print("SUCCESS: GPU training is enabled and working!")
except Exception as e:
    print(f"GPU Error: {e}")