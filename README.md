# Traffic Signal Optimization

## Setup

```
python -m venv env
env/bin/pip install torch torchvision numpy scipy pillow matplotlib
```

## Train

### Stage 1 - Vehicle Density Estimation
```
env/bin/python train/train_estimation.py
```

### Stage 2 - Flow Prediction (15min, 30min, 1hr)
```
env/bin/python train/train_prediction.py
```

## Test

### Test estimation model
```
env/bin/python test/test_estimation.py
```

### Test prediction models
```
env/bin/python test/test_prediction.py
```

### Test full pipeline (estimate + store + predict)
```
env/bin/python test/test_pipeline.py
```
