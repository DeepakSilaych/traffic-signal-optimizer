# Traffic Signal Optimization

## Setup

```
python -m venv env
env/bin/pip install torch torchvision numpy scipy pillow matplotlib
env/bin/pip install eclipse-sumo sumo-rl stable-baselines3 pettingzoo torch_geometric
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

### Generate SUMO network (30-junction Indian city)
```
export SUMO_HOME=$(env/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
env/bin/python sumo_config/generate_network.py
```

### Stage 3 - Signal Optimization (Hierarchical GNN + MAPPO)
```
export SUMO_HOME=$(env/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
env/bin/python train/train_signal.py
```

### Test signal optimizer
```
export SUMO_HOME=$(env/bin/python -c "import sumo; print(sumo.SUMO_HOME)")
env/bin/python test/test_signal.py
```
