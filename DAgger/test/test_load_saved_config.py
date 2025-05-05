import pickle

with open("/home/atari/workspace/DAgger/example/data/behavior_cloning/trot/May_05_2025_11_54_22/dataset/config.pkl", "rb") as f:
    cfg = pickle.load(f)

print(cfg)  # This will be an OmegaConf object (pretty-printable)
