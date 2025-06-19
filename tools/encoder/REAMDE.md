# 默认 simple_token 编码器
python main.py

# 切换到 latent_path_token 编码器
python main.py encoder=latent_path_token

# 开启 lane+vehicle 邻居提取，调整半径
python main.py encoder=neighbor_token extractor.sdc_radius=30
