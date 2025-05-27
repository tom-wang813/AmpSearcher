trainer/
│
├── core/
│   ├── base_trainer.py          # 基礎訓練邏輯
│   └── custom_trainer.py        # 自定義訓練邏輯
│
├── losses/
│   ├── base_loss.py             # 基礎損失計算邏輯
│   └── custom_loss.py           # 自定義損失計算邏輯
│
├── optimizers/
│   ├── base_optimizer.py        # 基礎優化器配置邏輯
│   ├── custom_optimizer.py      # 自定義優化器配置邏輯
│   ├── base_scheduler.py        # 基礎學習率調度器配置邏輯
│   └── custom_scheduler.py      # 自定義學習率調度器配置邏輯
│
├── data/
│   ├── base_data_loader.py      # 基礎數據加載邏輯
│   ├── custom_data_loader.py    # 自定義數據加載邏輯
│   ├── base_data_preprocessor.py# 基礎數據預處理邏輯
│   └── custom_data_preprocessor.py # 自定義數據預處理邏輯
│
├── utils/
│   ├── logger.py                # 日誌記錄工具
│   └── metrics.py               # 評估指標
│
└── __init__.py