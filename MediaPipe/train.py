"""
訓練安全帽 / 繩帶分類器（MobileNetV2 Transfer Learning）

資料夾結構：
  dataset/
    helmet/
      positive/   ← 戴安全帽的頭頂 ROI 圖片
      negative/   ← 未戴安全帽的頭頂 ROI 圖片
    strap/
      positive/   ← 繩帶已扣的下巴 ROI 圖片
      negative/   ← 繩帶未扣的下巴 ROI 圖片

執行方式：
  python train.py
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

IMG_SIZE   = (96, 96)
BATCH_SIZE = 16
EPOCHS     = 20

def build_model():
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False  # 凍結預訓練權重，只訓練頂層

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(2, activation="softmax"),  # [positive, negative]
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def count_images(data_dir):
    total = 0
    for root, _, files in os.walk(data_dir):
        total += sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return total

def train(data_dir: str, output_path: str):
    n = count_images(data_dir)
    print(f"[INFO] 資料集：{data_dir}，共 {n} 張圖片")
    if n < 20:
        print(f"[警告] 圖片數量過少（{n} 張），建議每類至少收集 50 張")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )

    train_gen = datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training"
    )
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation"
    )

    print(f"[INFO] 類別對應：{train_gen.class_indices}")

    model = build_model()

    callbacks = [
        EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(output_path, save_best_only=True, verbose=1),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print(f"[DONE] 模型已儲存：{output_path}\n")

if __name__ == "__main__":
    print("=== 訓練安全帽頭頂分類器 ===")
    train("dataset/helmet", "model_helmet.h5")

    print("=== 訓練繩帶扣合分類器 ===")
    train("dataset/strap", "model_strap.h5")
