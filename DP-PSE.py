import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os



# 1. T-Shape
def full_preprocessing_pipeline(file_path, window_size=20, sample_rate=0.5):
    print("1. 正在进行预处理：加载数据并全局打乱...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到数据集文件: {file_path}")

    df = pd.read_csv(file_path)


    df = df.sample(frac=1, random_state=42).reset_index(drop=True)


    if sample_rate < 1.0:
        df, _ = train_test_split(df, train_size=sample_rate, stratify=df['Label'], random_state=42)

    # Physical Semantic Mapping
    unique_ids = sorted(df['CAN_ID'].unique())
    id_to_idx = {can_id: i for i, can_id in enumerate(unique_ids)}
    joblib.dump(id_to_idx, 'id_to_idx_mapping.pkl')
    print(f"   - ID 映射表已导出，共 {len(unique_ids)} 个唯一 ID")

    df['ID_idx'] = df['CAN_ID'].map(id_to_idx)
    label_map = {'R': 0, 'DoS': 1, 'Fuzzy': 2, 'Gear': 3, 'RPM': 4}
    df['Label'] = df['Label'].map(label_map)

    data_cols = [f'DATA[{i}]' for i in range(8)]
    scaler = MinMaxScaler()
    df[data_cols] = scaler.fit_transform(df[data_cols])

    # 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    def create_sequences(data_df):
        ids = data_df['ID_idx'].values.astype('int32')
        payloads = data_df[data_cols].values.astype('float32')
        labels = data_df['Label'].values.astype('int32')
        num_samples = len(data_df) - window_size + 1

        X_id = np.zeros((num_samples, window_size), dtype='int32')
        X_pay = np.zeros((num_samples, window_size, 8), dtype='float32')
        y = np.zeros((num_samples,), dtype='int32')

        for i in range(num_samples):
            X_id[i] = ids[i: i + window_size]
            X_pay[i] = payloads[i: i + window_size]
            y[i] = labels[i + window_size - 1]
        return X_id, X_pay, y

    print("2. 生成滑动窗口序列 (构建 T-Shape 结构)...")
    X_id_train, X_pay_train, y_train = create_sequences(train_df)
    X_id_test, X_pay_test, y_test = create_sequences(test_df)

    return (X_id_train, X_pay_train, y_train), (X_id_test, X_pay_test, y_test), len(unique_ids), id_to_idx

# 2. Dual-Path PSE-IDS
def build_dual_path_ids(num_ids, window_size=10):
    """
    Path A: Temporal Path (LSTM)
    Path B: Semantic Path (Attention)
    """
    id_in = Input(shape=(window_size,), name='ID_Input')
    pay_in = Input(shape=(window_size, 8), name='Data_Input')

    # --- Stage 2: Physical Semantic Embedding ---
    id_emb = layers.Embedding(num_ids, 24,
                              embeddings_regularizer='l2',
                              name='Semantic_Embedding')(id_in)

    # --- Stage 3: Fused Physical Semantic Vector
    merged = layers.Concatenate(name='Feature_Vectorization')([id_emb, pay_in])

    # --- Stage 4:Preprocessing & Alignment---
    h_seq = layers.LayerNormalization(name='Input_Alignment')(merged)

    # --- A: Temporal Path
    path_a = layers.LSTM(32, return_sequences=True, dropout=0.2, name='Path_A_LSTM')(h_seq)
    path_a_pool = layers.GlobalAveragePooling1D(name='Temporal_Feature_Extract')(path_a)

    # --- B: Semantic Path
    path_b_attn = layers.MultiHeadAttention(num_heads=2, key_dim=8, name='Path_B_Attention')(h_seq, h_seq)
    # path_b_attn = layers.MultiHeadAttention(num_heads=2,
    #                                         key_dim=16,
    #                                         name='Path_B_Attention')(h_seq, h_seq)
    path_b_pool = layers.GlobalAveragePooling1D(name='Semantic_Feature_Extract')(path_b_attn)

    dual_fused = layers.Concatenate(name='Dual_Path_Fusion')([path_a_pool, path_b_pool])
    x = layers.Dense(16, activation='swish', name='Bottleneck_Projection')(dual_fused)
    # x = layers.Dense(32, activation='swish')(dual_fused)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(5, activation='softmax', name='Classification')(x)

    model = models.Model(inputs=[id_in, pay_in], outputs=out, name="DualPath_PSE_IDS")


    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
def visualize_results(model, history, id_map, X_test, y_test):
    print("\n3. 执行性能评估与可视化...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='#1f77b4', lw=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#ff7f0e', lw=2)
    plt.title('Model Loss Convergence')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc', color='#2ca02c', lw=2)
    plt.plot(history.history['val_accuracy'], label='Val Acc', color='#d62728', lw=2)
    plt.title('Model Accuracy Trends')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # t-SNE
    weights = model.get_layer('Semantic_Embedding').get_weights()[0]
    idx_to_id = {v: k for k, v in id_map.items()}
    tsne = TSNE(n_components=2, perplexity=min(30, len(weights) - 1), random_state=42, init='pca', learning_rate='auto')
    weights_2d = tsne.fit_transform(weights)

    plt.figure(figsize=(10, 7))
    plt.scatter(weights_2d[:, 0], weights_2d[:, 1], s=80, alpha=0.6, c='crimson', edgecolors='white')
    for i in range(min(100, len(weights_2d))):
        plt.annotate(hex(int(idx_to_id[i])), (weights_2d[i, 0], weights_2d[i, 1]), fontsize=7, alpha=0.8)
    plt.title('Learned Physical Semantic Map (t-SNE Clustering)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

    y_pred_probs = model.predict(X_test, batch_size=2048)
    y_pred = np.argmax(y_pred_probs, axis=1)
    class_names = ['Normal', 'DoS', 'Fuzzy', 'Gear', 'RPM']
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Dual-Path IDS Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("\n" + "=" * 80)
    print("=" * 80)
    acc = accuracy_score(y_test, y_pred)
    print(f" Overall Accuracy (总体准确率) : {acc:.6f}\n")

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

    print(f"{'Attack Type (攻击类型)':<22} | {'Precision (精确率)':<15} | {'Recall (召回率)':<15} | {'F1-Score':<15}")
    print("-" * 75)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<22} | {precision[i]:<15.4f} | {recall[i]:<15.4f} | {f1[i]:<15.4f}")
    print("-" * 75)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    print(f"{'Macro Avg (宏平均)':<22} | {macro_p:<15.4f} | {macro_r:<15.4f} | {macro_f1:<15.4f}")
    print("=" * 80 + "\n")


def calculate_ttf_metrics(model, X_test, y_test, window_size=10, avg_can_interval_ms=0.5):
    """
     ##Time To Detection
    """
    print("\n--- 开始计算 TTF (检测时延) 指标 ---")
    y_pred_probs = model.predict(X_test, batch_size=2048)
    y_pred = np.argmax(y_pred_probs, axis=1)

    detection_delays = []
    i = 0
    while i < len(y_test):
        if y_test[i] != 0:
            for j in range(i, len(y_test)):
                if y_pred[j] == y_test[j]:
                    delay_frames = j - i
                    ttf_ms = (delay_frames + window_size) * avg_can_interval_ms
                    detection_delays.append(ttf_ms)
                    break
            while j < len(y_test) and y_test[j] != 0:
                j += 1
            i = j
        else:
            i += 1

    avg_ttf = np.mean(detection_delays) if detection_delays else 0
    print(f" 平均检测时延 (TTD/TTF): {avg_ttf:.4f} ms")
    return avg_ttf

if __name__ == "__main__":
    import time

    data_path = '../DP-PSE/car_hacking_data_50.csv'
    WINDOW_SIZE = 10

    (X_id_tr, X_pay_tr, y_tr), (X_id_te, X_pay_te, y_te), n_ids, id_map = \
        full_preprocessing_pipeline(data_path, window_size=WINDOW_SIZE, sample_rate=0.5)

    model = build_dual_path_ids(n_ids, window_size=WINDOW_SIZE)

    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]

    print("\n4. 双路模型训练开始...")
    history = model.fit(
        [X_id_tr, X_pay_tr], y_tr,
        validation_data=([X_id_te, X_pay_te], y_te),
        epochs=10,
        batch_size=1024,
        callbacks=cb,
        verbose=1
    )

    visualize_results(model, history, id_map, [X_id_te, X_pay_te], y_te)

    print("\n--- 开始单帧推理延迟测试 (Inference Latency) ---")
    single_sample_id = tf.convert_to_tensor(X_id_te[:1], dtype=tf.int32)
    single_sample_pay = tf.convert_to_tensor(X_pay_te[:1], dtype=tf.float32)

    for _ in range(50):
        _ = model([single_sample_id, single_sample_pay], training=False)
    iterations = 1000
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = model([single_sample_id, single_sample_pay], training=False)
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / iterations) * 1000
    print(f" 平均单帧推理延迟: {avg_latency_ms:.4f} 毫秒/帧 (ms/frame)")

    base_ttf = calculate_ttf_metrics(model, [X_id_te, X_pay_te], y_te, window_size=WINDOW_SIZE, avg_can_interval_ms=0.5)
    print(f"\n 【系统最终安全评估】 总实战告警时延 = {base_ttf + avg_latency_ms:.4f} ms")

