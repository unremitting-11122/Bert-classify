import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tqdm import tqdm
from modelscope import snapshot_download  

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_thucnews_data(data_dir, max_samples_per_class=None):
    """加载THUCNews数据集"""
    category_map = {
        '财经': 'finance', 
        '彩票': 'lottery', 
        '房产': 'real_estate', 
        '股票': 'stock', 
        '家居': 'home', 
        '教育': 'education', 
        '科技': 'technology', 
        '社会': 'society', 
        '时尚': 'fashion', 
        '时政': 'politics', 
        '体育': 'sports', 
        '星座': 'constellation', 
        '游戏': 'game', 
        '娱乐': 'entertainment'
    }
    
    texts = []
    labels = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据集目录不存在: {data_dir}")
    
    category_dirs = glob.glob(os.path.join(data_dir, '*'))
    if not category_dirs:
        print(f"警告: 在目录 {data_dir} 中未找到子文件夹")
    
    for category_dir in category_dirs:
        category_name = os.path.basename(category_dir)
        if category_name not in category_map:
            print(f"跳过未知类别: {category_name}")
            continue
            
        label = category_map[category_name]
        print(f"加载 {label} 类别...")
        
        files = glob.glob(os.path.join(category_dir, '*.txt'))
        if not files:
            print(f"警告: 类别 {category_name} 下没有找到txt文件")
            continue
            
        print(f"找到 {len(files)} 个样本")
        
        if max_samples_per_class and len(files) > max_samples_per_class:
            files = files[:max_samples_per_class]
            print(f"限制样本数量为 {max_samples_per_class}")
            
        success_count = 0
        error_count = 0
        
        for file_path in tqdm(files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    title = f.readline().strip()
                    if title:
                        texts.append(title)
                        labels.append(label)
                        success_count += 1
                    else:
                        error_count += 1
            except Exception as e:
                error_count += 1
                print(f"读取文件 {file_path} 时出错: {e}")
        
        print(f"成功加载 {success_count} 个样本，{error_count} 个样本加载失败")
    
    unique_labels = sorted(list(set(labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    labels = [label_to_id[label] for label in labels]
    
    print(f"总共加载了 {len(texts)} 个样本，包含 {len(label_to_id)} 个类别")
    if len(texts) == 0:
        raise ValueError("未能加载任何数据样本，请检查数据集格式和路径")
    
    return texts, labels, label_to_id, id_to_label

def train_model(model, train_dataloader, val_dataloader, optimizer, device, epochs):
    """训练模型"""
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                
                logits = outputs.logits
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}')
        print('-' * 50)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_bert_thucnews_model.pth')
            print(f'Saved best model with accuracy: {best_val_acc:.4f}')
    
    plot_training_history(history)
    
    return model, all_labels, all_predictions

def plot_training_history(history):
    """绘制训练和验证损失、准确率曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('thucnews_training_history.png')
    plt.close()

def evaluate_model(model, test_dataloader, device, id_to_label):
    """评估模型"""
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    true_labels = [id_to_label[label_id] for label_id in all_labels]
    pred_labels = [id_to_label[label_id] for label_id in all_predictions]
    
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels))
    
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Accuracy: {accuracy:.4f}")
    
    plot_confusion_matrix(true_labels, pred_labels, list(id_to_label.values()))
    
    return accuracy

def plot_confusion_matrix(true_labels, pred_labels, labels):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('thucnews_confusion_matrix.png')
    plt.close()

def predict_category(text, model, tokenizer, label_to_id, id_to_label, device, max_length=128):
    """预测新闻标题的类别"""
    model.eval()
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
    
    predicted_label = id_to_label[predicted_class.item()]
    confidence_value = confidence.item()
    
    return {
        'predicted_category': predicted_label,
        'confidence': confidence_value,
        'probabilities': {id_to_label[i]: probabilities[0][i].item() for i in range(len(id_to_label))}
    }

def main():
    # 设置参数
    max_length = 64  
    batch_size = 32
    learning_rate = 1e-6
    epochs = 10
    data_dir = 'THUCNews'  # 请替换为实际数据集路径
    max_samples_per_class = 2000  
    model_cache_path = "./autod_tmp/new_folder"  # 指定 ModelScope 模型下载路径

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    print("加载数据集...")
    texts, labels, label_to_id, id_to_label = load_thucnews_data(data_dir, max_samples_per_class)
    
    # 划分数据集
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    print(f"测试集大小: {len(test_texts)}")
    
    
    # 加载 tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=len(label_to_id),
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)
    # ------------------- 修改结束 -------------------
    
    # 创建数据加载器
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("开始训练模型...")
    model, _, _ = train_model(model, train_dataloader, val_dataloader, optimizer, device, epochs)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_bert_thucnews_model.pth'))
    model.to(device)
    
    # 评估模型
    print("在测试集上评估模型...")
    evaluate_model(model, test_dataloader, device, id_to_label)

    print("开始导出ONNX模型...")
    onnx_path = "bert_classifier.onnx"
    dummy_input_ids = torch.zeros(1, max_length, dtype=torch.long, device=device)
    dummy_attention_mask = torch.zeros(1, max_length, dtype=torch.long, device=device)
    
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    
    print(f"ONNX模型已导出至: {onnx_path}")
    
    
if __name__ == "__main__":
    main()
