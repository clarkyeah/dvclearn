import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from dvclive import Live
from pathlib import Path

# 设置路径
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'raw' / 'train.csv'

def preprocess_data(df):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].copy()
    
    X['Age'].fillna(X['Age'].median(), inplace=True)
    X['Fare'].fillna(X['Fare'].median(), inplace=True)
    
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    
    return X

def main():
    # 读取数据
    df = pd.read_csv(DATA_PATH)
    X = preprocess_data(df)
    y = df['Survived']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 使用 dvclive 记录训练过程
    with Live() as live:
        # 记录参数
        live.log_params({
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': list(X.columns),
            'random_state': 42,
            'n_estimators': 60
        })
        
        # 训练模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 记录指标
        live.log_metric('accuracy', accuracy)
        live.log_metric('precision_0', report['0']['precision'])
        live.log_metric('recall_0', report['0']['recall'])
        live.log_metric('f1_0', report['0']['f1-score'])
        live.log_metric('precision_1', report['1']['precision'])
        live.log_metric('recall_1', report['1']['recall'])
        live.log_metric('f1_1', report['1']['f1-score'])
        
        # 特征重要性可视化
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.bar(feature_importance['feature'], feature_importance['importance'])
        plt.xticks(rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # 记录可视化
        live.log_image('feature_importance.png', plt.gcf())
        plt.close()

        print(f"模型准确率: {accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        print("\n特征重要性:")
        print(feature_importance)

if __name__ == "__main__":
    main() 