import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

def run_final_research_pipeline():
    # ۱. تنظیمات فایل ورودی
    DATA_FILE = "spanner_dataset_pro.csv"
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ ERROR: {DATA_FILE} not found! Please run data_generator.py first.")
        return

    print(f"🚀 Loading ELITE Dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"📊 Total Samples (Edges) for Analysis: {len(df)}")

    # ۲. آماده‌سازی ویژگی‌های مهندسی شده (Features)
    # ترکیب طول، رتبه نسبی و شکاف زاویه‌ای
    features = ['edge_length', 'relative_rank', 'angular_gap']
    X = df[features]
    y = df['is_in_spanner']

    # ۳. تقسیم داده‌ها به آموزش و تست (۸۰/۲۰)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Training AI with Full Geometric Intelligence...")
    # استفاده از ۱۰۰ درخت و متعادل‌سازی وزن کلاس‌ها برای دقت حداکثری
    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1 # استفاده از تمام هسته‌های CPU برای سرعت بالاتر
    )
    model.fit(X_train, y_train)

    # ۴. پیش‌بینی و تحلیل نتایج
    y_pred = model.predict(X_test)
    
    print("\n" + "💎" * 20)
    print("   FINAL RESEARCH RESULTS (Q1 GRADE)")
    print("💎" * 20)
    print(classification_report(y_test, y_pred))
    print("💎" * 20)

    # ۵. تحلیل اهمیت ویژگی‌ها (بسیار حیاتی برای مقاله)
    importances = model.feature_importances_
    print("\n🔍 Feature Importance Analysis:")
    for name, imp in zip(features, importances):
        print(f"   - {name}: {imp:.4f}")

    # ۶. رسم و ذخیره ماتریس اغتشاش حرفه‌ای
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma')
    plt.xlabel('Predicted by AI')
    plt.ylabel('Actual (Greedy Spanner)')
    plt.title('Final Confusion Matrix: ML vs Computational Geometry')
    plt.savefig('final_research_matrix.png')
    print("\n✅ Visualization saved as 'final_research_matrix.png'")

if __name__ == "__main__":
    run_final_research_pipeline()