import numpy as np                    # للعمليات الرياضية والمصفوفات
import pandas as pd                   # لقراءة البيانات ومعالجتها

import matplotlib.pyplot as plt       # لرسم المخططات البيانية
import seaborn as sns                 # لتحسين الرسومات البيانية

from sklearn.model_selection import train_test_split  # لتقسيم البيانات

from sklearn.metrics import classification_report, confusion_matrix
# classification_report : تقرير أداء النموذج
# confusion_matrix      : مصفوفة الأخطاء

from sklearn.ensemble import RandomForestClassifier
# نموذج Random Forest للتصنيف

from sklearn.metrics import accuracy_score
# لحساب دقة النموذج

# 1. قراءة ملف البيانات
data_set = pd.read_csv('iris_dataset.csv')

# 2. تحديد المدخلات (Features)
x = data_set.iloc[:, 0:4].values

# 3. تحديد المخرجات (Target)
y = data_set.iloc[:, -1].values

# 4. طباعة أبعاد بيانات الإدخال
print("x shape:", x.shape)

# 5. طباعة أبعاد بيانات الإخراج
print("y shape:", y.shape)

# 6. عرض بيانات الإدخال كاملة
print("x =\n", x)

# 7. عرض بيانات الإخراج كاملة
print("y =\n", y)

# 8. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=0
)

# 9. عرض أحجام المجموعات الناتجة
print("\n--- Train/Test Split Shapes ---")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 10. عرض بيانات التدريب
print("\n--- X_train & y_train ---")
print(X_train, y_train)

# 11. عرض بيانات الاختبار
print("\n--- X_test & y_test ---")
print(X_test, y_test)

# 12. إنشاء نموذج Random Forest
clf = RandomForestClassifier(max_depth=2, random_state=0)

# 13. تدريب النموذج على بيانات التدريب
print("\nTraining the model:")
print(clf.fit(X_train, y_train))

# 14. تنفيذ التوقعات على بيانات الاختبار
ll = clf.predict(X_test)

# 15. عرض نتائج التوقع
print("\nModel Predictions (ll):")
print(ll)

# 16. حفظ القيم الحقيقية
y_t = y_test

# 17. حفظ القيم المتوقعة
y_p = ll

# 18. عرض تقرير الأداء النهائي
print("\n--- Final Result ---")
print(classification_report(y_t, y_p))


# =============================================================
# الجزء الخاص بالقراف والرسومات البيانية
# =============================================================

# الرسمة الأولى: مصفوفة الارتباك (Confusion Matrix)
cm = confusion_matrix(y_t, y_p)
plt.figure(figsize=(6, 4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))

plt.title('Iris Classification Results (Confusion Matrix)')
plt.xlabel('Predicted Species (توقع النموذج)')
plt.ylabel('Actual Species (النوع الحقيقي)')
plt.show()  # ستفتح النافذة الأولى، وعند إغلاقها ستفتح الرسمة التالية تلقائياً


# الرسمة الثانية: توزيع الزهور بناءً على قياسات الأوراق (Scatter Plot)
plt.figure(figsize=(7, 5))
# تنبيه: تأكد من أن أسماء الأعمدة في ملف الـ CSV تطابق المكتوب هنا تماماً (Species و Petal measurements)
sns.scatterplot(data=data_set, x='petal length (cm)', y='petal width (cm)', hue='species', palette='Set1')
plt.title('Flower Distribution based on Petal Measurements')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()  # ستفتح النافذة الثانية، وعند إغلاقها ستفتح الرسمة الخطية الثالثة


# -------------------------------------------------------------
# الرسمة الثالثة والأخيرة: منحنى الأداء الخطي 
# -------------------------------------------------------------

# 1. حساب الدقة لتجربة قيم مختلفة لعدد الأشجار
tree_counts = [1, 5, 10, 25, 50, 100, 150, 200]
accuracy_results = []

for n in tree_counts:
    model_test = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)
    model_test.fit(X_train, y_train)
    preds_test = model_test.predict(X_test)
    accuracy_results.append(accuracy_score(y_test, preds_test))

# 2. رسم القراف الخطي الشبكي
plt.figure(figsize=(8, 4))
plt.plot(tree_counts, accuracy_results, marker='o', color='#107c41', linewidth=2, linestyle='-')

plt.title('Model Accuracy vs. Number of Trees (Random Forest)', fontsize=12, pad=15)
plt.xlabel('Number of Trees (عدد الأشجار داخل الغابة)')
plt.ylabel('Accuracy Score (مستوى الدقة)')
plt.grid(True, linestyle='--', alpha=0.6)  # لتظهر المربعات الخلفية الشبكية

plt.show()  # ستفتح نافذة الفيجر الثالثة والأخيرة فوراً!