#Librerias 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carga los datos
data = pd.read_csv("D:\\data ware\\Reservas\\hotel_bookings.csv")

# Muesetra información general y valores nulos
print(data.shape)
print(data.head(5))
print(data.info())
print(data.isnull().sum())

# Maneja valores nulos (reemplazar nulos con la moda de la columna)
data['children'] = data['children'].fillna(data['children'].mode()[0])
data['country'] = data['country'].fillna(data['country'].mode()[0])
data['agent'] = data['agent'].fillna(data['agent'].mode()[0])
data['company'] = data['company'].fillna(data['company'].mode()[0])

# Convierte variables categóricas en numéricas usando Label Encoding
label_encoders = {}
categorical_columns = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 
                       'distribution_channel', 'reserved_room_type', 'assigned_room_type', 
                       'deposit_type', 'customer_type', 'reservation_status']

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Convertir 'reservation_status_date' a datetime y luego a un valor numérico (timestamp)
data['reservation_status_date'] = pd.to_datetime(data['reservation_status_date'])
data['reservation_status_date'] = data['reservation_status_date'].apply(lambda x: x.timestamp())

# Selecciona las características y el objetivo
features = ['hotel', 'lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 
            'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 
            'babies', 'meal', 'country', 'market_segment', 'distribution_channel', 
            'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 
            'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type', 
            'agent', 'company', 'days_in_waiting_list', 'customer_type', 
            'adr', 'required_car_parking_spaces', 'total_of_special_requests']

X = data[features]
y = data['is_canceled']

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrena el modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Realiza predicciones con Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evalua el modelo Random Forest
print("Accuracy RF:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix RF:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report RF:\n", classification_report(y_test, y_pred_rf))

# Entrena el modelo SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)

# Realiza predicciones con SVC
y_pred_svc = svc_model.predict(X_test)

# Evaluar el modelo SVC
acc_svc = accuracy_score(y_test, y_pred_svc)
print("Accuracy SVC:", acc_svc)

# Matriz de confusión para SVC
cm_svc = confusion_matrix(y_test, y_pred_svc)
plt.figure(figsize=(10, 6))
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='coolwarm')
plt.title('Matriz de Confusión - SVC')
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.show()

# Visualiza la correlación entre las características
corr_ds = data[features + ['is_canceled']].corr()
top_corr = corr_ds.index
plt.figure(figsize=(20,20))
sns.heatmap(data[top_corr].corr(), annot=True, cmap='coolwarm')
plt.show()

# Crea un gráfico de conteo para la columna 'is_canceled'
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='is_canceled')
plt.title('Distribución de Cancelaciones de Reservas')
plt.xlabel('Cancelación')
plt.ylabel('Conteo')
plt.show()
