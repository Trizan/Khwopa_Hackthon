import joblib
import cv2
from skimage.feature import hog
from skimage import exposure

model = joblib.load('fused_svm_model.pkl')  
scaler = joblib.load('fused_scaler.pkl')  

def recognize(unseen_image):
	unseen_image = cv2.resize(unseen_image, (64, 64))

	features, _ = hog(unseen_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
							visualize=True)

	unseen_features = features.flatten()

	unseen_features_scaled = scaler.transform([unseen_features])

	prediction = model.predict(unseen_features_scaled)
	return prediction[0]