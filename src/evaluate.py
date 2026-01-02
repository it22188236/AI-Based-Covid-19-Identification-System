from tensorflow.keras.models import load_model
from src.model import grad_cam, overlay_heatmap
import os

model = load_model('models/trained_model.h5')
test_dir = 'data/processed/test/'
for cls in os.listdir(test_dir):
    for img in os.listdir(os.path.join(test_dir, cls)):
        img_path = os.path.join(test_dir, cls, img)
        img_array = np.expand_dims(cv2.resize(cv2.imread(img_path, 0), (224,224)), axis=0) / 255.0
        pred = model.predict(img_array)
        heatmap = grad_cam(model, img_array)
        overlay = overlay_heatmap(img_path, heatmap)
        cv2.imwrite(f'results/heatmaps/{img}_overlay.png', overlay)
        # Display: plt.imshow(overlay); plt.show() (or in notebook)

# Validation: Manually review or script to log feedback