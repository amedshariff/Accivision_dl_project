"""
Unit Tests for CNN + LSTM Classifier and Inference Modules
"""
import unittest
import torch
from models.cnn_model import CNNFeatureExtractor
from models.lstm_model import LSTMClassifier
from models.cnn_lstm_model import CNNLSTMModel
from inference.accident_predictor import AccidentPredictor

class TestClassifier(unittest.TestCase):
    def test_cnn_feature_extractor_shape(self):
        cnn = CNNFeatureExtractor(pretrained=True, freeze_backbone=True)
        dummy_frames = torch.zeros(2, 3, 224, 224)  # (B, C, H, W)
        feats = cnn(dummy_frames)
        self.assertEqual(feats.shape, (2, 512))

    def test_lstm_classifier_shape(self):
        lstm = LSTMClassifier(input_dim=512, hidden_dim=128, num_layers=1)
        dummy_seq = torch.zeros(2, 16, 512)  # (B, T, 512)
        logits = lstm(dummy_seq)
        self.assertEqual(logits.shape, (2, 1))

    def test_cnn_lstm_end_to_end(self):
        model = CNNLSTMModel(pretrained_cnn=True, freeze_cnn=True, hidden_dim=64, num_layers=1)
        dummy_video = torch.zeros(1, 4, 3, 224, 224)  # (B=1, T=4, C=3, H=224, W=224)
        prob = model.predict_probability(dummy_video)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_accident_predictor_output(self):
        predictor = AccidentPredictor()
        res = predictor.predict_video("data/raw/videos/accident/mock_accident.mp4")
        self.assertIn("is_accident", res)
        self.assertIn("verdict", res)
        self.assertIn("accident_probability", res)
        self.assertTrue(res["is_accident"])
        self.assertGreaterEqual(res["accident_probability"], 0.50)

if __name__ == "__main__":
    unittest.main()
