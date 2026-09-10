# cognitive_analyzer.py
import numpy as np


class CognitiveStateAnalyzer:
    def __init__(self):
        self.thresholds = {
            'attention': {'high': 70, 'medium': 40, 'low': 0},
            'workload': {'high': 60, 'medium': 30, 'low': 0},
            'drowsiness': {'high': 50, 'medium': 25, 'low': 0}
        }

    def analyze_cognitive_state(self, eeg_data):
        features = self._extract_features(eeg_data)

        attention_score = self._calculate_attention_formula(features)
        workload_score = self._calculate_workload_formula(features)
        drowsiness_score = self._calculate_drowsiness_formula(features)

        state = self._determine_overall_state(attention_score, workload_score, drowsiness_score)
        recommendations = self._generate_recommendations(state, attention_score, drowsiness_score)

        return {
            'attention': int(attention_score),
            'workload': int(workload_score),
            'drowsiness': int(drowsiness_score),
            'state': state,
            'recommendations': recommendations
        }

    def _calculate_attention_formula(self, features):
        beta_power = features.get('avg_beta', 50)
        alpha_power = features.get('avg_alpha', 35)

        if alpha_power < 1:
            alpha_power = 1

        beta_alpha_ratio = (beta_power / alpha_power)
        attention = (beta_alpha_ratio - 0.5) * (100 / 2.5)
        theta_influence = features.get('avg_theta', 40) * 0.3
        attention -= theta_influence

        return max(0, min(100, attention))

    def _calculate_drowsiness_formula(self, features):
        theta_power = features.get('avg_theta', 40)
        alpha_power = features.get('avg_alpha', 35)
        beta_power = features.get('avg_beta', 50)

        if alpha_power > 0:
            theta_alpha_ratio = theta_power / alpha_power
        else:
            theta_alpha_ratio = theta_power

        if beta_power > 0:
            theta_beta_ratio = theta_power / beta_power
        else:
            theta_beta_ratio = theta_power

        drowsiness = (theta_alpha_ratio * 25) + (theta_beta_ratio * 15)
        return max(0, min(100, drowsiness))

    def _calculate_workload_formula(self, features):
        beta_power = features.get('avg_beta', 50)
        alpha_power = features.get('avg_alpha', 35)

        workload_beta = beta_power * 1.2
        alpha_correction = (50 - alpha_power) * 0.5
        workload = workload_beta + alpha_correction

        return max(0, min(100, workload))

    def _extract_features(self, eeg_data):
        features = {}

        for band in ['alpha', 'beta', 'theta', 'delta']:
            band_values = []
            for key, value in eeg_data.items():
                if f' {band} m' in key and isinstance(value, (int, float)):
                    band_values.append(value)

            if band_values:
                features[f'avg_{band}'] = np.mean(band_values)
            else:
                default_values = {'alpha': 35, 'beta': 50, 'theta': 40, 'delta': 2000}
                features[f'avg_{band}'] = default_values[band]

        return features

    def _determine_overall_state(self, attention, workload, drowsiness):
        if drowsiness > 70:
            return "critical_drowsiness"
        elif drowsiness > 50:
            return "high_drowsiness"
        elif attention < 25:
            return "very_low_attention"
        elif attention < 40:
            return "low_attention"
        elif workload > 75:
            return "high_workload"
        elif attention > 70:
            return "high_attention"
        elif attention > 50:
            return "good_attention"
        else:
            return "normal"

    def _generate_recommendations(self, state, attention, drowsiness):
        recommendations = []

        if state == "critical_drowsiness":
            recommendations.extend(["Take a short break", "Practice active listening"])
        elif state == "high_drowsiness":
            recommendations.extend(["Take a short break", "Practice active listening"])
        elif state in ["very_low_attention", "low_attention"]:
            recommendations.extend(["Take a short break", "Practice active listening"])
        elif state == "high_workload":
            recommendations.extend(["Simplify the material", "Give time for comprehension"])
        elif state in ["high_attention", "good_attention"]:
            recommendations.extend(["Continue the lecture", "Encourage active participation"])
        else:
            recommendations.extend(["Continue as is", "Monitor the dynamics"])

        return recommendations