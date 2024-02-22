from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class DecisionTree_CRS:
    def __init__(self, training_data):
        self.clf = self._build_pipeline()
        self.features, self.labels = self._extract_features_labels(training_data)
        self._train_model()

    def _extract_features_labels(self, data):
        features = [
            [
                float(entry['reviews_count']) if entry['reviews_count'] else 0.0,
                entry['fuel_type'],
                float(entry['engine_displacement']) if entry['engine_displacement'] else 0.0,
                float(entry['no_cylinder']) if entry['no_cylinder'] else 0.0,
                float(entry['seating_capacity']) if entry['seating_capacity'] else 0.0,
                entry['transmission_type'],
                float(entry['fuel_tank_capacity']) if entry['fuel_tank_capacity'] else 0.0,
                entry['body_type'],
                float(entry['rating']) if entry['rating'] else 0.0,
                float(entry['starting_price']) if entry['starting_price'] else 0.0,
                float(entry['ending_price']) if entry['ending_price'] else 0.0,
                float(entry['max_torque_nm']) if entry['max_torque_nm'] else 0.0,
                float(entry['max_torque_rpm']) if entry['max_torque_rpm'] else 0.0,
                float(entry['max_power_bhp']) if entry['max_power_bhp'] else 0.0,
                float(entry['max_power_rp']) if entry['max_power_rp'] else 0.0,
            ]
            for entry in data
        ]
        labels = [entry['car_name'] for entry in data]
        return features, labels

    def _build_pipeline(self):
        # Define features and their types
        categorical_features = [1, 5, 7]  # Indices of categorical features
        numeric_features = [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14]  # Indices of numerical features

        # Create transformers for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('num', 'passthrough')
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create the pipeline with the preprocessor and the classifier
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', DecisionTreeClassifier())])
        return clf

    def _train_model(self):
        self.clf.fit(self.features, self.labels)

    def get_recommendation(self, user_input):
        recommendation = self.clf.predict(user_input)
        return recommendation[0]
