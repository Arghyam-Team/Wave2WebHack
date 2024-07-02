from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import water_reservoir, feature_descriptions
import dash_bootstrap_components as dbc



X_train, y_train, X_test, y_test = water_reservoir()
model = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_train, y_train)
explainer = ClassifierExplainer(model, X_test, y_test, cats=['Sex', 'Deck', 'Embarked'], descriptions=feature_descriptions, labels=['Not survived', 'Survived'])
ExplainerDashboard(explainer, model_summary=True, shap_interaction=False, shap_dependence=False, title="Water Reservoir Dashboard", name='Dash', description="This is a dashboard", bootstrap=dbc.themes.CYBORG, whatif=False).run()
